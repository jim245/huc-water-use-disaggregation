"""
Runs the spatial disaggregation pipeline for one case study and one year.

The pipeline reads water use totals per Hydrologic Unit Code (HUC) watershed. It writes a raster of water use per grid
cell, in million gallons per day (mgd). Run this script from the repository root, because all paths are relative to
the root.

Case studies (settings in raster_utilities.CASE_STUDIES):
    irr: Irrigation water consumption.
    pws-cu: Public water supply water consumption.
    pws-gw: Public water supply groundwater use.
    pws-sw: Public water supply surface water use.
    pws-tot: Public water supply total water use.

Terms:
    statistics: The reported water use total of each HUC, in mgd.
    template grid: The cell grid of all rasters in a run.
    ancillary layer: A raster of predictor data on the template grid, for example land cover.
    zone: The cells of the template grid that belong to one HUC. Each cell belongs to at most one zone.
    support: The cells that can receive water. Only irr uses a support limit.

Stages:
    1. Data preparation: Downloads the HUC boundaries, the NLCD rasters and the state boundaries for the figure.
       Builds the template raster. Merges the statistics with the HUC geometries. Prepares the ancillary layers of the
       case study on the template grid.
    2. Dasymetric mapping: Spreads each HUC total over its cells in proportion to the first ancillary layer.
    3. Pycnophylactic interpolation: Smooths the dasymetric estimates and rescales them to the HUC totals.
    4. Spatial disaggregation: Refines the estimates with alternating CNN and random forest iterations. Each iteration
       rescales the estimates to the HUC totals.
    5. Evaluation: Compares the HUC sums of the disaggregated raster with the statistics. A run below HUC-12 also
       scores its raster against the HUC-12 statistics (the holdout score).
    6. Figure: Renders the disaggregated raster.

This script reads the template grid once and rasterizes the HUC zones once, so all stages use the same zones. Stages
2-6 receive arrays from this script. Stage 5 reads the statistics again, for its comparison files. Stages 2-4 write
their output rasters, stage 5 writes comparison files and metrics, and stage 6 writes an image.

Stages 2-4 keep each HUC total only if the HUC has cells that can hold its water. A HUC with no such cells loses its
total, and no later stage restores it. Evaluation reports this water as missing.

Inputs: The pipeline downloads only the HUC boundaries, the NLCD rasters and the state boundaries. All other inputs
must be in input/ before the run.

HUC levels: One run uses one HUC level for all stages. Earlier runs used HUC-12 initial estimates and HUC-8
disaggregation. In tests on irr 2000 and pws-tot 2000, that setup lost the HUC-12 totals. Its placement scores at
HUC-12 were about equal to those of a run at level 8.
    12: The default level and the finest level. All source statistics have 12-digit HUC IDs.
    8: The holdout level. Run each case study and year at level 12 for the product, and at level 8 for the error
       estimate. Stages 1-4 of a level 8 run don't read the HUC-12 statistics, so the holdout score is out of sample.
       It measures placement from HUC-8 to HUC-12. It doesn't measure the cell error of the level 12 product.
    2, 4, 6 and 10: Available but not tested. The merge scripts sum the HUC-12 statistics to these levels.
"""

import argparse
import time
import warnings
from pathlib import Path

import dasymetric_mapping
import disaggregation
import evaluation
import pycnophylactic_interpolation
from utilities import (
    download_huc_boundaries,
    download_nlcd_rasters,
    figure_utilities,
    generate_irr_rasters,
    generate_pws_rasters,
    generate_template_raster,
    merge_irr_statistics,
    merge_pws_statistics,
    merge_pwscu_statistics,
    raster_utilities,
)

# rasterio 1.5.0 raises this in its compiled read() under NumPy 2.5, at each raster read. It has no effect now, but a
# later NumPy may turn it into an error
warnings.filterwarnings(
    "ignore",
    message="Setting the shape on a NumPy array has been deprecated",
    category=DeprecationWarning,
)
# Keras internals raise this under NumPy 2
warnings.filterwarnings(
    "ignore",
    message="__array__ implementation doesn't accept a copy keyword",
    category=DeprecationWarning,
)


def main(
    case_study: str,
    reprojected_crs: str,
    cell_size: int,
    huc2s: list[str],
    huc_level: int,
    year: int,
    detail_huc2s: list[str],
) -> None:
    """
    Runs all pipeline stages for one case study and one year.

    The module docstring describes the stages and the HUC levels. Stages 2-6 write their outputs to
    output/huc<huc_level>/<case_study>/<year>/, except metrics.csv, which is in output/huc<huc_level>/<case_study>/.

    Parameters:
        case_study (str): The case study to run. It must be a key of raster_utilities.CASE_STUDIES.
        reprojected_crs (str): The CRS of all spatial operations.
        cell_size (int): The output cell size, in CRS units.
        huc2s (list[str]): The HUC-2 watersheds of the run, without zero-padding (for example "1", not "01").
        huc_level (int): The HUC level of the statistics, the initial estimates, the disaggregation and the evaluation.
        year (int): The year to run.
        detail_huc2s (list[str]): The HUC-2 watersheds that the detail view of the figure zooms in on. They must be
            in huc2s. An empty list omits the detail view.
    """

    start_time = time.perf_counter()

    settings = raster_utilities.CASE_STUDIES[case_study]

    # stage 1: data preparation
    download_huc_boundaries.run(huc2s)

    template_raster_path = generate_template_raster.run(
        huc2s, reprojected_crs, cell_size
    )

    download_nlcd_rasters.run(year)

    # the figure is the last stage, so its download runs here. Otherwise a failed download would stop the run after
    # all other stages
    figure_utilities.download_state_boundaries()

    # the merge scripts accept a set of HUC levels. A run below HUC-12 also needs the HUC-12 statistics, which only
    # the holdout score reads
    merge_levels = {huc_level, raster_utilities.STATISTICS_HUC_LEVEL}
    match case_study:
        case "pws-cu" | "pws-gw" | "pws-sw" | "pws-tot":
            # pws-cu has its own script because its source values are monthly volumes, not monthly MGD
            if case_study == "pws-cu":
                merge_pwscu_statistics.run(
                    huc2s=huc2s,
                    huc_levels=merge_levels,
                    reprojected_crs=reprojected_crs,
                    year=year,
                )
            else:
                merge_pws_statistics.run(
                    case_study=case_study,
                    huc2s=huc2s,
                    huc_levels=merge_levels,
                    reprojected_crs=reprojected_crs,
                    year=year,
                )

            prepared_raster_paths = generate_pws_rasters.run(
                huc2s=huc2s,
                reprojected_crs=reprojected_crs,
                year=year,
                template_raster_path=template_raster_path,
            )

        case "irr":
            merge_irr_statistics.run(
                huc2s=huc2s,
                huc_levels=merge_levels,
                reprojected_crs=reprojected_crs,
                year=year,
            )

            prepared_raster_paths = generate_irr_rasters.run(
                huc2s=huc2s,
                reprojected_crs=reprojected_crs,
                year=year,
                template_raster_path=template_raster_path,
                huc_level=huc_level,
            )

        case _:
            # a case study in CASE_STUDIES has no branch here
            raise ValueError(f"Unhandled case study {case_study}!")

    # every stage works on the template grid, and read_raster raises an error for a raster on another grid
    grid = raster_utilities.Grid.from_raster(template_raster_path)
    prepared_layers = [
        raster_utilities.read_raster(prepared_raster_path, grid)
        for prepared_raster_path in prepared_raster_paths
    ]

    # read the statistics and rasterize their HUCs once, so every stage assigns each cell to the same HUC
    statistics = raster_utilities.read_statistics(
        huc2s=huc2s,
        huc_level=huc_level,
        case_study=case_study,
        year=year,
        columns=[f"huc{huc_level}", "water_mgd", "geometry"],
    )
    zones, zone_hucs, zone_totals = raster_utilities.build_zones(
        statistics, huc_level, grid
    )
    # later stages in this function need only the IDs and the totals, so this line releases the geometries
    del statistics

    # stage 2: dasymetric mapping. Layer 0 is the weight layer: NLCD planted/cultivated (irr) or NLCD developed (pws)
    dasymetric_raster = dasymetric_mapping.run(
        prepared_layers[0],
        zones,
        zone_totals,
        grid,
        case_study=case_study,
        year=year,
        huc_level=huc_level,
    )

    # stage 3: pycnophylactic interpolation
    pycnophylactic_raster = pycnophylactic_interpolation.run(
        dasymetric_raster,
        zones,
        zone_totals,
        grid,
        case_study=case_study,
        year=year,
        huc_level=huc_level,
        support=(
            prepared_layers[settings.support_layer]
            if settings.support_layer is not None
            else None
        ),
    )

    # stage 4: spatial disaggregation
    disaggregated_raster = disaggregation.run(
        prepared_layers,
        pycnophylactic_raster,
        zones,
        grid,
        case_study=case_study,
        year=year,
        huc_level=huc_level,
        support_layer=settings.support_layer,
    )

    # stage 5: evaluation. Each iteration rescales to these same zones, so scores near 1 show that the run keeps the
    # totals. They don't measure the placement of water within each HUC. The holdout score below measures placement
    evaluation.run(
        disaggregated_raster,
        zones,
        zone_hucs,
        huc2s,
        huc_level=huc_level,
        case_study=case_study,
        year=year,
    )

    if huc_level < raster_utilities.STATISTICS_HUC_LEVEL:
        evaluation.score_holdout(
            disaggregated_raster,
            pycnophylactic_raster,
            grid,
            huc2s,
            huc_level=huc_level,
            case_study=case_study,
            year=year,
        )

    # stage 6: figure. It runs after evaluation, so a plotting error can't lose the metrics
    figure_utilities.run(
        disaggregated_raster,
        grid,
        detail_huc2s,
        huc_level=huc_level,
        case_study=case_study,
        year=year,
    )

    end_time = time.perf_counter()
    print("Time (h):", (end_time - start_time) / 3600)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case_study",
        choices=list(raster_utilities.CASE_STUDIES),
        default="pws-tot",
        help="The case study to run. It selects the statistics and the ancillary layers.",
    )
    parser.add_argument(
        "--year",
        default=2000,
        type=int,
        help="The year to run. It selects the statistics and the ancillary layers. pws-cu covers 2009-2020, and all "
        "other case studies cover 2000-2020.",
    )
    parser.add_argument(
        "--huc2s",
        nargs="+",
        choices=raster_utilities.HUC2S,
        default=list(raster_utilities.HUC2S),
        metavar="HUC2",
        help='The HUC-2 watersheds of the run, from 1 to 18, without zero-padding (for example "1", not "01"). The '
        "default is all 18 (CONUS).",
    )
    parser.add_argument(
        "--huc_level",
        choices=raster_utilities.HUC_LEVELS,
        default=12,
        type=int,
        help="The HUC level of all stages. The source statistics stop at level 12. The project tested only levels 8 "
        "and 12.",
    )
    parser.add_argument(
        "--crs",
        default="EPSG:5070",
        help="The coordinate reference system of all spatial operations.",
    )
    parser.add_argument(
        "--cell_size",
        default=1000,
        type=int,
        help="The output cell size, in CRS units.",
    )
    parser.add_argument(
        "--detail_huc2s",
        nargs="*",
        default=["14", "15"],
        metavar="HUC2",
        help="The HUC-2 watersheds that the detail view of the figure zooms in on. They must be in --huc2s, so a run "
        "without 14 and 15 must set this option. Pass no values to omit the detail view.",
    )
    args = parser.parse_args()

    # all paths are relative to the repository root. From another folder, stage 1 would download several GB into a new
    # input/ there before a missing file stops the run
    repository_root = Path(__file__).resolve().parent
    if Path.cwd().resolve() != repository_root:
        parser.error(f"run main.py from the repository root, {repository_root}.")

    # the valid years depend on --case_study, so choices= can't express them. parser.error stops before any stage runs
    settings = raster_utilities.CASE_STUDIES[args.case_study]
    if not settings.first_year <= args.year <= settings.last_year:
        parser.error(
            f"--year: {args.case_study} statistics cover {settings.first_year}-{settings.last_year}, not {args.year}."
        )

    # choices= accepts a repeated value, and a repeated HUC-2 would count its statistics twice in evaluation
    if len(set(args.huc2s)) != len(args.huc2s):
        parser.error("--huc2s: a HUC-2 is repeated.")

    # only the run's HUC-2s are on the grid, so the detail view of another HUC-2 would be empty
    missing_huc2s = sorted(set(args.detail_huc2s) - set(args.huc2s))
    if missing_huc2s:
        parser.error(f"--detail_huc2s: {missing_huc2s} aren't in --huc2s.")

    main(
        case_study=args.case_study,
        reprojected_crs=args.crs,
        cell_size=args.cell_size,
        huc2s=args.huc2s,
        huc_level=args.huc_level,
        year=args.year,
        detail_huc2s=args.detail_huc2s,
    )
