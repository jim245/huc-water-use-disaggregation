"""
Orchestrator script for the spatial disaggregation pipeline.

Runs one case study and year through each stage in order:

1. Data preparation: downloads HUC boundaries and NLCD rasters, builds the template raster, merges the water use
   statistics with HUC geometries, and prepares the case study's ancillary rasters on the template grid.
2. Dasymetric mapping: spreads each HUC total over its cells in proportion to the first ancillary layer.
3. Pycnophylactic interpolation: smooths the dasymetric estimates while preserving each HUC total.
4. Spatial disaggregation: refines the estimates with alternating CNN and random forest iterations, preserving
   each HUC total.
5. Evaluation: compares the disaggregated raster's HUC sums with the statistics.
6. Figure: renders the disaggregated raster.

The template grid, the statistics and their HUC zones are read once here, and stages 2-5 pass arrays to each other.
Each stage still writes its output raster, and stages 2 and 3 read theirs back instead of recomputing it when it
already exists.
"""

import argparse
import time
import warnings

import numpy

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

# NumPy 2.5 deprecations raised inside third-party code (rasterio raster reads)
warnings.filterwarnings(
    "ignore",
    message="Setting the shape on a NumPy array has been deprecated",
    category=DeprecationWarning,
)
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
    bottom_panel_huc2s: list[str],
):
    """
    Orchestrates a full spatial disaggregation pipeline run.

    Parameters:
        case_study (str): The case study to run.
        reprojected_crs (str): The CRS used in all spatial operations.
        cell_size (int): The desired output cell size (in CRS units).
        huc2s (list[str]): The HUC-2 watersheds used in spatial operations.
        huc_level (int): The HUC scale used for statistics, initial estimates and spatial disaggregation.
        year (int): The year to run.
        bottom_panel_huc2s (list[str]): The HUC-2 watersheds shown in the figure's bottom panel; empty to omit it.
    """

    start_time = time.perf_counter()

    # the figure is the last stage, so its inputs are checked before anything runs
    figure_utilities.check_inputs(huc2s, bottom_panel_huc2s)

    # data layer preparation
    download_huc_boundaries.run(huc2s)

    template_raster_path = generate_template_raster.run(
        huc2s, reprojected_crs, cell_size
    )

    download_nlcd_rasters.run(year)

    match case_study:
        case "pws-cu" | "pws-gw" | "pws-sw" | "pws-tot":
            if case_study == "pws-cu":
                merge_pwscu_statistics.run(huc2s, {huc_level}, reprojected_crs, year)
            else:
                merge_pws_statistics.run(
                    case_study,
                    huc2s,
                    {huc_level},
                    reprojected_crs,
                    year,
                )

            prepared_raster_paths = generate_pws_rasters.run(
                huc2s, reprojected_crs, year, template_raster_path
            )

        case "irr":
            merge_irr_statistics.run(huc2s, {huc_level}, reprojected_crs, year)

            prepared_raster_paths = generate_irr_rasters.run(
                huc2s, reprojected_crs, year, template_raster_path
            )

        case _:
            raise Exception(f"Unhandled case study {case_study}!")

    # constrain irrigation disaggregation to ONLY valid cells in layer 0 (NLCD 81/82 - Planted/Cultivated)
    support_layer = 0 if case_study == "irr" else None

    # every stage works on the template grid
    grid = raster_utilities.Grid.from_raster(template_raster_path)
    prepared_layers = [
        raster_utilities.read_raster(prepared_raster_path, grid)
        for prepared_raster_path in prepared_raster_paths
    ]

    # read the statistics and rasterize their HUCs once, so every stage assigns each cell to the same HUC
    statistics = raster_utilities.read_statistics(
        huc2s,
        huc_level,
        case_study,
        year,
        [f"huc{huc_level}", "water_mgd", "geometry"],
    )
    # the raster stores EPSG:5070 as WKT, which to_crs doesn't treat as the same CRS: it would run an identity
    # transform that copies every polygon
    if statistics.crs != grid.crs:
        statistics = statistics.to_crs(grid.crs)
    zones, zone_hucs = raster_utilities.rasterize_zones(
        statistics.geometry,
        statistics[f"huc{huc_level}"],
        grid.shape,
        grid.transform,
    )
    water_use_lookup = dict(
        zip(
            statistics[f"huc{huc_level}"].astype(int),
            statistics["water_mgd"],
        )
    )
    del statistics

    # each zone's total, indexed like zones (index 0 is outside all HUCs)
    zone_totals = numpy.full(len(zone_hucs) + 1, numpy.nan)
    zone_totals[1:] = [water_use_lookup[huc] for huc in zone_hucs]

    # dasymetric mapping
    dasymetric_raster = dasymetric_mapping.run(
        prepared_layers[0],
        zones,
        zone_totals,
        grid,
        case_study,
        year,
        huc_level,
    )

    # pycnophylactic interpolation
    pycnophylactic_raster = pycnophylactic_interpolation.run(
        dasymetric_raster,
        zones,
        zone_totals,
        grid,
        case_study,
        year,
        huc_level,
        support=(prepared_layers[support_layer] if support_layer is not None else None),
    )

    # spatial disaggregation
    disaggregated_raster = disaggregation.run(
        prepared_layers,
        pycnophylactic_raster,
        zones,
        grid,
        case_study,
        year,
        huc_level,
        support_layer,
    )

    # evaluate results
    evaluation.run(
        disaggregated_raster,
        zones,
        zone_hucs,
        huc2s,
        huc_level,
        case_study,
        year,
    )

    # render figure
    figure_utilities.run(
        disaggregated_raster,
        grid,
        bottom_panel_huc2s,
        huc_level,
        case_study,
        year,
    )

    end_time = time.perf_counter()
    print("Time (h):", (end_time - start_time) / 3600)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case_study",
        choices=["irr", "pws-cu", "pws-gw", "pws-sw", "pws-tot"],
        default="pws-tot",
        help="The case study to run, determines input statistics and ancillary data layers.",
    )
    parser.add_argument(
        "--year",
        default=2000,
        type=int,
        help="The year that determines statistics and ancillary data layers.",
    )
    parser.add_argument(
        "--huc2s",
        nargs="+",
        default=[
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "10",
            "11",
            "12",
            "13",
            "14",
            "15",
            "16",
            "17",
            "18",
        ],  # CONUS
        help="The HUC-2 watersheds used for all spatial operations and calculations.",
    )
    parser.add_argument(
        "--huc_level",
        choices=[2, 4, 6, 8, 10, 12, 14, 16],
        default=12,
        type=int,
        help="The HUC scale used for statistics, initial estimates and spatial disaggregation.",
    )
    parser.add_argument(
        "--crs",
        default="EPSG:5070",
        help="The coordinate reference system used in all spatial operations.",
    )
    parser.add_argument(
        "--cell_size",
        default=1000,
        type=int,
        help="The desired output cell size (in CRS units).",
    )
    parser.add_argument(
        "--bottom_panel_huc2s",
        nargs="*",
        default=["14", "15"],
        help="The HUC-2 watersheds shown in the figure's bottom panel, any subset of --huc2s; pass none to omit it.",
    )
    args = parser.parse_args()
    main(
        args.case_study,
        args.crs,
        args.cell_size,
        args.huc2s,
        args.huc_level,
        args.year,
        args.bottom_panel_huc2s,
    )
