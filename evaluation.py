"""
Evaluates disaggregated rasters against aggregated original totals.

The disaggregated raster is summed over each HUC and compared with that HUC's water use statistic. Disaggregation
rescales every iteration to the totals at its own HUC level, and run() scores at that same level, so its scores are
close to perfect by construction. They show that totals are preserved, not how well water is placed within each HUC;
that needs scoring at a finer level (e.g. HUC-12 for a HUC-8 run; see FIXES.md, "HUC-12 accuracy of the final
product").
"""

import argparse
import os
from os import path
from typing import cast

import geopandas
import numpy
import pandas
from numpy import typing as numpy_typing
from sklearn import metrics

from utilities import raster_utilities


def compute_zonal_sums(
    predicted_data: numpy_typing.NDArray[numpy.float32],
    zones: numpy_typing.NDArray[numpy.int32],
    zone_hucs: numpy_typing.NDArray[numpy.int64],
    actual_values: geopandas.GeoDataFrame,
    huc_level: int,
) -> pandas.DataFrame:
    """
    Calculate predicted zonal water use sums.

    Cells are assigned to HUCs by the zones disaggregation preserved totals over (one HUC per cell), so the sums are the
    totals the pipeline preserved.

    Parameters:
        predicted_data (numpy_typing.NDArray[numpy.float32]): The predicted values, NaN as nodata.
        zones (numpy_typing.NDArray[numpy.int32]): The huc_level zone index raster, 0 outside all HUCs.
        zone_hucs (numpy_typing.NDArray[numpy.int64]): The HUC ID for each zone index, offset by one.
        actual_values (geopandas.GeoDataFrame): The huc_level statistics, with a huc<huc_level> ID column and water_mgd.
        huc_level (int): The HUC level to sum over.

    Returns:
        predicted_values (pandas.DataFrame): actual_values' huc<huc_level> IDs, in the same order, with water_mgd the
            raster's sum over each HUC; NaN for HUCs with no valid cells. Geometry is left out, so the national HUC-12
            polygons aren't copied (see FIXES.md, fix 29).
    """

    zone_count = len(zone_hucs) + 1

    valid = ~numpy.isnan(predicted_data)

    zone_sums = numpy.bincount(
        zones[valid], weights=predicted_data[valid], minlength=zone_count
    )
    zone_cell_counts = numpy.bincount(zones[valid], minlength=zone_count)

    # index 0 is outside all HUCs; HUCs without valid cells get NaN, so they're left out of the metrics
    huc_sums = pandas.Series(
        numpy.where(zone_cell_counts[1:] > 0, zone_sums[1:], numpy.nan),
        index=zone_hucs,
    )
    predicted_values = pandas.DataFrame(
        {
            f"huc{huc_level}": actual_values[f"huc{huc_level}"].values,
            "water_mgd": actual_values[f"huc{huc_level}"]
            .astype(numpy.int64)
            .map(huc_sums)
            .values,
        }
    )

    return predicted_values


def evaluate_zonal_sums(
    actual_values: geopandas.GeoDataFrame,
    predicted_zonal_sums: pandas.DataFrame,
    huc_level: int,
    case_study: str,
    year: int,
) -> dict[str, float]:
    """
    Calculate metrics from actual and predicted zonal sum values, and write the per-HUC comparison to
    output/huc<huc_level>/<case_study>/<year>/ as comparison.csv and comparison.gpkg.

    NRMSE and NMAE are normalized by the range (max - min) of the actual values. HUCs where either value is NaN are
    left out of the metrics.

    Water the map never places (see FIXES.md, fix 28) is reported separately: Unpredicted_HUCs and Unpredicted_mgd
    count the HUCs with actual water use but no valid predicted cells (the rows left out above), and Missing_mgd /
    Missing_pct the total actual minus the total predicted, which also counts water lost inside HUCs that still have
    a prediction.

    Parameters:
        actual_values (geopandas.GeoDataFrame): The actual zonal sum values, with the HUC polygons.
        predicted_zonal_sums (pandas.DataFrame): The predicted zonal sum values, without geometry.
        huc_level (int): The HUC level to use for zones.
        case_study (str): The case study to evaluate.
        year (int): The year to evaluate.

    Returns:
        metrics (dict[str, float]): The year's RMSE, MAE, NRMSE, NMAE, R2, Unpredicted_HUCs, Unpredicted_mgd,
            Missing_mgd and Missing_pct.
    """

    # only actual_values carries geometry, so the comparison holds a single copy of the polygons
    comparison = actual_values.merge(
        predicted_zonal_sums,
        on=f"huc{huc_level}",
        suffixes=("_actual", "_predicted"),
    )

    actual = cast(
        numpy_typing.NDArray[numpy.float32], comparison["water_mgd_actual"].values
    )
    predicted = cast(
        numpy_typing.NDArray[numpy.float32], comparison["water_mgd_predicted"].values
    )

    # water the map never places: HUCs without valid predicted cells, and the overall shortfall
    unpredicted = numpy.isnan(predicted) & (actual > 0)
    unpredicted_hucs = int(unpredicted.sum())
    unpredicted_mgd = float(actual[unpredicted].sum())
    actual_total = float(numpy.nansum(actual, dtype=numpy.float64))
    missing_mgd = actual_total - float(numpy.nansum(predicted, dtype=numpy.float64))
    missing_pct = 100.0 * missing_mgd / actual_total if actual_total != 0 else numpy.nan

    # Drop rows where either value is NaN
    mask = ~(numpy.isnan(actual) | numpy.isnan(predicted))
    actual = actual[mask]
    predicted = predicted[mask]

    root_mean_squared_error = numpy.sqrt(metrics.mean_squared_error(actual, predicted))
    mean_absolute_error = metrics.mean_absolute_error(actual, predicted)
    rng = numpy.ptp(actual)
    nrmse = root_mean_squared_error / rng if rng != 0 else numpy.nan
    nmae = mean_absolute_error / rng if rng != 0 else numpy.nan
    r2 = metrics.r2_score(actual, predicted)

    print(f"RMSE  : {root_mean_squared_error:.4f}")
    print(f"MAE   : {mean_absolute_error:.4f}")
    print(f"NRMSE : {nrmse:.4f}")
    print(f"NMAE  : {nmae:.4f}")
    print(f"R^2   : {r2:.4f}")
    print(
        f"Unpredicted : {unpredicted_hucs} HUCs with water, {unpredicted_mgd:.4f} mgd"
    )
    print(f"Missing     : {missing_mgd:.4f} mgd ({missing_pct:.4f}% of actual)")

    output_directory = path.join("output", f"huc{huc_level}", case_study, str(year))

    # False for HUCs with no valid predicted cells, so they can be mapped
    comparison["predicted"] = comparison["water_mgd_predicted"].notna()

    # predicted / actual; inf or NaN for HUCs with no actual water use
    comparison["ratio"] = (
        comparison["water_mgd_predicted"] / comparison["water_mgd_actual"]
    )

    comparison = comparison[
        [
            f"huc{huc_level}",
            "water_mgd_predicted",
            "water_mgd_actual",
            "predicted",
            "ratio",
            "geometry",
        ]
    ]

    os.makedirs(output_directory, exist_ok=True)

    # geometry and ratio are left out of the CSV (geometry was ~1 GB of WKT text); both are in the gpkg below
    comparison.drop(columns=["geometry", "ratio"]).to_csv(
        path.join(output_directory, "comparison.csv"),
        index=False,
    )

    comparison.to_file(path.join(output_directory, "comparison.gpkg"))

    return {
        "RMSE": float(root_mean_squared_error),
        "MAE": float(mean_absolute_error),
        "NRMSE": float(nrmse),
        "NMAE": float(nmae),
        "R2": float(r2),
        "Unpredicted_HUCs": float(unpredicted_hucs),
        "Unpredicted_mgd": unpredicted_mgd,
        "Missing_mgd": missing_mgd,
        "Missing_pct": float(missing_pct),
    }


def update_metrics_summary(
    metrics: dict[str, float], huc_level: int, case_study: str, year: int
):
    """
    Insert or replace a year's metrics in the case study summary CSV and recompute the average row.

    Parameters:
        metrics (dict[str, float]): The metrics for the year.
        huc_level (int): The HUC level the metrics were computed at.
        case_study (str): The case study the metrics belong to.
        year (int): The year the metrics belong to.
    """

    summary_path = path.join("output", f"huc{huc_level}", case_study, "metrics.csv")
    row = pandas.DataFrame([{"Year": str(year), **metrics}])

    # years are run separately, so merge with rows written by earlier runs
    if path.exists(summary_path):
        summary = pandas.read_csv(summary_path, dtype={"Year": str})
        summary = summary[~summary["Year"].isin([str(year), "Average"])]
        summary = pandas.concat([summary, row], ignore_index=True)
    else:
        summary = row

    summary = summary.sort_values("Year", key=lambda years: years.astype(int))

    average = summary.drop(columns="Year").mean().to_frame().T
    average.insert(0, "Year", "Average")

    pandas.concat([summary, average], ignore_index=True).to_csv(
        summary_path, index=False
    )


def run(
    predicted_data: numpy_typing.NDArray[numpy.float32],
    zones: numpy_typing.NDArray[numpy.int32],
    zone_hucs: numpy_typing.NDArray[numpy.int64],
    huc2s: list[str],
    huc_level: int,
    case_study: str,
    year: int,
):
    """
    Evaluate a disaggregated raster using zonal statistics.

    Writes the per-HUC comparison files (see evaluate_zonal_sums) and updates
    output/huc<huc_level>/<case_study>/metrics.csv.

    Parameters:
        predicted_data (numpy_typing.NDArray[numpy.float32]): The disaggregated raster, NaN as nodata.
        zones (numpy_typing.NDArray[numpy.int32]): The huc_level zone index raster disaggregation used, 0 outside all
            HUCs.
        zone_hucs (numpy_typing.NDArray[numpy.int64]): The HUC ID for each zone index, offset by one.
        huc2s (list[str]): The HUC-2s to evaluate.
        huc_level (int): The HUC level to evaluate.
        case_study (str): The case study to evaluate.
        year (int): The year to evaluate.
    """

    actual_values = raster_utilities.read_statistics(
        huc2s,
        huc_level,
        case_study,
        year,
        [f"huc{huc_level}", "water_mgd", "geometry"],
    )
    predicted_zonal_sums = compute_zonal_sums(
        predicted_data, zones, zone_hucs, actual_values, huc_level
    )
    metrics = evaluate_zonal_sums(
        actual_values, predicted_zonal_sums, huc_level, case_study, year
    )
    update_metrics_summary(metrics, huc_level, case_study, year)


def rescore(
    huc2s: list[str] = ["14", "15"],
    huc_level: int = 12,
    case_study: str = "pws-tot",
    year: int = 2000,
):
    """
    Evaluate an existing output/huc<huc_level>/<case_study>/<year>/<case_study>_huc<huc_level>_<year>.tif outside the
    pipeline, rebuilding the zones main.py would pass to run().

    Parameters:
        huc2s (list[str] = ["14", "15"]): The HUC-2s to evaluate.
        huc_level (int = 12): The HUC level to evaluate.
        case_study (str = "pws-tot"): The case study to evaluate.
        year (int = 2000): The year to evaluate.
    """

    predicted_raster_path = path.join(
        "output",
        f"huc{huc_level}",
        case_study,
        str(year),
        f"{case_study}_huc{huc_level}_{year}.tif",
    )
    grid = raster_utilities.Grid.from_raster(predicted_raster_path)

    zone_polygons = raster_utilities.read_statistics(
        huc2s, huc_level, case_study, year, [f"huc{huc_level}", "geometry"]
    )
    # the raster stores EPSG:5070 as WKT, which to_crs doesn't treat as the same CRS: it would run an identity
    # transform that copies every polygon (~8 GB at CONUS HUC-12; see FIXES.md, fix 29)
    if zone_polygons.crs != grid.crs:
        zone_polygons = zone_polygons.to_crs(grid.crs)
    zones, zone_hucs = raster_utilities.rasterize_zones(
        zone_polygons.geometry,
        zone_polygons[f"huc{huc_level}"],
        grid.shape,
        grid.transform,
    )
    del zone_polygons

    run(
        raster_utilities.read_raster(predicted_raster_path, grid),
        zones,
        zone_hucs,
        huc2s,
        huc_level,
        case_study,
        year,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rescore existing disaggregated rasters and update output/huc<huc_level>/<case_study>/metrics.csv."
    )
    parser.add_argument(
        "--case_study",
        choices=["irr", "pws-cu", "pws-gw", "pws-sw", "pws-tot"],
        default="pws-tot",
        help="The case study to evaluate.",
    )
    parser.add_argument(
        "--years",
        nargs="+",
        default=[2000],
        type=int,
        help="The years to evaluate; each needs output/huc<huc_level>/<case_study>/<year>/<case_study>_huc<huc_level>_<year>.tif.",
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
        help="The HUC-2 watersheds to evaluate.",
    )
    parser.add_argument(
        "--huc_level",
        choices=[2, 4, 6, 8, 10, 12, 14, 16],
        default=12,
        type=int,
        help="The HUC scale the raster was disaggregated at, and is evaluated at.",
    )
    args = parser.parse_args()
    for year in args.years:
        rescore(args.huc2s, args.huc_level, args.case_study, year)
