"""
Evaluates disaggregated rasters against aggregated original totals.

The disaggregated raster is summed over each HUC and compared with that HUC's water use statistic. Disaggregation
rescales every iteration to the totals at its own HUC level, and run() scores at that same level, so its scores are
close to perfect by construction. They show that totals are preserved, not how well water is placed within each HUC.

score_holdout measures placement. A run below HUC-12 never sees the HUC-12 totals, so its HUC-12 sums are an
out-of-sample test. The score covers one step of detail, for example HUC-8 to HUC-12. It is a proxy, and it doesn't
measure the cell error of a HUC-12 run, because no reference data exists at the cell level.
"""

import argparse
import os
from os import path
from typing import Optional, cast

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


def compute_metrics(
    actual: numpy_typing.NDArray[numpy.float64],
    predicted: numpy_typing.NDArray[numpy.float64],
) -> dict[str, float]:
    """
    Calculate metrics from actual and predicted per-HUC values.

    NRMSE and NMAE are normalized by the range (max - min) of the actual values. HUCs where either value is NaN are
    left out of the metrics.

    Water the map never places (see FIXES.md, fix 28) is reported separately: Unpredicted_HUCs and Unpredicted_mgd
    count the HUCs with actual water use but no valid predicted cells (the rows left out above), and Missing_mgd /
    Missing_pct the total actual minus the total predicted, which also counts water lost inside HUCs that still have
    a prediction.

    Parameters:
        actual (numpy_typing.NDArray[numpy.float64]): The actual value of each HUC.
        predicted (numpy_typing.NDArray[numpy.float64]): The predicted value of each HUC, in the same order.

    Returns:
        metrics (dict[str, float]): The RMSE, MAE, NRMSE, NMAE, R2, Unpredicted_HUCs, Unpredicted_mgd, Missing_mgd and
            Missing_pct.
    """

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


def evaluate_zonal_sums(
    actual_values: geopandas.GeoDataFrame,
    predicted_zonal_sums: pandas.DataFrame,
    huc_level: int,
    case_study: str,
    year: int,
    run_huc_level: Optional[int] = None,
    comparison_name: str = "comparison",
) -> dict[str, float]:
    """
    Calculate metrics from actual and predicted zonal sum values (see compute_metrics), and write the per-HUC
    comparison to output/huc<run_huc_level>/<case_study>/<year>/ as <comparison_name>.csv and <comparison_name>.gpkg.

    Parameters:
        actual_values (geopandas.GeoDataFrame): The actual zonal sum values, with the HUC polygons.
        predicted_zonal_sums (pandas.DataFrame): The predicted zonal sum values, without geometry.
        huc_level (int): The HUC level to use for zones.
        case_study (str): The case study to evaluate.
        year (int): The year to evaluate.
        run_huc_level (Optional[int] = None): The HUC level of the run, which selects the output directory. Equal to
            huc_level when None.
        comparison_name (str = "comparison"): The name of the comparison files.

    Returns:
        metrics (dict[str, float]): The year's metrics (see compute_metrics).
    """

    # only actual_values carries geometry, so the comparison holds a single copy of the polygons
    comparison = actual_values.merge(
        predicted_zonal_sums,
        on=f"huc{huc_level}",
        suffixes=("_actual", "_predicted"),
    )

    year_metrics = compute_metrics(
        cast(
            numpy_typing.NDArray[numpy.float64], comparison["water_mgd_actual"].values
        ),
        cast(
            numpy_typing.NDArray[numpy.float64],
            comparison["water_mgd_predicted"].values,
        ),
    )

    print(f"RMSE  : {year_metrics['RMSE']:.4f}")
    print(f"MAE   : {year_metrics['MAE']:.4f}")
    print(f"NRMSE : {year_metrics['NRMSE']:.4f}")
    print(f"NMAE  : {year_metrics['NMAE']:.4f}")
    print(f"R^2   : {year_metrics['R2']:.4f}")
    print(
        f"Unpredicted : {int(year_metrics['Unpredicted_HUCs'])} HUCs with water, "
        f"{year_metrics['Unpredicted_mgd']:.4f} mgd"
    )
    print(
        f"Missing     : {year_metrics['Missing_mgd']:.4f} mgd "
        f"({year_metrics['Missing_pct']:.4f}% of actual)"
    )

    output_directory = path.join(
        "output",
        f"huc{huc_level if run_huc_level is None else run_huc_level}",
        case_study,
        str(year),
    )

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
        path.join(output_directory, f"{comparison_name}.csv"),
        index=False,
    )

    comparison.to_file(path.join(output_directory, f"{comparison_name}.gpkg"))

    return year_metrics


def update_metrics_summary(
    year_metrics: dict[str, float],
    huc_level: int,
    case_study: str,
    year: int,
    summary_name: str = "metrics.csv",
) -> None:
    """
    Insert or replace a year's metrics in the case study summary CSV and recompute the average row.

    Parameters:
        year_metrics (dict[str, float]): The metrics for the year.
        huc_level (int): The HUC level of the run.
        case_study (str): The case study the metrics belong to.
        year (int): The year the metrics belong to.
        summary_name (str = "metrics.csv"): The name of the summary file.
    """

    summary_path = path.join("output", f"huc{huc_level}", case_study, summary_name)
    row = pandas.DataFrame([{"Year": str(year), **year_metrics}])

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
) -> None:
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
        huc2s=huc2s,
        huc_level=huc_level,
        case_study=case_study,
        year=year,
        columns=[f"huc{huc_level}", "water_mgd", "geometry"],
    )
    predicted_zonal_sums = compute_zonal_sums(
        predicted_data, zones, zone_hucs, actual_values, huc_level=huc_level
    )
    metrics = evaluate_zonal_sums(
        actual_values,
        predicted_zonal_sums,
        huc_level=huc_level,
        case_study=case_study,
        year=year,
    )
    update_metrics_summary(
        metrics, huc_level=huc_level, case_study=case_study, year=year
    )


def score_holdout(
    predicted_data: numpy_typing.NDArray[numpy.float32],
    initial_data: numpy_typing.NDArray[numpy.float32],
    grid: raster_utilities.Grid,
    huc2s: list[str],
    huc_level: int,
    case_study: str,
    year: int,
) -> None:
    """
    Scores a run below HUC-12 against the HUC-12 statistics, which the run didn't use.

    Writes holdout_comparison.csv and holdout_comparison.gpkg to output/huc<huc_level>/<case_study>/<year>/ and updates
    output/huc<huc_level>/<case_study>/holdout_metrics.csv. Most HUC-12s have no water and are easy to score, so the
    metrics also cover only the HUC-12s with water (Wet_*). Report these next to the metrics for all HUC-12s. The
    Initial_* metrics score the initial estimates, so the difference shows what disaggregation adds.

    Parameters:
        predicted_data (numpy_typing.NDArray[numpy.float32]): The disaggregated raster, NaN as nodata.
        initial_data (numpy_typing.NDArray[numpy.float32]): The initial estimates, NaN as nodata.
        grid (raster_utilities.Grid): The grid of both rasters.
        huc2s (list[str]): The HUC-2s to evaluate.
        huc_level (int): The HUC level of the run.
        case_study (str): The case study to evaluate.
        year (int): The year to evaluate.
    """

    scoring_level = raster_utilities.STATISTICS_HUC_LEVEL

    actual_values = raster_utilities.read_statistics(
        huc2s=huc2s,
        huc_level=scoring_level,
        case_study=case_study,
        year=year,
        columns=[f"huc{scoring_level}", "water_mgd", "geometry"],
    )
    zones, zone_hucs, _ = raster_utilities.build_zones(
        actual_values, scoring_level, grid
    )

    predicted_sums = compute_zonal_sums(
        predicted_data, zones, zone_hucs, actual_values, huc_level=scoring_level
    )
    initial_sums = compute_zonal_sums(
        initial_data, zones, zone_hucs, actual_values, huc_level=scoring_level
    )

    print(f"Holdout at HUC-{scoring_level}:")
    holdout_metrics = evaluate_zonal_sums(
        actual_values,
        predicted_sums,
        huc_level=scoring_level,
        case_study=case_study,
        year=year,
        run_huc_level=huc_level,
        comparison_name="holdout_comparison",
    )

    actual = actual_values["water_mgd"].to_numpy(dtype=numpy.float64)
    predicted = predicted_sums["water_mgd"].to_numpy(dtype=numpy.float64)
    initial = initial_sums["water_mgd"].to_numpy(dtype=numpy.float64)
    wet = actual > 0

    # the scored HUCs: those with a statistic and at least one valid cell
    holdout_metrics["HUCs"] = float((~numpy.isnan(actual + predicted)).sum())
    holdout_metrics["Wet_HUCs"] = float((wet & ~numpy.isnan(predicted)).sum())

    for prefix, values, selection in [
        ("Wet_", predicted, wet),
        ("Initial_", initial, numpy.ones_like(wet)),
        ("Initial_Wet_", initial, wet),
    ]:
        selected_metrics = compute_metrics(actual[selection], values[selection])
        for name in ["RMSE", "MAE", "R2"]:
            holdout_metrics[f"{prefix}{name}"] = selected_metrics[name]

    print(
        f"R^2 initial {holdout_metrics['Initial_R2']:.4f}, final {holdout_metrics['R2']:.4f}; "
        f"with water only: initial {holdout_metrics['Initial_Wet_R2']:.4f}, final {holdout_metrics['Wet_R2']:.4f}"
    )

    update_metrics_summary(
        holdout_metrics,
        huc_level=huc_level,
        case_study=case_study,
        year=year,
        summary_name="holdout_metrics.csv",
    )


def rescore(
    huc2s: list[str] = ["14", "15"],
    huc_level: int = 12,
    case_study: str = "pws-tot",
    year: int = 2000,
) -> None:
    """
    Evaluate an existing output/huc<huc_level>/<case_study>/<year>/<case_study>_huc<huc_level>_<year>.tif outside the
    pipeline, rebuilding the zones main.py would pass to run(). A run below HUC-12 also gets its holdout score, which
    needs the run's pycnophylactic-interpolation.tif.

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
        huc2s=huc2s,
        huc_level=huc_level,
        case_study=case_study,
        year=year,
        columns=[f"huc{huc_level}", "water_mgd", "geometry"],
    )
    zones, zone_hucs, _ = raster_utilities.build_zones(zone_polygons, huc_level, grid)
    del zone_polygons

    predicted_data = raster_utilities.read_raster(predicted_raster_path, grid)

    run(
        predicted_data,
        zones,
        zone_hucs,
        huc2s,
        huc_level=huc_level,
        case_study=case_study,
        year=year,
    )

    if huc_level < raster_utilities.STATISTICS_HUC_LEVEL:
        del zones, zone_hucs
        score_holdout(
            predicted_data,
            raster_utilities.read_raster(
                path.join(
                    path.dirname(predicted_raster_path),
                    "pycnophylactic-interpolation.tif",
                ),
                grid,
            ),
            grid,
            huc2s,
            huc_level=huc_level,
            case_study=case_study,
            year=year,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rescore existing disaggregated rasters and update output/huc<huc_level>/<case_study>/metrics.csv."
    )
    parser.add_argument(
        "--case_study",
        choices=list(raster_utilities.CASE_STUDIES),
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
        choices=raster_utilities.HUC2S,
        default=list(raster_utilities.HUC2S),
        metavar="HUC2",
        help="The HUC-2 watersheds to evaluate, from 1 to 18, without zero-padding. The default is all 18 (CONUS).",
    )
    parser.add_argument(
        "--huc_level",
        choices=raster_utilities.HUC_LEVELS,
        default=12,
        type=int,
        help="The HUC scale the raster was disaggregated at, and is evaluated at.",
    )
    args = parser.parse_args()

    # choices= accepts a repeated value, and a repeated HUC-2 would count its statistics twice
    if len(set(args.huc2s)) != len(args.huc2s):
        parser.error("--huc2s: a HUC-2 is repeated.")

    for year in args.years:
        rescore(
            huc2s=args.huc2s,
            huc_level=args.huc_level,
            case_study=args.case_study,
            year=year,
        )
