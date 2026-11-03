"""
Stage 5 of the pipeline: evaluation (see main.py).

Run this file from the repository root to score existing runs again (see rescore).
"""

import argparse
import os
from collections.abc import Sequence
from pathlib import Path

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
) -> numpy_typing.NDArray[numpy.float64]:
    """
    Sums a raster over each HUC of actual_values.

    Parameters:
        predicted_data (numpy_typing.NDArray[numpy.float32]): The predicted values, NaN as nodata.
        zones (numpy_typing.NDArray[numpy.int32]): The zone index raster of the HUCs in actual_values
            (raster_utilities.build_zones), 0 outside all HUCs.
        zone_hucs (numpy_typing.NDArray[numpy.int64]): The HUC ID for each zone index, offset by one.
        actual_values (geopandas.GeoDataFrame): The statistics, with a huc<huc_level> ID column.
        huc_level (int): The HUC level of actual_values.

    Returns:
        predicted_values (numpy_typing.NDArray[numpy.float64]): The sum of each HUC, in the row order of actual_values.
            NaN for a HUC with no valid cells.
    """

    zone_count = len(zone_hucs) + 1

    valid = ~numpy.isnan(predicted_data)

    zone_sums = numpy.bincount(zones[valid], weights=predicted_data[valid], minlength=zone_count)
    zone_cell_counts = numpy.bincount(zones[valid], minlength=zone_count)

    # index 0 is outside all HUCs
    huc_sums = pandas.Series(numpy.where(zone_cell_counts[1:] > 0, zone_sums[1:], numpy.nan), index=zone_hucs)

    return numpy.asarray(actual_values[f"huc{huc_level}"].astype(numpy.int64).map(huc_sums), dtype=numpy.float64)


def compute_metrics(
    actual: numpy_typing.NDArray[numpy.float64],
    predicted: numpy_typing.NDArray[numpy.float64],
) -> dict[str, float]:
    """
    Calculates the metrics of the predicted values of each HUC against the actual values.

    NRMSE and NMAE divide by the range (max - min) of the actual values. The error metrics leave out each HUC where one
    of the two values is NaN. Thus the other metrics report the water that the map doesn't place. Unpredicted_HUCs
    counts the HUCs that have actual water use and a NaN prediction (no valid cells), and Unpredicted_mgd is their
    actual water use. Missing_mgd is the total actual minus the total predicted, and Missing_pct is Missing_mgd as a
    percentage of the total actual. Missing_mgd also includes the water that a HUC with a prediction lost.

    Parameters:
        actual (numpy_typing.NDArray[numpy.float64]): The actual value of each HUC.
        predicted (numpy_typing.NDArray[numpy.float64]): The predicted value of each HUC, in the same order.

    Returns:
        metrics (dict[str, float]): The RMSE, MAE, NRMSE, NMAE, R2, Unpredicted_HUCs, Unpredicted_mgd, Missing_mgd and
            Missing_pct.
    """

    unpredicted = numpy.isnan(predicted) & (actual > 0)
    unpredicted_hucs = int(unpredicted.sum())
    unpredicted_mgd = float(actual[unpredicted].sum())
    actual_total = float(numpy.nansum(actual, dtype=numpy.float64))
    missing_mgd = actual_total - float(numpy.nansum(predicted, dtype=numpy.float64))
    missing_pct = 100.0 * missing_mgd / actual_total if actual_total != 0 else numpy.nan

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
    predicted: numpy_typing.NDArray[numpy.float64],
    *,
    huc_level: int,
    directory: Path,
    comparison_name: str = "comparison",
) -> dict[str, float]:
    """
    Calculates the metrics of the predicted HUC sums (see compute_metrics), and prints them.

    Writes the comparison of each HUC to <comparison_name>.csv and <comparison_name>.gpkg in directory. The predicted
    column is False for a HUC with no valid cells, so that a map can show these HUCs. Only the gpkg has the polygons and
    the ratio predicted / actual, which is inf or NaN for a HUC with no actual water use.

    Parameters:
        actual_values (geopandas.GeoDataFrame): The statistics, with the huc<huc_level>, water_mgd and geometry columns.
        predicted (numpy_typing.NDArray[numpy.float64]): The predicted sum of each HUC (compute_zonal_sums), in the row
            order of actual_values.
        huc_level (int): The HUC level of actual_values.
        directory (Path): The output directory (raster_utilities.run_directory).
        comparison_name (str = "comparison"): The name of the comparison files.

    Returns:
        metrics (dict[str, float]): The metrics of the year (see compute_metrics).
    """

    comparison = actual_values.rename(columns={"water_mgd": "water_mgd_actual"})
    comparison["water_mgd_predicted"] = predicted

    year_metrics = compute_metrics(comparison["water_mgd_actual"].to_numpy(dtype=numpy.float64), predicted)

    print(f"RMSE  : {year_metrics['RMSE']:.4f}")
    print(f"MAE   : {year_metrics['MAE']:.4f}")
    print(f"NRMSE : {year_metrics['NRMSE']:.4f}")
    print(f"NMAE  : {year_metrics['NMAE']:.4f}")
    print(f"R^2   : {year_metrics['R2']:.4f}")
    print(
        f"Unpredicted : {int(year_metrics['Unpredicted_HUCs'])} HUCs with water, "
        f"{year_metrics['Unpredicted_mgd']:.4f} mgd"
    )
    print(f"Missing     : {year_metrics['Missing_mgd']:.4f} mgd ({year_metrics['Missing_pct']:.4f}% of actual)")

    comparison["predicted"] = comparison["water_mgd_predicted"].notna()
    comparison["ratio"] = comparison["water_mgd_predicted"] / comparison["water_mgd_actual"]

    comparison = comparison[
        [f"huc{huc_level}", "water_mgd_predicted", "water_mgd_actual", "predicted", "ratio", "geometry"]
    ]

    # the WKT text of the polygons would make the CSV very large
    comparison.drop(columns=["geometry", "ratio"]).to_csv(directory / f"{comparison_name}.csv", index=False)

    comparison.to_file(directory / f"{comparison_name}.gpkg")

    return year_metrics


def update_metrics_summary(year_metrics: dict[str, float], *, year: int, summary_path: Path) -> None:
    """
    Inserts or replaces the metrics of one year in a summary CSV, and calculates the Average row again.

    Each year runs separately, so the summary keeps the rows of the other years. The function writes a temporary file
    and then replaces summary_path, so an interrupted write can't lose these rows.

    Parameters:
        year_metrics (dict[str, float]): The metrics of the year.
        year (int): The year of the metrics.
        summary_path (Path): The summary CSV, in output/huc<huc_level>/<case_study>/.
    """

    row = pandas.DataFrame([{"Year": str(year), **year_metrics}])

    if summary_path.exists():
        summary = pandas.read_csv(summary_path, dtype={"Year": str})
        summary = summary[~summary["Year"].isin([str(year), "Average"])]
        summary = pandas.concat([summary, row], ignore_index=True)
    else:
        summary = row

    summary = summary.sort_values("Year", key=lambda years: years.astype(int))

    average = summary.drop(columns="Year").mean().to_frame().T
    average.insert(0, "Year", "Average")

    # the process ID keeps concurrent runs from sharing a temporary file
    temporary_path = summary_path.with_name(f"{summary_path.name}.{os.getpid()}.tmp")
    pandas.concat([summary, average], ignore_index=True).to_csv(temporary_path, index=False)
    temporary_path.replace(summary_path)


def run(
    predicted_data: numpy_typing.NDArray[numpy.float32],
    zones: numpy_typing.NDArray[numpy.int32],
    zone_hucs: numpy_typing.NDArray[numpy.int64],
    huc2s: Sequence[str],
    *,
    case_study: str,
    year: int,
    huc_level: int,
) -> None:
    """
    Scores the disaggregated raster against the statistics at the HUC level of the run.

    Each disaggregation iteration rescales to these totals over the same zones, so the scores are near 1 by
    construction. They show that the run keeps the totals, not how well it places water in each HUC. score_holdout
    measures placement.

    Writes comparison.csv and comparison.gpkg to output/huc<huc_level>/<case_study>/<year>/ (see evaluate_zonal_sums),
    and updates output/huc<huc_level>/<case_study>/metrics.csv.

    Parameters:
        predicted_data (numpy_typing.NDArray[numpy.float32]): The disaggregated raster, NaN as nodata.
        zones (numpy_typing.NDArray[numpy.int32]): The huc_level zone index raster that disaggregation used, 0 outside
            all HUCs.
        zone_hucs (numpy_typing.NDArray[numpy.int64]): The HUC ID for each zone index, offset by one.
        huc2s (Sequence[str]): The HUC-2s of the run.
        case_study (str): The case study of the run.
        year (int): The year of the run.
        huc_level (int): The HUC level of the run.
    """

    directory = raster_utilities.run_directory(case_study=case_study, year=year, huc_level=huc_level)

    actual_values = raster_utilities.read_statistics(
        huc2s=huc2s,
        huc_level=huc_level,
        case_study=case_study,
        year=year,
        columns=[f"huc{huc_level}", "water_mgd", "geometry"],
    )
    predicted = compute_zonal_sums(predicted_data, zones, zone_hucs, actual_values, huc_level=huc_level)
    year_metrics = evaluate_zonal_sums(actual_values, predicted, huc_level=huc_level, directory=directory)
    update_metrics_summary(year_metrics, year=year, summary_path=directory.parent / "metrics.csv")


def score_holdout(
    predicted_data: numpy_typing.NDArray[numpy.float32],
    initial_data: numpy_typing.NDArray[numpy.float32],
    grid: raster_utilities.Grid,
    huc2s: Sequence[str],
    *,
    case_study: str,
    year: int,
    huc_level: int,
) -> None:
    """
    Scores a run below HUC-12 against the HUC-12 statistics, which the run doesn't use.

    Writes holdout_comparison.csv and holdout_comparison.gpkg to output/huc<huc_level>/<case_study>/<year>/ and updates
    output/huc<huc_level>/<case_study>/holdout_metrics.csv. Most HUC-12s have no water and are easy to score, so the
    metrics also cover only the HUC-12s with water (Wet_*). Report these next to the metrics for all HUC-12s. The
    Initial_* metrics score the initial estimates, so the difference shows what disaggregation adds.

    Parameters:
        predicted_data (numpy_typing.NDArray[numpy.float32]): The disaggregated raster, NaN as nodata.
        initial_data (numpy_typing.NDArray[numpy.float32]): The initial estimates, NaN as nodata.
        grid (raster_utilities.Grid): The grid of both rasters.
        huc2s (Sequence[str]): The HUC-2s of the run.
        case_study (str): The case study of the run.
        year (int): The year of the run.
        huc_level (int): The HUC level of the run.
    """

    directory = raster_utilities.run_directory(case_study=case_study, year=year, huc_level=huc_level)
    scoring_level = raster_utilities.STATISTICS_HUC_LEVEL

    actual_values = raster_utilities.read_statistics(
        huc2s=huc2s,
        huc_level=scoring_level,
        case_study=case_study,
        year=year,
        columns=[f"huc{scoring_level}", "water_mgd", "geometry"],
    )
    zones, zone_hucs, _ = raster_utilities.build_zones(actual_values, scoring_level, grid)

    predicted = compute_zonal_sums(predicted_data, zones, zone_hucs, actual_values, huc_level=scoring_level)
    initial = compute_zonal_sums(initial_data, zones, zone_hucs, actual_values, huc_level=scoring_level)

    print(f"Holdout at HUC-{scoring_level}:")
    holdout_metrics = evaluate_zonal_sums(
        actual_values,
        predicted,
        huc_level=scoring_level,
        directory=directory,
        comparison_name="holdout_comparison",
    )

    actual = actual_values["water_mgd"].to_numpy(dtype=numpy.float64)
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

    update_metrics_summary(holdout_metrics, year=year, summary_path=directory.parent / "holdout_metrics.csv")


def rescore(huc2s: Sequence[str], *, case_study: str, year: int, huc_level: int) -> None:
    """
    Scores an existing run again, with the zones that main.py gives to run. A run below HUC-12 also gets its holdout
    score.

    Reads output/huc<huc_level>/<case_study>/<year>/<case_study>_huc<huc_level>_<year>.tif and, for the holdout score,
    pycnophylactic-interpolation.tif in the same directory. Writes the files of run and score_holdout.

    Parameters:
        huc2s (Sequence[str]): The HUC-2s of the run, without zero-padding.
        case_study (str): The case study of the run.
        year (int): The year of the run.
        huc_level (int): The HUC level of the run.

    Raises:
        ValueError: The raster has values outside the HUCs of huc2s, or a HUC-2 in huc2s has no valid cells. Each one
            shows that huc2s isn't the set of HUC-2s of the run, and the metrics would then be incorrect.
    """

    directory = raster_utilities.run_directory(case_study=case_study, year=year, huc_level=huc_level)
    predicted_raster_path = directory / f"{case_study}_huc{huc_level}_{year}.tif"
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

    # stages 2-4 write NaN outside all HUCs, so a value there is in a HUC-2 of the run that huc2s doesn't have
    valid = ~numpy.isnan(predicted_data)
    if valid[zones == 0].any():
        raise ValueError(
            f"{predicted_raster_path} has values outside the HUCs of the HUC-2s {list(huc2s)}. "
            "Give the HUC-2s of the run."
        )

    # a HUC-2 that the run doesn't have is off the grid or NaN on the grid
    valid_hucs = zone_hucs[numpy.unique(zones[valid]) - 1]
    valid_huc2s = {str(huc2) for huc2 in (valid_hucs // 10 ** (huc_level - 2)).tolist()}
    missing_huc2s = sorted(set(huc2s) - valid_huc2s, key=int)
    if missing_huc2s:
        raise ValueError(
            f"The HUC-2s {missing_huc2s} have no valid cells in {predicted_raster_path}. Give the HUC-2s of the run."
        )
    del valid

    run(predicted_data, zones, zone_hucs, huc2s, case_study=case_study, year=year, huc_level=huc_level)

    if huc_level < raster_utilities.STATISTICS_HUC_LEVEL:
        del zones, zone_hucs
        score_holdout(
            predicted_data,
            raster_utilities.read_raster(directory / "pycnophylactic-interpolation.tif", grid),
            grid,
            huc2s,
            case_study=case_study,
            year=year,
            huc_level=huc_level,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scores existing disaggregated rasters again. Updates output/huc<huc_level>/<case_study>/"
        f"metrics.csv, and holdout_metrics.csv for a level below {raster_utilities.STATISTICS_HUC_LEVEL}."
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
        help="The years to evaluate. Each year needs "
        "output/huc<huc_level>/<case_study>/<year>/<case_study>_huc<huc_level>_<year>.tif.",
    )
    parser.add_argument(
        "--huc2s",
        nargs="+",
        choices=raster_utilities.HUC2S,
        default=list(raster_utilities.HUC2S),
        metavar="HUC2",
        help="The HUC-2 watersheds of the run, from 1 to 18, without zero-padding. The default is all 18 (CONUS).",
    )
    parser.add_argument(
        "--huc_level",
        choices=raster_utilities.HUC_LEVELS,
        default=12,
        type=int,
        help="The HUC level of the run. The scores use this level.",
    )
    args = parser.parse_args()

    # choices= accepts a repeated value, and a repeated HUC-2 would count its statistics twice
    if len(set(args.huc2s)) != len(args.huc2s):
        parser.error("--huc2s: a HUC-2 is repeated.")

    for year in args.years:
        rescore(args.huc2s, case_study=args.case_study, year=year, huc_level=args.huc_level)
