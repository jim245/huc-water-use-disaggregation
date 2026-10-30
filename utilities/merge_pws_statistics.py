"""
Utility for merging public water supply water use statistics with their associated HUC geometries.
"""

import calendar
import os
from os import path
from typing import Any, Optional

import geopandas
import pandas

from utilities import huc12_crosswalk


def run(
    case_study: str,
    huc2s: list[str] = ["14", "15"],
    huc_levels: set[int] = {8, 12},
    reprojected_crs: str = "EPSG:5070",
    year: Optional[int] = None,
):
    """
    Averages monthly water use statistics (MGD) annually and merges them with HUC geometries.

    The source CSV is wide: one row per year and month, one column per HUC-12, with values in monthly mean MGD.
    HUC-12 codes missing from the WBD shapefiles have their water moved to the codes that replaced them first
    (huc12_crosswalk).
    Writes input/statistics/<huc2>/<huc_level>/<case_study>/<year>.csv and .gpkg, with one water_mgd value per HUC.
    The geopackage is a left merge onto the HUC geometries, so HUCs without statistics get NaN. A year's files are
    rebuilt unless both exist, and nothing is read when every output exists.

    Parameters:
        case_study (str): The case study to run for. "pws-gw", "pws-sw" and "pws-tot" select the GW, SW and Tot
            source CSVs; any other value falls back to Tot.
        huc2s (list[str] = ["14", "15"]): A list of HUC2 numerical ID strings.
        huc_levels (set[int] = {8, 12}): A set of HUC levels to create statistics for.
        reprojected_crs (str = "EPSG:5070"): The CRS of the output geopackages.
        year (Optional[int] = None): The year to create statistics for; all years in the source CSV when None.
    """

    match case_study:
        case "pws-gw":
            stat = "GW"

        case "pws-sw":
            stat = "SW"

        case "pws-tot" | _:
            stat = "Tot"

    statistics_path = path.join(
        "input", "statistics", f"PS_HUC12_{stat}_2000_2020.csv"
    )

    # the source CSV (~20 s to read) is only read up front when the years must come from it
    statistics: Optional[pandas.DataFrame] = None
    if year is None:
        statistics = huc12_crosswalk.remap_columns(pandas.read_csv(statistics_path))
        years = list(statistics["Year"].unique())
    else:
        years = [year]

    # nothing is read when every output already exists
    pending = {
        (huc2, huc_level): missing_years(
            path.join("input", "statistics", huc2, str(huc_level), case_study), years
        )
        for huc2 in huc2s
        for huc_level in huc_levels
    }
    if not any(pending.values()):
        return

    if statistics is None:
        statistics = huc12_crosswalk.remap_columns(pandas.read_csv(statistics_path))

    for (huc2, huc_level), huc_years in pending.items():
        if not huc_years:
            continue

        huc_directory = path.join(
            "input", "statistics", huc2, str(huc_level), case_study
        )
        os.makedirs(huc_directory, exist_ok=True)

        # filter columns to dates and the relevant HUCs
        desired_columns = ["Year", "Month"] + [
            column
            for column in statistics.columns
            if str(column).startswith(huc2.zfill(2))
        ]
        filtered_statistics = statistics[desired_columns]

        huc_gdf = geopandas.read_file(
            path.join(
                "input",
                "HUC",
                huc2,
                "Shape",
                f"WBDHU{huc_level}.shp",
            )
        )

        for year in huc_years:
            year_df = filtered_statistics[filtered_statistics["Year"] == year]

            huc_columns = [column for column in year_df.columns if column.isdigit()]

            # Each row is one month's mean MGD, so the annual MGD is the
            # day-weighted mean of the months, not their sum (which is ~12x).
            days_in_month = year_df["Month"].map(
                lambda month: calendar.monthrange(int(year), int(month))[1]
            )
            yearly_means = year_df[huc_columns].mul(days_in_month, axis=0).sum(
                axis=0
            ) / (366 if calendar.isleap(int(year)) else 365)

            # truncate HUC codes to huc_level
            huc_level_columns = {huc: huc[:huc_level] for huc in huc_columns}
            water_use = (
                yearly_means.groupby(pandas.Series(huc_level_columns))
                .sum()
                .reset_index()
            )
            water_use.columns = [f"huc{huc_level}", "water_mgd"]

            water_use.to_csv(path.join(huc_directory, f"{year}.csv"), index=False)

            merged_gdf = huc_gdf.merge(water_use, on=f"huc{huc_level}", how="left")
            merged_gdf = merged_gdf.to_crs(reprojected_crs)
            merged_gdf.to_file(path.join(huc_directory, f"{year}.gpkg"))


def missing_years(directory: str, years: list[Any]) -> list[Any]:
    """
    Lists the years whose statistics are incomplete in a directory: a year is complete only when both its csv and its
    gpkg exist.

    Parameters:
        directory (str): The statistics directory.
        years (list[Any]): The years to check.

    Returns:
        missing (list[Any]): The incomplete years, in the given order.
    """

    return [
        year
        for year in years
        if not (
            path.exists(path.join(directory, f"{year}.csv"))
            and path.exists(path.join(directory, f"{year}.gpkg"))
        )
    ]


if __name__ == "__main__":
    run()
