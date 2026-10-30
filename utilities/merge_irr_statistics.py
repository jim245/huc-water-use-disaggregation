"""
Utility for merging irrigation water use statistics with their associated HUC geometries.
"""

import os
from os import path
from typing import Any, Optional

import geopandas
import pandas

from utilities import huc12_crosswalk


def run(
    huc2s: list[str] = ["14", "15"],
    huc_levels: set[int] = {8, 12},
    reprojected_crs: str = "EPSG:5070",
    year: Optional[int] = None,
):
    """
    Sums annual HUC-12 water use statistics to each HUC level and merges them with HUC geometries.

    The source CSV is wide: one row per year, one column per HUC-12, with values in annual MGD. It is read with
    read_statistics, which zeroes sentinel codes and remaps HUC-12 codes missing from the WBD shapefiles. Writes
    input/statistics/<huc2>/<huc_level>/irr/<year>.csv and .gpkg, with one water_mgd value per HUC. The geopackage is
    a left merge onto the HUC geometries, so HUCs without statistics get NaN. A year's files are rebuilt unless both
    exist, and nothing is read when every output exists.

    Parameters:
        huc2s (list[str] = ["14", "15"]): A list of HUC2 numerical ID strings.
        huc_levels (set[int] = {8, 12}): A set of HUC levels to create statistics for.
        reprojected_crs (str = "EPSG:5070"): The CRS of the output geopackages.
        year (Optional[int] = None): The year to create statistics for; all years in the source CSV when None.
    """

    statistics_path = path.join(
        "input", "statistics", "Irr_CU_HUC12_Tot_annual_2000_2020.csv"
    )

    # the source CSV is only read up front when the years must come from it
    statistics: Optional[pandas.DataFrame] = None
    if year is None:
        statistics = read_statistics(statistics_path)
        years = list(statistics["Year"].unique())
    else:
        years = [year]

    # nothing is read when every output already exists
    pending = {
        (huc2, huc_level): missing_years(
            path.join("input", "statistics", huc2, str(huc_level), "irr"), years
        )
        for huc2 in huc2s
        for huc_level in huc_levels
    }
    if not any(pending.values()):
        return

    if statistics is None:
        statistics = read_statistics(statistics_path)

    for (huc2, huc_level), huc_years in pending.items():
        if not huc_years:
            continue

        huc_directory = path.join("input", "statistics", huc2, str(huc_level), "irr")
        os.makedirs(huc_directory, exist_ok=True)

        # filter columns to years and the relevant HUCs
        desired_columns = ["Year"] + [
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
            yearly_sums = year_df[huc_columns].sum(axis=0)

            # truncate HUC codes to huc_level
            huc_level_columns = {huc: huc[:huc_level] for huc in huc_columns}
            water_use = (
                yearly_sums.groupby(pandas.Series(huc_level_columns))
                .sum()
                .reset_index()
            )
            water_use.columns = [f"huc{huc_level}", "water_mgd"]

            water_use.to_csv(path.join(huc_directory, f"{year}.csv"), index=False)

            merged_gdf = huc_gdf.merge(water_use, on=f"huc{huc_level}", how="left")
            merged_gdf = merged_gdf.to_crs(reprojected_crs)
            merged_gdf.to_file(path.join(huc_directory, f"{year}.gpkg"))


def read_statistics(statistics_path: str) -> pandas.DataFrame:
    """
    Reads the irrigation statistics CSV, zeroes its sentinel codes and moves the water of HUC-12 codes missing from
    the WBD shapefiles to the codes that replaced them (huc12_crosswalk).

    Parameters:
        statistics_path (str): The path to the source CSV.

    Returns:
        statistics (pandas.DataFrame): The wide statistics, one row per year and one column per HUC-12.
    """

    statistics = pandas.read_csv(statistics_path)

    # 999: HUC-12 has no irrigated land. 888: HUC-12 has irrigated land but OpenET returned no ET value.
    # both are treated as no irrigation, and must be zeroed before the crosswalk splits values
    huc_columns = [column for column in statistics.columns if column.isdigit()]
    statistics[huc_columns] = statistics[huc_columns].replace([888, 999], 0)

    return huc12_crosswalk.remap_columns(statistics)


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
