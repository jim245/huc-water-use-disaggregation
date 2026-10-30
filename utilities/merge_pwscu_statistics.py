"""
Utility for merging public water supply consumptive use statistics with their associated HUC geometries.
"""

import calendar
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
    Sums monthly consumptive use volumes annually, converts them to MGD and merges them with HUC geometries.

    The source CSV is long: one row per HUC-12 and month (HUC12, year, month, ps_cu_m3_mtot), with values in total
    cubic meters for the month. The 12 months are summed to an annual volume, then converted to million gallons per
    day. HUC-12 codes missing from the WBD shapefiles have their water moved to the codes that replaced them first
    (huc12_crosswalk). Writes input/statistics/<huc2>/<huc_level>/pws-cu/<year>.csv and .gpkg, with one water_mgd value per HUC.
    The geopackage is a left merge onto the HUC geometries, so HUCs without statistics get NaN. A year's files are
    rebuilt unless both exist, and nothing is read when every output exists.

    Parameters:
        huc2s (list[str] = ["14", "15"]): A list of HUC2 numerical ID strings.
        huc_levels (set[int] = {8, 12}): A set of HUC levels to create statistics for.
        reprojected_crs (str = "EPSG:5070"): The CRS of the output geopackages.
        year (Optional[int] = None): The year to create statistics for; all years in the source CSV when None.
    """

    statistics_path = path.join("input", "statistics", "ps_mon_CU_ETr.csv")

    # the source CSV is only read up front when the years must come from it
    statistics: Optional[pandas.DataFrame] = None
    if year is None:
        statistics = huc12_crosswalk.remap_rows(
            pandas.read_csv(statistics_path, dtype={"HUC12": str}),
            "HUC12",
            "ps_cu_m3_mtot",
        )
        years = list(statistics["year"].unique())
    else:
        years = [year]

    # nothing is read when every output already exists
    pending = {
        (huc2, huc_level): missing_years(
            path.join("input", "statistics", huc2, str(huc_level), "pws-cu"), years
        )
        for huc2 in huc2s
        for huc_level in huc_levels
    }
    if not any(pending.values()):
        return

    if statistics is None:
        statistics = huc12_crosswalk.remap_rows(
            pandas.read_csv(statistics_path, dtype={"HUC12": str}),
            "HUC12",
            "ps_cu_m3_mtot",
        )

    for (huc2, huc_level), huc_years in pending.items():
        if not huc_years:
            continue

        huc_directory = path.join(
            "input", "statistics", huc2, str(huc_level), "pws-cu"
        )
        os.makedirs(huc_directory, exist_ok=True)

        # filter columns to dates and the relevant HUCs
        filtered_statistics = statistics[
            statistics["HUC12"].str.startswith(huc2.zfill(2))
        ]

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
            year_df = filtered_statistics[filtered_statistics["year"] == year]

            # truncate HUC codes to huc_level
            year_df = year_df.assign(truncated=year_df["HUC12"].str[:huc_level])
            annual_sums = year_df.groupby("truncated")["ps_cu_m3_mtot"].sum()
            water_use = annual_sums.reset_index()
            water_use.columns = [f"huc{huc_level}", "water_mgd"]

            # convert water use from cubic meters per year to million gallons per day
            water_use["water_mgd"] = (water_use["water_mgd"] * 264.1720523581) / (
                (366 if calendar.isleap(year) else 365) * 1_000_000
            )

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
