"""
Utility for merging irrigation water use statistics with their associated HUC geometries.
"""

from collections.abc import Collection, Sequence
from pathlib import Path

import pandas

from utilities import huc12_crosswalk, statistics_utilities

SOURCE_PATH = Path("input", "statistics", "Irr_CU_HUC12_Tot_annual_2000_2020.csv")


def run(
    huc2s: Sequence[str] = ("14", "15"),
    huc_levels: Collection[int] = (8, 12),
    reprojected_crs: str = "EPSG:5070",
    year: int | None = None,
) -> None:
    """
    Merges the irrigation statistics with the HUC geometries (statistics_utilities.merge), into
    input/statistics/<huc2>/<huc_level>/irr/<year>.csv and .gpkg.

    Parameters:
        huc2s (Sequence[str] = ("14", "15")): The HUC-2 IDs, without zero-padding.
        huc_levels (Collection[int] = (8, 12)): The HUC levels to write.
        reprojected_crs (str = "EPSG:5070"): The CRS of the output geopackages.
        year (int | None = None): The year to write. All years in the source when None.
    """

    statistics_utilities.merge("irr", SOURCE_PATH, read_annual_mgd, huc2s, huc_levels, reprojected_crs, year)


def read_annual_mgd(source_path: Path) -> pandas.DataFrame:
    """
    Reads the irrigation statistics, zeroes their sentinel codes and applies huc12_crosswalk.remap_columns.

    The source is wide: one row per year, one column per HUC-12, with values in annual mgd.

    Parameters:
        source_path (Path): The source CSV.

    Returns:
        annual_mgd (pandas.DataFrame): One row per year (the index) and one column per HUC-12, in mgd.
    """

    # 999: the HUC-12 has no irrigated land. 888: the HUC-12 has irrigated land, but OpenET returned no ET value.
    # both mean no irrigation. Zero them before the crosswalk splits values
    statistics = pandas.read_csv(source_path, index_col="Year").replace([888, 999], 0)

    return huc12_crosswalk.remap_columns(statistics)


if __name__ == "__main__":
    run()
