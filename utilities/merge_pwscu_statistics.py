"""
Utility for merging public water supply consumptive use statistics with their associated HUC geometries.
"""

from collections.abc import Collection, Sequence
from pathlib import Path

import pandas

from utilities import huc12_crosswalk, statistics_utilities

SOURCE_PATH = Path("input", "statistics", "ps_mon_CU_ETr.csv")

GALLONS_PER_CUBIC_METER = 264.1720523581


def run(
    huc2s: Sequence[str] = ("14", "15"),
    huc_levels: Collection[int] = (8, 12),
    reprojected_crs: str = "EPSG:5070",
    year: int | None = None,
) -> None:
    """
    Merges the public water supply consumptive use statistics with the HUC geometries (statistics_utilities.merge),
    into input/statistics/<huc2>/<huc_level>/pws-cu/<year>.csv and .gpkg. A HUC with no statistics gets 0.

    Parameters:
        huc2s (Sequence[str] = ("14", "15")): The HUC-2 IDs, without zero-padding.
        huc_levels (Collection[int] = (8, 12)): The HUC levels to write.
        reprojected_crs (str = "EPSG:5070"): The CRS of the output geopackages.
        year (int | None = None): The year to write. All years in the source when None.
    """

    # the source has rows only for HUC-12s with public supply withdrawals (pws-tot > 0), so no rows means no use
    statistics_utilities.merge(
        "pws-cu",
        SOURCE_PATH,
        read_annual_mgd,
        huc2s,
        huc_levels,
        reprojected_crs,
        year,
        absent_is_zero=True,
    )


def read_annual_mgd(source_path: Path) -> pandas.DataFrame:
    """
    Reads the consumptive use statistics, applies huc12_crosswalk.remap_rows, sums the months of each year and
    converts the sums to mgd.

    The source is long: one row per HUC-12 and month (HUC12, year, month, ps_cu_m3_mtot), with values in total cubic
    meters for the month.

    Parameters:
        source_path (Path): The source CSV.

    Returns:
        annual_mgd (pandas.DataFrame): One row per year (the index) and one column per HUC-12, in mgd. NaN where a
            HUC-12 has no rows in a year.
    """

    statistics = huc12_crosswalk.remap_rows(
        pandas.read_csv(source_path, dtype={"HUC12": str}), "HUC12", "ps_cu_m3_mtot"
    )

    # remap_rows can give a code several rows for one month. The sum merges them
    annual_m3 = statistics.groupby(["year", "HUC12"])["ps_cu_m3_mtot"].sum().unstack()
    days = annual_m3.index.map(statistics_utilities.days_in_year).to_series(index=annual_m3.index)

    return (annual_m3 * GALLONS_PER_CUBIC_METER).div(days * 1_000_000, axis=0)


if __name__ == "__main__":
    run()
