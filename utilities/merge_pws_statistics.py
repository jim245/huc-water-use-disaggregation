"""
Utility for merging public water supply water use statistics with their associated HUC geometries.
"""

import calendar
from collections.abc import Collection, Sequence
from pathlib import Path

import pandas

from utilities import huc12_crosswalk, statistics_utilities

SOURCE_PATHS = {
    case_study: Path("input", "statistics", f"PS_HUC12_{stat}_2000_2020.csv")
    for case_study, stat in (("pws-gw", "GW"), ("pws-sw", "SW"), ("pws-tot", "Tot"))
}


def run(
    case_study: str,
    huc2s: Sequence[str] = ("14", "15"),
    huc_levels: Collection[int] = (8, 12),
    reprojected_crs: str = "EPSG:5070",
    year: int | None = None,
) -> None:
    """
    Merges the public water supply statistics of a case study with the HUC geometries (statistics_utilities.merge),
    into input/statistics/<huc2>/<huc_level>/<case_study>/<year>.csv and .gpkg.

    Parameters:
        case_study (str): A key of SOURCE_PATHS.
        huc2s (Sequence[str] = ("14", "15")): The HUC-2 IDs, without zero-padding.
        huc_levels (Collection[int] = (8, 12)): The HUC levels to write.
        reprojected_crs (str = "EPSG:5070"): The CRS of the output geopackages.
        year (int | None = None): The year to write. All years in the source when None.

    Raises:
        ValueError: case_study isn't a key of SOURCE_PATHS.
    """

    if case_study not in SOURCE_PATHS:
        raise ValueError(f"no public water supply source for {case_study!r}")

    statistics_utilities.merge(
        case_study,
        SOURCE_PATHS[case_study],
        read_annual_mgd,
        huc2s,
        huc_levels,
        reprojected_crs,
        year,
    )


def read_annual_mgd(source_path: Path) -> pandas.DataFrame:
    """
    Reads the public water supply statistics, applies huc12_crosswalk.remap_columns and averages the months of each
    year.

    The source is wide: one row per year and month, one column per HUC-12, with values in monthly mean mgd.

    Parameters:
        source_path (Path): The source CSV.

    Returns:
        annual_mgd (pandas.DataFrame): One row per year (the index) and one column per HUC-12, in mgd.
    """

    # read_csv gives one block per column, and copy merges them. Without it, each operation below runs once per column
    statistics = huc12_crosswalk.remap_columns(pandas.read_csv(source_path)).copy()
    huc_columns = [column for column in statistics.columns if column.isdigit()]

    # each row is the mean mgd of one month. The annual mgd is the day-weighted mean of the months, not their sum
    annual_mgd = {}
    for year in statistics["Year"].unique():
        year_df = statistics.loc[statistics["Year"] == year]
        days_in_month = pandas.Series(
            [calendar.monthrange(int(year), int(month))[1] for month in year_df["Month"]], index=year_df.index
        )
        annual_mgd[year] = year_df[huc_columns].mul(days_in_month, axis=0).sum(
            axis=0
        ) / statistics_utilities.days_in_year(int(year))

    return pandas.DataFrame(annual_mgd).T


if __name__ == "__main__":
    run("pws-tot")
