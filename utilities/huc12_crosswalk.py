"""
Utility for moving water use statistics from HUC-12 codes missing in the downloaded WBD to the HUC-12s that replaced
them (see huc12_crosswalk.csv).
"""

from os import path

import pandas

CROSSWALK_PATH = path.join(path.dirname(__file__), "huc12_crosswalk.csv")


def load() -> pandas.DataFrame:
    """
    Reads the HUC-12 crosswalk and computes each new code's share of its old code's water.

    The share is the overlap area divided by the old code's summed overlaps, so the shares of each old code sum to 1
    and no water is lost or created. A split code's water is divided by area.

    Returns:
        crosswalk (pandas.DataFrame): One row per old and new code pair, with old_huc12, new_huc12 and share.
    """

    crosswalk = pandas.read_csv(
        CROSSWALK_PATH, comment="#", dtype={"old_huc12": str, "new_huc12": str}
    )
    crosswalk["share"] = crosswalk["overlap_sqkm"] / crosswalk.groupby("old_huc12")[
        "overlap_sqkm"
    ].transform("sum")

    return crosswalk[["old_huc12", "new_huc12", "share"]]


def remap_columns(statistics: pandas.DataFrame) -> pandas.DataFrame:
    """
    Moves the water of old HUC-12 columns in a wide statistics table (one column per HUC-12) to their new columns.

    Each old column is dropped and its values, times the share, are added to the new columns, which are created when
    the table has none. Missing values count as 0. Every value must be water (no sentinel codes), since it is split.

    Parameters:
        statistics (pandas.DataFrame): The wide statistics, with HUC-12 code strings as column names.

    Returns:
        remapped (pandas.DataFrame): The statistics without old HUC-12 columns.
    """

    crosswalk = load()
    crosswalk = crosswalk[crosswalk["old_huc12"].isin(statistics.columns)]
    if crosswalk.empty:
        return statistics

    moved = statistics[crosswalk["old_huc12"]].fillna(0) * crosswalk["share"].to_numpy()
    moved.columns = crosswalk["new_huc12"]
    # a new code that receives water from several old codes gets their sum
    moved = moved.T.groupby(level=0).sum().T

    remapped = statistics.drop(columns=crosswalk["old_huc12"].unique())
    # new codes that already have statistics get the moved water added; the rest are appended in one step, since
    # inserting columns one by one fragments the frame (pandas PerformanceWarning)
    existing = [huc12 for huc12 in moved.columns if huc12 in remapped.columns]
    added = [huc12 for huc12 in moved.columns if huc12 not in remapped.columns]
    remapped[existing] = remapped[existing].fillna(0) + moved[existing]

    return pandas.concat([remapped, moved[added]], axis=1)


def remap_rows(
    statistics: pandas.DataFrame, huc12_column: str, value_column: str
) -> pandas.DataFrame:
    """
    Moves the water of old HUC-12 rows in a long statistics table (one row per HUC-12 and date) to their new codes.

    Each old row is replaced by one row per new code, with the value times the share. Rows of a new code that already
    has statistics are not merged, so callers must sum by code.

    Parameters:
        statistics (pandas.DataFrame): The long statistics, with HUC-12 code strings in huc12_column.
        huc12_column (str): The column of HUC-12 codes.
        value_column (str): The column of water values.

    Returns:
        remapped (pandas.DataFrame): The statistics without old HUC-12 rows.
    """

    crosswalk = load()
    is_old = statistics[huc12_column].isin(crosswalk["old_huc12"])
    if not is_old.any():
        return statistics

    moved = statistics[is_old].merge(
        crosswalk, left_on=huc12_column, right_on="old_huc12"
    )
    moved[huc12_column] = moved["new_huc12"]
    moved[value_column] = moved[value_column] * moved["share"]

    return pandas.concat(
        [statistics[~is_old], moved[statistics.columns]], ignore_index=True
    )
