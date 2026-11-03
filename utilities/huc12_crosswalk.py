"""
Utility for moving water use statistics from HUC-12 codes that are not in the downloaded WBD to the HUC-12s that
replaced them (see huc12_crosswalk.csv).
"""

from pathlib import Path

import pandas

CROSSWALK_PATH = Path(__file__).with_name("huc12_crosswalk.csv")


def load() -> pandas.DataFrame:
    """
    Reads the HUC-12 crosswalk and calculates the share of the old code's water that goes to each new code.

    The share is the overlap area divided by the sum of the old code's overlaps. Thus the shares of each old code sum
    to 1, and the crosswalk keeps the total water. Area divides the water of a split code.

    Returns:
        crosswalk (pandas.DataFrame): One row per pair of old and new code, with old_huc12, new_huc12 and share.
    """

    crosswalk = pandas.read_csv(CROSSWALK_PATH, comment="#", dtype={"old_huc12": str, "new_huc12": str})
    crosswalk["share"] = crosswalk["overlap_sqkm"] / crosswalk.groupby("old_huc12")["overlap_sqkm"].transform("sum")

    return crosswalk[["old_huc12", "new_huc12", "share"]]


def remap_columns(statistics: pandas.DataFrame) -> pandas.DataFrame:
    """
    Moves the water of old HUC-12 columns in a wide statistics table (one column per HUC-12) to their new columns.

    The function removes each old column and adds its values, times the share, to the new columns. It makes a new
    column when the table does not have it. A missing value counts as 0. Every value must be water, not a sentinel
    code, because the function splits it.

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
    # a new code that gets water from more than one old code gets the sum
    moved = moved.T.groupby(level=0).sum().T

    remapped = statistics.drop(columns=crosswalk["old_huc12"].unique())
    existing = [huc12 for huc12 in moved.columns if huc12 in remapped.columns]
    added = [huc12 for huc12 in moved.columns if huc12 not in remapped.columns]
    remapped[existing] = remapped[existing].fillna(0) + moved[existing]

    # add the new columns in one step. One insert per column makes a fragmented, slow frame
    return pandas.concat([remapped, moved[added]], axis=1)


def remap_rows(statistics: pandas.DataFrame, huc12_column: str, value_column: str) -> pandas.DataFrame:
    """
    Moves the water of old HUC-12 rows in a long statistics table (one row per HUC-12 and date) to their new codes.

    The function replaces each old row with one row per new code, with the value times the share. It does not merge
    these rows with the rows of a new code that has statistics, so the caller must sum by code.

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

    moved = statistics[is_old].merge(crosswalk, left_on=huc12_column, right_on="old_huc12")
    moved[huc12_column] = moved["new_huc12"]
    moved[value_column] = moved[value_column] * moved["share"]

    return pandas.concat([statistics[~is_old], moved[statistics.columns]], ignore_index=True)
