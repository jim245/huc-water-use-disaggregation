"""
Utility for merging annual HUC-12 water use statistics with HUC geometries. The merge_*_statistics scripts read their
sources and call merge.
"""

import calendar
import tempfile
from collections.abc import Callable, Collection, Sequence
from pathlib import Path

import geopandas
import pandas

from utilities import huc12_crosswalk, raster_utilities


def merge(
    case_study: str,
    source_path: Path,
    read_annual_mgd: Callable[[Path], pandas.DataFrame],
    huc2s: Sequence[str],
    huc_levels: Collection[int],
    reprojected_crs: str,
    year: int | None,
    absent_is_zero: bool = False,
) -> None:
    """
    Sums annual HUC-12 statistics to each HUC level and merges them with the HUC geometries.

    Writes input/statistics/<huc2>/<huc_level>/<case_study>/<year>.csv and .gpkg, with one water_mgd value per HUC.
    The geopackage is a left merge onto the HUC geometries, so HUCs without statistics get NaN, or 0 with
    absent_is_zero. A year's files are rebuilt unless both are newer than the source, the crosswalk and the HUC
    boundaries. The source is read only when a file must be rebuilt, or when year is None.

    Parameters:
        case_study (str): The case study, for the output directory.
        source_path (Path): The source statistics file.
        read_annual_mgd (Callable[[Path], pandas.DataFrame]): Reads source_path. Returns one row per year (the index)
            and one column per HUC-12 code string, in annual mgd. The crosswalk must already be applied.
        huc2s (Sequence[str]): The HUC-2 IDs, without zero-padding.
        huc_levels (Collection[int]): The HUC levels to write.
        reprojected_crs (str): The CRS of the output geopackages.
        year (int | None): The year to write. All years in the source when None.
        absent_is_zero (bool = False): Give 0, not NaN, to a HUC with no statistics.

    Raises:
        ValueError: The source has no rows for year, or a HUC code in the source has no polygon. Without a polygon,
            the gpkg would silently lose that code's water.
    """

    annual_mgd: pandas.DataFrame | None = None
    if year is None:
        annual_mgd = read_annual_mgd(source_path)
        years = annual_mgd.index.tolist()
    else:
        years = [year]

    # (huc2, huc_level, output directory, WBD shapefile, years to rebuild)
    pending: list[tuple[str, int, Path, Path, list[int]]] = []
    for huc2 in huc2s:
        for huc_level in huc_levels:
            directory = Path("input", "statistics", huc2, str(huc_level), case_study)
            huc_path = Path("input", "HUC", huc2, "Shape", f"WBDHU{huc_level}.shp")
            sources = [source_path, huc12_crosswalk.CROSSWALK_PATH, huc_path]
            pending.append((huc2, huc_level, directory, huc_path, stale_years(directory, years, sources)))
    if not any(huc_years for *_, huc_years in pending):
        return

    if annual_mgd is None:
        annual_mgd = read_annual_mgd(source_path)
    if absent_years := [pending_year for pending_year in years if pending_year not in annual_mgd.index]:
        raise ValueError(f"{source_path} has no rows for {absent_years}")

    for huc2, huc_level, directory, huc_path, huc_years in pending:
        if not huc_years:
            continue

        directory.mkdir(parents=True, exist_ok=True)
        huc_column = f"huc{huc_level}"

        huc_gdf = geopandas.read_file(huc_path).to_crs(reprojected_crs)

        region_mgd = annual_mgd[[column for column in annual_mgd.columns if column.startswith(huc2.zfill(2))]]
        truncated = pandas.Series({huc12: huc12[:huc_level] for huc12 in region_mgd.columns})

        if unmatched := sorted(set(truncated) - set(huc_gdf[huc_column])):
            raise ValueError(f"{source_path}: HUC-{huc_level} codes with no polygon in HUC-2 {huc2}: {unmatched}")

        for huc_year in huc_years:
            water_use = region_mgd.loc[huc_year].groupby(truncated).sum().reset_index()
            water_use.columns = [huc_column, "water_mgd"]

            if absent_is_zero:
                absent = huc_gdf.loc[~huc_gdf[huc_column].isin(water_use[huc_column]), huc_column]
                water_use = pandas.concat(
                    [water_use, pandas.DataFrame({huc_column: absent, "water_mgd": 0.0})]
                ).sort_values(huc_column, ignore_index=True)

            merged_gdf = huc_gdf.merge(water_use, on=huc_column, how="left")
            write_year(water_use, merged_gdf, directory, huc_year)


def write_year(water_use: pandas.DataFrame, merged_gdf: geopandas.GeoDataFrame, directory: Path, year: int) -> None:
    """
    Writes <year>.csv and <year>.gpkg to directory.

    Both are written to a temporary directory, then moved, the gpkg last. stale_years accepts any file that exists,
    so a file must appear only when complete.

    Parameters:
        water_use (pandas.DataFrame): The csv rows.
        merged_gdf (geopandas.GeoDataFrame): The gpkg features. The layer takes its name from the file, the year.
        directory (Path): The output directory.
        year (int): The statistics year.
    """

    with tempfile.TemporaryDirectory(dir=directory) as temporary_directory:
        csv_path = Path(temporary_directory, f"{year}.csv")
        gpkg_path = Path(temporary_directory, f"{year}.gpkg")
        water_use.to_csv(csv_path, index=False)
        merged_gdf.to_file(gpkg_path)

        csv_path.replace(directory / csv_path.name)
        gpkg_path.replace(directory / gpkg_path.name)


def stale_years(directory: Path, years: Sequence[int], source_paths: Sequence[Path]) -> list[int]:
    """
    Lists the years whose csv or gpkg in directory is missing or older than a source.

    Parameters:
        directory (Path): The statistics directory.
        years (Sequence[int]): The years to check.
        source_paths (Sequence[Path]): The files the statistics are built from.

    Returns:
        stale (list[int]): The years to rebuild, in the given order.
    """

    sources = [str(source_path) for source_path in source_paths]

    return [
        year
        for year in years
        if not all(
            raster_utilities.is_newer(str(directory / f"{year}.{extension}"), sources) for extension in ("csv", "gpkg")
        )
    ]


def days_in_year(year: int) -> int:
    """
    Gives the number of days in a year.

    Parameters:
        year (int): The year.

    Returns:
        days (int): 366 in a leap year, else 365.
    """

    return 366 if calendar.isleap(year) else 365
