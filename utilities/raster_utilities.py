"""
Utility for handling common raster operations.
"""

import glob
import os
from dataclasses import dataclass
from os import path
from typing import Any, Optional, cast

# Activating the conda environment sets PROJ_NETWORK=ON, which lets PROJ download grid-based datum shifts and moves
# WGS84 sources (NLCD, DARTE, GDP, WorldPop) ~0.3 m relative to running .cenv/bin/python directly (FIXES.md fix 23).
# Pinning it off keeps every launch on the same transformation, independent of internet access and PROJ's grid cache
os.environ["PROJ_NETWORK"] = "OFF"

# isort: split

import affine
import geopandas
import numpy
import pandas
import rasterio
import shapely
from numpy.typing import NDArray
from rasterio import enums, features, io
from rasterio import transform as rasterio_transform
from rasterio import warp

# all source statistics have 12-digit HUC IDs. The merge scripts sum them to coarser levels by truncation, so they
# can't supply levels 14 and 16. The project tested only levels 8 and 12
HUC_LEVELS = [2, 4, 6, 8, 10, 12]
STATISTICS_HUC_LEVEL = 12


@dataclass(frozen=True)
class CaseStudy:
    """
    The settings of one case study. main.py selects the merge and raster preparation scripts for each case study.

    Attributes:
        sector (str): The water use sector, for the figure title.
        measure (str): The quantity that the statistics report, for the figure title and label.
        first_year (int): The first year in the source statistics.
        last_year (int): The last year in the source statistics.
        support_layer (Optional[int] = None): The index of the prepared layer whose cells > 0 can receive water. All
            cells can receive water when None.
    """

    sector: str
    measure: str
    first_year: int
    last_year: int
    support_layer: Optional[int] = None


CASE_STUDIES = {
    # layer 0 is NLCD 81/82 (planted/cultivated). Without a support limit, the final irr 2000 map had 9.5% of its water
    # on cells with no cropland
    "irr": CaseStudy("Irrigation", "Water Consumption", 2000, 2020, support_layer=0),
    # pws has no support limit: its layers are already clipped to the water service areas, and the final pws-tot 2000
    # map had only 1.07% of its water on cells with no developed land
    "pws-cu": CaseStudy("Public Water Supply", "Water Consumption", 2009, 2020),
    "pws-gw": CaseStudy("Public Water Supply", "Groundwater Use", 2000, 2020),
    "pws-sw": CaseStudy("Public Water Supply", "Surface Water Use", 2000, 2020),
    "pws-tot": CaseStudy("Public Water Supply", "Total Water Use", 2000, 2020),
}


@dataclass(frozen=True)
class Grid:
    """
    The cell grid shared by the template, every prepared raster and every pipeline output.

    Attributes:
        transform (affine.Affine): The grid's affine transform.
        crs (Any): The grid's CRS.
        shape (tuple[int, int]): The grid's (height, width).
    """

    transform: affine.Affine
    crs: Any
    shape: tuple[int, int]

    @classmethod
    def from_raster(cls, raster_path: str) -> "Grid":
        """
        Reads the grid of an existing raster (the template, in main.py).

        Parameters:
            raster_path (str): The path to the raster.

        Returns:
            grid (Grid): The raster's grid.
        """

        with rasterio.open(raster_path) as raster:
            return cls(
                cast(affine.Affine, raster.transform),
                raster.crs,
                cast(tuple[int, int], raster.shape),
            )

    def profile(self) -> dict[str, Any]:
        """
        Builds the rasterio profile of a single-band float32 GeoTIFF on this grid, with NaN as nodata.

        Returns:
            profile (dict[str, Any]): The profile.
        """

        return {
            "driver": "GTiff",
            "height": self.shape[0],
            "width": self.shape[1],
            "count": 1,
            "dtype": numpy.float32,
            "nodata": numpy.nan,
            "crs": self.crs,
            "transform": self.transform,
        }


def clip_raster(
    raster_path: str, raster_mask: NDArray[numpy.bool_], output_path: str
) -> str:
    """
    Clips the supplied raster using the provided mask.

    Cells outside the mask are set to the raster's nodata (0 if it has none); the raster's extent is unchanged.

    Parameters:
        raster_path (str): The raster path to clip.
        raster_mask (NDArray[numpy.bool_]): True for the cells to keep, on the raster's grid (see rasterize_mask).
        output_path (str): The path to the output file.

    Returns:
        output_path (str): The path to the output file.
    """

    with rasterio.open(raster_path) as original:
        if original.shape != raster_mask.shape:
            raise ValueError(f"{raster_path} and the mask aren't on the same grid")
        raster = cast(NDArray[Any], original.read(1))
        nodata = cast(Optional[float], original.nodata)
        profile = cast(dict[str, Any], original.profile)

    raster[~raster_mask] = nodata if nodata is not None else 0

    with rasterio.open(output_path, "w", **profile) as output_raster:
        output_raster.write(raster, 1)

    return output_path


def extrapolate_raster(raster_path: str, target_year: int) -> str:
    """
    Extrapolates a raster set to the target year using pixel-wise linear regression over sibling rasters.

    raster_path is the (missing) path for target_year and must contain the year; its first occurrence is replaced
    with a wildcard to find siblings. Every sibling with an earlier year and the same shape as the latest one is used
    to fit a straight line per pixel. Earlier "-predicted.tif" outputs are skipped (their year doesn't parse), so
    predictions never feed later predictions. Pixels that are nodata in any year are NaN. The trend can go negative:
    log1p normalization clamps negatives to 0, min-max normalization does not.

    Parameters:
        raster_path (str): The path the raster for target_year would have.
        target_year (int): The year to extrapolate to.

    Returns:
        predicted_raster_path (str): The path to the output file, raster_path with a "-predicted.tif" suffix.
    """

    known_raster_paths = find_sibling_rasters(raster_path, target_year)

    if not known_raster_paths:
        raise FileNotFoundError(
            f"No sibling rasters found for extrapolation of {raster_path!r} "
            f"with a year < {target_year}."
        )

    years = sorted(known_raster_paths.keys())

    with rasterio.open(known_raster_paths[years[-1]]) as ref:
        ref_shape = (ref.height, ref.width)

    arrays = []
    matched_years = []
    profile: Optional[dict[str, Any]] = None
    nodata_val = None

    for year in years:
        with rasterio.open(known_raster_paths[year]) as src:
            if (src.height, src.width) != ref_shape:
                continue
            if profile is None:
                profile = src.profile.copy()
                nodata_val = src.nodata
            arrays.append(src.read(1).astype(numpy.float64))
            matched_years.append(year)

    stacked = numpy.stack(arrays, axis=0)  # (T, H, W)
    T, H, W = stacked.shape

    # pixels that are valid across all known years
    if nodata_val is not None and not numpy.isnan(float(nodata_val)):
        valid_mask = numpy.all(stacked != nodata_val, axis=0)
    else:
        valid_mask = numpy.all(~numpy.isnan(stacked), axis=0)

    # design matrix [year, 1] shape (T, 2)
    X = numpy.stack(
        [numpy.array(matched_years, dtype=numpy.float64), numpy.ones(T)], axis=1
    )

    # solve for coefficients over valid pixels only: (2, N_valid)
    flat = stacked[:, valid_mask]
    coefficients, _, _, _ = numpy.linalg.lstsq(X, flat, rcond=None)

    result = numpy.full((H, W), numpy.nan, dtype=numpy.float32)
    result[valid_mask] = (coefficients[0] * target_year + coefficients[1]).astype(
        numpy.float32
    )

    assert profile is not None
    profile.update(dtype=numpy.float32, nodata=numpy.nan)
    os.makedirs(path.dirname(raster_path), exist_ok=True)
    predicted_raster_path = path.join(
        path.dirname(raster_path),
        path.splitext(path.basename(raster_path))[0] + "-predicted.tif",
    )

    with rasterio.open(predicted_raster_path, "w", **profile) as output_raster:
        output_raster.write(result, 1)

    return predicted_raster_path


def filter_raster(raster_path: str, filters: list[int], output_path: str) -> str:
    """
    Filters the provided raster with the supplied filters.

    Cells whose value is in filters keep that value (e.g. their NLCD class code); all other cells become 0. The result
    is not a 1/0 mask.

    Parameters:
        raster_path (str): The path to the raster to filter.
        filters (list[int]): The values to filter the raster band for.
        output_path (str): The path to the output file.

    Returns:
        output_path (str): The path to the output file.
    """

    original = cast(io.DatasetReader, rasterio.open(raster_path))
    data = cast(numpy.ndarray, original.read(1))
    profile = cast(dict[str, Any], original.profile)
    original.close()

    # filter the raster to cells containing values only in filters
    filtered_raster = numpy.where(numpy.isin(data, filters), data, 0)

    outfile = cast(io.DatasetWriter, rasterio.open(output_path, "w", **profile))
    outfile.write(filtered_raster, 1)
    outfile.close()

    return output_path


def find_sibling_rasters(raster_path: str, target_year: int) -> dict[int, str]:
    """
    Finds the rasters of earlier years in the same series as raster_path.

    raster_path must contain target_year; its first occurrence is replaced with a wildcard. Matches whose year part
    isn't an integer (e.g. earlier "-predicted.tif" outputs) are skipped.

    Parameters:
        raster_path (str): The path the raster for target_year would have.
        target_year (int): The year whose earlier siblings to find.

    Returns:
        known_raster_paths (dict[int, str]): The path of each earlier year's raster, by year.
    """

    year_str = str(target_year)
    glob_pattern = raster_path.replace(year_str, "*", 1)
    prefix_len = raster_path.index(year_str)
    suffix_len = len(raster_path) - prefix_len - len(year_str)

    known_raster_paths: dict[int, str] = {}
    for match in glob.glob(glob_pattern):
        candidate_year_str = match[
            prefix_len : len(match) - suffix_len if suffix_len else None
        ]
        try:
            candidate_year = int(candidate_year_str)
        except ValueError:
            continue
        if candidate_year < target_year:
            known_raster_paths[candidate_year] = match

    return known_raster_paths


def generate_mask(
    primary_mask_paths: list[str],
    secondary_mask_paths: Optional[list[str]] = None,
    reprojected_crs: str = "EPSG:5070",
) -> geopandas.GeoDataFrame:
    """
    Generates a geometric mask from a set of primary and secondary masks.

    The result is the primary polygons (not dissolved), clipped to the union of the secondary polygons when any are
    given.

    Parameters:
        primary_mask_paths (list[str]): A list of paths to the primary masking shapefiles.
        secondary_mask_paths (Optional[list[str]] = None): A list of paths to other masking shapefiles.
        reprojected_crs (str = "EPSG:5070"): The CRS of the generated mask.

    Returns:
        return (geopandas.GeoDataFrame): The GeoDataFrame mask.
    """

    primary_masks = [
        geopandas.read_file(primary, columns=["geometry"]).to_crs(reprojected_crs)
        for primary in primary_mask_paths
    ]

    primary_mask = pandas.concat(primary_masks)
    primary_mask = geopandas.GeoDataFrame(primary_mask, geometry="geometry")

    valid_primary = shapely.make_valid(primary_mask.geometry.values)
    try:
        primary_union = shapely.coverage_union_all(valid_primary)
    except shapely.errors.GEOSException:
        primary_union = shapely.unary_union(valid_primary)

    if secondary_mask_paths:
        # a GeoSeries with a CRS is reprojected to each file's CRS; a bare geometry would be assumed to be in it
        primary_union_series = geopandas.GeoSeries([primary_union], crs=reprojected_crs)
        secondary_masks = [
            geopandas.read_file(
                secondary, columns=["geometry"], mask=primary_union_series
            ).to_crs(reprojected_crs)
            for secondary in secondary_mask_paths
        ]

        secondary_mask = pandas.concat(secondary_masks)
        secondary_mask = geopandas.GeoDataFrame(secondary_mask, geometry="geometry")

        valid_secondary = shapely.make_valid(secondary_mask.geometry.values)
        try:
            secondary_union = shapely.coverage_union_all(valid_secondary)
        except shapely.errors.GEOSException:
            secondary_union = shapely.unary_union(valid_secondary)
        primary_mask = primary_mask.clip(secondary_union)

    return primary_mask


def huc2_cache_key(huc2s: list[str]) -> str:
    """
    Builds a filename-safe key identifying a HUC-2 selection, for naming files cached per selection.

    Parameters:
        huc2s (list[str]): A list of HUC2 numerical ID strings.

    Returns:
        return (str): The key, e.g. "huc2-14-15". Order, duplicates and zero padding don't change it.
    """

    return "huc2-" + "-".join(
        str(huc2) for huc2 in sorted({int(huc2) for huc2 in huc2s})
    )


def is_newer(output_path: str, source_paths: list[str]) -> bool:
    """
    Checks whether a cached file exists and was written after all of its sources were last modified.

    Parameters:
        output_path (str): The cached file.
        source_paths (list[str]): The files it was built from.

    Returns:
        current (bool): False when the file is missing, a source is missing or newer, or there are no sources.
    """

    if not source_paths or not path.exists(output_path):
        return False
    if not all(path.exists(source_path) for source_path in source_paths):
        return False

    return path.getmtime(output_path) > max(
        path.getmtime(source_path) for source_path in source_paths
    )


def normalize_raster(
    raster_path: str,
    output_path: str,
    method: str = "minmax",
) -> str:
    """
    Normalizes the supplied raster to [0, 1].

    "minmax" scales valid cells linearly from their minimum to their maximum. "log1p" clamps negatives to 0, applies
    log1p, then scales the same way. Nodata cells are NaN in the output.

    Parameters:
        raster_path (str): The path to the input raster.
        output_path (str): The path to the output file.
        method (str = "minmax"): The normalization method, one of "minmax" or "log1p".

    Returns:
        output_path (str): The path to the output file.
    """

    with rasterio.open(raster_path) as src:
        data = src.read(1).astype(numpy.float32)
        profile = src.profile.copy()
        nodata = src.nodata

    # build a mask of cells that carry real data
    if nodata is not None and not numpy.isnan(float(nodata)):
        valid = data != nodata
    else:
        valid = ~numpy.isnan(data)

    result = numpy.full(data.shape, numpy.nan, dtype=numpy.float32)

    if method == "minmax":
        low = float(numpy.nanmin(data[valid]))
        high = float(numpy.nanmax(data[valid]))
        if high > low:
            result[valid] = (data[valid] - low) / (high - low)
        else:
            result[valid] = 0.0

    elif method == "log1p":
        # clip negatives to 0 before log1p (counts/densities are non-negative)
        log_data = numpy.log1p(numpy.maximum(data, 0.0))
        low = float(numpy.nanmin(log_data[valid]))
        high = float(numpy.nanmax(log_data[valid]))
        if high > low:
            result[valid] = (log_data[valid] - low) / (high - low)
        else:
            result[valid] = 0.0

    else:
        raise ValueError(
            f"Unknown normalization method {method!r}. "
            "Choose from 'minmax' or 'log1p'."
        )

    # clamp to [0, 1] to guard against floating-point drift
    numpy.clip(result, 0.0, 1.0, out=result)
    result[~valid] = numpy.nan

    profile.update(dtype=numpy.float32, nodata=numpy.nan)
    with rasterio.open(output_path, "w", **profile) as output_raster:
        output_raster.write(result, 1)

    return output_path


def prepare_raster_layer(
    output_directory: str,
    raster_path: str,
    reprojected_crs: str,
    year: int,
    raster_title: str,
    template_raster_path: str,
    raster_mask: NDArray[numpy.bool_],
    filter_keys: Optional[list[int]] = None,
    resampling_method: enums.Resampling = enums.Resampling.sum,
    normalization_method: str = "log1p",
    mask_key: str = "",
) -> str:
    """
    Prepares a raster for use in the disaggregation model.

    Steps, each writing an intermediate file under output_directory:
      1. Extrapolate the raster from earlier years if raster_path doesn't exist (see extrapolate_raster).
      2. Filter to filter_keys, if given (filtered/).
      3. Reproject and resample onto the template grid (reprojected/).
      4. Clip to raster_mask (clipped/).
      5. Normalize (normalized/<normalization_method>/).
    Steps 1-3 are cached: the reprojected file is named after everything it depends on (year, title, filter keys,
    resampling method and template) and reused while it's newer than its sources (the raster, or the earlier years an
    extrapolated raster is fitted to, and the template). Steps 4-5 depend on the mask and always rerun.

    Parameters:
        output_directory (str): The output directory of the prepared raster.
        raster_path (str): The unprepared raster path.
        reprojected_crs (str): The target CRS for the generated rasters.
        year (int): The year to generate rasters for.
        raster_title (str): The title of the raster for use in the filename.
        template_raster_path (str): The path to the template raster.
        raster_mask (NDArray[numpy.bool_]): True for the template cells to keep (see rasterize_mask).
        filter_keys (Optional[list[int]] = None): A list of integer values to filter the raster to.
        resampling_method (enums.Resampling = enums.Resampling.sum): The raster resampling method to use.
        normalization_method (str = "log1p"): The normalization method to use.
        mask_key (str = ""): A name for the mask, added to the clipped and normalized filenames. Use it when the mask
            changes between runs, so that concurrent runs with different masks don't overwrite each other's files.

    Returns:
        normalized_raster_path (str): The path to the prepared raster.
    """

    # e.g. [81, 82] -> "81-82", keeping brackets and spaces out of filenames
    filter_key = (
        "" if filter_keys is None else "_" + "-".join(str(key) for key in filter_keys)
    )
    template_key = path.splitext(path.basename(template_raster_path))[0]
    mask_suffix = f"_{mask_key}" if mask_key else ""

    reprojected_directory = path.join(output_directory, "reprojected")
    os.makedirs(reprojected_directory, exist_ok=True)
    reprojected_raster_path = path.join(
        reprojected_directory,
        f"{year}_{raster_title}{filter_key}_{resampling_method.name}_{template_key}.tif",
    )

    source_paths = (
        [raster_path]
        if path.exists(raster_path)
        else list(find_sibling_rasters(raster_path, year).values())
    )

    if not is_newer(reprojected_raster_path, source_paths + [template_raster_path]):
        if not path.exists(raster_path):
            raster_path = extrapolate_raster(raster_path, year)

        if filter_keys is not None:
            filtered_directory = path.join(output_directory, "filtered")
            os.makedirs(filtered_directory, exist_ok=True)

            raster_path = filter_raster(
                raster_path,
                filter_keys,
                path.join(filtered_directory, f"{year}_{raster_title}{filter_key}.tif"),
            )

        reproject_raster(
            raster_path,
            reprojected_crs,
            reprojected_raster_path,
            template_raster_path,
            resampling_method=resampling_method,
        )

    clipped_directory = path.join(output_directory, "clipped")
    os.makedirs(clipped_directory, exist_ok=True)

    clipped_raster_path = clip_raster(
        reprojected_raster_path,
        raster_mask,
        path.join(clipped_directory, f"{year}_{raster_title}{mask_suffix}.tif"),
    )

    normalized_directory = path.join(
        output_directory, "normalized", normalization_method
    )
    os.makedirs(normalized_directory, exist_ok=True)

    normalized_raster_path = normalize_raster(
        clipped_raster_path,
        path.join(normalized_directory, f"{year}_{raster_title}{mask_suffix}.tif"),
        normalization_method,
    )

    return normalized_raster_path


def rasterize_mask(geometries: geopandas.GeoSeries, grid: Grid) -> NDArray[numpy.bool_]:
    """
    Rasterizes a geometric mask onto the grid: True for every cell the geometries touch.

    This is the rasterization rasterio.mask.mask does with all_touched=True, so clipping with the result keeps the
    same cells.

    Parameters:
        geometries (geopandas.GeoSeries): The mask geometries.
        grid (Grid): The grid to rasterize onto.

    Returns:
        mask (NDArray[numpy.bool_]): The mask raster.
    """

    return features.geometry_mask(
        geometries.to_crs(grid.crs),
        out_shape=grid.shape,
        transform=grid.transform,
        all_touched=True,
        invert=True,
    )


def rasterize_zones(
    geometries: geopandas.GeoSeries,
    zone_ids: Any,
    shape: tuple[int, int],
    transform: affine.Affine,
) -> tuple[NDArray[numpy.int32], NDArray[numpy.int64]]:
    """
    Rasterizes polygons into compact zone indices for vectorized per-zone operations.

    Cells are labeled 0 outside all polygons and i for the zone zone_ids_unique[i - 1], so per-zone
    sums are numpy.bincount(zones.ravel(), weights, minlength=len(zone_ids_unique) + 1) and per-zone
    values broadcast back to cells as values[zones]. Polygons sharing an ID share one zone.

    Parameters:
        geometries (geopandas.GeoSeries): The zone polygons, in the raster CRS.
        zone_ids (Any): The integer-like zone ID of each polygon.
        shape (tuple[int, int]): The (height, width) of the output raster.
        transform (affine.Affine): The transform of the output raster.

    Returns:
        zones (NDArray[numpy.int32]): The zone index raster.
        zone_ids_unique (NDArray[numpy.int64]): The zone ID for each index, offset by one.
    """

    codes, zone_ids_unique = pandas.factorize(
        numpy.asarray(zone_ids).astype(numpy.int64)
    )
    values = (codes + 1).tolist()

    # rasterio holds every shape as nested Python coordinate tuples until it burns them (~7 GB for HUC-2s 14-18 at
    # HUC-12), so burn in consecutive batches of ~1M coordinates; later polygons still overwrite earlier ones, as in
    # one call (see FIXES.md, fix 29)
    batches = numpy.cumsum(shapely.get_num_coordinates(geometries.values)) // 1_000_000
    starts = numpy.flatnonzero(numpy.diff(batches, prepend=-1))
    ends = numpy.append(starts[1:], len(values))

    zones = numpy.zeros(shape, dtype=numpy.int32)
    for start, end in zip(starts, ends):
        features.rasterize(
            zip(geometries.iloc[start:end], values[start:end]),
            out=zones,
            transform=transform,
            all_touched=True,
        )

    return zones, numpy.asarray(zone_ids_unique, dtype=numpy.int64)


def read_raster(raster_path: str, grid: Grid) -> NDArray[Any]:
    """
    Reads the first band of a raster that must lie on the given grid.

    Parameters:
        raster_path (str): The path to the raster.
        grid (Grid): The grid the raster must be on.

    Returns:
        data (NDArray[Any]): The band, in the raster's own dtype.
    """

    with rasterio.open(raster_path) as raster:
        if raster.shape != grid.shape or raster.transform != grid.transform:
            raise ValueError(f"{raster_path} isn't on the template grid")
        return raster.read(1)


def read_statistics(
    huc2s: list[str],
    huc_level: int,
    case_study: str,
    year: int,
    columns: Optional[list[str]] = None,
) -> geopandas.GeoDataFrame:
    """
    Reads the merged water use statistics (input/statistics/<huc2>/<huc_level>/<case_study>/<year>.gpkg) of the
    given HUC-2s, concatenated in huc2s order. The order decides which HUC gets a shared boundary cell when the
    polygons are rasterized (the later one), so every stage must read them in the same order.

    Parameters:
        huc2s (list[str]): A list of HUC2 numerical ID strings.
        huc_level (int): The HUC level of the statistics.
        case_study (str): The case study.
        year (int): The statistics year.
        columns (Optional[list[str]] = None): The columns to read. All when None.

    Returns:
        statistics (geopandas.GeoDataFrame): The statistics, with a fresh index.
    """

    return geopandas.GeoDataFrame(
        pandas.concat(
            [
                geopandas.read_file(
                    path.join(
                        "input",
                        "statistics",
                        huc2,
                        str(huc_level),
                        case_study,
                        f"{year}.gpkg",
                    ),
                    columns=columns,
                )
                for huc2 in huc2s
            ],
            ignore_index=True,
        )
    )


def reproject_raster(
    raster_path: str,
    reprojected_crs: str,
    output_path: str,
    reference_raster_path: str,
    dtype: type = numpy.float32,
    nodata: Any = numpy.nan,
    resampling_method: enums.Resampling = enums.Resampling.mode,
) -> str:
    """
    Reprojects (and resamples if necessary) the provided raster onto the reference raster's grid.

    Parameters:
        raster_path (str): The path to the raster to reproject.
        reprojected_crs (str): The target CRS for the generated rasters.
        output_path (str): The path to the output file.
        reference_raster_path (str): The path to the reference raster.
        dtype (type = numpy.float32): The datatype the raster should be in.
        nodata (Any = numpy.nan): The nodata value to use in the raster.
        resampling_method (enums.Resampling = enums.Resampling.mode): The raster resampling method to use.

    Returns:
        output_path (str): The path to the output file.
    """

    with rasterio.open(raster_path) as original:
        source = cast(numpy.ndarray, original.read(1))
        profile = cast(dict[str, Any], original.profile)
        source_transform = cast(affine.Affine, original.transform)
        source_crs = original.crs
    profile.update(compress="lzw", driver="GTiff")

    # replace the source's own nodata value with nodata, so missing cells enter resampling as NaN
    old_nodata = profile.get("nodata")
    if old_nodata is not None and old_nodata != nodata:
        if numpy.isnan(old_nodata):
            mask = numpy.isnan(source)
        else:
            mask = source == old_nodata
        source = numpy.where(mask, nodata, source)
    profile.update(nodata=nodata)

    source = source.astype(dtype, copy=False)
    profile.update(dtype=dtype)

    with rasterio.open(reference_raster_path) as reference:
        transform = cast(affine.Affine, reference.transform)
        width = cast(int, reference.width)
        height = cast(int, reference.height)

    profile.update(crs=reprojected_crs, height=height, transform=transform, width=width)

    destination = numpy.empty((height, width), dtype)

    warp.reproject(
        source,
        destination,
        src_transform=source_transform,
        src_crs=source_crs,
        dst_transform=transform,
        dst_crs=reprojected_crs,
        resampling=resampling_method,
    )

    outfile = cast(io.DatasetWriter, rasterio.open(output_path, "w", **profile))
    outfile.write(destination, 1)
    outfile.close()

    return output_path


def templatize_geometries(
    geometry_paths: list[str], reprojected_crs: str, cell_size: int, output_path: str
) -> str:
    """
    Rasterizes the combined extent of the provided geometries into a binary template raster.

    The raster covers the geometries' bounding box, with 1 in cells touching a geometry and 0 (nodata) elsewhere.

    Parameters:
        geometry_paths (list[str]): A list of filepaths to geometry files.
        reprojected_crs (str): The target CRS for the produced template.
        cell_size (int): The target cell size (in CRS units) for the produced template.
        output_path (str): The path to the output file.

    Returns:
        output_path (str): The path to the output file.
    """

    geometries = pandas.concat(
        [geopandas.read_file(geometry_path) for geometry_path in geometry_paths]
    )
    geometries = geopandas.GeoDataFrame(geometries, geometry="geometry")
    geometries.to_crs(reprojected_crs, inplace=True)

    min_x, min_y, max_x, max_y = geometries.total_bounds
    raster_width = int(abs(max_x - min_x) / cell_size)
    raster_height = int(abs(max_y - min_y) / cell_size)

    # calculate the template transform from the new bounds and cell size
    new_transform = rasterio_transform.from_origin(min_x, max_y, cell_size, cell_size)

    raster = cast(
        NDArray[numpy.uint8],
        features.rasterize(
            [(geometry, 1) for geometry in geometries.geometry],
            (raster_height, raster_width),
            nodata=0,
            transform=new_transform,
            all_touched=True,
            dtype=numpy.uint8,
        ),
    )
    raster_metadata: dict[str, Any] = {
        "count": 1,
        "crs": geometries.crs,
        "driver": "GTiff",
        "dtype": numpy.uint8,
        "height": raster_height,
        "nodata": 0,
        "transform": new_transform,
        "width": raster_width,
    }

    with rasterio.open(output_path, "w", **raster_metadata) as output_raster:
        output_raster.write(raster, 1)

    return output_path


def write_raster(output_path: str, data: NDArray[numpy.float32], grid: Grid) -> str:
    """
    Writes a single-band float32 GeoTIFF on the grid, with NaN as nodata.

    The function writes to a temporary file in the same directory, then renames it. An interrupted write therefore
    can't leave a partial raster at output_path.

    Parameters:
        output_path (str): The path to the output file.
        data (NDArray[numpy.float32]): The (height, width) values.
        grid (Grid): The grid the values are on.

    Returns:
        output_path (str): The path to the output file.
    """

    # the process ID keeps concurrent runs from sharing a temporary file
    temporary_path = f"{output_path}.{os.getpid()}.tmp"
    with rasterio.open(temporary_path, "w", **grid.profile()) as output_raster:
        output_raster.write(data, 1)
    os.replace(temporary_path, output_path)

    return output_path
