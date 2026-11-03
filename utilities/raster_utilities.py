"""
Utility for handling common raster operations.
"""

import glob
import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

# Activating the conda environment sets PROJ_NETWORK=ON. PROJ can then download grid-based datum shifts, which move the
# WGS84 sources (NLCD, DARTE, GDP, WorldPop) relative to a run of .cenv/bin/python without activation. Pinning it off
# keeps every run on the same transformation, with or without internet access and PROJ's grid cache
os.environ["PROJ_NETWORK"] = "OFF"

# isort: split

import affine
import geopandas
import numpy
import pandas
import rasterio
import shapely
from numpy.typing import NDArray
from rasterio import enums, features, warp
from rasterio import transform as rasterio_transform
from shapely.geometry import MultiPolygon, Polygon

HUC_LEVELS = [2, 4, 6, 8, 10, 12]
STATISTICS_HUC_LEVEL = 12

# the HUC-2 regions of the conterminous US, without zero-padding. The source statistics have no regions above 18
HUC2S: tuple[str, ...] = tuple(str(huc2) for huc2 in range(1, 19))


@dataclass(frozen=True)
class CaseStudy:
    """
    The settings of one case study. main.py selects the merge and raster preparation scripts for each case study.

    Attributes:
        sector (str): The water use sector, for the figure title.
        measure (str): The quantity that the statistics report, for the figure title and label.
        first_year (int): The first year in the source statistics.
        last_year (int): The last year in the source statistics.
        support_layer (int | None = None): The index of the prepared layer whose cells > 0 can receive water. All
            cells can receive water when None.
    """

    sector: str
    measure: str
    first_year: int
    last_year: int
    support_layer: int | None = None


CASE_STUDIES = {
    # layer 0 is NLCD 81/82 (planted/cultivated). Without a support limit, irr puts water on cells with no cropland
    "irr": CaseStudy("Irrigation", "Water Consumption", 2000, 2020, support_layer=0),
    # pws has no support limit, because its layers are already clipped to the water service areas
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
    def from_raster(cls, raster_path: str | Path) -> "Grid":
        """
        Reads the grid of an existing raster (the template, in main.py).

        Parameters:
            raster_path (str | Path): The path to the raster.

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


def extrapolate_raster(raster_path: Path, target_year: int) -> Path:
    """
    Extrapolates a raster set to the target year using pixel-wise linear regression over sibling rasters.

    raster_path is the (missing) path for target_year and must contain the year. Its first occurrence is replaced
    with a wildcard to find siblings. Every sibling with an earlier year and the same shape and transform as the latest
    one is used to fit a straight line per pixel. Earlier "-predicted.tif" outputs are skipped (their year doesn't
    parse), so predictions never feed later predictions. Pixels that are nodata in any year are NaN. The trend can go
    negative: log1p normalization clamps negatives to 0, and min-max normalization doesn't.

    Parameters:
        raster_path (Path): The path the raster for target_year would have.
        target_year (int): The year to extrapolate to.

    Returns:
        predicted_raster_path (Path): The path to the output file, raster_path with a "-predicted.tif" suffix.
    """

    known_raster_paths = find_sibling_rasters(str(raster_path), target_year)

    if not known_raster_paths:
        raise FileNotFoundError(
            f"No sibling rasters found for extrapolation of {str(raster_path)!r} with a year < {target_year}."
        )

    years = sorted(known_raster_paths.keys())

    with rasterio.open(known_raster_paths[years[-1]]) as reference:
        reference_grid = (reference.shape, reference.transform)

    arrays = []
    matched_years = []
    profile: dict[str, Any] | None = None
    nodata_val = None

    for year in years:
        with rasterio.open(known_raster_paths[year]) as src:
            if (src.shape, src.transform) != reference_grid:
                continue
            if profile is None:
                profile = src.profile.copy()
                nodata_val = src.nodata
            arrays.append(src.read(1).astype(numpy.float64))
            matched_years.append(year)

    stacked = numpy.stack(arrays, axis=0)  # (years, height, width)

    # pixels that are valid across all known years
    if nodata_val is not None and not numpy.isnan(float(nodata_val)):
        valid_mask = numpy.all(stacked != nodata_val, axis=0)
    else:
        valid_mask = numpy.all(~numpy.isnan(stacked), axis=0)

    # design matrix [year, 1] shape (years, 2)
    design = numpy.stack(
        [numpy.array(matched_years, dtype=numpy.float64), numpy.ones(len(matched_years))], axis=1
    )

    # solve for coefficients over valid pixels only: (2, N_valid)
    coefficients, _, _, _ = numpy.linalg.lstsq(design, stacked[:, valid_mask], rcond=None)

    result = numpy.full(valid_mask.shape, numpy.nan, dtype=numpy.float32)
    result[valid_mask] = (coefficients[0] * target_year + coefficients[1]).astype(numpy.float32)

    assert profile is not None
    profile.update(dtype=numpy.float32, nodata=numpy.nan)
    raster_path.parent.mkdir(parents=True, exist_ok=True)
    predicted_raster_path = raster_path.with_name(f"{raster_path.stem}-predicted.tif")
    _write_atomically(predicted_raster_path, result, profile)

    return predicted_raster_path


def filter_raster(raster_path: Path, filters: Sequence[int], output_path: Path) -> Path:
    """
    Filters the provided raster with the supplied filters.

    Cells whose value is in filters become 1, and all other cells become 0. Average resampling then gives the share of
    each target cell in the kept classes. Keeping the class codes would weight one class over another by its code.

    Parameters:
        raster_path (Path): The path to the raster to filter.
        filters (Sequence[int]): The values to keep.
        output_path (Path): The path to the output file.

    Returns:
        output_path (Path): The path to the output file.
    """

    with rasterio.open(raster_path) as original:
        data = cast(NDArray[Any], original.read(1))
        profile = cast(dict[str, Any], original.profile)

    _write_atomically(output_path, numpy.isin(data, filters).astype(data.dtype), profile)

    return output_path


def find_sibling_rasters(raster_path: str, target_year: int) -> dict[int, str]:
    """
    Finds the rasters of earlier years in the same series as raster_path.

    raster_path must contain target_year. Its first occurrence is replaced with a wildcard. Matches whose year part
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
        candidate_year_str = match[prefix_len : len(match) - suffix_len if suffix_len else None]
        try:
            candidate_year = int(candidate_year_str)
        except ValueError:
            continue
        if candidate_year < target_year:
            known_raster_paths[candidate_year] = match

    return known_raster_paths


def generate_mask(
    primary_mask_paths: Sequence[Path],
    secondary_mask_paths: Sequence[Path] = (),
    reprojected_crs: str = "EPSG:5070",
) -> geopandas.GeoDataFrame:
    """
    Generates a geometric mask from a set of primary and secondary masks.

    The result is the primary polygons (not dissolved), clipped to the union of the secondary polygons when any are
    given.

    Parameters:
        primary_mask_paths (Sequence[Path]): The paths to the primary masking shapefiles.
        secondary_mask_paths (Sequence[Path] = ()): The paths to other masking shapefiles.
        reprojected_crs (str = "EPSG:5070"): The CRS of the generated mask.

    Returns:
        return (geopandas.GeoDataFrame): The GeoDataFrame mask.
    """

    primary_masks = [
        geopandas.read_file(primary, columns=["geometry"]).to_crs(reprojected_crs) for primary in primary_mask_paths
    ]

    primary_mask = pandas.concat(primary_masks)
    primary_mask = geopandas.GeoDataFrame(primary_mask, geometry="geometry")

    valid_primary = shapely.make_valid(primary_mask.geometry.values)
    try:
        primary_union = shapely.coverage_union_all(valid_primary)
    except shapely.errors.GEOSException:
        primary_union = shapely.unary_union(valid_primary)

    if secondary_mask_paths:
        # a GeoSeries with a CRS is reprojected to each file's CRS. A bare geometry would be assumed to be in it
        primary_union_series = geopandas.GeoSeries([primary_union], crs=reprojected_crs)
        secondary_masks = [
            geopandas.read_file(secondary, columns=["geometry"], mask=primary_union_series).to_crs(reprojected_crs)
            for secondary in secondary_mask_paths
        ]

        secondary_mask = pandas.concat(secondary_masks)
        secondary_mask = geopandas.GeoDataFrame(secondary_mask, geometry="geometry")

        valid_secondary = shapely.make_valid(secondary_mask.geometry.values)
        try:
            secondary_union = shapely.coverage_union_all(valid_secondary)
        except shapely.errors.GEOSException:
            secondary_union = shapely.unary_union(valid_secondary)
        primary_mask = primary_mask.clip(cast(Polygon | MultiPolygon, secondary_union))

    return primary_mask


def huc2_cache_key(huc2s: Sequence[str]) -> str:
    """
    Builds a filename-safe key identifying a HUC-2 selection, for naming files cached per selection.

    Parameters:
        huc2s (Sequence[str]): The HUC-2 IDs.

    Returns:
        return (str): The key, e.g. "huc2-14-15". Order, duplicates and zero padding don't change it.
    """

    return "huc2-" + "-".join(str(huc2) for huc2 in sorted({int(huc2) for huc2 in huc2s}))


def is_newer(output_path: str | Path, source_paths: Sequence[str | Path]) -> bool:
    """
    Checks whether a cached file exists and was written after all of its sources were last modified.

    Parameters:
        output_path (str | Path): The cached file.
        source_paths (Sequence[str | Path]): The files it was built from.

    Returns:
        current (bool): False when the file is missing, a source is missing or newer, or there are no sources.
    """

    output_path = Path(output_path)
    source_paths = [Path(source_path) for source_path in source_paths]

    if not source_paths or not output_path.exists():
        return False
    if not all(source_path.exists() for source_path in source_paths):
        return False

    return output_path.stat().st_mtime > max(source_path.stat().st_mtime for source_path in source_paths)


def normalize(data: NDArray[numpy.float32], method: str) -> NDArray[numpy.float32]:
    """
    Scales the valid cells of a layer to [0, 1].

    "minmax" scales valid cells linearly from their minimum to their maximum. "log1p" clamps negatives to 0, applies
    log1p, then scales the same way. NaN cells stay NaN.

    Parameters:
        data (NDArray[numpy.float32]): The layer, NaN as nodata.
        method (str): The normalization method, one of "minmax" or "log1p".

    Returns:
        result (NDArray[numpy.float32]): The normalized layer.
    """

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
        log_data = numpy.log1p(numpy.maximum(data, 0.0))
        low = float(numpy.nanmin(log_data[valid]))
        high = float(numpy.nanmax(log_data[valid]))
        if high > low:
            result[valid] = (log_data[valid] - low) / (high - low)
        else:
            result[valid] = 0.0

    else:
        raise ValueError(f"Unknown normalization method {method!r}. Choose from 'minmax' or 'log1p'.")

    # clamp to [0, 1] to guard against floating-point drift
    numpy.clip(result, 0.0, 1.0, out=result)
    result[~valid] = numpy.nan

    return result


def prepare_raster_layer(
    output_directory: Path,
    raster_path: Path,
    reprojected_crs: str,
    year: int,
    raster_title: str,
    template_raster_path: str,
    raster_mask: NDArray[numpy.bool_],
    resampling_method: enums.Resampling,
    filter_keys: Sequence[int] | None = None,
    normalization_method: str = "log1p",
) -> NDArray[numpy.float32]:
    """
    Prepares an ancillary layer on the template grid.

    Steps:
      1. Extrapolate the raster from earlier years if raster_path doesn't exist (see extrapolate_raster).
      2. Filter to filter_keys, if given. Writes output_directory/filtered/.
      3. Reproject and resample onto the template grid. Writes output_directory/reprojected/.
      4. Set the cells outside raster_mask to NaN.
      5. Normalize (see normalize).
    Steps 1-3 are cached. The reprojected file is named after its inputs: year, title, filter keys, resampling method
    and template. It's reused while it's newer than its sources: the raster (or the earlier years that an extrapolated
    raster is fitted to) and the template. The name doesn't change with the code, so delete the reprojected files
    after a change to steps 1-3. Steps 4-5 run in memory on every call, because concurrent runs with other masks would
    otherwise share their files.

    Parameters:
        output_directory (Path): The directory of the cached files.
        raster_path (Path): The unprepared raster path.
        reprojected_crs (str): The CRS of the template.
        year (int): The year of the raster.
        raster_title (str): The title of the raster for use in the filename.
        template_raster_path (str): The path to the template raster.
        raster_mask (NDArray[numpy.bool_]): True for the template cells to keep (see rasterize_mask).
        resampling_method (enums.Resampling): The resampling method. Use sum for a total per cell, and average for a
            value per area.
        filter_keys (Sequence[int] | None = None): The values to keep (see filter_raster). All values when None.
        normalization_method (str = "log1p"): The normalization method (see normalize).

    Returns:
        layer (NDArray[numpy.float32]): The prepared layer, NaN as nodata.
    """

    # e.g. (81, 82) -> "81-82", keeping brackets and spaces out of filenames
    filter_key = "" if filter_keys is None else "_" + "-".join(str(key) for key in filter_keys)
    template_key = Path(template_raster_path).stem

    reprojected_directory = output_directory / "reprojected"
    reprojected_directory.mkdir(parents=True, exist_ok=True)
    reprojected_raster_path = (
        reprojected_directory / f"{year}_{raster_title}{filter_key}_{resampling_method.name}_{template_key}.tif"
    )

    source_paths = (
        [raster_path] if raster_path.exists() else list(find_sibling_rasters(str(raster_path), year).values())
    )

    if not is_newer(reprojected_raster_path, [*source_paths, template_raster_path]):
        if not raster_path.exists():
            raster_path = extrapolate_raster(raster_path, year)

        if filter_keys is not None:
            filtered_directory = output_directory / "filtered"
            filtered_directory.mkdir(parents=True, exist_ok=True)

            raster_path = filter_raster(
                raster_path, filter_keys, filtered_directory / f"{year}_{raster_title}{filter_key}.tif"
            )

        reproject_raster(
            raster_path,
            reprojected_crs,
            reprojected_raster_path,
            template_raster_path,
            resampling_method=resampling_method,
        )

    layer = read_raster(reprojected_raster_path, Grid.from_raster(template_raster_path))
    layer[~raster_mask] = numpy.nan

    return normalize(layer, normalization_method)


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

    Cells are labeled 0 outside all polygons, and i for the zone zone_ids_unique[i - 1]. Per-zone sums are therefore
    numpy.bincount(zones.ravel(), weights, minlength=len(zone_ids_unique) + 1), and per-zone values broadcast back to
    cells as values[zones]. Polygons sharing an ID share one zone.

    Parameters:
        geometries (geopandas.GeoSeries): The zone polygons, in the raster CRS.
        zone_ids (Any): The integer-like zone ID of each polygon.
        shape (tuple[int, int]): The (height, width) of the output raster.
        transform (affine.Affine): The transform of the output raster.

    Returns:
        zones (NDArray[numpy.int32]): The zone index raster.
        zone_ids_unique (NDArray[numpy.int64]): The zone ID for each index, offset by one.
    """

    codes, zone_ids_unique = pandas.factorize(numpy.asarray(zone_ids).astype(numpy.int64))
    values = (codes + 1).tolist()

    # rasterio holds every shape as nested Python coordinate tuples until it burns them, so burn in consecutive batches
    # of about 1M coordinates. Later polygons still overwrite earlier ones, as in one call
    batches = numpy.cumsum(shapely.get_num_coordinates(geometries.values)) // 1_000_000
    starts = numpy.flatnonzero(numpy.diff(batches, prepend=-1))
    ends = numpy.append(starts[1:], len(values))

    zones = numpy.zeros(shape, dtype=numpy.int32)
    for start, end in zip(starts, ends, strict=True):
        features.rasterize(
            zip(geometries.iloc[start:end], values[start:end], strict=True),
            out=zones,
            transform=transform,
            all_touched=True,
        )

    return zones, numpy.asarray(zone_ids_unique, dtype=numpy.int64)


def read_raster(raster_path: str | Path, grid: Grid) -> NDArray[Any]:
    """
    Reads the first band of a raster that must lie on the given grid.

    Parameters:
        raster_path (str | Path): The path to the raster.
        grid (Grid): The grid the raster must be on.

    Returns:
        data (NDArray[Any]): The band, in the raster's own dtype.
    """

    with rasterio.open(raster_path) as raster:
        if raster.shape != grid.shape or raster.transform != grid.transform:
            raise ValueError(f"{raster_path} isn't on the template grid")
        return raster.read(1)


def read_statistics(
    huc2s: Sequence[str],
    huc_level: int,
    case_study: str,
    year: int,
    columns: list[str] | None = None,
) -> geopandas.GeoDataFrame:
    """
    Reads the merged water use statistics (input/statistics/<huc2>/<huc_level>/<case_study>/<year>.gpkg) of the
    given HUC-2s, sorted by HUC ID.

    The row order decides which HUC gets a shared boundary cell, because rasterize_zones gives the cell to the later
    polygon. The sort makes the zones independent of the order of huc2s and of the row order of the WBD shapefiles. A
    WBD release can change that row order and keep the same HUC IDs and polygons.

    Parameters:
        huc2s (Sequence[str]): The HUC-2 IDs.
        huc_level (int): The HUC level of the statistics.
        case_study (str): The case study.
        year (int): The statistics year.
        columns (list[str] | None = None): The columns to read. All when None. A list must include huc<huc_level>.

    Returns:
        statistics (geopandas.GeoDataFrame): The statistics, sorted by HUC ID, with a fresh index.
    """

    statistics = pandas.concat(
        [
            geopandas.read_file(
                Path("input", "statistics", huc2, str(huc_level), case_study, f"{year}.gpkg"), columns=columns
            )
            for huc2 in huc2s
        ],
        ignore_index=True,
    )

    return geopandas.GeoDataFrame(statistics.sort_values(f"huc{huc_level}", ignore_index=True))


def build_zones(
    statistics: geopandas.GeoDataFrame, huc_level: int, grid: Grid
) -> tuple[NDArray[numpy.int32], NDArray[numpy.int64], NDArray[numpy.float64]]:
    """
    Rasterizes the HUCs of read_statistics onto the grid, and gives the water use total of each zone.

    Every use of the same statistics must get its zones from this function, so all stages assign each cell to the same
    HUC.

    Parameters:
        statistics (geopandas.GeoDataFrame): The statistics, with the huc<huc_level>, water_mgd and geometry columns.
        huc_level (int): The HUC level of the statistics.
        grid (Grid): The grid of the zones.

    Returns:
        zones (NDArray[numpy.int32]): The zone index raster (rasterize_zones), 0 outside all HUCs.
        zone_hucs (NDArray[numpy.int64]): The HUC ID for each zone index, offset by one.
        zone_totals (NDArray[numpy.float64]): The water use total of each zone, in mgd, indexed like zones. Index 0 is
            NaN.
    """

    # the raster stores EPSG:5070 as WKT, which to_crs doesn't treat as the same CRS. Without this test, to_crs would
    # run an identity transform that copies every polygon, several GB at CONUS HUC-12
    if statistics.crs != grid.crs:
        statistics = statistics.to_crs(grid.crs)
    zones, zone_hucs = rasterize_zones(
        statistics.geometry,
        statistics[f"huc{huc_level}"],
        grid.shape,
        grid.transform,
    )

    # the statistics store HUC IDs as strings with leading zeros, and zone_hucs holds integers
    water_use_lookup = dict(
        zip(
            statistics[f"huc{huc_level}"].astype(int),
            statistics["water_mgd"],
            strict=True,
        )
    )

    zone_totals = numpy.full(len(zone_hucs) + 1, numpy.nan)
    zone_totals[1:] = [water_use_lookup[huc] for huc in zone_hucs]

    return zones, zone_hucs, zone_totals


def reproject_raster(
    raster_path: Path,
    reprojected_crs: str,
    output_path: Path,
    reference_raster_path: str,
    dtype: type = numpy.float32,
    nodata: Any = numpy.nan,
    resampling_method: enums.Resampling = enums.Resampling.mode,
) -> Path:
    """
    Reprojects (and resamples if necessary) the provided raster onto the reference raster's grid.

    Parameters:
        raster_path (Path): The path to the raster to reproject.
        reprojected_crs (str): The target CRS for the generated rasters.
        output_path (Path): The path to the output file.
        reference_raster_path (str): The path to the reference raster.
        dtype (type = numpy.float32): The datatype the raster should be in.
        nodata (Any = numpy.nan): The nodata value to use in the raster.
        resampling_method (enums.Resampling = enums.Resampling.mode): The raster resampling method to use.

    Returns:
        output_path (Path): The path to the output file.
    """

    with rasterio.open(raster_path) as original:
        source = cast(NDArray[Any], original.read(1))
        profile = cast(dict[str, Any], original.profile)
        source_transform = cast(affine.Affine, original.transform)
        source_crs = original.crs
    profile.update(compress="lzw", driver="GTiff")

    # replace the source's own nodata value with nodata, so missing cells enter resampling as NaN. The cast comes first,
    # because numpy.where with a NaN turns an integer source into float64, twice the size of float32
    old_nodata = profile.get("nodata")
    if old_nodata is not None:
        missing = numpy.isnan(source) if numpy.isnan(old_nodata) else source == old_nodata
        source = source.astype(dtype, copy=False)
        source[missing] = nodata
    else:
        source = source.astype(dtype, copy=False)
    profile.update(nodata=nodata, dtype=dtype)

    with rasterio.open(reference_raster_path) as reference:
        transform = cast(affine.Affine, reference.transform)
        width = cast(int, reference.width)
        height = cast(int, reference.height)

    profile.update(crs=reprojected_crs, height=height, transform=transform, width=width)

    destination = numpy.empty((height, width), dtype)

    # without src_nodata, one nodata source cell makes the whole target cell nodata under sum and average. Without
    # dst_nodata, target cells outside the source get 0, not nodata
    warp.reproject(
        source,
        destination,
        src_transform=source_transform,
        src_crs=source_crs,
        src_nodata=nodata,
        dst_transform=transform,
        dst_crs=reprojected_crs,
        dst_nodata=nodata,
        resampling=resampling_method,
    )

    _write_atomically(output_path, destination, profile)

    return output_path


def templatize_geometries(
    geometry_paths: Sequence[str], reprojected_crs: str, cell_size: int, output_path: str
) -> str:
    """
    Rasterizes the combined extent of the provided geometries into a binary template raster.

    The grid starts at the top-left corner of the geometries' bounding box and holds only whole cells. It can stop less
    than one cell short of the east and south bounds. Cells touching a geometry hold 1, and other cells hold 0 (nodata).
    An interrupted write can't leave a partial raster at output_path (see _write_atomically).

    Parameters:
        geometry_paths (Sequence[str]): The paths to the geometry files.
        reprojected_crs (str): The target CRS for the produced template.
        cell_size (int): The target cell size (in CRS units) for the produced template.
        output_path (str): The path to the output file.

    Returns:
        output_path (str): The path to the output file.
    """

    geometries = pandas.concat([geopandas.read_file(geometry_path) for geometry_path in geometry_paths])
    geometries = geopandas.GeoDataFrame(geometries, geometry="geometry")
    geometries.to_crs(reprojected_crs, inplace=True)

    min_x, min_y, max_x, max_y = geometries.total_bounds
    raster_width = int(abs(max_x - min_x) / cell_size)
    raster_height = int(abs(max_y - min_y) / cell_size)

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

    _write_atomically(output_path, raster, raster_metadata)

    return output_path


def write_mask(mask: geopandas.GeoDataFrame, output_path: Path) -> None:
    """
    Writes a geometric mask to a GeoPackage. The layer takes its name from the file.

    The file is written in a temporary directory, then moved, because is_newer accepts any file that exists.

    Parameters:
        mask (geopandas.GeoDataFrame): The mask.
        output_path (Path): The path to the output file.
    """

    with tempfile.TemporaryDirectory(dir=output_path.parent) as temporary_directory:
        temporary_path = Path(temporary_directory, output_path.name)
        mask.to_file(temporary_path)
        temporary_path.replace(output_path)


def write_raster(output_path: str, data: NDArray[numpy.float32], grid: Grid) -> str:
    """
    Writes a single-band float32 GeoTIFF on the grid, with NaN as nodata.

    An interrupted write can't leave a partial raster at output_path (see _write_atomically).

    Parameters:
        output_path (str): The path to the output file.
        data (NDArray[numpy.float32]): The (height, width) values.
        grid (Grid): The grid the values are on.

    Returns:
        output_path (str): The path to the output file.
    """

    _write_atomically(output_path, data, grid.profile())

    return output_path


def _write_atomically(output_path: str | Path, data: NDArray[Any], profile: dict[str, Any]) -> None:
    """
    Writes a single-band raster to a temporary file in the same directory, then renames it to output_path.

    is_newer accepts any file that exists, so a cached raster must appear only when complete.

    Parameters:
        output_path (str | Path): The path to the output file.
        data (NDArray[Any]): The (height, width) values.
        profile (dict[str, Any]): The rasterio profile of the output file.
    """

    # the process ID keeps concurrent runs from sharing a temporary file
    temporary_path = Path(f"{output_path}.{os.getpid()}.tmp")
    with rasterio.open(temporary_path, "w", **profile) as output_raster:
        output_raster.write(data, 1)
    temporary_path.replace(output_path)
