"""
Prepares an initial estimate raster using one iteration of pycnophylactic interpolation.
"""

import os
from os import path
from typing import Optional

import numpy
from numpy import typing as numpy_typing
from scipy import ndimage

from utilities import raster_utilities


def run(
    dasymetric_raster: numpy_typing.NDArray[numpy.float32],
    zones: numpy_typing.NDArray[numpy.int32],
    zone_totals: numpy_typing.NDArray[numpy.float64],
    grid: raster_utilities.Grid,
    case_study: str,
    year: int,
    huc_level: int = 12,
    support: Optional[numpy_typing.NDArray[numpy.float32]] = None,
) -> numpy_typing.NDArray[numpy.float32]:
    """
    Smooths dasymetric initial estimates while preserving each HUC's total.

    Runs a single pass of Tobler's pycnophylactic method, which normally repeats smoothing and rescaling until
    convergence:
      1. Replace each cell with the mean of its valid 4-neighbors (the cell itself is excluded).
      2. If a support raster is given, zero cells that can't receive water.
      3. HUCs left with nothing fall back to their dasymetric values (unsmoothed), so step 4 can preserve their total.
      4. Rescale each HUC so its cells sum to its known total.
    The output is written to output/huc<huc_level>/<case_study>/<year>/pycnophylactic-interpolation.tif and read back
    instead if it already exists.

    Parameters:
        dasymetric_raster (numpy_typing.NDArray[numpy.float32]): The dasymetric initial estimates.
        zones (numpy_typing.NDArray[numpy.int32]): The huc_level zone index raster (raster_utilities.rasterize_zones),
            0 outside all HUCs.
        zone_totals (numpy_typing.NDArray[numpy.float64]): Each zone's water use statistic, indexed like zones.
        grid (raster_utilities.Grid): The grid all of the rasters are on.
        case_study (str): The case study to run for.
        year (int): The statistics year.
        huc_level (int = 12): The HUC level of the statistics.
        support (Optional[numpy_typing.NDArray[numpy.float32]] = None): A layer whose cells > 0 may receive water (the
            cultivated-land layer for irrigation). Without it, smoothing moves water onto neighboring cells with no
            cropland. All cells may receive water when None.

    Returns:
        pycnophylactic_raster (numpy_typing.NDArray[numpy.float32]): The smoothed initial estimates.
    """

    output_directory = path.join("output", f"huc{huc_level}", case_study, str(year))
    os.makedirs(output_directory, exist_ok=True)

    pycnophylactic_raster_path = path.join(
        output_directory, "pycnophylactic-interpolation.tif"
    )

    if path.exists(pycnophylactic_raster_path):
        return raster_utilities.read_raster(pycnophylactic_raster_path, grid)

    dasymetric_data = dasymetric_raster.astype(numpy.float32)
    zone_count = len(zone_totals)

    neighbor_mask = numpy.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=numpy.float64)

    dasymetric_nan_mask = numpy.isnan(dasymetric_data)

    # neighbor averaging: NaN-aware mean of the 4 neighbors, as sum of valid neighbors / count of valid neighbors
    neighbor_sums = ndimage.convolve(
        numpy.where(dasymetric_nan_mask, 0.0, dasymetric_data.astype(numpy.float64)),
        neighbor_mask,
        mode="constant",
        cval=0.0,
    )
    neighbor_counts = ndimage.convolve(
        (~dasymetric_nan_mask).astype(numpy.float64),
        neighbor_mask,
        mode="constant",
        cval=0.0,
    )
    pycnophylactic_raster = numpy.full(grid.shape, numpy.nan, dtype=numpy.float32)
    numpy.divide(
        neighbor_sums,
        neighbor_counts,
        out=pycnophylactic_raster,
        where=neighbor_counts > 0,
        casting="same_kind",
    )

    # nodata cells next to valid cells received a neighbor mean above; set them back to nodata
    pycnophylactic_raster[dasymetric_nan_mask] = numpy.nan

    # keep water off cells that can't receive it, before mass preservation so HUC totals stay exact
    if support is not None:
        supported = numpy.nan_to_num(support, nan=0.0) > 0
        pycnophylactic_raster[~supported & ~dasymetric_nan_mask] = 0.0

    # the 4-neighbor mean excludes the cell itself, so water on isolated cells (isolated cropland, or the few valid
    # cells of a HUC mostly outside the analysis mask) moves onto NaN, unsupported or other HUCs' cells. A HUC left
    # with nothing can't be rescaled to its total, so it falls back to its dasymetric values, which hold only on
    # supported cells for irrigation (the support layer is the dasymetric weight layer)
    smoothed_sums = numpy.bincount(
        zones.ravel(),
        weights=numpy.nan_to_num(pycnophylactic_raster, nan=0.0).ravel(),
        minlength=zone_count,
    )
    dasymetric_sums = numpy.bincount(
        zones.ravel(),
        weights=numpy.nan_to_num(dasymetric_data, nan=0.0).ravel(),
        minlength=zone_count,
    )
    fallback_zones = (smoothed_sums <= 0) & (dasymetric_sums > 0)
    fallback_zones[0] = False
    fallback_cells = fallback_zones[zones] & ~dasymetric_nan_mask
    pycnophylactic_raster[fallback_cells] = dasymetric_data[fallback_cells]
    print(
        f"Pycnophylactic fallback: {int(fallback_zones.sum())} HUCs kept their dasymetric values, "
        f"{dasymetric_sums[fallback_zones].sum():.4f} mgd"
    )

    # mass preservation: rescale each HUC so its cells sum to the known total
    polygon_sums = numpy.bincount(
        zones.ravel(),
        weights=numpy.nan_to_num(pycnophylactic_raster, nan=0.0).ravel(),
        minlength=zone_count,
    )
    # HUCs without any signal (and cells outside all HUCs) are left unscaled; their cells are all 0 or NaN, so their
    # totals are lost
    zone_ratios = numpy.ones(zone_count)
    numpy.divide(zone_totals, polygon_sums, out=zone_ratios, where=polygon_sums > 0)
    zone_ratios[0] = 1.0

    pycnophylactic_raster *= zone_ratios[zones]

    raster_utilities.write_raster(
        pycnophylactic_raster_path, pycnophylactic_raster, grid
    )

    return pycnophylactic_raster
