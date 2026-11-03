"""
Stage 3 of the pipeline: pycnophylactic interpolation (see main.py).
"""

import numpy
from numpy import typing as numpy_typing
from scipy import ndimage

from utilities import raster_utilities


def run(
    dasymetric_raster: numpy_typing.NDArray[numpy.float32],
    zones: numpy_typing.NDArray[numpy.int32],
    zone_totals: numpy_typing.NDArray[numpy.float64],
    grid: raster_utilities.Grid,
    *,
    case_study: str,
    year: int,
    huc_level: int,
    support: numpy_typing.NDArray[numpy.float32] | None = None,
) -> numpy_typing.NDArray[numpy.float32]:
    """
    Smooths the dasymetric estimates and keeps the total of each HUC.

    Runs one pass of Tobler's pycnophylactic method. The full method repeats the smoothing and the rescale until they
    converge.
      1. Replace each cell with the mean of its valid 4-neighbors. The mean doesn't include the cell itself.
      2. If support is given, set the cells that can't receive water to 0.
      3. Give each HUC that has no water after step 2 its dasymetric values, so step 4 can keep its total.
      4. Rescale each HUC so its cells sum to its total.
    Cells that are NaN in the dasymetric estimates stay NaN. Each run writes the result to
    output/huc<huc_level>/<case_study>/<year>/pycnophylactic-interpolation.tif. The stage doesn't reuse an existing
    file, because the file can't show which inputs made it, and the computation takes seconds.

    Parameters:
        dasymetric_raster (numpy_typing.NDArray[numpy.float32]): The dasymetric estimates (dasymetric_mapping.run).
        zones (numpy_typing.NDArray[numpy.int32]): The huc_level zone index raster (raster_utilities.rasterize_zones),
            0 outside all HUCs.
        zone_totals (numpy_typing.NDArray[numpy.float64]): The water use statistic of each zone, indexed like zones.
        grid (raster_utilities.Grid): The grid of all the rasters.
        case_study (str): The case study of the run.
        year (int): The statistics year.
        huc_level (int): The HUC level of the statistics.
        support (numpy_typing.NDArray[numpy.float32] | None = None): A layer whose cells > 0 can receive water. All
            cells can receive water when None.

    Returns:
        pycnophylactic_raster (numpy_typing.NDArray[numpy.float32]): The initial estimates.
    """

    zone_count = len(zone_totals)

    neighbor_mask = numpy.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=numpy.float64)

    dasymetric_nan_mask = numpy.isnan(dasymetric_raster)

    neighbor_sums = ndimage.convolve(
        numpy.where(dasymetric_nan_mask, 0.0, dasymetric_raster.astype(numpy.float64)),
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
    pycnophylactic_raster[dasymetric_nan_mask] = numpy.nan

    # this must come before the rescale, or the HUC sums no longer match the totals
    if support is not None:
        supported = numpy.nan_to_num(support, nan=0.0) > 0
        pycnophylactic_raster[~supported & ~dasymetric_nan_mask] = 0.0

    # the mean doesn't include the cell itself, so the water of an isolated cell moves onto NaN, unsupported or other
    # HUCs' cells. The dasymetric values obey the support limit only because the irr support layer is the weight layer
    smoothed_sums = numpy.bincount(
        zones.ravel(),
        weights=numpy.nan_to_num(pycnophylactic_raster, nan=0.0).ravel(),
        minlength=zone_count,
    )
    dasymetric_sums = numpy.bincount(
        zones.ravel(),
        weights=numpy.nan_to_num(dasymetric_raster, nan=0.0).ravel(),
        minlength=zone_count,
    )
    fallback_zones = (smoothed_sums <= 0) & (dasymetric_sums > 0)
    fallback_zones[0] = False
    fallback_cells = fallback_zones[zones] & ~dasymetric_nan_mask
    pycnophylactic_raster[fallback_cells] = dasymetric_raster[fallback_cells]
    print(
        f"Pycnophylactic fallback: {int(fallback_zones.sum())} HUCs kept their dasymetric values, "
        + f"{dasymetric_sums[fallback_zones].sum():.4f} mgd"
    )

    polygon_sums = numpy.bincount(
        zones.ravel(),
        weights=numpy.nan_to_num(pycnophylactic_raster, nan=0.0).ravel(),
        minlength=zone_count,
    )
    zone_ratios = numpy.ones(zone_count)
    numpy.divide(zone_totals, polygon_sums, out=zone_ratios, where=polygon_sums > 0)
    zone_ratios[0] = 1.0

    pycnophylactic_raster *= zone_ratios[zones]

    raster_utilities.write_raster(
        raster_utilities.run_directory(case_study=case_study, year=year, huc_level=huc_level)
        / "pycnophylactic-interpolation.tif",
        pycnophylactic_raster,
        grid,
    )

    return pycnophylactic_raster
