"""
Prepares an initial estimate raster using dasymetric mapping.
"""

import os
from os import path

import numpy
from numpy import typing as numpy_typing

from utilities import raster_utilities


def run(
    weights: numpy_typing.NDArray[numpy.float32],
    zones: numpy_typing.NDArray[numpy.int32],
    zone_totals: numpy_typing.NDArray[numpy.float64],
    grid: raster_utilities.Grid,
    case_study: str,
    year: int,
    huc_level: int = 12,
) -> numpy_typing.NDArray[numpy.float32]:
    """
    Spatially disaggregates water use statistics using a prepared raster as weights.

    Each HUC's total is spread over its cells in proportion to the weight layer: cell = total * a_cell / sum(a_HUC).
    Cells are NaN outside all HUCs, where the weight layer is nodata, in HUCs without statistics, and in HUCs whose
    weights sum to 0 (that HUC's total is dropped). Each run recomputes the output and writes it to
    output/huc<huc_level>/<case_study>/<year>/dasymetric-mapping.tif. The stage doesn't reuse an existing file, because
    the file can't show which statistics and weights made it, and the recomputation takes seconds.

    Parameters:
        weights (numpy_typing.NDArray[numpy.float32]): The weight layer (the first prepared raster), NaN as nodata.
        zones (numpy_typing.NDArray[numpy.int32]): The huc_level zone index raster (raster_utilities.rasterize_zones),
            0 outside all HUCs.
        zone_totals (numpy_typing.NDArray[numpy.float64]): Each zone's water use statistic, indexed like zones.
        grid (raster_utilities.Grid): The grid all of the above are on.
        case_study (str): The case study to run for.
        year (int): The statistics year.
        huc_level (int = 12): The HUC level of the statistics.

    Returns:
        dasymetric_raster (numpy_typing.NDArray[numpy.float32]): The initial estimates.
    """

    output_directory = path.join("output", f"huc{huc_level}", case_study, str(year))
    os.makedirs(output_directory, exist_ok=True)

    dasymetric_raster_path = path.join(output_directory, "dasymetric-mapping.tif")

    zone_count = len(zone_totals)

    # sum ancillary data per HUC in a single pass
    ancillary_sums = numpy.bincount(
        zones.ravel(),
        weights=numpy.nan_to_num(weights, nan=0.0).ravel(),
        minlength=zone_count,
    )

    # per-HUC scale factor, NaN for HUCs without ancillary signal and outside all HUCs
    zone_scale = numpy.full(zone_count, numpy.nan)
    numpy.divide(zone_totals, ancillary_sums, out=zone_scale, where=ancillary_sums > 0)
    zone_scale[0] = numpy.nan

    # produce dasymetric HUC values
    dasymetric_raster = (weights * zone_scale[zones]).astype(numpy.float32)

    raster_utilities.write_raster(dasymetric_raster_path, dasymetric_raster, grid)

    return dasymetric_raster
