"""
Stage 2 of the pipeline: dasymetric mapping (see main.py).
"""

import numpy
from numpy import typing as numpy_typing

from utilities import raster_utilities


def run(
    weights: numpy_typing.NDArray[numpy.float32],
    zones: numpy_typing.NDArray[numpy.int32],
    zone_totals: numpy_typing.NDArray[numpy.float64],
    grid: raster_utilities.Grid,
    *,
    case_study: str,
    year: int,
    huc_level: int,
) -> numpy_typing.NDArray[numpy.float32]:
    """
    Spreads each HUC total over the cells of the HUC in proportion to the weight layer.

    A cell gets total * a_cell / sum(a_HUC). Cells are NaN outside all HUCs, where the weight layer is nodata, in HUCs
    with no statistics, and in HUCs whose weights sum to 0. Each run writes the result to
    output/huc<huc_level>/<case_study>/<year>/dasymetric-mapping.tif. The stage doesn't reuse an existing file, because
    the file can't show which statistics and weights made it, and the computation takes seconds.

    Parameters:
        weights (numpy_typing.NDArray[numpy.float32]): The weight layer (prepared layer 0), NaN as nodata.
        zones (numpy_typing.NDArray[numpy.int32]): The huc_level zone index raster (raster_utilities.rasterize_zones),
            0 outside all HUCs.
        zone_totals (numpy_typing.NDArray[numpy.float64]): The water use statistic of each zone, indexed like zones.
        grid (raster_utilities.Grid): The grid of all the arrays above.
        case_study (str): The case study of the run.
        year (int): The statistics year.
        huc_level (int): The HUC level of the statistics.

    Returns:
        dasymetric_raster (numpy_typing.NDArray[numpy.float32]): The dasymetric estimates.
    """

    zone_count = len(zone_totals)

    ancillary_sums = numpy.bincount(
        zones.ravel(), weights=numpy.nan_to_num(weights, nan=0.0).ravel(), minlength=zone_count
    )

    zone_scale = numpy.full(zone_count, numpy.nan)
    numpy.divide(zone_totals, ancillary_sums, out=zone_scale, where=ancillary_sums > 0)
    zone_scale[0] = numpy.nan

    dasymetric_raster = (weights * zone_scale[zones]).astype(numpy.float32)

    raster_utilities.write_raster(
        raster_utilities.run_directory(case_study=case_study, year=year, huc_level=huc_level)
        / "dasymetric-mapping.tif",
        dasymetric_raster,
        grid,
    )

    return dasymetric_raster
