"""
Utility to generate the ancillary layers for public water supply disaggregation.
"""

from collections.abc import Sequence
from pathlib import Path

import geopandas
import numpy
from numpy.typing import NDArray
from rasterio import enums

from utilities import generate_template_raster, raster_utilities


def run(
    huc2s: Sequence[str] = ("14", "15"),
    reprojected_crs: str = "EPSG:5070",
    year: int = 2000,
    template_raster_path: str | None = None,
) -> list[NDArray[numpy.float32]]:
    """
    Prepares the ancillary layers for public water supply spatial disaggregation.

    Every layer is on the template grid and is NaN outside the mask. The mask is the selected HUC-2s, clipped to the
    union of the public water supply service areas (WSA_v1). Public supply therefore goes only into a service area.
    Building the mask takes minutes, so it's cached as input/masks/pws/<huc2 key>_<crs>.gpkg and rebuilt when a source
    shapefile is newer. When a layer has no file for the year, prepare_raster_layer extrapolates one from the years
    that do exist.

    Parameters:
        huc2s (Sequence[str] = ("14", "15")): The HUC-2 IDs, without zero-padding.
        reprojected_crs (str = "EPSG:5070"): The CRS of the template.
        year (int = 2000): The year to prepare.
        template_raster_path (str | None = None): The path to the template raster. Defaults to the 1000m template for
            huc2s and reprojected_crs.

    Returns:
        prepared_layers (list[NDArray[numpy.float32]]): The layers, in the order [NLCD developed, DARTE, GDP,
            WorldPop]. Index 0 is the dasymetric weight layer in main.py.
    """

    if template_raster_path is None:
        template_raster_path = generate_template_raster.run(huc2s, reprojected_crs)

    boundary_paths = [Path("input", "HUC", huc2, "Shape", "WBDHU2.shp") for huc2 in huc2s]
    wsa_paths = [Path("input", "WSA_v1", "WSA_v1.shp")]

    mask_directory = Path("input", "masks", "pws")
    mask_directory.mkdir(parents=True, exist_ok=True)
    mask_path = (
        mask_directory / f"{raster_utilities.huc2_cache_key(huc2s)}_{reprojected_crs.replace(':', '')}.gpkg"
    )

    if raster_utilities.is_newer(mask_path, [*boundary_paths, *wsa_paths]):
        mask = geopandas.read_file(mask_path)
    else:
        mask = raster_utilities.generate_mask(boundary_paths, wsa_paths, reprojected_crs)
        raster_utilities.write_mask(mask, mask_path)

    raster_mask = raster_utilities.rasterize_mask(
        mask.geometry, raster_utilities.Grid.from_raster(template_raster_path)
    )

    nlcd_directory = Path("input", "NLCD", str(year))
    darte_directory = Path("input", "DARTE")
    gdp_directory = Path("input", "GDP")
    worldpop_directory = Path("input", "WorldPop")

    return [
        # NLCD 21-24 (developed)
        raster_utilities.prepare_raster_layer(
            nlcd_directory,
            nlcd_directory / f"Annual_NLCD_LndCov_{year}_CU_C1V1.tif",
            reprojected_crs,
            year,
            "NLCD-developed",
            template_raster_path,
            raster_mask,
            enums.Resampling.average,
            filter_keys=(21, 22, 23, 24),
            normalization_method="minmax",
        ),
        # DARTE on-road CO2 emissions, a proxy for traffic and activity. The files end in 2017
        raster_utilities.prepare_raster_layer(
            darte_directory,
            darte_directory / f"onroad_{year}.tif",
            reprojected_crs,
            year,
            "DARTE",
            template_raster_path,
            raster_mask,
            enums.Resampling.sum,
        ),
        # 1 km revised real GDP from calibrated nighttime lights. The source covers 1992-2019
        raster_utilities.prepare_raster_layer(
            gdp_directory,
            gdp_directory / f"{year}GDP.tif",
            reprojected_crs,
            year,
            "GDP",
            template_raster_path,
            raster_mask,
            enums.Resampling.sum,
        ),
        # WorldPop people per cell, aggregated to 1 km
        raster_utilities.prepare_raster_layer(
            worldpop_directory,
            worldpop_directory / f"usa_ppp_{year}_1km_Aggregated.tif",
            reprojected_crs,
            year,
            "WorldPop",
            template_raster_path,
            raster_mask,
            enums.Resampling.sum,
        ),
    ]


if __name__ == "__main__":
    run()
