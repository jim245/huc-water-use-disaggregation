"""
Utility to generate the rasters required for public water supply disaggregation.
"""

import os
from os import path
from typing import Optional

import geopandas
from rasterio import enums

from utilities import generate_template_raster, raster_utilities


def run(
    huc2s: list[str] = ["14", "15"],
    reprojected_crs: str = "EPSG:5070",
    year: int = 2000,
    template_raster_path: Optional[str] = None,
) -> list[str]:
    """
    Prepares rasters for use in public water supply spatial disaggregation.

    Every layer is aligned to the template and clipped to the mask: the selected HUC-2s clipped to the public water
    supply service areas (WSA_v1), so cells outside a service area are nodata. The mask is cached as
    input/masks/pws/<huc2 key>_<crs>.gpkg and rebuilt when a source shapefile is newer. When a layer has no file for the year,
    prepare_raster_layer extrapolates one from the years that do exist.

    Parameters:
        huc2s (list[str] = ["14", "15"]): A list of HUC2 numerical ID strings.
        reprojected_crs (str = "EPSG:5070"): The target CRS for the generated rasters.
        year (int = 2000): The year to generate rasters for.
        template_raster_path (Optional[str] = None): The path to the template raster. Defaults to the 1000m
            template for huc2s and reprojected_crs.

    Returns:
        prepared_raster_paths (list[str]): The prepared raster paths, in the order [NLCD developed, DARTE, GDP,
            WorldPop]. Index 0 is the dasymetric mapping weight layer.
    """

    if template_raster_path is None:
        template_raster_path = generate_template_raster.run(huc2s, reprojected_crs)

    # HUC-2s clipped to the union of the water service areas, so public supply is only placed inside a service area.
    # Building it takes minutes, so it's cached per HUC-2 selection and CRS while newer than its source shapefiles
    huc2_paths = [
        path.join("input", "HUC", huc2, "Shape", "WBDHU2.shp") for huc2 in huc2s
    ]
    wsa_paths = [path.join("input", "WSA_v1", "WSA_v1.shp")]

    pws_mask_directory = path.join("input", "masks", "pws")
    os.makedirs(pws_mask_directory, exist_ok=True)
    pws_mask_path = path.join(
        pws_mask_directory,
        f"{raster_utilities.huc2_cache_key(huc2s)}_{reprojected_crs.replace(':', '')}.gpkg",
    )

    if raster_utilities.is_newer(pws_mask_path, huc2_paths + wsa_paths):
        pws_mask = geopandas.read_file(pws_mask_path)
    else:
        pws_mask = raster_utilities.generate_mask(
            huc2_paths, wsa_paths, reprojected_crs
        )
        pws_mask.to_file(pws_mask_path)

    # rasterized once for all layers
    raster_mask = raster_utilities.rasterize_mask(
        pws_mask.geometry, raster_utilities.Grid.from_raster(template_raster_path)
    )

    prepared_raster_paths = list[str]()

    # generate NLCD raster (21-24 - Developed)
    nlcd_directory = path.join("input", "NLCD", str(year))
    prepared_raster_paths.append(
        raster_utilities.prepare_raster_layer(
            nlcd_directory,
            path.join(nlcd_directory, f"Annual_NLCD_LndCov_{year}_CU_C1V1.tif"),
            reprojected_crs,
            year,
            "NLCD-developed",
            template_raster_path,
            raster_mask,
            [21, 22, 23, 24],
            resampling_method=enums.Resampling.average,
            normalization_method="minmax",
        )
    )

    # generate DARTE raster (on-road CO2 emissions, a proxy for traffic and activity; available through 2017, later
    # years are extrapolated)
    darte_directory = path.join("input", "DARTE")
    prepared_raster_paths.append(
        raster_utilities.prepare_raster_layer(
            darte_directory,
            path.join(darte_directory, f"onroad_{year}.tif"),
            reprojected_crs,
            year,
            "DARTE",
            template_raster_path,
            raster_mask,
        )
    )

    # generate GDP raster (1 km revised real GDP from calibrated nighttime lights, 1992-2019; later years are
    # extrapolated)
    gdp_directory = path.join("input", "GDP")
    prepared_raster_paths.append(
        raster_utilities.prepare_raster_layer(
            gdp_directory,
            path.join(gdp_directory, f"{year}GDP.tif"),
            reprojected_crs,
            year,
            "GDP",
            template_raster_path,
            raster_mask,
        )
    )

    # generate WorldPop raster (people per pixel, aggregated to 1 km)
    worldpop_directory = path.join("input", "WorldPop")
    prepared_raster_paths.append(
        raster_utilities.prepare_raster_layer(
            worldpop_directory,
            path.join(worldpop_directory, f"usa_ppp_{year}_1km_Aggregated.tif"),
            reprojected_crs,
            year,
            "WorldPop",
            template_raster_path,
            raster_mask,
        )
    )

    return prepared_raster_paths


if __name__ == "__main__":
    run()
