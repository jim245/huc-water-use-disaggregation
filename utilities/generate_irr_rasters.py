"""
Utility to generate the rasters required for irrigation disaggregation.
"""

import os
from os import path
from typing import Optional

import geopandas
import pandas
from rasterio import enums

from utilities import generate_template_raster, raster_utilities


def run(
    huc2s: list[str] = ["14", "15"],
    reprojected_crs: str = "EPSG:5070",
    year: int = 2000,
    template_raster_path: Optional[str] = None,
    huc_level: int = 12,
) -> list[str]:
    """
    Prepares rasters for use in irrigation spatial disaggregation.

    Every layer is aligned to the template and clipped to the mask: the union of the huc_level HUCs with irrigation > 0
    for the year, so cells outside them are nodata. The mask must use huc_level. A mask from finer statistics would
    tell a holdout run which of the finer HUCs have no water. When a layer has no file for the year,
    prepare_raster_layer extrapolates one from the years that do exist.

    Parameters:
        huc2s (list[str] = ["14", "15"]): A list of HUC2 numerical ID strings.
        reprojected_crs (str = "EPSG:5070"): The target CRS for the generated rasters.
        year (int = 2000): The year to generate rasters for.
        template_raster_path (Optional[str] = None): The path to the template raster. Defaults to the 1000m
            template for huc2s and reprojected_crs.
        huc_level (int = 12): The HUC level of the statistics that the mask uses.

    Returns:
        prepared_raster_paths (list[str]): The prepared raster paths, in the order [NLCD planted/cultivated, CDL,
            PRISM, slope]. Index 0 is the dasymetric mapping weight layer and the support layer in main.py.
    """

    if template_raster_path is None:
        template_raster_path = generate_template_raster.run(huc2s, reprojected_crs)

    irrigation_mask_directory = path.join("input", "masks", "irr")
    os.makedirs(irrigation_mask_directory, exist_ok=True)

    # union of the huc_level HUCs with irrigation > 0 for the year. The cache is per HUC level and HUC-2 selection,
    # since the mask only covers the selected HUC-2s, and it's rebuilt when the statistics are newer
    huc2_key = raster_utilities.huc2_cache_key(huc2s)
    irr_mask_path = path.join(
        irrigation_mask_directory, f"{year}_huc{huc_level}_{huc2_key}.gpkg"
    )
    statistics_paths = [
        path.join("input", "statistics", huc2, str(huc_level), "irr", f"{year}.gpkg")
        for huc2 in huc2s
    ]

    if not raster_utilities.is_newer(irr_mask_path, statistics_paths):
        irr_mask = geopandas.GeoDataFrame(
            pandas.concat(
                [
                    geopandas.read_file(statistics_path)
                    for statistics_path in statistics_paths
                ]
            )
        )
        irr_mask = irr_mask[irr_mask["water_mgd"] > 0]
        irr_mask = irr_mask[["geometry"]].dissolve()
        irr_mask = irr_mask[["geometry"]]
        irr_mask.to_file(irr_mask_path)
        irr_mask = irr_mask.geometry
    else:
        irr_mask = geopandas.read_file(irr_mask_path).geometry

    # rasterized once for all layers
    raster_mask = raster_utilities.rasterize_mask(
        irr_mask, raster_utilities.Grid.from_raster(template_raster_path)
    )

    # list to hold final product rasters
    prepared_raster_paths = list[str]()

    # generate NLCD raster (81 - Pasture/Hay, 82 - Cultivated Crops)
    nlcd_directory = path.join("input", "NLCD", str(year))
    prepared_raster_paths.append(
        raster_utilities.prepare_raster_layer(
            nlcd_directory,
            path.join(nlcd_directory, f"Annual_NLCD_LndCov_{year}_CU_C1V1.tif"),
            reprojected_crs,
            year,
            "NLCD-planted_cultivated",
            template_raster_path,
            raster_mask,
            [81, 82],
            mask_key=f"huc{huc_level}",
            resampling_method=enums.Resampling.average,
            normalization_method="minmax",
        )
    )

    # generate CDL raster (CONUS-wide CDL starts in 2008; earlier years use the 2008 layer)
    cdl_directory = path.join("input", "CDL", str(year if year > 2007 else 2008))
    prepared_raster_paths.append(
        raster_utilities.prepare_raster_layer(
            cdl_directory,
            path.join(cdl_directory, f"{year if year > 2007 else 2008}_30m_cdls.tif"),
            reprojected_crs,
            year if year > 2007 else 2008,
            "CDL",
            template_raster_path,
            raster_mask,
            mask_key=f"huc{huc_level}",
            resampling_method=enums.Resampling.mode,
        )
    )

    # generate PRISM precipitation raster (annual total, 30 arc-second)
    prism_directory = path.join("input", "PRISM", str(year))
    prepared_raster_paths.append(
        raster_utilities.prepare_raster_layer(
            prism_directory,
            path.join(prism_directory, f"prism_ppt_us_30s_{year}.tif"),
            reprojected_crs,
            year,
            "PRISM",
            template_raster_path,
            raster_mask,
            mask_key=f"huc{huc_level}",
        )
    )

    # generate LandFire slope raster (a single static 2020 product, used for every year)
    landfire_directory = path.join("input", "LandFire", "LF2020_SlpD_CONUS", "Tif")
    prepared_raster_paths.append(
        raster_utilities.prepare_raster_layer(
            landfire_directory,
            path.join(landfire_directory, "LF2020_SlpD_CONUS.tif"),
            reprojected_crs,
            2020,
            "Slope-Degrees",
            template_raster_path,
            raster_mask,
            mask_key=f"huc{huc_level}",
            resampling_method=enums.Resampling.average,
            normalization_method="minmax",
        )
    )

    return prepared_raster_paths


if __name__ == "__main__":
    run()
