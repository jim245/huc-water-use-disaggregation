"""
Utility to generate the ancillary layers for irrigation disaggregation.
"""

from collections.abc import Sequence
from pathlib import Path

import geopandas
import numpy
import pandas
from numpy.typing import NDArray
from rasterio import enums

from utilities import generate_template_raster, raster_utilities

# CONUS-wide CDL starts in this year. Earlier years use its layer
CDL_FIRST_YEAR = 2008


def run(
    huc2s: Sequence[str] = ("14", "15"),
    reprojected_crs: str = "EPSG:5070",
    year: int = 2000,
    template_raster_path: str | None = None,
    huc_level: int = 12,
) -> list[NDArray[numpy.float32]]:
    """
    Prepares the ancillary layers for irrigation spatial disaggregation.

    Every layer is on the template grid and is NaN outside the mask: the union of the huc_level HUCs with irrigation > 0
    in the year. The mask must use huc_level. A mask from finer statistics would tell a holdout run which of the finer
    HUCs have no water. The mask is cached as input/masks/irr/<year>_huc<huc_level>_<huc2 key>.gpkg, and rebuilt when
    the statistics are newer. When a layer has no file for the year, prepare_raster_layer extrapolates one from the
    years that do exist.

    Parameters:
        huc2s (Sequence[str] = ("14", "15")): The HUC-2 IDs, without zero-padding.
        reprojected_crs (str = "EPSG:5070"): The CRS of the template.
        year (int = 2000): The year to prepare.
        template_raster_path (str | None = None): The path to the template raster. Defaults to the 1000m template for
            huc2s and reprojected_crs.
        huc_level (int = 12): The HUC level of the statistics that the mask uses.

    Returns:
        prepared_layers (list[NDArray[numpy.float32]]): The layers, in the order [NLCD planted/cultivated, CDL, PRISM,
            slope]. Index 0 is the dasymetric weight layer in main.py and the irr support layer in CASE_STUDIES.
    """

    if template_raster_path is None:
        template_raster_path = generate_template_raster.run(huc2s, reprojected_crs)

    mask_directory = Path("input", "masks", "irr")
    mask_directory.mkdir(parents=True, exist_ok=True)
    mask_path = mask_directory / f"{year}_huc{huc_level}_{raster_utilities.huc2_cache_key(huc2s)}.gpkg"
    statistics_paths = [Path("input", "statistics", huc2, str(huc_level), "irr", f"{year}.gpkg") for huc2 in huc2s]

    if raster_utilities.is_newer(mask_path, statistics_paths):
        mask = geopandas.read_file(mask_path).geometry
    else:
        statistics = geopandas.GeoDataFrame(
            pandas.concat([geopandas.read_file(statistics_path) for statistics_path in statistics_paths])
        )
        mask_gdf = statistics.loc[statistics["water_mgd"] > 0, ["geometry"]].dissolve()
        raster_utilities.write_mask(mask_gdf, mask_path)
        mask = mask_gdf.geometry

    raster_mask = raster_utilities.rasterize_mask(mask, raster_utilities.Grid.from_raster(template_raster_path))

    nlcd_directory = Path("input", "NLCD", str(year))
    cdl_year = max(year, CDL_FIRST_YEAR)
    cdl_directory = Path("input", "CDL", str(cdl_year))
    prism_directory = Path("input", "PRISM", str(year))
    landfire_directory = Path("input", "LandFire", "LF2020_SlpD_CONUS", "Tif")

    return [
        # NLCD 81 (pasture/hay) and 82 (cultivated crops)
        raster_utilities.prepare_raster_layer(
            nlcd_directory,
            nlcd_directory / f"Annual_NLCD_LndCov_{year}_CU_C1V1.tif",
            reprojected_crs,
            year,
            "NLCD-planted_cultivated",
            template_raster_path,
            raster_mask,
            enums.Resampling.average,
            filter_keys=(81, 82),
            normalization_method="max",
        ),
        raster_utilities.prepare_raster_layer(
            cdl_directory,
            cdl_directory / f"{cdl_year}_30m_cdls.tif",
            reprojected_crs,
            cdl_year,
            "CDL",
            template_raster_path,
            raster_mask,
            enums.Resampling.mode,
        ),
        # PRISM annual precipitation, 30 arc-seconds
        raster_utilities.prepare_raster_layer(
            prism_directory,
            prism_directory / f"prism_ppt_us_30s_{year}.tif",
            reprojected_crs,
            year,
            "PRISM",
            template_raster_path,
            raster_mask,
            enums.Resampling.average,
        ),
        # LandFire slope in degrees, a single 2020 product used for every year
        raster_utilities.prepare_raster_layer(
            landfire_directory,
            landfire_directory / "LF2020_SlpD_CONUS.tif",
            reprojected_crs,
            2020,
            "Slope-Degrees",
            template_raster_path,
            raster_mask,
            enums.Resampling.average,
            normalization_method="minmax",
        ),
    ]


if __name__ == "__main__":
    run()
