"""
Creates a raster template from the HUC-2 geometry boundaries.
"""

import os
from os import path

from utilities import raster_utilities


def run(
    huc2s: list[str] = ["14", "15"],
    reprojected_crs: str = "EPSG:5070",
    cell_size: int = 1000,
) -> str:
    """
    Creates a raster template from the supplied HUC2s.

    The template covers the bounding box of the HUC-2s, with 1 in cells touching a HUC-2 polygon and 0 (nodata)
    elsewhere. All prepared rasters are aligned to this grid. It is cached in input/templates/ by cell size, CRS and
    HUC-2 selection.

    Parameters:
        huc2s (list[str] = ["14", "15"]): A list of HUC2 numerical ID strings.
        reprojected_crs (str = "EPSG:5070"): The target CRS for the generated template.
        cell_size (int = 1000): The target cell size (in CRS units) for the generated template.

    Returns:
        output_path (str): The filepath to the generated template raster.
    """

    output_directory = path.join("input", "templates")
    
    # cached per grid definition, so a different selection, CRS or cell size builds its own template
    crs_key = reprojected_crs.replace(":", "")
    huc2_key = raster_utilities.huc2_cache_key(huc2s)
    output_path = path.join(output_directory, f"{cell_size}m_{crs_key}_{huc2_key}.tif")

    if path.exists(output_path):
        return output_path

    os.makedirs(output_directory, exist_ok=True)

    huc_directory = path.join("input", "HUC")

    raster_utilities.templatize_geometries(
        [path.join(huc_directory, huc2, "Shape", "WBDHU2.shp") for huc2 in huc2s],
        reprojected_crs,
        cell_size,
        output_path,
    )

    return output_path


if __name__ == "__main__":
    run()
