"""
Builds the template raster from the HUC-2 boundaries.
"""

from collections.abc import Sequence
from pathlib import Path

from utilities import raster_utilities


def run(
    huc2s: Sequence[str] = ("14", "15"),
    reprojected_crs: str = "EPSG:5070",
    cell_size: int = 1000,
) -> str:
    """
    Builds the template raster of the HUC-2 selection.

    Writes input/templates/<cell_size>m_<crs>_<huc2 key>.tif, and reuses it while it's newer than the HUC-2 boundaries.
    The pipeline reads only the grid of the template, not its cell values.

    Parameters:
        huc2s (Sequence[str] = ("14", "15")): The HUC-2 IDs, without zero-padding.
        reprojected_crs (str = "EPSG:5070"): The CRS of the template.
        cell_size (int = 1000): The cell size, in CRS units.

    Returns:
        output_path (str): The path to the template raster.
    """

    crs_key = reprojected_crs.replace(":", "")
    huc2_key = raster_utilities.huc2_cache_key(huc2s)
    output_directory = Path("input", "templates")
    output_path = str(output_directory / f"{cell_size}m_{crs_key}_{huc2_key}.tif")
    boundary_paths = [
        str(Path("input", "HUC", huc2, "Shape", "WBDHU2.shp")) for huc2 in huc2s
    ]

    if raster_utilities.is_newer(output_path, boundary_paths):
        return output_path

    output_directory.mkdir(parents=True, exist_ok=True)

    raster_utilities.templatize_geometries(
        boundary_paths,
        reprojected_crs,
        cell_size,
        output_path,
    )

    return output_path


if __name__ == "__main__":
    run()
