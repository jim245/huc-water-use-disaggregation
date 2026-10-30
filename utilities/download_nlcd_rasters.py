"""
Utility for fetching MRLC Annual NLCD land cover rasters.
"""

import os
import zipfile
from os import path
from urllib import request


def run(year: int = 2000):
    """
    Downloads the MRLC Annual NLCD Land Cover raster for CONUS.

    The GeoTIFF is extracted to input/NLCD/<year>/. The download is skipped if it is already there.

    Parameters:
        year (int = 2000): The year to download rasters for.
    """

    year_directory = path.join("input", "NLCD", str(year))
    os.makedirs(year_directory, exist_ok=True)

    if path.exists(path.join(year_directory, f"Annual_NLCD_LndCov_{year}_CU_C1V1.tif")):
        return

    archive_filepath = path.join(
        year_directory, f"Annual_NLCD_LndCov_{year}_CU_C1V1.zip"
    )

    with request.urlopen(
        f"https://www.mrlc.gov/downloads/sciweb1/shared/mrlc/data-bundles/Annual_NLCD_LndCov_{year}_CU_C1V1.zip"
    ) as response, open(archive_filepath, "wb") as f:
        f.write(response.read())

    with zipfile.ZipFile(archive_filepath) as archive:
        archive.extractall(year_directory)

    os.remove(archive_filepath)


if __name__ == "__main__":
    run()
