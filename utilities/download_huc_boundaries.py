"""
Utility for fetching HUC watershed boundary shapefiles from the USGS Watershed Boundary Dataset (WBD).
"""

import os
import zipfile
from os import path
from urllib import request


def run(huc2s: list[str] = ["14", "15"]):
    """
    Downloads HUC watershed boundary shapefiles.

    For each HUC-2, extracts the WBDHU* shapefiles (HUC levels 2 through 16) to input/HUC/<huc2>/Shape/. The
    directory name is the HUC-2 ID as given, without zero-padding (e.g. "1", not "01"). HUC-2s whose Shape directory
    already exists are skipped.

    Parameters:
        huc2s (list[str] = ["14", "15"]): A list of HUC2 numerical ID strings.
    """

    for huc2 in huc2s:
        huc_directory = path.join("input", "HUC", huc2)
        os.makedirs(huc_directory, exist_ok=True)

        if path.exists(path.join(huc_directory, "Shape")):
            continue

        archive_filepath = path.join(huc_directory, "archive.zip")

        with request.urlopen(
            f"https://prd-tnm.s3.amazonaws.com/StagedProducts/Hydrography/WBD/HU2/Shape/WBD_{huc2.zfill(2)}_HU2_Shape.zip"
        ) as response, open(archive_filepath, "wb") as f:
            f.write(response.read())

        with zipfile.ZipFile(archive_filepath) as archive:
            for file in archive.namelist():
                if file.startswith("Shape/WBDHU"):
                    archive.extract(file, huc_directory)

        os.remove(archive_filepath)


if __name__ == "__main__":
    run()
