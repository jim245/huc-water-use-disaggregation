"""
Utility for fetching HUC watershed boundary shapefiles from the USGS Watershed Boundary Dataset (WBD).
"""

import io
import tempfile
import zipfile
from collections.abc import Sequence
from pathlib import Path
from urllib import request

from utilities import raster_utilities

# applies to each socket read, not to the whole download
DOWNLOAD_TIMEOUT_SECONDS = 60

EXTRACTED_PREFIXES = tuple(f"Shape/WBDHU{huc_level}." for huc_level in raster_utilities.HUC_LEVELS)


def run(huc2s: Sequence[str] = ("14", "15")) -> None:
    """
    Downloads HUC watershed boundary shapefiles.

    For each HUC-2, extracts the EXTRACTED_PREFIXES shapefiles to input/HUC/<huc2>/Shape/.

    The results use the WBD files downloaded on 2026-11-04. The source has only the latest release.

    Parameters:
        huc2s (Sequence[str] = ("14", "15")): The HUC-2 IDs, without zero-padding.
    """

    for huc2 in huc2s:
        huc_directory = Path("input", "HUC", huc2)
        shape_directory = huc_directory / "Shape"
        huc_directory.mkdir(parents=True, exist_ok=True)

        if shape_directory.exists():
            continue

        # Shape must appear only when complete, because the check above skips an existing Shape
        with tempfile.TemporaryDirectory(dir=huc_directory) as temporary_directory:
            with request.urlopen(
                f"https://prd-tnm.s3.amazonaws.com/StagedProducts/Hydrography/WBD/HU2/Shape/WBD_{huc2.zfill(2)}_HU2_Shape.zip",
                timeout=DOWNLOAD_TIMEOUT_SECONDS,
            ) as response:
                archive_data = response.read()

            # not written to disk: NFS can keep a deleted file, and that stops the temporary directory cleanup
            with zipfile.ZipFile(io.BytesIO(archive_data)) as archive:
                for file in archive.namelist():
                    if file.startswith(EXTRACTED_PREFIXES):
                        archive.extract(file, temporary_directory)

            try:
                Path(temporary_directory, "Shape").rename(shape_directory)
            except OSError:
                # a concurrent run renamed its complete Shape directory first
                if not shape_directory.exists():
                    raise


if __name__ == "__main__":
    run()
