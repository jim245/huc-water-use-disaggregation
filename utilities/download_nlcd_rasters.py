"""
Utility for fetching MRLC Annual NLCD land cover rasters.
"""

import io
import tempfile
import zipfile
from pathlib import Path
from urllib import request

# applies to each socket read, not to the whole download
DOWNLOAD_TIMEOUT_SECONDS = 60


def run(year: int = 2000) -> None:
    """
    Downloads the MRLC Annual NLCD Land Cover raster for CONUS.

    Extracts the GeoTIFF and its metadata files to input/NLCD/<year>/. Skips the download if the GeoTIFF is already
    there.

    The results use the files downloaded on 2026-02-13 and 2026-02-16. These match the MRLC archives of 2025-06-10.

    Parameters:
        year (int = 2000): The year to download rasters for.
    """

    year_directory = Path("input", "NLCD", str(year))
    tif_path = year_directory / f"Annual_NLCD_LndCov_{year}_CU_C1V1.tif"
    year_directory.mkdir(parents=True, exist_ok=True)

    if tif_path.exists():
        return

    # the GeoTIFF must appear only when complete, because the check above skips an existing GeoTIFF
    with tempfile.TemporaryDirectory(dir=year_directory) as temporary_directory:
        with request.urlopen(
            f"https://www.mrlc.gov/downloads/sciweb1/shared/mrlc/data-bundles/{tif_path.stem}.zip",
            timeout=DOWNLOAD_TIMEOUT_SECONDS,
        ) as response:
            archive_data = response.read()

        # not written to disk: NFS can keep a deleted file, and that stops the temporary directory cleanup
        with zipfile.ZipFile(io.BytesIO(archive_data)) as archive:
            archive.extractall(temporary_directory)

        # the GeoTIFF moves last
        for file in sorted(Path(temporary_directory).iterdir(), key=lambda file: file.name == tif_path.name):
            # a concurrent run moved its complete files first. Replacing them can break that run's reads on NFS
            if not (year_directory / file.name).exists():
                file.rename(year_directory / file.name)


if __name__ == "__main__":
    run()
