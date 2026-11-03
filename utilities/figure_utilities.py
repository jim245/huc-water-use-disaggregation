"""
Utility for rendering maps of water use rasters.

Each raster is a column of the figure. The overview on top shows the full grid, for example CONUS. The optional detail
view below it zooms in on a set of HUC-2 watersheds, for example 14 and 15 (the Colorado River Basin).
"""

import argparse
import glob
import io
import string
import tempfile
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any, NamedTuple
from urllib import request

import geopandas
import matplotlib
import matplotlib.pyplot as plt
import numpy
import pandas
from matplotlib.colors import LogNorm
from matplotlib.layout_engine import ConstrainedLayoutEngine
from numpy.typing import NDArray
from rasterio import transform as rasterio_transform
from rasterio.features import geometry_mask
from shapely.geometry import MultiPolygon, Polygon, box

from utilities import raster_utilities

STATE_BOUNDARIES_PATH = Path("input", "USA", "cb_2024_us_state_500k.shp")

# applies to each socket read, not to the whole download
DOWNLOAD_TIMEOUT_SECONDS = 60

# the color of the detail area's outline on the overview. None omits the outline
DETAIL_OUTLINE_COLOR: str | None = "red"
CMAP = "viridis_r"
LOG = True
# LOG only: the bottom of the colorbar. Values in (0, LOG_VMIN) show at this value
LOG_VMIN = 1e-3
ZERO_COLOR = "lightgrey"  # LOG only: the color of cells with values <= 0
# the top of the colorbar, as a percentile of the valid values. Higher values get the top color. None uses the maximum
VMAX_PERCENTILE: float | None = 100


class _DetailView(NamedTuple):
    """
    The HUC-2 watersheds that the detail view zooms in on.

    Attributes:
        outside (NDArray[numpy.bool_]): True for the grid cells outside the watersheds.
        outline (geopandas.GeoDataFrame): The dissolved outline of the watersheds.
        view_bounds (NDArray[numpy.float64]): The (minx, miny, maxx, maxy) bounds of the watersheds.
    """

    outside: NDArray[numpy.bool_]
    outline: geopandas.GeoDataFrame
    view_bounds: NDArray[numpy.float64]


def download_state_boundaries() -> None:
    """
    Downloads the Census cartographic boundary shapefile of the US states.

    Extracts the shapefile and its metadata files to the directory of STATE_BOUNDARIES_PATH. Skips the download if the
    .shp file is already there.

    The figures use the files downloaded on 2026-05-26. Their .shp, .shx, .dbf, .prj and .cpg files match the Census
    archive of 2025-04-10.
    """

    directory = STATE_BOUNDARIES_PATH.parent
    directory.mkdir(parents=True, exist_ok=True)

    if STATE_BOUNDARIES_PATH.exists():
        return

    # the .shp file must appear only when complete, because the check above skips an existing .shp file
    with tempfile.TemporaryDirectory(dir=directory) as temporary_directory:
        with request.urlopen(
            f"https://www2.census.gov/geo/tiger/GENZ2024/shp/{STATE_BOUNDARIES_PATH.stem}.zip",
            timeout=DOWNLOAD_TIMEOUT_SECONDS,
        ) as response:
            archive_data = response.read()

        # not written to disk: NFS can keep a deleted file, and that stops the temporary directory cleanup
        with zipfile.ZipFile(io.BytesIO(archive_data)) as archive:
            archive.extractall(temporary_directory)

        # the .shp file moves last
        for file in sorted(
            Path(temporary_directory).iterdir(),
            key=lambda file: file.name == STATE_BOUNDARIES_PATH.name,
        ):
            # a concurrent run moved its complete files first. Replacing them can break that run's reads on NFS
            if not (directory / file.name).exists():
                file.rename(directory / file.name)


def run(
    raster: NDArray[numpy.float32],
    grid: raster_utilities.Grid,
    detail_huc2s: Sequence[str],
    huc_level: int,
    case_study: str,
    year: int,
) -> Path:
    """
    Renders a disaggregated raster, with its title and label taken from the case study and year.

    The figure is written to output/huc<huc_level>/<case_study>/<year>/<case_study>_huc<huc_level>_<year>.png.

    Parameters:
        raster (NDArray[numpy.float32]): The disaggregated raster, NaN as nodata.
        grid (raster_utilities.Grid): The grid the raster is on.
        detail_huc2s (Sequence[str]): The HUC-2 watersheds that the detail view zooms in on. An empty sequence
            omits the detail view.
        huc_level (int): The HUC scale used for spatial disaggregation.
        case_study (str): The case study.
        year (int): The year.

    Returns:
        output_path (Path): The path to the figure.
    """

    sector = raster_utilities.CASE_STUDIES[case_study].sector
    measure = raster_utilities.CASE_STUDIES[case_study].measure

    return render(
        [raster],
        grid,
        detail_huc2s,
        f"Spatially Disaggregated {year} USGS {sector} {measure}",
        f"{measure} (Million Gallons per Day per cell)",
        Path(
            "output",
            f"huc{huc_level}",
            case_study,
            str(year),
            f"{case_study}_huc{huc_level}_{year}.png",
        ),
    )


def render(
    columns: Sequence[NDArray[numpy.float32]],
    grid: raster_utilities.Grid,
    detail_huc2s: Sequence[str],
    title: str,
    colorbar_label: str,
    output_path: Path,
) -> Path:
    """
    Renders one column of panels per raster, all sharing one color scale.

    Parameters:
        columns (Sequence[NDArray[numpy.float32]]): The rasters, NaN as nodata, one per column. At least one.
        grid (raster_utilities.Grid): The grid every raster is on.
        detail_huc2s (Sequence[str]): The HUC-2 watersheds that the detail views zoom in on. An empty sequence
            omits the detail views.
        title (str): The figure title.
        colorbar_label (str): The colorbar label.
        output_path (Path): The path to the figure.

    Returns:
        output_path (Path): The path to the figure.
    """

    left, bottom, right, top = rasterio_transform.array_bounds(
        *grid.shape, grid.transform
    )
    extent = [left, right, bottom, top]
    grid_box = box(left, bottom, right, top)

    download_state_boundaries()
    states = geopandas.read_file(STATE_BOUNDARIES_PATH).to_crs(grid.crs)
    states = states[states.intersects(grid_box)].clip(grid_box)

    detail_view = _read_detail_view(detail_huc2s, grid) if detail_huc2s else None

    all_valid = numpy.concatenate([d[~numpy.isnan(d)].ravel() for d in columns])
    vmin = float(all_valid.min()) if all_valid.size else 0.0
    if not all_valid.size:
        vmax = 1.0
    elif VMAX_PERCENTILE is None:
        vmax = float(all_valid.max())
    else:
        vmax = float(numpy.percentile(all_valid, VMAX_PERCENTILE))
    if LOG:
        # LogNorm can't draw a scale whose top is <= 0
        vmax = max(vmax, LOG_VMIN)

    imshow_kw: dict[str, Any] = {"origin": "upper", "extent": extent}
    if LOG:
        imshow_kw["cmap"] = matplotlib.colormaps[CMAP].with_extremes(under=ZERO_COLOR)
        imshow_kw["norm"] = LogNorm(vmin=LOG_VMIN, vmax=vmax)
    else:
        imshow_kw.update(cmap=CMAP, vmin=vmin, vmax=vmax)

    n_cols = len(columns)
    n_rows = 1 if detail_view is None else 2
    col_width = 9.25
    col_gap = 1.5
    fig_width = col_width * n_cols + col_gap * (n_cols - 1)
    fig_height = col_width * (top - bottom) / (right - left) * n_rows

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(fig_width, fig_height),
        layout=ConstrainedLayoutEngine(hspace=0.01, wspace=0.02),
        squeeze=False,
    )

    panel_letters = iter(string.ascii_uppercase)
    overview_images = []
    for column, data in enumerate(columns):
        if LOG:
            # LogNorm draws values <= 0 like NaN, so they move below LOG_VMIN to take the "under" color. Small positive
            # values move up to LOG_VMIN, so they don't read as zero
            data = numpy.where(data <= 0, LOG_VMIN / 10, numpy.maximum(data, LOG_VMIN))

        ax_overview = axes[0][column]
        states.plot(ax=ax_overview, facecolor="none", edgecolor="black", linewidth=0.4)
        overview_images.append(ax_overview.imshow(data, **imshow_kw))
        if detail_view is not None and DETAIL_OUTLINE_COLOR:
            detail_view.outline.plot(
                ax=ax_overview,
                facecolor="none",
                edgecolor=DETAIL_OUTLINE_COLOR,
                linewidth=1.5,
                zorder=3,
            )
        ax_overview.set_xlim(left, right)
        ax_overview.set_ylim(bottom, top)
        _finish_panel(ax_overview, next(panel_letters))

        if detail_view is not None:
            ax_detail = axes[1][column]
            detail_view.outline.plot(
                ax=ax_detail, facecolor="none", edgecolor="black", linewidth=0.4
            )
            ax_detail.imshow(
                numpy.where(detail_view.outside, numpy.nan, data), **imshow_kw
            )
            ax_detail.set_xlim(detail_view.view_bounds[0], detail_view.view_bounds[2])
            ax_detail.set_ylim(detail_view.view_bounds[1], detail_view.view_bounds[3])
            ax_detail.set_aspect("equal", adjustable="datalim")
            _finish_panel(ax_detail, next(panel_letters))

    cbar = fig.colorbar(
        overview_images[-1],
        ax=axes.ravel().tolist(),
        fraction=0.02 * (2 / n_cols),
        pad=0.04,
        # arrows only for ends that some pixels actually fall beyond (zeros below
        # the log floor, values above the percentile cap)
        extend=["neither", "min", "max", "both"][
            bool(LOG and (all_valid <= 0).any())
            + 2 * bool(all_valid.size and all_valid.max() > vmax)
        ],
    )
    if VMAX_PERCENTILE is not None:
        # label the cap so readers can see where the top color starts
        lo = LOG_VMIN if LOG else vmin
        ticks = [t for t in cbar.get_ticks() if lo <= t < vmax * 0.8]
        labels = [
            f"$10^{{{round(numpy.log10(t))}}}$" if LOG else f"{t:g}" for t in ticks
        ]
        cbar.set_ticks(ticks + [vmax], labels=labels + [f"{vmax:.2g}"])
    cbar.set_label(colorbar_label, fontsize=20)
    cbar.ax.tick_params(labelsize=18)

    fig.suptitle(title, fontweight="bold", fontsize=24, wrap=True)

    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved to {output_path}")

    return output_path


def _read_detail_view(
    detail_huc2s: Sequence[str], grid: raster_utilities.Grid
) -> _DetailView:
    """
    Reads the HUC-2 boundaries of the detail view from input/HUC/<huc2>/Shape/WBDHU2.shp.

    Parameters:
        detail_huc2s (Sequence[str]): The HUC-2 watersheds that the detail view zooms in on.
        grid (raster_utilities.Grid): The grid of the rasters.

    Returns:
        detail_view (_DetailView): The cells outside the watersheds, their outline and their bounds.
    """

    huc2_boundaries = geopandas.GeoDataFrame(
        pandas.concat(
            [
                geopandas.read_file(
                    Path("input", "HUC", huc2, "Shape", "WBDHU2.shp")
                ).to_crs(grid.crs)
                for huc2 in detail_huc2s
            ]
        ),
        geometry="geometry",
    )
    outside = geometry_mask(
        huc2_boundaries.geometry,
        transform=grid.transform,
        invert=False,
        out_shape=grid.shape,
    )
    return _DetailView(
        outside, _dissolve_outline(huc2_boundaries), huc2_boundaries.total_bounds
    )


def _finish_panel(ax: plt.Axes, letter: str) -> None:
    """
    Clears a panel's ticks and draws its letter in the top left corner.

    Parameters:
        ax (plt.Axes): The panel.
        letter (str): The panel's letter.
    """

    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(
        0.01,
        0.99,
        f"{letter}.",
        transform=ax.transAxes,
        fontsize=18,
        fontweight="bold",
        va="top",
        ha="left",
    )


def _dissolve_outline(gdf: geopandas.GeoDataFrame) -> geopandas.GeoDataFrame:
    """
    Dissolves polygons into one outline.

    Adjacent mask polygons don't share vertices exactly, so dissolving leaves sliver holes along their common border
    that would show as lines. This function deletes all holes.

    Parameters:
        gdf (geopandas.GeoDataFrame): The polygons.

    Returns:
        outline (geopandas.GeoDataFrame): The dissolved outline.
    """

    def _drop_holes(geom: Polygon | MultiPolygon) -> Polygon | MultiPolygon:
        if isinstance(geom, Polygon):
            return Polygon(geom.exterior)
        return MultiPolygon([Polygon(p.exterior) for p in geom.geoms])

    outline = gdf.dissolve()
    outline["geometry"] = outline.geometry.map(_drop_holes)
    return outline


def _expand_paths(raster_paths: Sequence[str]) -> list[str]:
    """
    Expands glob patterns in a list of raster paths.

    Parameters:
        raster_paths (Sequence[str]): The paths, globs allowed.

    Returns:
        expanded_paths (list[str]): The matching paths. A path that matches nothing stays as given.
    """

    expanded_paths: list[str] = []
    for raster_path in raster_paths:
        # not Path.glob, which doesn't take absolute patterns
        matches = glob.glob(raster_path)
        expanded_paths.extend(sorted(matches) if matches else [raster_path])
    return expanded_paths


def _sum_rasters(
    raster_paths: Sequence[str], grid: raster_utilities.Grid
) -> NDArray[numpy.float32]:
    """
    Sums rasters cell by cell, treating NaN as nodata.

    Parameters:
        raster_paths (Sequence[str]): The rasters to sum.
        grid (raster_utilities.Grid): The grid they must be on.

    Returns:
        summed (NDArray[numpy.float32]): The sum, NaN where every raster is NaN.
    """

    layers = numpy.stack(
        [
            raster_utilities.read_raster(raster_path, grid).astype(numpy.float32)
            for raster_path in raster_paths
        ]
    )
    summed = numpy.nansum(layers, axis=0)
    summed[numpy.isnan(layers).all(axis=0)] = numpy.nan
    return summed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Render existing rasters, for figures the pipeline doesn't make."
    )
    parser.add_argument(
        "--left",
        nargs="+",
        required=True,
        help="The rasters summed into the left column (globs allowed).",
    )
    parser.add_argument(
        "--right",
        nargs="+",
        help="The rasters summed into the right column (globs allowed). Omit for one column.",
    )
    parser.add_argument(
        "--detail_huc2s",
        nargs="*",
        default=["14", "15"],
        help="The HUC-2 watersheds that the detail views zoom in on. Pass no values to omit the detail views.",
    )
    parser.add_argument("--title", required=True, help="The figure title.")
    parser.add_argument("--colorbar_label", required=True, help="The colorbar label.")
    parser.add_argument("--output", required=True, help="The path to the figure.")
    args = parser.parse_args()

    column_paths = [_expand_paths(args.left)]
    if args.right:
        column_paths.append(_expand_paths(args.right))

    figure_grid = raster_utilities.Grid.from_raster(column_paths[0][0])
    render(
        [_sum_rasters(paths, figure_grid) for paths in column_paths],
        figure_grid,
        args.detail_huc2s,
        args.title,
        args.colorbar_label,
        Path(args.output),
    )
