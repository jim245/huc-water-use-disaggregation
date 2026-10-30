"""
Utility for rendering maps of water use rasters.

Each map is a column: the full grid on top, and optionally a zoomed view of a set of HUC-2 watersheds below it.
"""

import argparse
import glob
import string
from os import path

import geopandas
import matplotlib
import matplotlib.pyplot as plt
import numpy
import pandas
from matplotlib.colors import LogNorm
from numpy.typing import NDArray
from rasterio import transform as rasterio_transform
from rasterio.features import geometry_mask
from shapely.geometry import MultiPolygon, Polygon, box

from utilities import raster_utilities

STATE_BOUNDARIES_PATH = path.join("input", "USA", "cb_2024_us_state_500k.shp")

# the (sector, measure) each case study's statistics report
CASE_STUDY_LABELS = {
    "irr": ("Irrigation", "Water Consumption"),
    "pws-cu": ("Public Water Supply", "Water Consumption"),
    "pws-gw": ("Public Water Supply", "Groundwater Use"),
    "pws-sw": ("Public Water Supply", "Surface Water Use"),
    "pws-tot": ("Public Water Supply", "Total Water Use"),
}

MASK_OVERLAY_COLOR = "red"  # outline of mask region on top panels; None to omit
CMAP = "viridis_r"
LOG = True
LOG_VMIN = (
    1e-3  # LOG only: bottom of the colorbar; values in (0, LOG_VMIN) are drawn at it
)
ZERO_COLOR = "lightgrey"  # LOG only: color for zero-valued pixels
VMAX_PERCENTILE = (
    100  # top of the colorbar; higher values share its color. None for max
)


def check_inputs(huc2s: list[str], bottom_panel_huc2s: list[str]):
    """
    Fails before a pipeline run starts if its figure couldn't be rendered at the end.

    Parameters:
        huc2s (list[str]): The HUC-2 watersheds of the run.
        bottom_panel_huc2s (list[str]): The HUC-2 watersheds shown in the bottom panels.
    """

    if not path.exists(STATE_BOUNDARIES_PATH):
        raise FileNotFoundError(f"{STATE_BOUNDARIES_PATH} doesn't exist!")

    # only the run's HUC-2s are downloaded and on the grid
    missing_huc2s = sorted(set(bottom_panel_huc2s) - set(huc2s))
    if missing_huc2s:
        raise ValueError(f"Bottom panel HUC-2s {missing_huc2s} aren't in --huc2s!")


def run(
    raster: NDArray[numpy.float32],
    grid: raster_utilities.Grid,
    bottom_panel_huc2s: list[str],
    huc_level: int,
    case_study: str,
    year: int,
) -> str:
    """
    Renders a disaggregated raster, with its title and label taken from the case study and year.

    The figure is written to output/huc<huc_level>/<case_study>/<year>/<case_study>_huc<huc_level>_<year>.png.

    Parameters:
        raster (NDArray[numpy.float32]): The disaggregated raster, NaN as nodata.
        grid (raster_utilities.Grid): The grid the raster is on.
        bottom_panel_huc2s (list[str]): The HUC-2 watersheds shown in the bottom panel; empty to omit it.
        huc_level (int): The HUC scale used for spatial disaggregation.
        case_study (str): The case study.
        year (int): The year.

    Returns:
        output_path (str): The path to the figure.
    """

    sector, measure = CASE_STUDY_LABELS[case_study]
    units = "log10 Million Gallons per Day" if LOG else "Million Gallons per Day"

    return render(
        [raster],
        grid,
        bottom_panel_huc2s,
        f"Spatially Disaggregated {year} USGS {sector} {measure}",
        f"{measure} ({units})",
        path.join(
            "output",
            f"huc{huc_level}",
            case_study,
            str(year),
            f"{case_study}_huc{huc_level}_{year}.png",
        ),
    )


def render(
    columns: list[NDArray[numpy.float32]],
    grid: raster_utilities.Grid,
    bottom_panel_huc2s: list[str],
    title: str,
    colorbar_label: str,
    output_path: str,
) -> str:
    """
    Renders one column of panels per raster, all sharing one color scale.

    Parameters:
        columns (list[NDArray[numpy.float32]]): The rasters, NaN as nodata, one per column.
        grid (raster_utilities.Grid): The grid every raster is on.
        bottom_panel_huc2s (list[str]): The HUC-2 watersheds shown in the bottom panels; empty to omit them.
        title (str): The figure title.
        colorbar_label (str): The colorbar label.
        output_path (str): The path to the figure.

    Returns:
        output_path (str): The path to the figure.
    """

    has_bottom_panels = bool(bottom_panel_huc2s)
    left, bottom, right, top = rasterio_transform.array_bounds(
        *grid.shape, grid.transform
    )
    extent = [left, right, bottom, top]

    states = geopandas.read_file(STATE_BOUNDARIES_PATH).to_crs(grid.crs)
    states = states[states.intersects(box(left, bottom, right, top))].clip(
        box(left, bottom, right, top)
    )

    if has_bottom_panels:
        mask_gdf = geopandas.GeoDataFrame(
            pandas.concat(
                [
                    geopandas.read_file(
                        path.join("input", "HUC", huc2, "Shape", "WBDHU2.shp")
                    ).to_crs(grid.crs)
                    for huc2 in bottom_panel_huc2s
                ]
            ),
            geometry="geometry",
        )
        outside = geometry_mask(
            mask_gdf.geometry,
            transform=grid.transform,
            invert=False,
            out_shape=grid.shape,
        )
        mask_outline = _dissolve_outline(mask_gdf)
        view_bounds = mask_gdf.total_bounds  # (minx, miny, maxx, maxy)

    all_valid = numpy.concatenate([d[~numpy.isnan(d)].ravel() for d in columns])
    vmin = float(all_valid.min()) if all_valid.size else 0.0
    if not all_valid.size:
        vmax = 1.0
    elif VMAX_PERCENTILE is None:
        vmax = float(all_valid.max())
    else:
        vmax = float(numpy.percentile(all_valid, VMAX_PERCENTILE))

    imshow_kw = dict(origin="upper", extent=extent)
    if LOG:
        imshow_kw["cmap"] = matplotlib.colormaps[CMAP].with_extremes(under=ZERO_COLOR)
        imshow_kw["norm"] = LogNorm(vmin=LOG_VMIN, vmax=vmax)
    else:
        imshow_kw.update(cmap=CMAP, vmin=vmin, vmax=vmax)

    n_cols = len(columns)
    n_rows = 2 if has_bottom_panels else 1
    col_width = 9.25  # matches (20 - 1.5) / 2 from the original two-column layout
    fig_width = col_width * n_cols + 1.5 * (n_cols - 1)
    fig_height = col_width * (top - bottom) / (right - left) * n_rows

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(fig_width, fig_height),
        layout="constrained",
        squeeze=False,
    )
    fig.get_layout_engine().set(hspace=0.01, wspace=0.02)

    panel_letters = iter(string.ascii_uppercase)
    for column, data in enumerate(columns):
        if LOG:
            # zeros fall below LOG_VMIN and take the colormap's "under" color; small
            # positive values are raised to LOG_VMIN so they don't read as zero
            data = numpy.where((data > 0) & (data < LOG_VMIN), LOG_VMIN, data)

        ax_top = axes[0][column]
        states.plot(ax=ax_top, facecolor="none", edgecolor="black", linewidth=0.4)
        im = ax_top.imshow(data, **imshow_kw)
        if has_bottom_panels and MASK_OVERLAY_COLOR:
            mask_outline.plot(
                ax=ax_top,
                facecolor="none",
                edgecolor=MASK_OVERLAY_COLOR,
                linewidth=1.5,
                zorder=3,
            )
        ax_top.set_xlim(left, right)
        ax_top.set_ylim(bottom, top)
        _finish_panel(ax_top, next(panel_letters))

        if has_bottom_panels:
            ax_bottom = axes[1][column]
            mask_outline.plot(
                ax=ax_bottom, facecolor="none", edgecolor="black", linewidth=0.4
            )
            ax_bottom.imshow(numpy.where(outside, numpy.nan, data), **imshow_kw)
            ax_bottom.set_xlim(view_bounds[0], view_bounds[2])
            ax_bottom.set_ylim(view_bounds[1], view_bounds[3])
            ax_bottom.set_aspect("equal", adjustable="datalim")
            _finish_panel(ax_bottom, next(panel_letters))

    cbar = fig.colorbar(
        im,
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
        # label the cap so readers can see where the top color begins
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


def _finish_panel(ax: plt.Axes, letter: str):
    """
    Removes a panel's ticks and draws its letter in the top left corner.

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
    that would be drawn as lines. Holes are dropped.

    Parameters:
        gdf (geopandas.GeoDataFrame): The polygons.

    Returns:
        outline (geopandas.GeoDataFrame): The dissolved outline.
    """

    def _drop_holes(geom):
        if geom.geom_type == "Polygon":
            return Polygon(geom.exterior)
        return MultiPolygon([Polygon(p.exterior) for p in geom.geoms])

    outline = gdf.dissolve()
    outline["geometry"] = outline.geometry.apply(_drop_holes)
    return outline


def _expand_paths(raster_paths: list[str]) -> list[str]:
    """
    Expands glob patterns in a list of raster paths.

    Parameters:
        raster_paths (list[str]): The paths, globs allowed.

    Returns:
        expanded_paths (list[str]): The matching paths; a path that matches nothing is kept as given.
    """

    expanded_paths = []
    for raster_path in raster_paths:
        matches = glob.glob(raster_path)
        expanded_paths.extend(sorted(matches) if matches else [raster_path])
    return expanded_paths


def _sum_rasters(
    raster_paths: list[str], grid: raster_utilities.Grid
) -> NDArray[numpy.float32]:
    """
    Sums rasters cell by cell, treating NaN as nodata.

    Parameters:
        raster_paths (list[str]): The rasters to sum.
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
        help="The rasters summed into the right column (globs allowed); omit for one column.",
    )
    parser.add_argument(
        "--bottom_panel_huc2s",
        nargs="*",
        default=["14", "15"],
        help="The HUC-2 watersheds shown in the bottom panels; pass none to omit them.",
    )
    parser.add_argument("--title", required=True, help="The figure title.")
    parser.add_argument(
        "--colorbar_label", required=True, help="The colorbar label."
    )
    parser.add_argument("--output", required=True, help="The path to the figure.")
    args = parser.parse_args()

    column_paths = [_expand_paths(args.left)]
    if args.right:
        column_paths.append(_expand_paths(args.right))

    figure_grid = raster_utilities.Grid.from_raster(column_paths[0][0])
    render(
        [_sum_rasters(paths, figure_grid) for paths in column_paths],
        figure_grid,
        args.bottom_panel_huc2s,
        args.title,
        args.colorbar_label,
        args.output,
    )
