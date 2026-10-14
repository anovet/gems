"""Preview maps: pictures of fault probability maps, known faults and feature bands.

Used by the runner (one preview image per run) and by the exploration notebook.
Figures are built with the object-oriented matplotlib API on the non-interactive
Agg canvas, so nothing here touches pyplot's global state or needs a display.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from gems.raster import Grid, read_grid

INSET_PIXELS = 150  # zoomed inset side: 15 km at 100 m, so single fault traces are visible
OVERVIEW_PIXELS = 900  # known faults are max-pooled to about this many pixels across
PROBABILITY_CMAP = "magma"
FAULT_COLOUR = "#00e5ff"
_FAULT_CMAP = ListedColormap([FAULT_COLOUR])


@dataclass(frozen=True)
class Window:
    """A rectangle of grid pixels: rows [row, row + height), cols [col, col + width)."""

    row: int
    col: int
    height: int
    width: int

    @property
    def slices(self) -> tuple[slice, slice]:
        return slice(self.row, self.row + self.height), slice(self.col, self.col + self.width)


def save_preview(
    path: str | Path,
    probabilities: np.ndarray,
    faults: np.ndarray,
    study_area: np.ndarray,
    grid: Grid,
    title: str | None = None,
) -> Path:
    """Write a PNG of the fault probability map with known faults overlaid and a zoomed inset."""
    path = Path(path)
    fig = preview_figure(probabilities, faults, study_area, grid, title=title)
    FigureCanvasAgg(fig)
    fig.savefig(path, dpi=100)
    return path


def preview_figure(
    probabilities: np.ndarray,
    faults: np.ndarray,
    study_area: np.ndarray,
    grid: Grid,
    title: str | None = None,
    inset: Window | None = None,
) -> Figure:
    """Study-area overview of the fault probability map, known faults on top, plus a zoomed inset."""
    inset = inset or choose_inset(faults, study_area)
    fig = Figure(figsize=(16, 8), layout="constrained")
    overview, zoom = fig.subplots(1, 2, width_ratios=[1.15, 1])

    image = _draw_probabilities(overview, probabilities, study_area, grid, _bbox(study_area))
    _draw_faults(overview, faults, grid, _bbox(study_area), pool=True)
    x0, y0, x1, y1 = _extent_xy(grid, inset)
    overview.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="white", lw=1.5))
    overview.set_title("Fault probability map, known faults overlaid")

    _draw_probabilities(zoom, probabilities, study_area, grid, inset)
    _draw_faults(zoom, faults, grid, inset, pool=False, alpha=0.45)
    zoom.set_title(f"Zoomed inset ({inset.height * grid.res[1] / 1000:g} km, 100 m pixels)")

    fig.colorbar(image, ax=[overview, zoom], shrink=0.8, label="Fault probability")
    fig.suptitle(title or "Fault probability map")
    return fig


def known_faults_figure(faults: np.ndarray, study_area: np.ndarray, grid: Grid) -> Figure:
    """The known faults over the study area."""
    fig = Figure(figsize=(9, 9), layout="constrained")
    ax = fig.subplots()
    window = _bbox(study_area)
    _draw_mask(ax, study_area, grid, window, colour="#d9d9d9")
    _draw_faults(ax, faults, grid, window, pool=True, colour="black")
    ax.ticklabel_format(style="plain", useOffset=False)
    ax.set_title(f"Known faults over the study area ({faults.sum():,} fault pixels)")
    return fig


def feature_bands_figure(
    bands: np.ndarray, names: list[str], study_area: np.ndarray, grid: Grid, columns: int = 3
) -> Figure:
    """A grid of feature band maps, each stretched to its own 2nd-98th percentile."""
    rows = -(-len(names) // columns)
    fig = Figure(figsize=(5 * columns, 5 * rows), layout="constrained")
    axes = np.atleast_1d(fig.subplots(rows, columns)).ravel()
    window = _bbox(study_area)
    for ax, band, name in zip(axes, bands, names):
        values = np.where(study_area, band, np.nan)
        lo, hi = np.nanpercentile(values, [2, 98]) if np.isfinite(values).any() else (0, 1)
        image = ax.imshow(
            values[window.slices], extent=_extent(grid, window), cmap="viridis", vmin=lo, vmax=hi
        )
        fig.colorbar(image, ax=ax, shrink=0.7)
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes[len(names) :]:
        ax.set_axis_off()
    return fig


def read_fault_probability_map(path: str | Path) -> tuple[np.ndarray, Grid]:
    """Read a fault probability map GeoTIFF (e.g. a run's submission.tif) with nodata as NaN."""
    with rasterio.open(path) as src:
        probabilities = src.read(1).astype(np.float32, copy=False)
        nodata = src.nodata
    if nodata is not None and not np.isnan(nodata):
        probabilities[probabilities == nodata] = np.nan
    return probabilities, read_grid(Path(path))


def choose_inset(faults: np.ndarray, study_area: np.ndarray, size: int = INSET_PIXELS) -> Window:
    """The square window with the most known-fault pixels (searched on a half-window stride)."""
    rows, cols = faults.shape
    h, w = min(size, rows), min(size, cols)
    counts = np.pad(faults.astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    step = max(1, size // 2)
    starts_r = sorted({*range(0, rows - h + 1, step), rows - h})
    starts_c = sorted({*range(0, cols - w + 1, step), cols - w})
    r0 = np.array(starts_r)[:, None]
    c0 = np.array(starts_c)[None, :]
    totals = counts[r0 + h, c0 + w] - counts[r0, c0 + w] - counts[r0 + h, c0] + counts[r0, c0]
    if totals.max() == 0:  # no known faults: centre on the study area instead
        rr, cc = np.nonzero(study_area) if study_area.any() else ([rows // 2], [cols // 2])
        r = int(np.clip(np.median(rr) - h // 2, 0, rows - h))
        c = int(np.clip(np.median(cc) - w // 2, 0, cols - w))
        return Window(r, c, h, w)
    i, j = np.unravel_index(np.argmax(totals), totals.shape)
    return Window(starts_r[i], starts_c[j], h, w)


def _draw_probabilities(ax, probabilities, study_area, grid, window):
    values = np.where(study_area, probabilities, np.nan)[window.slices]
    _draw_mask(ax, study_area, grid, window, colour="#3a3a3a")
    image = ax.imshow(
        values, extent=_extent(grid, window), cmap=PROBABILITY_CMAP, vmin=0, vmax=1,
        interpolation="antialiased",
    )
    ax.set_xlabel("Easting (m, EPSG:32611)")
    ax.set_ylabel("Northing (m)")
    ax.ticklabel_format(style="plain", useOffset=False)
    return image


def _draw_faults(ax, faults, grid, window, pool, colour=FAULT_COLOUR, alpha=1.0):
    """Overlay known-fault pixels. In overviews, max-pool so thin fault traces survive downsampling."""
    cut = faults[window.slices]
    factor = max(1, -(-max(cut.shape) // OVERVIEW_PIXELS)) if pool else 1
    if factor > 1:
        h, w = cut.shape
        padded = np.zeros((-(-h // factor) * factor, -(-w // factor) * factor), dtype=bool)
        padded[:h, :w] = cut
        cut = padded.reshape(padded.shape[0] // factor, factor, -1, factor).any(axis=(1, 3))
        x0, x1, y0, y1 = _extent(grid, window)
        extent = (x0, x0 + cut.shape[1] * factor * grid.res[0], y1 - cut.shape[0] * factor * grid.res[1], y1)
    else:
        extent = _extent(grid, window)
    ax.imshow(
        np.ma.masked_where(~cut, cut), extent=extent, cmap=ListedColormap([colour]),
        interpolation="nearest", vmin=0, vmax=1, alpha=alpha,
    )
    ax.set_xlim(*_extent(grid, window)[:2])
    ax.set_ylim(*_extent(grid, window)[2:])


def _draw_mask(ax, mask, grid, window, colour):
    cut = mask[window.slices]
    ax.imshow(
        np.ma.masked_where(~cut, cut), extent=_extent(grid, window), cmap=ListedColormap([colour]),
        interpolation="nearest", vmin=0, vmax=1,
    )


def _bbox(mask: np.ndarray) -> Window:
    """Smallest window holding every True pixel (the whole grid if there are none)."""
    rows, cols = np.any(mask, axis=1), np.any(mask, axis=0)
    if not rows.any():
        return Window(0, 0, *mask.shape)
    r0, r1 = np.flatnonzero(rows)[[0, -1]]
    c0, c1 = np.flatnonzero(cols)[[0, -1]]
    return Window(int(r0), int(c0), int(r1 - r0 + 1), int(c1 - c0 + 1))


def _extent_xy(grid: Grid, window: Window) -> tuple[float, float, float, float]:
    """(x_min, y_min, x_max, y_max) of a window in map coordinates."""
    x0, y1 = grid.transform @ (window.col, window.row)
    x1, y0 = grid.transform @ (window.col + window.width, window.row + window.height)
    return x0, y0, x1, y1


def _extent(grid: Grid, window: Window) -> tuple[float, float, float, float]:
    """imshow extent (left, right, bottom, top) of a window."""
    x0, y0, x1, y1 = _extent_xy(grid, window)
    return x0, x1, y0, y1
