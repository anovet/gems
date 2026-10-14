"""Raster I/O: the reference grid, feature bands and the study area."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from rasterio.crs import CRS


@dataclass(frozen=True)
class Grid:
    """The pixel grid every raster in a run shares (CRS, transform, shape)."""

    crs: CRS
    transform: Affine
    height: int
    width: int

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    @property
    def res(self) -> tuple[float, float]:
        return (abs(self.transform.a), abs(self.transform.e))

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return rasterio.transform.array_bounds(self.height, self.width, self.transform)[0:4]


def read_grid(path: Path) -> Grid:
    with rasterio.open(path) as src:
        return Grid(src.crs, src.transform, src.height, src.width)


def band_names(path: Path) -> list[str]:
    """Feature band names, from per-band `band_name` metadata."""
    with rasterio.open(path) as src:
        return [src.tags(i).get("band_name", f"band_{i}") for i in range(1, src.count + 1)]


def read_feature_bands(path: Path, names: list[str] | None = None) -> tuple[np.ndarray, list[str]]:
    """Read feature bands by name as float32 (bands, rows, cols), with nodata as NaN.

    `names=None` reads every band in file order.
    """
    available = band_names(path)
    names = list(available) if names is None else list(names)
    missing = [n for n in names if n not in available]
    if missing:
        raise ValueError(f"Feature bands not found in {path}: {missing}. Available: {available}")
    indexes = [available.index(n) + 1 for n in names]
    with rasterio.open(path) as src:
        data = src.read(indexes).astype(np.float32, copy=False)
        nodata = src.nodata
    if nodata is not None:
        data[data == np.float32(nodata)] = np.nan
    return data, names


def read_labels(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read the known-fault labels.

    Returns (known_faults, study_area): two bool grids. The study area is the label
    raster's valid (non-nodata) pixels; known faults are pixels labelled 1.
    """
    with rasterio.open(path) as src:
        labels = src.read(1)
        nodata = src.nodata
    study_area = labels != nodata if nodata is not None else np.ones(labels.shape, dtype=bool)
    known_faults = (labels == 1) & study_area
    return known_faults, study_area
