"""Synthetic dataset fixture that mimics the real challenge rasters.

Never touches the real data/ folder: everything is written under tmp_path.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

CRS = "EPSG:32611"
PIXEL = 100.0
FEATURE_NODATA = float(np.finfo(np.float32).min)  # same as the real features raster
LABEL_NODATA = -1
ROWS, COLS = 60, 80
ORIGIN_X, ORIGIN_Y = 300_000.0, 4_200_000.0
BAND_NAMES = ["mag_anom", "iso_grav_anom_slope", "geod_shearrate"]
FAULT_BAND = "iso_grav_anom_slope"  # the band that correlates with fault traces


@dataclass
class SyntheticDataset:
    root: Path
    features: Path
    labels: Path
    config: Path
    runs_dir: Path
    study_area: np.ndarray  # bool, True inside the study area
    faults: np.ndarray  # bool, True on known-fault pixels


def _study_area(rows: int, cols: int) -> np.ndarray:
    """Irregular blob with a nodata border on every side."""
    r, c = np.mgrid[0:rows, 0:cols]
    ellipse = ((r - rows / 2) / (rows * 0.42)) ** 2 + ((c - cols / 2) / (cols * 0.45)) ** 2 <= 1
    notch = (r < rows // 3) & (c > cols * 0.6)  # bite out a corner to make it irregular
    return ellipse & ~notch


def _fault_traces(rows: int, cols: int) -> np.ndarray:
    faults = np.zeros((rows, cols), dtype=bool)
    faults[np.arange(5, 55), np.arange(10, 60)] = True  # diagonal
    faults[30, 5:75] = True  # east-west
    faults[8:52, 20] = True  # north-south
    return faults


def write_dataset(root: Path, seed: int = 0) -> SyntheticDataset:
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True, exist_ok=True)
    transform = from_origin(ORIGIN_X, ORIGIN_Y, PIXEL, PIXEL)
    study = _study_area(ROWS, COLS)
    faults = _fault_traces(ROWS, COLS) & study

    bands = rng.normal(size=(len(BAND_NAMES), ROWS, COLS)).astype(np.float32)
    bands[BAND_NAMES.index(FAULT_BAND)] += 3.0 * faults
    bands[:, ~study] = FEATURE_NODATA
    # A few study-area pixels missing one band, like the real data.
    holes = study & (rng.random((ROWS, COLS)) < 0.02)
    bands[0, holes] = FEATURE_NODATA

    features = root / "features.tif"
    with rasterio.open(
        features, "w", driver="GTiff", width=COLS, height=ROWS, count=len(BAND_NAMES),
        dtype="float32", crs=CRS, transform=transform, nodata=FEATURE_NODATA,
    ) as dst:
        dst.write(bands)
        for i, name in enumerate(BAND_NAMES, start=1):
            dst.update_tags(i, band_name=name, description=f"synthetic {name}")
            dst.set_band_description(i, f"{name} - synthetic {name}")

    labels_arr = np.full((ROWS, COLS), LABEL_NODATA, dtype=np.int8)
    labels_arr[study] = 0
    labels_arr[faults] = 1
    labels = root / "labels.tif"
    with rasterio.open(
        labels, "w", driver="GTiff", width=COLS, height=ROWS, count=1,
        dtype="int8", crs=CRS, transform=transform, nodata=LABEL_NODATA,
    ) as dst:
        dst.write(labels_arr, 1)

    runs_dir = root / "runs"
    config = root / "config.toml"
    bands_toml = "[" + ", ".join(f'"{b}"' for b in BAND_NAMES) + "]"
    config.write_text(
        f"""\
[run]
name = "synthetic"
iteration = "V1"

[data]
features = '{features.as_posix()}'
labels = '{labels.as_posix()}'
bands = {bands_toml}

[output]
runs_dir = '{runs_dir.as_posix()}'
""",
        encoding="utf-8",
    )
    return SyntheticDataset(root, features, labels, config, runs_dir, study, faults)


@pytest.fixture
def synthetic(tmp_path) -> SyntheticDataset:
    return write_dataset(tmp_path / "synthetic")
