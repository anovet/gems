"""Submission writer and validator.

A submission is a fault probability map in the exact challenge format: the
reference grid (CRS, 100 m resolution, bounds, shape), one float32 band, values
in [0, 1], NaN exactly outside the study area.
"""

from pathlib import Path

import numpy as np
import rasterio

from gems.raster import Grid, read_grid, read_labels


def write_submission(path: Path, probabilities: np.ndarray, grid: Grid) -> Path:
    """Write a fault probability map as a single-band float32 GeoTIFF with NaN nodata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w", driver="GTiff", width=grid.width, height=grid.height, count=1,
        dtype="float32", crs=grid.crs, transform=grid.transform, nodata=np.nan,
        compress="lzw",
    ) as dst:
        dst.write(probabilities.astype(np.float32), 1)
    return path


def validate_submission(path: Path, reference: Path, labels: Path) -> list[str]:
    """Check a GeoTIFF against the challenge format.

    `reference` supplies the grid (normally the features raster) and `labels`
    the study area. Returns human-readable problems; empty means valid.
    """
    ref = read_grid(reference)
    _, study_area = read_labels(labels)
    problems: list[str] = []

    with rasterio.open(path) as src:
        if src.crs != ref.crs:
            problems.append(f"CRS is {src.crs}, expected {ref.crs}.")
        res = (abs(src.transform.a), abs(src.transform.e))
        if not np.allclose(res, ref.res):
            problems.append(f"Resolution is {res}, expected {ref.res}.")
        if (src.height, src.width) != ref.shape:
            problems.append(f"Shape is {(src.height, src.width)} (rows, cols), expected {ref.shape}.")
        if not np.allclose(tuple(src.bounds), ref.bounds):
            problems.append(f"Bounds are {tuple(src.bounds)}, expected {ref.bounds}.")
        if src.count != 1:
            problems.append(f"Has {src.count} bands, expected exactly 1.")
        dtype = src.dtypes[0]
        if dtype != "float32":
            problems.append(f"Data type is {dtype}, expected float32.")
        if problems and (src.height, src.width) != ref.shape:
            return problems  # pixel checks are meaningless on the wrong shape
        values = src.read(1, masked=False).astype(np.float64)
        nodata = src.nodata

    if nodata is not None and not np.isnan(nodata):
        problems.append(f"Nodata value is {nodata}, expected NaN.")
        values[values == nodata] = np.nan

    is_nan = np.isnan(values)
    inside = values[study_area & ~is_nan]
    missing = int(np.count_nonzero(study_area & is_nan))
    if missing:
        problems.append(f"{missing} pixels inside the study area are NaN; every study-area pixel needs a value.")
    stray = int(np.count_nonzero(~study_area & ~is_nan))
    if stray:
        problems.append(f"{stray} pixels outside the study area are not NaN; they must be NaN.")
    out_of_range = int(np.count_nonzero((inside < 0) | (inside > 1)))
    if out_of_range:
        problems.append(
            f"{out_of_range} study-area values are outside [0, 1] "
            f"(min {inside.min():.4g}, max {inside.max():.4g})."
        )
    return problems
