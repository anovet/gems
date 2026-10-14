"""Exact local implementation of the challenge metric, the distance-weighted Tversky index (DTI).

Follows the published formula (see docs/agents/background.md):

    TP_w = sum over fault-trace pixels g of  max_{x: d(x,g) <= R} p(x) k(d(x,g))
    FN_w = sum over fault-trace pixels g of  1 - (that same max)
    FP_w = sum over pixels x with p(x) > 0 of  p(x) [1 - max_g k(d(x,g))]
    DTI  = TP_w / (TP_w + alpha FP_w + beta FN_w + epsilon)

with the triangular kernel k(d) = max(1 - d/R, 0) and Euclidean distances between pixel
centres in metres. Only pixels inside the study area and outside the optional exclusion mask
take part, as predictions and as fault-trace pixels alike.

The maxima are computed exactly by enumerating every pixel offset with k > 0.
"""

from dataclasses import dataclass

import numpy as np

ALPHA = 0.2
BETA = 0.8
DISTANCE_TOLERANCE_M = 300.0
EPSILON = 1e-6
PIXEL_SIZE_M = 100.0


@dataclass(frozen=True)
class DtiScore:
    """The DTI and its weighted true-positive, false-positive and false-negative components."""

    dti: float
    tp_w: float
    fp_w: float
    fn_w: float


def score_dti(
    probability: np.ndarray,
    fault_trace: np.ndarray,
    study_area: np.ndarray,
    exclusion: np.ndarray | None = None,
    *,
    alpha: float = ALPHA,
    beta: float = BETA,
    distance_tolerance: float = DISTANCE_TOLERANCE_M,
    epsilon: float = EPSILON,
    pixel_size: float = PIXEL_SIZE_M,
) -> DtiScore:
    """Score a fault probability map against fault-trace pixels with the DTI.

    probability: 2-D grid of values in [0, 1]; anything outside the scored pixels (e.g. NaN
        outside the study area) is ignored.
    fault_trace: 2-D grid, truthy where a fault trace passes through the pixel.
    study_area: 2-D boolean grid, True where the pixel is in the study area.
    exclusion: optional 2-D boolean grid of pixels to ignore entirely (e.g. known faults when
        imitating the official round).
    distance_tolerance, pixel_size: in metres.
    """
    shape = np.shape(probability)
    for name, array in (("fault_trace", fault_trace), ("study_area", study_area), ("exclusion", exclusion)):
        if array is not None and np.shape(array) != shape:
            raise ValueError(f"{name} has shape {np.shape(array)}, expected {shape}")

    scored = np.asarray(study_area, dtype=bool)
    if exclusion is not None:
        scored = scored & ~np.asarray(exclusion, dtype=bool)
    p = np.where(scored, np.asarray(probability, dtype=np.float32), np.float32(0.0))
    g = scored & np.asarray(fault_trace, dtype=bool)

    offsets = _kernel_offsets(distance_tolerance, pixel_size)
    best_credit = _neighbourhood_max(p, offsets)  # max p(x) k(d(x, g)) around each pixel
    best_k = _neighbourhood_max(g.astype(np.float32), offsets)  # max k to the nearest fault trace

    tp_w = float(best_credit[g].sum(dtype=np.float64))
    fn_w = float(g.sum()) - tp_w
    predicted = p > 0
    fp_w = float((p[predicted] * (1.0 - best_k[predicted])).sum(dtype=np.float64))

    dti = tp_w / (tp_w + alpha * fp_w + beta * fn_w + epsilon)
    return DtiScore(dti=dti, tp_w=tp_w, fp_w=fp_w, fn_w=fn_w)


def _kernel_offsets(distance_tolerance: float, pixel_size: float) -> list[tuple[int, int, np.float32]]:
    """Every (row, col) offset with a positive triangular-kernel weight, and that weight."""
    reach = int(np.ceil(distance_tolerance / pixel_size))
    offsets = []
    for dr in range(-reach, reach + 1):
        for dc in range(-reach, reach + 1):
            k = 1.0 - pixel_size * np.hypot(dr, dc) / distance_tolerance
            if k > 0:
                offsets.append((dr, dc, np.float32(k)))
    return offsets


def _neighbourhood_max(values: np.ndarray, offsets: list[tuple[int, int, np.float32]]) -> np.ndarray:
    """For each pixel, the max over offsets of k * values at that offset (zero beyond the grid)."""
    reach = max(max(abs(dr), abs(dc)) for dr, dc, _ in offsets)
    rows, cols = values.shape
    padded = np.pad(values, reach)
    result = np.zeros_like(values)
    weighted = np.empty_like(values)
    for dr, dc, k in offsets:
        window = padded[reach + dr : reach + dr + rows, reach + dc : reach + dc + cols]
        np.multiply(window, k, out=weighted)
        np.maximum(result, weighted, out=result)
    return result
