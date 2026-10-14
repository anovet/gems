"""Exact local implementation of the challenge metric, the distance-weighted Tversky index (DTI).

Follows the published formula (see docs/agents/background.md):

    TP_w = sum over fault-trace pixels g of  max_{x: d(x,g) <= R} p(x) k(d(x,g))
    FN_w = sum over fault-trace pixels g of  1 - (that same max)
    FP_w = sum over pixels x with p(x) > 0 of  p(x) [1 - max_g k(d(x,g))]
    DTI  = TP_w / (TP_w + alpha FP_w + beta FN_w + epsilon)

with the triangular kernel k(d) = max(1 - d/R, 0) and Euclidean distances between pixel
centres in metres. Only pixels inside the study area and outside the optional exclusion mask
take part, as predictions and as fault-trace pixels alike.

The maxima are computed exactly by enumerating every pixel offset with k > 0. `RegionScorer`
scores several regions (e.g. spatial folds) at once and is fast for scoring many maps;
`score_dti` is the single-region case of it.
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
    scorer = RegionScorer(
        fault_trace, np.where(scored, 0, -1).astype(np.int8), alpha=alpha, beta=beta,
        distance_tolerance=distance_tolerance, epsilon=epsilon, pixel_size=pixel_size,
    )
    return scorer.score(probability)[0]


class RegionScorer:
    """The DTI of each region of a grid, for scoring many fault probability maps quickly.

    `regions` labels every pixel with a region number 0..n-1, or -1 for pixels that are not
    scored. Each region is scored as its own study area: its predictions earn credit only from
    its own fault-trace pixels, and vice versa. Everything that does not depend on the
    probabilities is computed once: each pixel's false-positive weight, and each fault-trace
    pixel's neighbours (within the distance tolerance and in the same region) with their kernel
    weights. Scoring a map then costs one pass over the scored pixels.
    """

    def __init__(
        self,
        fault_trace: np.ndarray,
        regions: np.ndarray,
        *,
        alpha: float = ALPHA,
        beta: float = BETA,
        distance_tolerance: float = DISTANCE_TOLERANCE_M,
        epsilon: float = EPSILON,
        pixel_size: float = PIXEL_SIZE_M,
    ):
        regions = np.asarray(regions)
        if np.shape(fault_trace) != regions.shape:
            raise ValueError(f"fault_trace has shape {np.shape(fault_trace)}, expected {regions.shape}")
        self.alpha, self.beta, self.epsilon = alpha, beta, epsilon
        self.shape = regions.shape
        self.n_regions = int(regions.max()) + 1 if regions.size else 0
        scored = regions >= 0
        self._pixels = np.flatnonzero(scored)  # flat grid index of each scored pixel
        self._region = regions.ravel()[self._pixels].astype(np.intp)
        position = np.full(regions.shape, -1, dtype=np.int64)  # scored-pixel number of each pixel
        position.ravel()[self._pixels] = np.arange(len(self._pixels))

        rows, cols = np.nonzero(scored & np.asarray(fault_trace, dtype=bool))
        self._trace_region = regions[rows, cols].astype(np.intp)
        self._n_trace = np.bincount(self._trace_region, minlength=self.n_regions).astype(np.float64)
        offsets = _kernel_offsets(distance_tolerance, pixel_size)
        best_k = np.zeros(regions.shape, dtype=np.float32)  # kernel weight to the nearest fault trace
        # Neighbours outside the grid or region point at scored pixel 0 with weight 0.
        self._neighbour = np.zeros((len(rows), len(offsets)), dtype=np.int64)
        self._neighbour_k = np.zeros((len(rows), len(offsets)), dtype=np.float32)
        n_rows, n_cols = regions.shape
        for j, (dr, dc, k) in enumerate(offsets):
            r, c = rows + dr, cols + dc
            inside = (r >= 0) & (r < n_rows) & (c >= 0) & (c < n_cols)
            same = np.zeros(len(rows), dtype=bool)
            same[inside] = regions[r[inside], c[inside]] == self._trace_region[inside]
            r, c = r[same], c[same]
            best_k[r, c] = np.maximum(best_k[r, c], k)
            self._neighbour[same, j] = position[r, c]
            self._neighbour_k[same, j] = k
        self._fp_weight = 1.0 - best_k.ravel()[self._pixels]

    def pixel_values(self, probability: np.ndarray) -> np.ndarray:
        """The probabilities of the scored pixels (row-major order) as float32, NaN as 0."""
        if np.shape(probability) != self.shape:
            raise ValueError(f"probability has shape {np.shape(probability)}, expected {self.shape}")
        values = np.asarray(probability, dtype=np.float32).ravel()[self._pixels]
        return np.nan_to_num(values, nan=0.0, copy=False)

    def score(self, probability: np.ndarray) -> list[DtiScore]:
        """The DTI of each region for a fault probability map."""
        return self.score_pixels(self.pixel_values(probability))

    def score_pixels(self, values: np.ndarray, at: np.ndarray | None = None) -> list[DtiScore]:
        """The DTI of each region, given the values of the scored pixels.

        values: aligned with `pixel_values`, or, with `at`, the values of the scored pixels
            numbered `at` (indices into `pixel_values`), every other pixel being 0. Scoring
            only the nonzero pixels this way gives exactly the full-map score, faster.
        """
        values = np.asarray(values, dtype=np.float32)
        if at is None:
            dense, region, fp_weight = values, self._region, self._fp_weight
        else:
            dense = np.zeros(len(self._pixels), dtype=np.float32)
            dense[at] = values
            region, fp_weight = self._region[at], self._fp_weight[at]
        fp_w = np.bincount(region, weights=values * fp_weight, minlength=self.n_regions)
        credit = (dense[self._neighbour] * self._neighbour_k).max(axis=1, initial=0.0)
        tp_w = np.bincount(self._trace_region, weights=credit, minlength=self.n_regions)
        fn_w = self._n_trace - tp_w
        dti = tp_w / (tp_w + self.alpha * fp_w + self.beta * fn_w + self.epsilon)
        return [DtiScore(float(d), float(t), float(f), float(n)) for d, t, f, n in zip(dti, tp_w, fp_w, fn_w)]


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
