"""Post-processing: map raw model probabilities to submitted values, tuned on the local score.

The DTI makes false alarms cheap (alpha = 0.2) and misses expensive (beta = 0.8), so raw
probabilities are rarely the best values to submit. The transform family is

    value = clip(scale * p ** power, 0, 1)   where p >= threshold, else 0

and the identity (threshold 0, power 1, scale 1) is always a candidate, so tuning can never
lower the local score it is tuned on. Tuning is a grid search over the out-of-fold predictions
scored per spatial fold: mildly optimistic, since the same folds choose and score the transform
(no nested cross-validation; accepted for V1).
"""

from dataclasses import asdict, dataclass
from itertools import product
from typing import Any

import numpy as np

from gems.dti import ALPHA, BETA, DISTANCE_TOLERANCE_M, EPSILON, PIXEL_SIZE_M, _kernel_offsets, _neighbourhood_max

TUNING_NOTE = (
    "Post-processing was tuned by grid search on the same out-of-fold predictions it is scored "
    "on (no nested cross-validation), so the post-processed local score is mildly optimistic."
)


@dataclass(frozen=True)
class Transform:
    threshold: float = 0.0
    power: float = 1.0
    scale: float = 1.0

    def __call__(self, probability: np.ndarray) -> np.ndarray:
        """Apply to a fault probability map; NaN (outside the study area) stays NaN."""
        p = np.asarray(probability)
        keep = p >= self.threshold  # False for NaN
        out = np.zeros_like(p)
        out[keep] = np.clip(self.scale * p[keep] ** self.power, 0.0, 1.0)
        out[np.isnan(p)] = np.nan
        return out

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


IDENTITY = Transform()


def tune_transform(
    out_of_fold: np.ndarray, faults: np.ndarray, folds: np.ndarray, search_space: dict[str, Any]
) -> Transform:
    """The transform from `search_space` (plus the identity) with the best mean local score.

    `search_space` has lists `thresholds`, `powers` and `scales`; every combination is tried.
    Ties go to the identity, then to the earliest combination.
    """
    scorer = _FoldScorer(out_of_fold, faults, folds)
    best, best_score = IDENTITY, scorer.mean_score(IDENTITY)
    for threshold in search_space["thresholds"]:
        selected = scorer.select(threshold)
        for power in search_space["powers"]:
            powered = selected.p ** np.float32(power)
            for scale in search_space["scales"]:
                values = np.minimum(np.float32(scale) * powered, np.float32(1.0))
                score = scorer.mean_score_of(selected, values, Transform(threshold, power, scale))
                if score > best_score:
                    best, best_score = Transform(float(threshold), float(power), float(scale)), score
    return best


@dataclass
class _Selection:
    p: np.ndarray  # probabilities of the scored pixels at or above the threshold
    fp_weight: np.ndarray  # 1 - kernel weight to the nearest fault trace in the same fold
    fold: np.ndarray


class _FoldScorer:
    """Fast, exact per-fold DTI of transformed out-of-fold predictions.

    Everything that does not depend on the submitted values is computed once: each pixel's
    false-positive weight, and each fault-trace pixel's neighbours (within the distance
    tolerance and in the same spatial fold) with their kernel weights. Equivalent to
    `gems.folds.local_scores(transform(out_of_fold), faults, folds)`.
    """

    def __init__(self, out_of_fold: np.ndarray, faults: np.ndarray, folds: np.ndarray):
        offsets = _kernel_offsets(DISTANCE_TOLERANCE_M, PIXEL_SIZE_M)
        self.n_folds = int(folds.max()) + 1
        scored = folds >= 0
        trace = faults.astype(bool) & scored

        best_k = np.zeros(folds.shape, dtype=np.float32)
        for k in range(self.n_folds):
            in_fold = folds == k
            best_k[in_fold] = _neighbourhood_max((trace & in_fold).astype(np.float32), offsets)[in_fold]
        self.p = out_of_fold[scored].astype(np.float32)
        self.fp_weight = 1.0 - best_k[scored]
        self.fold = folds[scored]

        rows, cols = np.nonzero(trace)
        trace_fold = folds[rows, cols]
        n_rows, n_cols = folds.shape
        self.neighbour_p = np.zeros((len(rows), len(offsets)), dtype=np.float32)
        self.neighbour_k = np.zeros((len(rows), len(offsets)), dtype=np.float32)
        for j, (dr, dc, k) in enumerate(offsets):
            r, c = rows + dr, cols + dc
            inside = (r >= 0) & (r < n_rows) & (c >= 0) & (c < n_cols)
            same_fold = np.zeros(len(rows), dtype=bool)
            same_fold[inside] = folds[r[inside], c[inside]] == trace_fold[inside]
            self.neighbour_p[same_fold, j] = out_of_fold[r[same_fold], c[same_fold]]
            self.neighbour_k[same_fold, j] = k
        self.trace_fold = trace_fold
        self.n_trace = np.bincount(trace_fold, minlength=self.n_folds).astype(np.float64)

    def select(self, threshold: float) -> _Selection:
        keep = self.p >= threshold
        return _Selection(self.p[keep], self.fp_weight[keep], self.fold[keep])

    def mean_score(self, transform: Transform) -> float:
        selected = self.select(transform.threshold)
        values = np.clip(transform.scale * selected.p**transform.power, 0.0, 1.0).astype(np.float32)
        return self.mean_score_of(selected, values, transform)

    def mean_score_of(self, selected: _Selection, values: np.ndarray, transform: Transform) -> float:
        fp_w = np.bincount(selected.fold, weights=values * selected.fp_weight, minlength=self.n_folds)
        credit = transform(self.neighbour_p) * self.neighbour_k
        tp_w = np.bincount(self.trace_fold, weights=credit.max(axis=1, initial=0.0), minlength=self.n_folds)
        fn_w = self.n_trace - tp_w
        dti = tp_w / (tp_w + ALPHA * fp_w + BETA * fn_w + EPSILON)
        return float(dti.mean())
