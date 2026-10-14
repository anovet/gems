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

from gems.dti import RegionScorer

TUNING_NOTE = (
    "Post-processing was tuned by grid search on the same out-of-fold predictions it is scored "
    "on (no nested cross-validation), so the post-processed local score is mildly optimistic."
)


@dataclass(frozen=True)
class PostProcessing:
    """One member of the post-processing family; call it on a fault probability map."""

    threshold: float = 0.0
    power: float = 1.0
    scale: float = 1.0

    def __call__(self, probability: np.ndarray) -> np.ndarray:
        """Apply to fault probabilities (any shape); NaN (outside the study area) stays NaN."""
        p = np.asarray(probability)
        keep = p >= self.threshold  # False for NaN
        out = np.zeros_like(p)
        out[keep] = np.clip(self.scale * p[keep] ** self.power, 0.0, 1.0)
        out[np.isnan(p)] = np.nan
        return out

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


IDENTITY = PostProcessing()


def tune_post_processing(
    out_of_fold: np.ndarray, known_faults: np.ndarray, folds: np.ndarray, search_space: dict[str, Any]
) -> PostProcessing:
    """The post-processing from `search_space` (plus the identity) with the best mean local score.

    `search_space` has lists `thresholds`, `powers` and `scales`; every combination is tried.
    Ties go to the identity, then to the earliest combination. Candidates are scored with the
    same `RegionScorer` as `gems.folds.local_scores`, so the chosen one's local score is exactly
    the one tuning saw, and never below the identity's (the raw local score).
    """
    scorer = RegionScorer(known_faults, folds)
    p = scorer.pixel_values(out_of_fold)

    def mean_local_score(candidate: PostProcessing, at: np.ndarray) -> float:
        """`at`: the pixels at or above the candidate's threshold; all others become 0."""
        return float(np.mean([s.dti for s in scorer.score_pixels(candidate(p[at]), at=at)]))

    best, best_score = IDENTITY, mean_local_score(IDENTITY, np.flatnonzero(p >= IDENTITY.threshold))
    for threshold in search_space["thresholds"]:
        at = np.flatnonzero(p >= threshold)
        for power, scale in product(search_space["powers"], search_space["scales"]):
            candidate = PostProcessing(float(threshold), float(power), float(scale))
            score = mean_local_score(candidate, at)
            if score > best_score:
                best, best_score = candidate, score
    return best
