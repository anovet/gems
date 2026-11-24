"""Post-processing: map raw model probabilities to submitted values, tuned on the withheld-fault score.

The DTI makes false alarms cheap (alpha = 0.2) and misses expensive (beta = 0.8), so raw
probabilities are rarely the best values to submit. The post-processing family is

    value = clip(scale * p ** power, 0, 1)   where p >= threshold, else 0

and the identity (threshold 0, power 1, scale 1) is always a candidate, so tuning can never
lower the score it is tuned on. Tuning is a grid search over a set of scored maps (ADR 0006:
each withheld group's map, scored on its withheld faults). It is mildly optimistic, since the
same maps choose and score the post-processing (no nested cross-validation).
"""

from dataclasses import asdict, dataclass
from itertools import product
from typing import Any

import numpy as np

from gems.dti import RegionScorer

TUNING_NOTE = (
    "Post-processing was tuned by grid search on the same withheld-fault maps it is scored on "
    "(no nested cross-validation), so the post-processed withheld-fault score is mildly optimistic."
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
    scored_maps: list[tuple[RegionScorer, np.ndarray]], search_space: dict[str, Any]
) -> PostProcessing:
    """The post-processing from `search_space` (plus the identity) with the best mean DTI.

    `scored_maps` pairs each fault probability map with the scorer that scores it; the mean is
    taken over every region of every scorer. `search_space` has lists `thresholds`, `powers`
    and `scales`; every combination is tried. Ties go to the identity, then to the earliest
    combination. Scoring a candidate the same way afterwards gives exactly the score tuning
    saw, and never less than the identity's (the raw score).
    """
    pixels = [(scorer, scorer.pixel_values(probability)) for scorer, probability in scored_maps]

    def at_or_above(threshold: float) -> list[np.ndarray]:
        """Per map, the pixels at or above `threshold`; all others become 0."""
        return [np.flatnonzero(p >= threshold) for _, p in pixels]

    def mean_score(candidate: PostProcessing, ats: list[np.ndarray]) -> float:
        dti = []
        for (scorer, p), at in zip(pixels, ats):
            dti += [s.dti for s in scorer.score_pixels(candidate(p[at]), at=at)]
        return float(np.mean(dti))

    best, best_score = IDENTITY, mean_score(IDENTITY, at_or_above(IDENTITY.threshold))
    for threshold in search_space["thresholds"]:
        ats = at_or_above(threshold)
        for power, scale in product(search_space["powers"], search_space["scales"]):
            candidate = PostProcessing(float(threshold), float(power), float(scale))
            score = mean_score(candidate, ats)
            if score > best_score:
                best, best_score = candidate, score
    return best
