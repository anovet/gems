"""Training-pixel sampling: keep every known-fault pixel, subsample the rest."""

import numpy as np


def sample_training_pixels(
    known_faults: np.ndarray, eligible: np.ndarray, *, negative_ratio: float, rng: np.random.Generator
) -> np.ndarray:
    """Boolean mask of training pixels drawn from `eligible`.

    All eligible known-fault pixels are kept; non-fault pixels are sampled without replacement,
    `negative_ratio` per fault pixel (or all of them, if there are fewer).
    """
    positives = eligible & known_faults
    negative_idx = np.flatnonzero(eligible & ~known_faults)
    n_neg = min(len(negative_idx), int(round(negative_ratio * positives.sum())))
    chosen = rng.choice(negative_idx, size=n_neg, replace=False)
    mask = positives.copy()
    mask.flat[chosen] = True
    return mask
