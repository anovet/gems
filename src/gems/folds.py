"""Spatial folds (ADR 0001): square blocks of the study area held out together.

A fold's *region* is the study-area pixels of its blocks. When training for a fold, pixels
within the distance tolerance of that region are dropped, so fault traces crossing a block
edge don't leak into training.
"""

import numpy as np
from scipy import ndimage

from gems.dti import DISTANCE_TOLERANCE_M, PIXEL_SIZE_M, RegionScorer

NO_FOLD = -1


def assign_spatial_folds(
    study_area: np.ndarray,
    *,
    block_size: float,
    n_folds: int,
    seed: int,
    pixel_size: float = PIXEL_SIZE_M,
) -> np.ndarray:
    """Spatial fold of every pixel (0..n_folds-1), NO_FOLD outside the study area.

    The grid is cut into square blocks of `block_size` metres. Blocks containing study area are
    shuffled with `seed` and dealt to folds in turn, so fold sizes differ by at most one block.
    """
    block_px = max(1, round(block_size / pixel_size))
    n_block_cols = -(-study_area.shape[1] // block_px)
    r, c = np.indices(study_area.shape)
    block_ids = (r // block_px) * n_block_cols + (c // block_px)

    blocks = np.unique(block_ids[study_area])
    if len(blocks) < n_folds:
        raise ValueError(
            f"Only {len(blocks)} blocks of {block_size} m contain study area; "
            f"need at least n_folds={n_folds}. Use a smaller block size."
        )
    shuffled = np.random.default_rng(seed).permutation(blocks)
    fold_of_block = np.full(block_ids.max() + 1, NO_FOLD, dtype=np.int16)
    fold_of_block[shuffled] = np.arange(len(shuffled)) % n_folds

    folds = fold_of_block[block_ids]
    folds[~study_area] = NO_FOLD
    return folds


def training_region(
    folds: np.ndarray,
    held_out: int,
    *,
    distance_tolerance: float = DISTANCE_TOLERANCE_M,
    pixel_size: float = PIXEL_SIZE_M,
) -> np.ndarray:
    """Study-area pixels usable for training when `held_out` is the held-out spatial fold.

    Excludes the held-out region and every pixel within `distance_tolerance` (Euclidean,
    between pixel centres) of it.
    """
    region = folds == held_out
    radius = int(distance_tolerance // pixel_size)
    dy, dx = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    disk = (dy**2 + dx**2) * pixel_size**2 <= distance_tolerance**2
    near = ndimage.binary_dilation(region, structure=disk)
    return (folds != NO_FOLD) & ~near


def local_scores(probability: np.ndarray, known_faults: np.ndarray, folds: np.ndarray) -> list[float]:
    """The DTI of each spatial fold, scored on that fold's region against its known faults."""
    return [score.dti for score in RegionScorer(known_faults, folds).score(probability)]
