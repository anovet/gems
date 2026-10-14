"""Spatial folds: the leakage guarantee, which the run's outputs can't show directly."""

import numpy as np
from scipy import ndimage

from conftest import COLS, ROWS, _study_area
from gems.folds import assign_spatial_folds, training_region


def test_no_training_pixel_lies_within_300_m_of_the_held_out_fold():
    study = _study_area(ROWS, COLS)
    folds = assign_spatial_folds(study, block_size=2000.0, n_folds=5, seed=3)

    for k in range(5):
        train = training_region(folds, k)
        metres_to_fold = ndimage.distance_transform_edt(folds != k) * 100.0

        assert train.any()
        assert not (train & ~study).any()
        assert metres_to_fold[train].min() > 300.0
        # Only the buffer is dropped: pixels beyond it are still used.
        assert np.array_equal(train, study & (metres_to_fold > 300.0))


def test_spatial_folds_are_balanced_by_block_count():
    study = np.ones((100, 100), dtype=bool)
    folds = assign_spatial_folds(study, block_size=1000.0, n_folds=5, seed=0)  # 100 blocks

    assert np.array_equal(np.bincount(folds.ravel()), [2000] * 5)
