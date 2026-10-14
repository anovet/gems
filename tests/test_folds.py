"""Spatial folds: the leakage guarantee, which the run's outputs can't show directly.

A deliberate exception to spec #1's two-seam rule (runner and DTI scorer only): ticket #4's
criterion "no training pixel within 300 m of the held-out fold" is not observable through the
runner's outputs, so it is checked here on `gems.folds` directly.
"""

import numpy as np
from conftest import study_area_mask
from scipy import ndimage

from gems.folds import assign_spatial_folds, training_region


def test_no_training_pixel_lies_within_300_m_of_the_held_out_fold():
    study = study_area_mask()
    folds = assign_spatial_folds(study, block_size=2000.0, n_folds=5, seed=3)

    for k in range(5):
        train = training_region(folds, k)
        metres_to_fold = ndimage.distance_transform_edt(folds != k) * 100.0

        assert train.any()
        assert not (train & ~study).any()
        assert metres_to_fold[train].min() > 300.0
        # Only pixels within the distance tolerance are dropped: those beyond it are still used.
        assert np.array_equal(train, study & (metres_to_fold > 300.0))


def test_spatial_folds_are_balanced_by_block_count():
    study = np.ones((100, 100), dtype=bool)
    folds = assign_spatial_folds(study, block_size=1000.0, n_folds=5, seed=0)  # 100 blocks

    assert np.array_equal(np.bincount(folds.ravel()), [2000] * 5)
