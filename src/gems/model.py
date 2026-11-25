"""Model: fit on training pixels, predict a fault probability for any pixels.

Scikit-learn's histogram gradient boosting on each pixel's feature vector (native NaN
handling); V2's neighbourhood features bring in the spatial context. The interface is
grid-shaped (a feature stack plus pixel masks) so that a V3 U-Net can implement the same
`fit` / `predict` pair.
"""

from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from gems.features import FeatureStack

PREDICT_CHUNK = 1_000_000  # pixels per predict call, to bound memory on the full grid


class PixelGradientBoosting:
    def __init__(self, **params: Any):
        self.params = params
        self._clf: HistGradientBoostingClassifier | None = None

    def fit(self, features: FeatureStack, known_faults: np.ndarray, train: np.ndarray) -> PixelGradientBoosting:
        """Train on the pixels in `train`, with known faults as the positive class."""
        self._clf = HistGradientBoostingClassifier(**self.params)
        self._clf.fit(features.rows(train), known_faults[train])
        return self

    def predict(self, features: FeatureStack, where: np.ndarray) -> np.ndarray:
        """Fault probability for the pixels in `where`, as a float32 grid that is NaN elsewhere."""
        if self._clf is None:
            raise RuntimeError("fit() the model before predict()")
        out = np.full(where.shape, np.nan, dtype=np.float32)
        idx, cols = np.flatnonzero(where), features.positions(where)
        for start in range(0, len(idx), PREDICT_CHUNK):
            chunk = slice(start, start + PREDICT_CHUNK)
            out.flat[idx[chunk]] = self._clf.predict_proba(features.values[:, cols[chunk]].T)[:, 1]
        return out


def build_model(params: dict[str, Any]) -> PixelGradientBoosting:
    """The gradient-boosted tree model, with hyperparameters from the config's [model] section."""
    return PixelGradientBoosting(**params)
