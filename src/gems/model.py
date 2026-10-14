"""Model: fit on training pixels, predict a fault probability for any pixels.

V1 is scikit-learn's histogram gradient boosting on per-pixel features (native NaN handling).
The interface is grid-shaped (a features stack plus pixel masks) so that a V3 U-Net can
implement the same `fit` / `predict` pair.
"""

from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from gems.features import pixel_rows

PREDICT_CHUNK = 1_000_000  # pixels per predict call, to bound memory on the full grid


class PixelGradientBoosting:
    def __init__(self, **params: Any):
        self.params = params
        self._clf: HistGradientBoostingClassifier | None = None

    def fit(self, features: np.ndarray, known_faults: np.ndarray, train: np.ndarray) -> "PixelGradientBoosting":
        """Train on the pixels in `train`, with known faults as the positive class."""
        self._clf = HistGradientBoostingClassifier(**self.params)
        self._clf.fit(pixel_rows(features, train), known_faults[train])
        return self

    def predict(self, features: np.ndarray, where: np.ndarray) -> np.ndarray:
        """Fault probability for the pixels in `where`, as a float32 grid that is NaN elsewhere."""
        if self._clf is None:
            raise RuntimeError("fit() the model before predict()")
        out = np.full(where.shape, np.nan, dtype=np.float32)
        idx = np.flatnonzero(where)
        flat = features.reshape(features.shape[0], -1)
        for start in range(0, len(idx), PREDICT_CHUNK):
            chunk = idx[start : start + PREDICT_CHUNK]
            out.flat[chunk] = self._clf.predict_proba(flat[:, chunk].T)[:, 1]
        return out


def build_model(params: dict[str, Any]) -> PixelGradientBoosting:
    """The V1 model, with hyperparameters from the config's [model] section."""
    return PixelGradientBoosting(**params)
