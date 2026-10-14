"""Features: turn feature bands into the model's inputs.

V1 uses the raw feature bands as they are (ADR 0003: per-pixel only, a known limitation).
V2 neighbourhood features replace `compute_features` and keep its contract: full-grid bands
in, a (features, rows, cols) float32 stack with NaN for missing values out.
"""

import numpy as np


def compute_features(bands: np.ndarray, band_names: list[str]) -> np.ndarray:
    """V1: the raw feature bands, missing values left as NaN."""
    return bands


def pixel_rows(features: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """The feature vectors of the pixels in `mask`, as an (n_pixels, n_features) matrix."""
    return features[:, mask].T
