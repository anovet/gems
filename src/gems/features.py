"""Features: turn feature bands into the model's inputs.

V1 used the raw feature bands only (ADR 0003). V2 adds multi-scale neighbourhood features per
band, chosen in the config's [features] section; every scale is a length in metres:

- `anomaly`: the value minus its focal mean over a square window of about that width.
- `std`: the focal standard deviation over the same window (texture, roughness).
- `gradient`: gradient magnitude after Gaussian smoothing (edges, steps across a fault).
- `ridge`: the dominant Hessian eigenvalue, signed (linear ridges < 0, linear troughs > 0).
- `coherence`: structure-tensor coherence in [0, 1], 1 where the gradients share one
  direction over that window (a lineament).

Every filter is NaN-aware (normalized convolution): missing pixels are ignored, and a pixel
gets a value from its valid neighbours even if its own bands are missing. The features are
kept only for the pixels in a `domain` mask (the study area), which is all the model ever
sees, so the stack stays about 2.4 times smaller than full grids.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import ndimage

FEATURE_KINDS = ("anomaly", "std", "gradient", "ridge", "coherence")
GRADIENT_SIGMA_PX = 1.0  # pre-smoothing of the gradients a structure tensor is built from

DEFAULT_FEATURES: dict[str, Any] = {
    "raw": True,
    "neighbourhood_bands": None,  # None = every band read; else a subset of band names
    **{f"{kind}_scales": [] for kind in FEATURE_KINDS},  # metres; empty = feature kind off
}


@dataclass
class FeatureStack:
    """Feature values of the pixels in `domain`, as an (n_features, n_domain_pixels) matrix."""

    values: np.ndarray
    names: list[str]
    domain: np.ndarray  # bool grid

    def __post_init__(self) -> None:
        self._index = np.full(self.domain.size, -1, dtype=np.int64)
        self._index[np.flatnonzero(self.domain)] = np.arange(int(self.domain.sum()))

    @property
    def n_features(self) -> int:
        return self.values.shape[0]

    def positions(self, mask: np.ndarray) -> np.ndarray:
        """Columns of `values` of the pixels in `mask`, in flat grid order. `mask` ⊆ domain."""
        if mask.shape != self.domain.shape:
            raise ValueError(f"Mask shape {mask.shape} does not match the grid {self.domain.shape}")
        if (mask & ~self.domain).any():
            raise ValueError("Mask has pixels outside the feature domain (the study area)")
        return self._index[np.flatnonzero(mask)]

    def rows(self, mask: np.ndarray) -> np.ndarray:
        """The feature vectors of the pixels in `mask`, as an (n_pixels, n_features) matrix."""
        return self.values[:, self.positions(mask)].T


def compute_features(
    bands: np.ndarray,
    band_names: list[str],
    domain: np.ndarray,
    *,
    pixel_size: float,
    settings: dict[str, Any] | None = None,
) -> FeatureStack:
    """Raw and neighbourhood features of every band, for the pixels in `domain`."""
    settings = {**DEFAULT_FEATURES, **(settings or {})}
    unknown = set(settings) - set(DEFAULT_FEATURES)
    if unknown:
        raise ValueError(f"Unknown [features] settings: {sorted(unknown)}")
    neighbourhood = settings["neighbourhood_bands"]
    neighbourhood = list(band_names) if neighbourhood is None else list(neighbourhood)
    missing = [n for n in neighbourhood if n not in band_names]
    if missing:
        raise ValueError(f"neighbourhood_bands not among the bands read: {missing}")

    plan = feature_names(band_names, settings)
    values = np.empty((len(plan), int(domain.sum())), dtype=np.float32)
    i = 0
    for b, name in enumerate(band_names):
        x = bands[b].astype(np.float64)
        if settings["raw"]:
            values[i] = x[domain]
            i += 1
        if name not in neighbourhood:
            continue
        for kind in FEATURE_KINDS:
            for scale in settings[f"{kind}_scales"]:
                values[i] = _FILTERS[kind](x, scale / pixel_size)[domain]
                i += 1
    assert i == len(plan)
    return FeatureStack(values, plan, domain)


def feature_names(band_names: list[str], settings: dict[str, Any]) -> list[str]:
    """Names of the features `compute_features` makes, in stack order, e.g. `rtp:anomaly@1000`."""
    settings = {**DEFAULT_FEATURES, **settings}
    neighbourhood = settings["neighbourhood_bands"]
    neighbourhood = band_names if neighbourhood is None else neighbourhood
    names = []
    for name in band_names:
        if settings["raw"]:
            names.append(name)
        if name in neighbourhood:
            names.extend(
                f"{name}:{kind}@{scale:g}" for kind in FEATURE_KINDS for scale in settings[f"{kind}_scales"]
            )
    if not names:
        raise ValueError("The [features] settings select no features")
    return names


# --- Filters. Each takes a float64 grid with NaN for missing values and a scale in pixels. ---


def window_size(scale_px: float) -> int:
    """Odd square window about `scale_px` wide, at least 3: 3 px -> 3, 10 px -> 11, 30 px -> 31."""
    return max(3, round(scale_px) | 1)


def gaussian_sigma(scale_px: float) -> float:
    """Gaussian sigma for a feature `scale_px` wide: a quarter of the width."""
    return scale_px / 4


def nan_filter(x: np.ndarray, filt: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """Normalized convolution: `filt` over the valid pixels only, NaN where none are in reach."""
    valid = np.isfinite(x)
    num = filt(np.where(valid, x, 0.0))
    den = filt(valid.astype(np.float64))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 1e-6, num / den, np.nan)


def focal_mean(x: np.ndarray, scale_px: float) -> np.ndarray:
    size = window_size(scale_px)
    return nan_filter(x, lambda a: ndimage.uniform_filter(a, size, mode="constant"))


def smooth(x: np.ndarray, sigma_px: float) -> np.ndarray:
    return nan_filter(x, lambda a: ndimage.gaussian_filter(a, sigma_px, mode="constant", truncate=3.0))


def anomaly(x: np.ndarray, scale_px: float) -> np.ndarray:
    return x - focal_mean(x, scale_px)


def focal_std(x: np.ndarray, scale_px: float) -> np.ndarray:
    # Centre first: E[x^2] - E[x]^2 loses precision on bands with large offsets.
    centred = x - np.nanmean(x)
    mean = focal_mean(centred, scale_px)
    var = focal_mean(centred**2, scale_px) - mean**2
    return np.sqrt(np.clip(var, 0, None))


def _gradients(x: np.ndarray, sigma_px: float) -> tuple[np.ndarray, np.ndarray]:
    s = smooth(x, sigma_px)
    gy, gx = np.gradient(s)
    return gy, gx


def gradient_magnitude(x: np.ndarray, scale_px: float) -> np.ndarray:
    gy, gx = _gradients(x, gaussian_sigma(scale_px))
    return np.hypot(gx, gy)


def ridge(x: np.ndarray, scale_px: float) -> np.ndarray:
    """Signed Hessian eigenvalue of largest magnitude: < 0 along ridges, > 0 along troughs."""
    gy, gx = _gradients(x, gaussian_sigma(scale_px))
    gyy, gyx = np.gradient(gy)
    gxy, gxx = np.gradient(gx)
    hxy = (gxy + gyx) / 2
    mid = (gxx + gyy) / 2
    half_gap = np.sqrt(((gxx - gyy) / 2) ** 2 + hxy**2)
    high, low = mid + half_gap, mid - half_gap
    return np.where(np.abs(high) >= np.abs(low), high, low)


def coherence(x: np.ndarray, scale_px: float) -> np.ndarray:
    """Structure-tensor coherence in [0, 1] over a Gaussian window of the given width."""
    gy, gx = _gradients(x, GRADIENT_SIGMA_PX)
    sigma = gaussian_sigma(scale_px)
    jxx, jyy, jxy = (
        smooth(gx * gx, sigma),
        smooth(gy * gy, sigma),
        smooth(gx * gy, sigma),
    )
    trace = jxx + jyy
    gap = np.sqrt((jxx - jyy) ** 2 + 4 * jxy**2)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(trace > 0, gap / trace, 0.0).clip(0, 1) * np.where(np.isfinite(trace), 1, np.nan)


_FILTERS: dict[str, Callable[[np.ndarray, float], np.ndarray]] = {
    "anomaly": anomaly,
    "std": focal_std,
    "gradient": gradient_magnitude,
    "ridge": ridge,
    "coherence": coherence,
}
