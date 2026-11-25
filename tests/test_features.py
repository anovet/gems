import numpy as np
import pytest

from gems.features import (
    anomaly,
    coherence,
    compute_features,
    feature_names,
    focal_mean,
    focal_std,
    gradient_magnitude,
    ridge,
    window_size,
)

ALL_KINDS = {
    "anomaly_scales": [300.0, 1000.0],
    "std_scales": [1000.0],
    "gradient_scales": [300.0],
    "ridge_scales": [1000.0],
    "coherence_scales": [1000.0],
}


def test_window_sizes_round_to_odd_widths():
    assert [window_size(s) for s in (1, 3, 10, 30)] == [3, 3, 11, 31]


def test_focal_mean_ignores_missing_pixels():
    x = np.full((9, 9), 2.0)
    x[4, 4] = np.nan
    x[0, :] = np.nan

    mean = focal_mean(x, 3)

    assert mean[4, 4] == pytest.approx(2.0)  # a missing pixel gets its neighbours' mean
    assert mean[0, 4] == pytest.approx(2.0)
    assert np.isfinite(mean).all()


def test_filters_are_nan_where_no_valid_pixel_is_in_reach():
    x = np.full((20, 20), np.nan)
    x[:, :5] = 1.0

    assert np.isnan(focal_mean(x, 3)[:, 10:]).all()
    assert np.isnan(gradient_magnitude(x, 3)[:, 15:]).all()


def test_anomaly_and_std_of_a_constant_field_are_zero():
    x = np.full((15, 15), 5e6)  # earthquake bands reach about 5e6

    assert np.allclose(anomaly(x, 3), 0)
    assert np.allclose(focal_std(x, 10), 0)


def test_std_matches_numpy_inside_the_grid():
    x = np.random.default_rng(0).normal(1e6, 3.0, size=(21, 21))

    assert focal_std(x, 3)[10, 10] == pytest.approx(x[9:12, 9:12].std(), rel=1e-6)


def test_a_step_has_its_largest_gradient_on_the_step():
    x = np.zeros((30, 30))
    x[:, 15:] = 1.0

    grad = gradient_magnitude(x, 3)

    assert set(np.argmax(grad, axis=1)) <= {14, 15}


def test_ridge_is_negative_on_a_ridge_and_positive_in_a_trough():
    x = np.zeros((30, 30))
    x[:, 15] = 1.0

    assert ridge(x, 4)[15, 15] < 0
    assert ridge(-x, 4)[15, 15] > 0


def test_coherence_is_high_along_a_line_and_low_in_noise():
    rng = np.random.default_rng(0)
    line = np.zeros((40, 40))
    line[:, 20] = 1.0
    noise = rng.normal(size=(40, 40))

    assert coherence(line, 10)[20, 20] > 0.9
    assert coherence(noise, 10)[10:30, 10:30].mean() < 0.5
    assert 0 <= np.nanmin(coherence(noise, 10)) and np.nanmax(coherence(noise, 10)) <= 1


def test_feature_names_follow_stack_order():
    names = feature_names(["a", "b"], {**ALL_KINDS, "neighbourhood_bands": ["b"]})

    assert names == [
        "a",
        "b",
        "b:anomaly@300",
        "b:anomaly@1000",
        "b:std@1000",
        "b:gradient@300",
        "b:ridge@1000",
        "b:coherence@1000",
    ]


def test_compute_features_keeps_domain_pixels_only():
    rng = np.random.default_rng(0)
    bands = rng.normal(size=(2, 12, 14)).astype(np.float32)
    domain = np.zeros((12, 14), dtype=bool)
    domain[2:10, 3:12] = True

    stack = compute_features(bands, ["a", "b"], domain, pixel_size=100.0, settings=ALL_KINDS)

    assert stack.values.shape == (len(stack.names), domain.sum())
    assert stack.values.dtype == np.float32
    assert np.array_equal(stack.rows(domain)[:, 0], bands[0][domain])
    one = np.zeros_like(domain)
    one[5, 7] = True
    assert np.array_equal(stack.rows(one)[0], stack.values[:, stack.positions(one)[0]])


def test_default_settings_reproduce_the_raw_bands():
    bands = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    domain = np.ones((3, 4), dtype=bool)

    stack = compute_features(bands, ["a", "b"], domain, pixel_size=100.0)

    assert stack.names == ["a", "b"]
    assert np.array_equal(stack.rows(domain), bands.reshape(2, -1).T)


def test_rows_outside_the_domain_are_an_error():
    domain = np.zeros((4, 4), dtype=bool)
    domain[1:3, 1:3] = True
    stack = compute_features(np.ones((1, 4, 4), np.float32), ["a"], domain, pixel_size=100.0)

    with pytest.raises(ValueError, match="outside"):
        stack.rows(np.ones((4, 4), dtype=bool))


def test_bad_settings_are_errors():
    bands, domain = np.ones((1, 4, 4), np.float32), np.ones((4, 4), dtype=bool)

    with pytest.raises(ValueError, match="Unknown"):
        compute_features(bands, ["a"], domain, pixel_size=100.0, settings={"slope_scales": [1.0]})
    with pytest.raises(ValueError, match="neighbourhood_bands"):
        compute_features(
            bands,
            ["a"],
            domain,
            pixel_size=100.0,
            settings={"neighbourhood_bands": ["z"]},
        )
    with pytest.raises(ValueError, match="no features"):
        compute_features(bands, ["a"], domain, pixel_size=100.0, settings={"raw": False})
