"""Behaviour of the distance-weighted Tversky index (DTI) scorer on small hand-checkable grids."""

import time

import numpy as np
import pytest

from gems.dti import RegionScorer, score_dti


def grid(shape=(11, 11)):
    return np.zeros(shape, dtype=np.float32)


def whole_study_area(shape=(11, 11)):
    return np.ones(shape, dtype=bool)


def test_perfect_prediction_scores_about_one():
    fault_trace = np.zeros((11, 11), dtype=bool)
    fault_trace[5, 2:9] = True

    score = score_dti(fault_trace.astype(np.float32), fault_trace, whole_study_area())

    assert score.dti == pytest.approx(1.0, abs=1e-6)
    assert score.tp_w == pytest.approx(7.0)
    assert score.fp_w == pytest.approx(0.0)
    assert score.fn_w == pytest.approx(0.0)


def test_all_zeros_prediction_scores_zero():
    fault_trace = np.zeros((11, 11), dtype=bool)
    fault_trace[5, 2:9] = True

    score = score_dti(grid(), fault_trace, whole_study_area())

    assert score.dti == 0.0
    assert score.fn_w == pytest.approx(7.0)


@pytest.mark.parametrize(("pixels_away", "expected_credit"), [(0, 1.0), (1, 2 / 3), (2, 1 / 3), (3, 0.0)])
def test_prediction_near_a_fault_trace_earns_distance_weighted_credit(pixels_away, expected_credit):
    fault_trace = np.zeros((11, 11), dtype=bool)
    fault_trace[5, 5] = True
    probability = grid()
    probability[5, 5 + pixels_away] = 1.0

    score = score_dti(probability, fault_trace, whole_study_area())

    assert score.tp_w == pytest.approx(expected_credit, abs=1e-6)
    assert score.fn_w == pytest.approx(1 - expected_credit, abs=1e-6)
    # the same prediction counts as a false positive by the share it is not credited
    assert score.fp_w == pytest.approx(1 - expected_credit, abs=1e-6)


def test_diagonal_distances_are_euclidean_between_pixel_centres():
    fault_trace = np.zeros((11, 11), dtype=bool)
    fault_trace[5, 5] = True
    probability = grid()
    probability[6, 6] = 1.0  # sqrt(2) * 100 m away

    score = score_dti(probability, fault_trace, whole_study_area())

    assert score.tp_w == pytest.approx(1 - np.sqrt(2) / 3, abs=1e-6)


def test_fault_trace_credit_is_the_best_nearby_prediction_not_the_sum():
    fault_trace = np.zeros((11, 11), dtype=bool)
    fault_trace[5, 5] = True
    probability = grid()
    probability[5, 4] = 0.9  # 0.9 * 2/3 = 0.6
    probability[5, 6] = 0.9
    probability[5, 7] = 1.0  # 1.0 * 1/3

    score = score_dti(probability, fault_trace, whole_study_area())

    assert score.tp_w == pytest.approx(0.6, abs=1e-6)


def test_false_positive_far_from_any_fault_adds_alpha_times_p_to_denominator():
    fault_trace = np.zeros((11, 11), dtype=bool)
    fault_trace[1, 1] = True
    probability = grid()
    probability[1, 1] = 1.0
    probability[9, 9] = 0.5

    score = score_dti(probability, fault_trace, whole_study_area())

    assert score.fp_w == pytest.approx(0.5)
    assert score.dti == pytest.approx(1.0 / (1.0 + 0.2 * 0.5 + 1e-6))


def test_alpha_and_beta_weight_false_positives_and_false_negatives():
    fault_trace = np.zeros((11, 11), dtype=bool)
    fault_trace[1, 1:3] = True
    probability = grid()
    probability[1, 1] = 1.0  # hits one fault-trace pixel and is 100 m from the other
    probability[9, 9] = 1.0  # a pure false positive

    score = score_dti(probability, fault_trace, whole_study_area(), alpha=0.5, beta=0.5, epsilon=0.0)

    # TP_w = 1 + 2/3, FN_w = 1/3, FP_w = 1
    assert score.dti == pytest.approx((5 / 3) / (5 / 3 + 0.5 * 1 + 0.5 * 1 / 3), abs=1e-6)


def _scene():
    """A fault trace and a prediction straddling a study-area edge and an excluded patch."""
    fault_trace = np.zeros((11, 11), dtype=bool)
    fault_trace[2:9, 5] = True
    probability = grid()
    probability[2:9, 5] = 0.8
    probability[3, 3] = 0.4
    study_area = whole_study_area()
    study_area[:, 8:] = False
    exclusion = np.zeros((11, 11), dtype=bool)
    exclusion[7:, :] = True
    return probability, fault_trace, study_area, exclusion


def test_values_outside_the_study_area_do_not_change_the_score():
    probability, fault_trace, study_area, exclusion = _scene()
    baseline = score_dti(probability, fault_trace, study_area, exclusion)

    outside = ~study_area
    probability[outside] = np.nan  # as in a submission
    altered_trace = fault_trace.copy()
    altered_trace[outside] = True
    probability_ones = np.where(outside, 1.0, probability)

    assert score_dti(probability, fault_trace, study_area, exclusion) == baseline
    assert score_dti(probability_ones, altered_trace, study_area, exclusion) == baseline


def test_values_inside_the_exclusion_mask_do_not_change_the_score():
    probability, fault_trace, study_area, exclusion = _scene()
    baseline = score_dti(probability, fault_trace, study_area, exclusion)

    probability[exclusion] = 1.0
    fault_trace[exclusion] = ~fault_trace[exclusion]

    assert score_dti(probability, fault_trace, study_area, exclusion) == baseline


def test_excluded_fault_trace_pixels_are_not_scored():
    probability, fault_trace, study_area, _ = _scene()
    without_exclusion = score_dti(probability, fault_trace, study_area)
    exclusion = fault_trace.copy()

    score = score_dti(probability, fault_trace, study_area, exclusion)

    assert without_exclusion.tp_w > 0
    assert score.tp_w == 0.0
    assert score.fn_w == 0.0
    assert score.fp_w == pytest.approx(0.4)


def test_each_region_is_scored_on_its_own_fault_traces():
    fault_trace = np.zeros((11, 11), dtype=bool)
    fault_trace[5, 5] = True
    regions = np.zeros((11, 11), dtype=np.int16)
    regions[:, 6:] = 1  # the fault trace is in region 0, right at the border
    probability = grid()
    probability[5, 5] = 0.5  # on the fault trace
    probability[5, 6] = 1.0  # 100 m away, but across the border in region 1

    region_0, region_1 = RegionScorer(fault_trace, regions).score(probability)

    assert region_0.tp_w == pytest.approx(0.5)  # no credit from the prediction in region 1
    assert region_0.fp_w == pytest.approx(0.0)
    assert region_1.tp_w == 0.0 and region_1.fn_w == 0.0
    assert region_1.fp_w == pytest.approx(1.0)  # no fault trace in its own region
    assert region_1.dti == 0.0


def _random_scene(seed=0, shape=(40, 50)):
    rng = np.random.default_rng(seed)
    probability = rng.random(shape, dtype=np.float32)
    probability[rng.random(shape) < 0.5] = 0.0
    fault_trace = rng.random(shape) < 0.05
    regions = rng.integers(-1, 3, size=shape).astype(np.int16)  # -1: not scored
    regions[:20, :25] = 0  # contiguous blocks as well as scattered pixels
    regions[20:, 25:] = 2
    return probability, fault_trace, regions


def test_scoring_regions_together_matches_scoring_each_region_alone():
    probability, fault_trace, regions = _random_scene()

    together = RegionScorer(fault_trace, regions).score(probability)

    for k, score in enumerate(together):
        alone = score_dti(probability, fault_trace, regions == k)
        assert score.dti == pytest.approx(alone.dti, rel=1e-6)
        assert score.tp_w == pytest.approx(alone.tp_w, rel=1e-6)
        assert score.fp_w == pytest.approx(alone.fp_w, rel=1e-6)
        assert score.fn_w == pytest.approx(alone.fn_w, rel=1e-6)


def test_scoring_only_the_nonzero_pixels_gives_exactly_the_full_map_score():
    probability, fault_trace, regions = _random_scene(seed=1)
    scorer = RegionScorer(fault_trace, regions)
    values = scorer.pixel_values(probability)
    at = np.flatnonzero(values > 0.3)
    thresholded = np.where(probability > 0.3, probability, 0.0)

    assert scorer.score_pixels(values[at], at=at) == scorer.score(thresholded)


def test_scores_a_full_size_grid_in_seconds():
    rng = np.random.default_rng(0)
    shape = (3730, 3292)
    probability = rng.random(shape, dtype=np.float32)
    fault_trace = rng.random(shape) < 0.012
    study_area = rng.random(shape) < 0.9

    start = time.perf_counter()
    score = score_dti(probability, fault_trace, study_area)
    elapsed = time.perf_counter() - start

    assert 0.0 < score.dti < 1.0
    assert elapsed < 5.0
