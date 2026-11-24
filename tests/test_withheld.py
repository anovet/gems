import numpy as np
import pytest
from conftest import write_dataset
from scipy import ndimage

from gems.dti import score_dti
from gems.withheld import (
    NO_GROUP,
    assign_withheld_groups,
    constant_map_scores,
    given_faults,
    withheld_fault_scores,
    withheld_scorers,
)


def _separate_traces(n: int, shape=(40, 200), gap: int = 10) -> np.ndarray:
    """`n` short horizontal fault traces in a row, `gap` pixels apart."""
    known = np.zeros(shape, dtype=bool)
    for i in range(n):
        known[20, i * gap : i * gap + 5] = True
    return known


def test_every_known_fault_pixel_gets_a_group_and_nothing_else_does():
    known = _separate_traces(12)

    groups = assign_withheld_groups(known, cluster_size=0.0, n_groups=4, seed=0)

    assert set(np.unique(groups[known]).tolist()) == {0, 1, 2, 3}
    assert np.all(groups[~known] == NO_GROUP)


def test_a_fault_trace_is_withheld_whole():
    known = _separate_traces(12)
    known[21, 4] = True  # touches its trace diagonally: still the same 8-connected trace

    groups = assign_withheld_groups(known, cluster_size=0.0, n_groups=4, seed=0)

    traces, n = ndimage.label(known, structure=np.ones((3, 3)))
    assert n == 12
    for t in range(1, n + 1):
        assert len(np.unique(groups[traces == t])) == 1


def test_traces_in_the_same_cluster_block_are_withheld_together():
    known = _separate_traces(20, shape=(40, 200), gap=10)  # trace centroids 10 px = 1 km apart

    groups = assign_withheld_groups(known, cluster_size=5000.0, n_groups=2, seed=0)  # 50 px blocks

    trace_groups = [int(groups[20, i * 10]) for i in range(20)]
    for block in range(4):  # traces 0-4 have centroids in block 0, 5-9 in block 1, ...
        assert len(set(trace_groups[block * 5 : block * 5 + 5])) == 1
    assert set(trace_groups) == {0, 1}


def test_the_seed_changes_the_groups_and_fixes_them():
    known = _separate_traces(12)

    a = assign_withheld_groups(known, cluster_size=0.0, n_groups=3, seed=0)
    b = assign_withheld_groups(known, cluster_size=0.0, n_groups=3, seed=0)
    c = assign_withheld_groups(known, cluster_size=0.0, n_groups=3, seed=1)

    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_too_few_traces_or_cluster_blocks_is_an_error():
    known = _separate_traces(3)

    with pytest.raises(ValueError, match="fault traces"):
        assign_withheld_groups(known, cluster_size=0.0, n_groups=5, seed=0)
    with pytest.raises(ValueError, match="cluster size"):
        assign_withheld_groups(_separate_traces(12), cluster_size=100_000.0, n_groups=2, seed=0)


def test_given_faults_are_the_known_faults_outside_the_withheld_group():
    known = _separate_traces(12)
    groups = assign_withheld_groups(known, cluster_size=0.0, n_groups=3, seed=0)

    given = given_faults(known, groups, 1)

    assert np.array_equal(given, known & (groups != 1))
    assert given.sum() < known.sum()


def _withheld_setup():
    known = _separate_traces(12)
    study = np.ones(known.shape, dtype=bool)
    groups = assign_withheld_groups(known, cluster_size=0.0, n_groups=3, seed=0)
    return known, study, groups, withheld_scorers(known, study, groups)


def test_predicting_exactly_the_withheld_faults_scores_about_one():
    _, _, groups, scorers = _withheld_setup()

    maps = [(groups == k).astype(np.float32) for k in range(3)]

    assert withheld_fault_scores(maps, scorers) == pytest.approx([1.0] * 3, abs=1e-6)


def test_predictions_on_given_faults_are_neither_rewarded_nor_penalised():
    known, _, groups, scorers = _withheld_setup()

    for k, scorer in enumerate(scorers):
        on_given = given_faults(known, groups, k).astype(np.float32)
        score = scorer.score(on_given)[0]
        assert (score.tp_w, score.fp_w) == (0.0, 0.0)


def test_constant_map_scores_match_scoring_a_map_of_ones_with_given_faults_excluded():
    known, study, groups, scorers = _withheld_setup()

    expected = [
        score_dti(np.ones(known.shape, np.float32), groups == k, study, exclusion=given_faults(known, groups, k)).dti
        for k in range(3)
    ]

    assert constant_map_scores(study, scorers) == pytest.approx(expected)


def test_the_synthetic_dataset_has_enough_separate_fault_traces(tmp_path):
    data = write_dataset(tmp_path)

    groups = assign_withheld_groups(data.known_faults, cluster_size=1000.0, n_groups=3, seed=0)

    assert set(np.unique(groups[data.known_faults]).tolist()) == {0, 1, 2}
