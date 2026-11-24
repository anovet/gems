"""Withheld faults (ADR 0006): known faults withheld from training and scored as if new.

The leaderboard scores new faults with known faults excluded, after the model has trained on
every known fault. To imitate that locally, the known faults are cut into fault traces
(8-connected components) and the traces are dealt into groups. Traces are dealt in spatial
clusters: every trace whose centroid falls in the same square cluster block goes to the same
group, so a withheld trace usually has no given trace nearby, like most new faults.

For each group, a model is trained on the whole study area with that group's pixels labelled
as non-fault (as a new fault is in our labels), and its fault probability map is scored only
on the withheld traces, with the other known faults excluded from scoring.
"""

import numpy as np
from scipy import ndimage

from gems.dti import PIXEL_SIZE_M, RegionScorer

NO_GROUP = -1
EIGHT_CONNECTED = np.ones((3, 3), dtype=bool)


def assign_withheld_groups(
    known_faults: np.ndarray,
    *,
    cluster_size: float,
    n_groups: int,
    seed: int,
    pixel_size: float = PIXEL_SIZE_M,
) -> np.ndarray:
    """Withheld group (0..n_groups-1) of every known-fault pixel, NO_GROUP elsewhere.

    Each fault trace (8-connected component of known faults) goes to the group of the cluster
    block holding its centroid. Blocks are `cluster_size` metres square, shuffled with `seed`
    and dealt to groups in turn. A `cluster_size` of 0 deals every trace on its own.
    """
    traces, n_traces = ndimage.label(known_faults, structure=EIGHT_CONNECTED)
    if n_traces < n_groups:
        raise ValueError(f"Only {n_traces} fault traces; need at least n_groups={n_groups}.")
    if cluster_size > 0:
        centroids = np.array(ndimage.center_of_mass(known_faults, traces, range(1, n_traces + 1)))
        block_px = cluster_size / pixel_size
        block = np.floor(centroids / block_px).astype(np.int64)
        _, cluster = np.unique(block, axis=0, return_inverse=True)
        cluster = cluster.ravel()
    else:
        cluster = np.arange(n_traces)
    n_clusters = int(cluster.max()) + 1
    if n_clusters < n_groups:
        raise ValueError(
            f"Only {n_clusters} cluster blocks of {cluster_size} m hold fault traces; "
            f"need at least n_groups={n_groups}. Use a smaller cluster size."
        )
    group_of_cluster = np.empty(n_clusters, dtype=np.int16)
    group_of_cluster[np.random.default_rng(seed).permutation(n_clusters)] = np.arange(n_clusters) % n_groups
    group_of_trace = np.concatenate([[NO_GROUP], group_of_cluster[cluster]]).astype(np.int16)
    return group_of_trace[traces]


def given_faults(known_faults: np.ndarray, groups: np.ndarray, withheld: int) -> np.ndarray:
    """The known faults the model may train on when group `withheld` is withheld."""
    return known_faults & (groups != withheld)


def withheld_scorers(known_faults: np.ndarray, study_area: np.ndarray, groups: np.ndarray) -> list[RegionScorer]:
    """One DTI scorer per withheld group.

    Group k's scorer counts only group k's traces as fault traces and ignores the other known
    faults entirely, as the leaderboard ignores known faults when scoring new ones.
    """
    n_groups = int(groups.max()) + 1
    scorers = []
    for k in range(n_groups):
        withheld = groups == k
        scored = study_area & ~(known_faults & ~withheld)
        scorers.append(RegionScorer(withheld, np.where(scored, 0, -1).astype(np.int8)))
    return scorers


def withheld_fault_scores(probabilities: list[np.ndarray], scorers: list[RegionScorer]) -> list[float]:
    """The DTI of each group's fault probability map on its withheld faults."""
    return [scorer.score(p)[0].dti for p, scorer in zip(probabilities, scorers, strict=True)]


def constant_map_scores(study_area: np.ndarray, scorers: list[RegionScorer]) -> list[float]:
    """The DTI of a map of 1 over the whole study area, per group: the no-skill baseline."""
    ones = np.where(study_area, 1.0, np.nan).astype(np.float32)
    return [scorer.score(ones)[0].dti for scorer in scorers]
