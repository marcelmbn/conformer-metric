from itertools import combinations, permutations

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from conformer_metric import (
    auc,
    cluster_labels,
    mean_pairwise,
    n_unique,
    pairwise_distances,
    rao_diversity,
    rarefaction,
    unique_curve,
    validate_distances,
)


@pytest.fixture
def distances():
    # Six-pair example from the original discussion: total 7.4 angstroms.
    return np.array(
        [[0, 0.2, 1.1, 2.0], [0.2, 0, 1.0, 1.9], [1.1, 1.0, 0, 1.2], [2.0, 1.9, 1.2, 0]]
    )


def test_known_mean(distances):
    assert mean_pairwise(distances) == pytest.approx(7.4 / 6)


def test_pairwise_coordinates_and_call_count():
    coords = [np.zeros((2, 3)), np.ones((2, 3)), np.full((2, 3), 2.0)]
    calls = []

    def toy_distance(a, b):
        calls.append((a, b))
        return float(np.linalg.norm(a - b))

    result = pairwise_distances(iter(coords), toy_distance)
    assert len(calls) == 3
    assert_allclose(result, np.sqrt(6) * np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]]))


def test_arbitrary_conformer_objects():
    assert_array_equal(
        pairwise_distances(["a", "abc"], lambda a, b: float(abs(len(a) - len(b)))),
        [[0, 2], [2, 0]],
    )


@pytest.mark.parametrize("value", [-1, np.nan, np.inf, 1j, [1]])
def test_bad_callback(value):
    with pytest.raises(ValueError):
        pairwise_distances([0, 1], lambda a, b: value)


@pytest.mark.parametrize(
    "matrix",
    [
        [],
        [0, 1],
        [[0, 1, 2], [1, 0, 3]],
        [[0, 1], [2, 0]],
        [[0, -1], [-1, 0]],
        [[1, 0], [0, 0]],
        [[0, np.inf], [np.inf, 0]],
        [[0, np.nan], [np.nan, 0]],
        [[0, 1j], [1j, 0]],
    ],
)
def test_bad_matrix(matrix):
    with pytest.raises(ValueError):
        validate_distances(matrix)


def test_validation_canonicalizes_roundoff_without_mutation():
    d = np.array([[1e-13, -1e-13, 2], [-1e-13, 0, 1], [2 + 1e-13, 1, 0]])
    original = d.copy()
    result = validate_distances(d)
    assert_array_equal(d, original)
    assert_array_equal(result, result.T)
    assert_array_equal(result.diagonal(), [0, 0, 0])
    assert result.min() == 0
    result[0, 2] = 99
    assert_array_equal(d, original)


def test_no_triangle_inequality_requirement():
    assert mean_pairwise([[0, 1, 10], [1, 0, 1], [10, 1, 0]]) == 4


def test_empty_and_singleton():
    empty = pairwise_distances([], lambda a, b: 0)
    assert empty.shape == (0, 0)
    assert n_unique(empty, 0) == 0
    assert_array_equal(cluster_labels(empty, 0), [])
    assert_array_equal(unique_curve(empty, [0, 1]).counts, [0, 0])
    assert rao_diversity([[0]]) == 0
    assert n_unique([[0]], 0) == 1
    for d in (empty, [[0]]):
        with pytest.raises(ValueError):
            mean_pairwise(d)
    with pytest.raises(ValueError):
        rao_diversity(empty)


def test_rao_known_populations():
    d = [[0, 0.2, 1.1], [0.2, 0, 1.0], [1.1, 1.0, 0]]
    w = np.array([98.0, 1.0, 1.0])
    assert rao_diversity(d, w) == pytest.approx(0.02568)
    assert rao_diversity(d, [0.98, 0.01, 0.01]) == pytest.approx(0.02568)
    assert_array_equal(w, [98, 1, 1])
    assert rao_diversity(d, [1, 0, 0]) == 0


def test_rao_equal_weights_finite_size_factor(distances):
    expected = 3 / 4 * mean_pairwise(distances)
    assert rao_diversity(distances) == pytest.approx(expected)
    assert rao_diversity(distances, [1] * 4) == pytest.approx(expected)
    assert rao_diversity(distances, [1e308] * 4) == pytest.approx(expected)


@pytest.mark.parametrize(
    "weights", [[0, 0], [-1, 2], [np.nan, 1], [np.inf, 1], [1], [[1, 1]], [1j, 1]]
)
def test_bad_weights(weights):
    with pytest.raises(ValueError):
        rao_diversity([[0, 2], [2, 0]], weights)


def test_zero_duplicates_and_inclusive_threshold():
    d = [[0, 0, 2], [0, 0, 2], [2, 2, 0]]
    assert mean_pairwise(d) == pytest.approx(4 / 3)
    assert_array_equal(cluster_labels(d, 0), [0, 0, 1])
    assert n_unique(d, 2) == 1


def test_single_linkage_chaining_and_input_order():
    d = np.array([[0, 0.4, 0.8, 2], [0.4, 0, 0.4, 2], [0.8, 0.4, 0, 2], [2, 2, 2, 0]])
    original = cluster_labels(d, 0.5)
    assert_array_equal(original, [0, 0, 0, 1])
    for perm in permutations(range(4)):
        labels = cluster_labels(d[np.ix_(perm, perm)], 0.5)
        expected = original[list(perm)]
        assert_array_equal(labels[:, None] == labels, expected[:, None] == expected)


@pytest.mark.parametrize("delta", [-0.1, np.nan, np.inf, 1j, [1]])
def test_bad_cutoffs(delta):
    with pytest.raises(ValueError):
        n_unique([[0]], delta)


def test_cutoff_curve(distances):
    curve = unique_curve(distances, [0, 0.2, 1.0, 1.2, 2])
    assert_array_equal(curve.counts, [4, 3, 2, 1, 1])
    assert curve.auc() == pytest.approx(3.8)
    assert curve.auc(normalize=True) == pytest.approx(1.9)


@pytest.mark.parametrize("grid", [[], [[0, 1]], [-1, 0], [0, 0], [1, 0], [0, np.nan], [0, 1j]])
def test_bad_cutoff_grid(grid):
    with pytest.raises(ValueError):
        unique_curve([[0]], grid)


def test_auc_irregular_grid():
    assert auc([1, 2, 4], [2, 2, 0]) == 4
    assert auc([1, 2, 4], [2, 2, 0], normalize=True) == pytest.approx(4 / 3)


@pytest.mark.parametrize(
    ("x", "y"),
    [
        ([0], [1]),
        ([0, 0], [1, 1]),
        ([1, 0], [1, 1]),
        ([0, 1], [1]),
        ([0, 1], [0, np.inf]),
        ([0, 1], [0, 1j]),
    ],
)
def test_bad_auc(x, y):
    with pytest.raises(ValueError):
        auc(x, y)


def test_rarefaction_reproducibility_and_full_ensemble(distances):
    first = rarefaction(distances, [2, 3, 4], repeats=50, seed=123)
    second = rarefaction(distances, [2, 3, 4], repeats=50, seed=123)
    assert_array_equal(first.samples, second.samples)
    assert first.samples.shape == (3, 50)
    assert_allclose(first.samples[-1], mean_pairwise(distances))
    assert_allclose(first.mean, first.samples.mean(axis=1))
    assert_allclose(first.std, first.samples.std(axis=1))
    # Every size-two draw must be one of the six actual pair distances.
    assert set(first.samples[0]) <= set(distances[np.triu_indices(4, 1)])


def test_rarefaction_custom_coverage_statistic(distances):
    result = rarefaction(
        distances, [1, 2, 4], statistic=lambda d: float(n_unique(d, 0.2)), repeats=1, seed=0
    )
    assert result.samples[0, 0] == 1
    assert result.samples[-1, 0] == 3
    assert_array_equal(result.std, [0, 0, 0])


def test_rarefaction_subsets_have_original_order(distances):
    # An order-sensitive statistic with distinct pair values; full size preserves input.
    result = rarefaction(distances, [4], statistic=lambda d: float(d[0, 1]), repeats=8)
    assert_array_equal(result.samples, np.full((1, 8), 0.2))


def test_rarefaction_does_not_modify_global_rng(distances):
    np.random.seed(123)
    expected = np.random.random()
    np.random.seed(123)
    rarefaction(distances, [2], repeats=2, seed=42)
    assert np.random.random() == expected


@pytest.mark.parametrize("sizes", [[], [0], [5], [2.5], [2, 2], [3, 2], [True], [1]])
def test_bad_rarefaction_sizes(distances, sizes):
    with pytest.raises(ValueError):
        rarefaction(distances, sizes)


@pytest.mark.parametrize("repeats", [0, -1, 1.5, True])
def test_bad_repeats(distances, repeats):
    with pytest.raises(ValueError):
        rarefaction(distances, [2], repeats=repeats)


@pytest.mark.parametrize("value", [np.nan, np.inf, 1j, [1]])
def test_bad_subsample_statistic(distances, value):
    with pytest.raises(ValueError):
        rarefaction(distances, [2], statistic=lambda d: value, repeats=1)


def test_exact_subset_mean_is_size_independent(distances):
    # Exhaustive oracle: no stochastic tolerance or implementation-derived expected value.
    for size in (2, 3, 4):
        scores = [
            mean_pairwise(distances[np.ix_(subset, subset)])
            for subset in combinations(range(4), size)
        ]
        assert np.mean(scores) == pytest.approx(7.4 / 6)


def test_cutoff_monotonicity_for_random_dissimilarities():
    rng = np.random.default_rng(7)
    for _ in range(10):
        a = rng.random((10, 10))
        d = (a + a.T) / 2
        np.fill_diagonal(d, 0)
        curve = unique_curve(d, np.linspace(0, 1, 21))
        assert curve.counts[0] == 10
        assert curve.counts[-1] == 1
        assert np.all(np.diff(curve.counts) <= 0)


def test_public_metrics_preserve_input(distances):
    original = distances.copy()
    mean_pairwise(distances)
    rao_diversity(distances)
    n_unique(distances, 0.2)
    unique_curve(distances, [0, 0.2])
    rarefaction(distances, [2, 4], repeats=3)
    assert_array_equal(distances, original)
