"""Distance-agnostic conformer diversity. Distances keep their input units."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from operator import index

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "validate_distances",
    "pairwise_distances",
    "mean_pairwise",
    "rao_diversity",
    "cluster_labels",
    "n_unique",
    "unique_curve",
    "auc",
    "rarefaction",
    "CutoffCurve",
    "RarefactionCurve",
]


def validate_distances(distances: ArrayLike) -> NDArray[np.float64]:
    """Return a copy of a finite, nonnegative, symmetric, zero-diagonal matrix.

    Absolute roundoff up to 1e-12 is accepted and canonicalized. No triangle
    inequality is required. An empty ensemble has shape (0, 0).
    """
    if np.iscomplexobj(distances):
        raise ValueError("distances must be real")
    d = np.array(distances, dtype=float, copy=True)
    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distances must be a square matrix")
    if not np.isfinite(d).all() or np.any(d < -1e-12):
        raise ValueError("distances must be finite and nonnegative")
    if not np.allclose(d, d.T, rtol=0, atol=1e-12):
        raise ValueError("distances must be symmetric")
    if np.any(np.abs(d.diagonal()) > 1e-12):
        raise ValueError("distances must have a zero diagonal")
    d = np.maximum(d / 2 + d.T / 2, 0)
    np.fill_diagonal(d, 0)
    return d


def pairwise_distances[T](
    conformers: Iterable[T], distance: Callable[[T, T], float]
) -> NDArray[np.float64]:
    """Evaluate a symmetric distance once per i < j; diagonal is zero.

    Conformers can be coordinate arrays or arbitrary objects. The caller's
    distance must be symmetric and return a finite, nonnegative real scalar.
    """
    items = list(conformers)
    d = np.zeros((len(items), len(items)), dtype=float)
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            value = np.asarray(distance(items[i], items[j]))
            if value.ndim != 0 or np.iscomplexobj(value):
                raise ValueError("distance must return a real scalar")
            value = float(value)
            if not np.isfinite(value) or value < 0:
                raise ValueError("distance must return a finite nonnegative value")
            d[i, j] = d[j, i] = value
    return d


def mean_pairwise(distances: ArrayLike) -> float:
    """Mean over unordered, distinct pairs. Requires at least two conformers."""
    d = validate_distances(distances)
    n = len(d)
    if n < 2:
        raise ValueError("mean pairwise distance requires at least two conformers")
    pairs = d[np.triu_indices(n, k=1)]
    return float(np.sum(pairs / len(pairs)))


def rao_diversity(distances: ArrayLike, weights: ArrayLike | None = None) -> float:
    """Q = p.T @ D @ p for independent draws with replacement.

    Nonnegative weights are normalized; None gives equal populations. With
    equal populations Q = (N - 1) / N * mean_pairwise(D). Singleton Q is zero.
    """
    d = validate_distances(distances)
    n = len(d)
    if n == 0:
        raise ValueError("Rao diversity requires a nonempty ensemble")
    if weights is None:
        p = np.full(n, 1 / n)
    else:
        if np.iscomplexobj(weights):
            raise ValueError("weights must be real")
        p = np.array(weights, dtype=float, copy=True)
        if p.shape != (n,) or not np.isfinite(p).all() or np.any(p < 0):
            raise ValueError("weights must be a finite nonnegative vector of length N")
        if p.max() == 0:
            raise ValueError("weights must have positive total mass")
        p /= p.max()  # Avoid overflow when normalizing large weights.
        p /= p.sum()
    return float(p @ d @ p)


def _cutoff(delta: float) -> float:
    value = np.asarray(delta)
    if value.ndim != 0 or np.iscomplexobj(value):
        raise ValueError("cutoff must be a real scalar")
    delta = float(value)
    if not np.isfinite(delta) or delta < 0:
        raise ValueError("cutoff must be finite and nonnegative")
    return delta


def _labels(d: NDArray[np.float64], delta: float) -> NDArray[np.int64]:
    labels = np.full(len(d), -1, dtype=np.int64)
    group = 0
    for root in range(len(d)):
        if labels[root] != -1:
            continue
        labels[root] = group
        stack = [root]
        while stack:
            i = stack.pop()
            neighbors = np.flatnonzero((labels == -1) & (d[i] <= delta))
            labels[neighbors] = group
            stack.extend(neighbors.tolist())
        group += 1
    return labels


def cluster_labels(distances: ArrayLike, delta: float) -> NDArray[np.int64]:
    """Single-linkage components of edges d_ij <= delta.

    Labels start at zero, ordered by the smallest input index in each component.
    Chaining is intentional: a cluster need not have diameter <= delta.
    """
    return _labels(validate_distances(distances), _cutoff(delta))


def n_unique(distances: ArrayLike, delta: float) -> int:
    """Count single-linkage clusters at an inclusive cutoff; empty returns 0."""
    labels = cluster_labels(distances, delta)
    return int(labels.max() + 1) if len(labels) else 0


def _grid(values: ArrayLike, name: str) -> NDArray[np.float64]:
    if np.iscomplexobj(values):
        raise ValueError(f"{name} must be real")
    x = np.array(values, dtype=float, copy=True)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError(f"{name} must be a nonempty finite vector")
    if np.any(np.diff(x) <= 0):
        raise ValueError(f"{name} must be strictly increasing")
    return x


def auc(x: ArrayLike, y: ArrayLike, *, normalize: bool = False) -> float:
    """Trapezoidal AUC on the supplied grid; optionally divide by its span.

    This interpolates linearly between observations, including cutoff counts.
    At least two strictly increasing finite x values are required.
    """
    x = _grid(x, "x")
    if np.iscomplexobj(y):
        raise ValueError("y must be real")
    y = np.asarray(y, dtype=float)
    if len(x) < 2 or y.shape != x.shape or not np.isfinite(y).all():
        raise ValueError("AUC requires at least two finite, matched x/y values")
    area = float(np.sum(np.diff(x) * (y[:-1] / 2 + y[1:] / 2)))
    return area / float(x[-1] - x[0]) if normalize else area


@dataclass(frozen=True, slots=True)
class CutoffCurve:
    """N_unique(delta) evaluated on an explicit cutoff grid."""

    cutoffs: NDArray[np.float64]
    counts: NDArray[np.int64]

    def auc(self, *, normalize: bool = False) -> float:
        return auc(self.cutoffs, self.counts, normalize=normalize)


def unique_curve(distances: ArrayLike, cutoffs: ArrayLike) -> CutoffCurve:
    """Coverage/diversity-versus-cutoff curve using single linkage."""
    d = validate_distances(distances)
    x = _grid(cutoffs, "cutoffs")
    if np.any(x < 0):
        raise ValueError("cutoffs must be nonnegative")
    counts = []
    for delta in x:
        labels = _labels(d, float(delta))
        counts.append(int(labels.max() + 1) if len(labels) else 0)
    return CutoffCurve(x, np.asarray(counts, dtype=np.int64))


@dataclass(frozen=True, slots=True)
class RarefactionCurve:
    """Samples have shape (number of sizes, repeats); spread is not a CI."""

    sizes: NDArray[np.int64]
    samples: NDArray[np.float64]

    @property
    def mean(self) -> NDArray[np.float64]:
        return self.samples.mean(axis=1)

    @property
    def std(self) -> NDArray[np.float64]:
        """Population SD (ddof=0) across subsamples, zero for one repeat."""
        return self.samples.std(axis=1)


def rarefaction(
    distances: ArrayLike,
    sizes: Iterable[int],
    *,
    statistic: Callable[[NDArray[np.float64]], float] = mean_pairwise,
    repeats: int = 100,
    seed: int | None = None,
) -> RarefactionCurve:
    """Repeated uniform subsampling without replacement, independently per size.

    A statistic receives the submatrix in original input order. The default
    measures mean pairwise distance; lambda d: n_unique(d, delta) measures
    coverage. Every requested size must be <= the original ensemble size.
    """
    d = validate_distances(distances)
    requested = list(sizes)
    if any(isinstance(s, (bool, np.bool_)) for s in requested):
        raise ValueError("sizes must be integers, not booleans")
    try:
        values = [index(s) for s in requested]
        count = index(repeats)
    except TypeError as exc:
        raise ValueError("sizes and repeats must be integers") from exc
    if isinstance(repeats, (bool, np.bool_)) or count < 1:
        raise ValueError("repeats must be a positive integer")
    if not values or any(s < 1 or s > len(d) for s in values):
        raise ValueError("sizes must be nonempty and between 1 and N")
    if any(a >= b for a, b in zip(values, values[1:], strict=False)):
        raise ValueError("sizes must be strictly increasing")
    if statistic is mean_pairwise and values[0] < 2:
        raise ValueError("mean pairwise rarefaction requires sizes >= 2")
    rng = np.random.default_rng(seed)
    samples = np.empty((len(values), count))
    for row, size in enumerate(values):
        for repeat in range(count):
            chosen = np.sort(rng.choice(len(d), size=size, replace=False))
            value = np.asarray(statistic(d[np.ix_(chosen, chosen)]))
            if value.ndim != 0 or np.iscomplexobj(value) or not np.isfinite(value):
                raise ValueError("statistic must return a finite real scalar")
            samples[row, repeat] = float(value)
    return RarefactionCurve(np.asarray(values, dtype=np.int64), samples)
