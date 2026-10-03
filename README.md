# conformer-metric

A lean, typed Python library for **ensemble diversity**, with NumPy and upstream
iRMSD as required runtime dependencies. Molecular distances use iRMSD by default.
Python 3.12+, `uv`, Hatchling, pytest, Ruff, and ty.

## Scientific definitions

The primary score is the mean distance between distinct conformer indices:

\[
\bar d = \frac{2}{N(N-1)}\sum_{i<j}d_{ij}.
\]

Duplicates remain observations and contribute zero distances. For iid samples
from a fixed distribution, this is an unbiased estimate of the expected distance
between independent conformers. Adding samples does not mechanically increase
its expectation; sampling depth still affects which basins are discovered.

For population weights \(w_i\geq0\), normalized to \(p_i=w_i/\sum_jw_j\),
Rao diversity measures independent draws **with replacement**:

\[
Q=\sum_{i,j}p_ip_jd_{ij}.
\]

Equal weights give \(Q=(N-1)\bar d/N\), so the two scores differ at finite
ensemble size. Supply externally computed Boltzmann populations when meaningful.

\(N_\mathrm{unique}(\delta)\) is a **coverage diagnostic**, defined here as the
number of connected components in the graph with edges \(d_{ij}\leq\delta\)
(single linkage). This is invariant to input ordering and monotone in cutoff.
It allows chaining: if A–B and B–C are within cutoff, all three belong to one
cluster even when A–C exceeds cutoff. It is not a count of mutually separated
representatives or an implementation of Pracht pruning. At zero cutoff, exact
duplicates merge; an empty ensemble has zero clusters.

Use the same distance and units to compare algorithms on the same molecule.
For aggregation across heterogeneous molecules, the motivating benchmark uses
externally supplied TFD as a normalized alternative and iRMSD as a complementary
score. Neither subsampling nor division by N removes molecular size/flexibility
effects. Raw cluster counts should not rank different molecules.

The originating discussion and its design rationale are summarized in
[Project context](docs/context.md).

## Setup and usage

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```sh
uv sync --locked
uv run python examples/basic.py
```

Calculate molecular distances directly from coordinates; no backend extra or
distance callback is needed:

```python
import numpy as np
from conformer_metric import mean_pairwise, pairwise_distances

water = np.array([[0, 0, 0], [0.9572, 0, 0], [-0.239987, 0.927297, 0]])
D = pairwise_distances([water, water + [3, 2, 1]], atomic_numbers=[8, 1, 1])
assert mean_pairwise(D) < 1e-7
```

All diversity and coverage functions consume the resulting distance matrix.
They also accept precomputed distances:

```python
import numpy as np
from conformer_metric import (
    mean_pairwise,
    rao_diversity,
    rarefaction,
    n_unique,
    unique_curve,
)

# Precomputed iRMSD (angstroms), TFD, or another symmetric dissimilarity.
D = np.array([[0, 0.2, 1.1], [0.2, 0, 1.0], [1.1, 1.0, 0]])
print(mean_pairwise(D))  # 0.766666...
print(rao_diversity(D, weights=[98, 1, 1]))  # 0.02568
print(n_unique(D, delta=0.2))  # 2

coverage = unique_curve(D, [0, 0.2, 1.0, 1.5])
print(coverage.counts)  # [3, 2, 1, 1]
print(coverage.auc(normalize=True))  # AUC / cutoff span

diversity = rarefaction(D, [2, 3], repeats=200, seed=7)
print(diversity.mean, diversity.std)
print(np.quantile(diversity.samples, [0.05, 0.95], axis=1))

coverage_by_size = rarefaction(
    D,
    [1, 2, 3],
    statistic=lambda sub: n_unique(sub, 0.2),
    seed=7,
)
print(coverage_by_size.mean)
```

To use another metric, supply an explicit distance callable:

```python
from conformer_metric import pairwise_distances

# D = pairwise_distances(molecule_objects, distance=external_tfd)

# Runnable toy example with vector Euclidean distance (not molecular RMSD):
D = pairwise_distances(
    np.array([[0.0, 0.0], [3.0, 4.0]]),
    distance=lambda a, b: float(np.linalg.norm(a - b)),
)
assert mean_pairwise(D) == 5.0
```

The callback is evaluated once per unordered pair and mirrored. It must be
symmetric, pure, and return a finite nonnegative scalar. A custom callback is
responsible for alignment, atom correspondence, symmetry/permutation handling,
and units. `atomic_numbers` and non-default `inversion` cannot be combined with
a custom callback. All built-in RMSD calculation uses upstream iRMSD; no
reimplementation of the Pracht algorithm is bundled.

## Upstream iRMSD

The built-in adapter uses [pprcht/irmsd](https://github.com/pprcht/irmsd), via its
[coordinate API](https://pprcht.github.io/irmsd/generated/irmsd.html#irmsd.get_irmsd).
It is installed by `uv sync --locked` or `pip install conformer-metric`.
The explicit iRMSD helper is also available at the package's top level:

```python
from conformer_metric import irmsd_distances

water = np.array([[0, 0, 0], [0.9572, 0, 0], [-0.239987, 0.927297, 0]])
D = irmsd_distances([8, 1, 1], [water, water + [3, 2, 1]])
assert mean_pairwise(D) < 1e-7
```

Coordinates have shape `(conformers, atoms, 3)` in Å. Atomic numbers can be a
shared `(atoms,)` vector or a `(conformers, atoms)` array for different atom
orders. These inputs apply to both `pairwise_distances` and `irmsd_distances`.
All conformers must describe the same molecule; equal formulas alone
do not establish equal connectivity. Select any heavy atoms before calling.
`inversion="off"` is the default; `"auto"`/`"on"` forward upstream flags 0/1.
Backend 0.1.2 passes rotation, translation, permutation, and geometry-change
checks here, but a forced-inversion check with all unique canonical ranks fails
upstream. That case is recorded as a strict expected failure, not corrected by
the adapter. Integration tests run as part of the standard test suite and fail
if the required backend is missing.

For `irmsd.Molecule` objects, use `pairwise_distances(molecules,
lambda a, b: float(irmsd.get_irmsd_molecule(a, b, iinversion=2)[0]))`.
The upstream backend has its own LGPL license and native-library requirements.
Run `uv run python examples/with_irmsd.py` for a complete example.

## Interpretation and API boundaries

- `rarefaction` draws uniformly **without replacement**, independently at each
  size and repeat. It returns all draws, their mean, and SD (`ddof=0`). Quantile
  bands describe subset variability, not confidence intervals for a population.
  Sizes must be strictly increasing integers from 1 to N (at least 2 for the
  default mean score). Weights affect `rao_diversity`, not subset selection.
- For mean distance, the exact average over all size-n subsets equals the full
  ensemble mean for every n >= 2. Thus mean rarefaction primarily shows sampling
  variability; it is not an expected increasing discovery curve.
- Clustering is recomputed within each subset, so omitted bridging conformers
  can split single-linkage clusters. This is induced-subset coverage, not
  rarefaction of fixed cluster labels, and need not increase monotonically in n.
- Equal output counts do not imply equal search budgets. Measuring sampling
  efficiency versus computational budget needs separate ensembles/checkpoints
  collected at matched search effort; this library summarizes those inputs.
- `unique_curve` is the cutoff-dependent coverage/diversity curve. Mean distance
  and Rao diversity themselves have no cutoff. `auc(x, y)` and `curve.auc()` use
  **trapezoidal linear interpolation** on the supplied grid, not exact integration
  of the clustering step function. AUC depends on cutoff range, grid, units, and
  ensemble size; `normalize=True` divides only by the x span.
- Matrices must be square, real, finite, nonnegative, symmetric, and have zero
  diagonal. Absolute roundoff <= 1e-12 is canonicalized by averaging symmetric
  entries, clipping tiny negatives, and zeroing the diagonal. Triangle inequality
  is not required. Functions copy inputs. No distance rescaling is performed.
- Mean distance is undefined for N < 2 and raises `ValueError`. Rao diversity
  requires N >= 1 and positive total weight (singleton Q = 0). AUC needs at least
  two strictly increasing grid points.
- Storage and matrix construction are O(N²); clustering is O(N²) per cutoff.
  Distance computation is usually the expensive step. Precompute once and reuse.

## Development

```sh
uv sync --locked
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run ty check
uv build
```

`uv.lock` records the development environment; published package requirements
remain ranges. See the [uv project workflow](https://docs.astral.sh/uv/guides/projects/).
