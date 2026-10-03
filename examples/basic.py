"""Run with: uv run python examples/basic.py."""

import numpy as np

from conformer_metric import (
    mean_pairwise,
    n_unique,
    pairwise_distances,
    rao_diversity,
    rarefaction,
    unique_curve,
)

water = np.array([[0, 0, 0], [0.9572, 0, 0], [-0.239987, 0.927297, 0]])
stretched = water.copy()
stretched[1] *= 1.2
D = pairwise_distances([water, water + [3, 2, 1], stretched], atomic_numbers=[8, 1, 1])
print("iRMSD matrix (Å):", D)
print("Mean pairwise iRMSD (Å):", mean_pairwise(D))
print("Population-weighted Rao diversity:", rao_diversity(D, [98, 1, 1]))
curve = unique_curve(D, [0, 1e-6, 0.05, 0.1])
print("Cutoffs / coverage:", curve.cutoffs, curve.counts)
print("Normalized trapezoidal AUC:", curve.auc(normalize=True))
for statistic in (mean_pairwise, lambda sub: n_unique(sub, 1e-6)):
    result = rarefaction(D, [2, 3], statistic=statistic, repeats=200, seed=7)
    print("Sizes / subset mean / SD:", result.sizes, result.mean, result.std)
