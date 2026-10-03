"""Run with: uv run python examples/basic.py."""

import numpy as np

from conformer_metric import mean_pairwise, n_unique, rao_diversity, rarefaction, unique_curve

D = np.array([[0, 0.2, 1.1], [0.2, 0, 1.0], [1.1, 1.0, 0]])
print("Mean pairwise distance:", mean_pairwise(D))
print("Population-weighted Rao diversity:", rao_diversity(D, [98, 1, 1]))
curve = unique_curve(D, [0, 0.2, 1.0, 1.5])
print("Cutoffs / coverage:", curve.cutoffs, curve.counts)
print("Normalized trapezoidal AUC:", curve.auc(normalize=True))
for statistic in (mean_pairwise, lambda sub: n_unique(sub, 0.2)):
    result = rarefaction(D, [2, 3], statistic=statistic, repeats=200, seed=7)
    print("Sizes / subset mean / SD:", result.sizes, result.mean, result.std)
