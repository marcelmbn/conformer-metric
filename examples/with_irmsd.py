"""Run with: uv run --extra irmsd python examples/with_irmsd.py."""

import numpy as np

from conformer_metric import mean_pairwise, unique_curve
from conformer_metric.backends import irmsd_distances

water = np.array([[0, 0, 0], [0.9572, 0, 0], [-0.239987, 0.927297, 0]])
stretched = water.copy()
stretched[1] *= 1.2
distances = irmsd_distances([8, 1, 1], [water, water + [3, 2, 1], stretched])
print("iRMSD matrix (Å):", distances)
print("Mean pairwise iRMSD (Å):", mean_pairwise(distances))
print("Coverage:", unique_curve(distances, [0, 1e-6, 0.05, 0.1]).counts)
