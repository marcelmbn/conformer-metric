"""Shared matrix construction for iRMSD and explicit distance callbacks."""

from collections.abc import Callable, Iterable

import numpy as np
from numpy.typing import NDArray


def callback_distances[T](
    conformers: Iterable[T], distance: Callable[[T, T], float]
) -> NDArray[np.float64]:
    """Evaluate a symmetric distance once per unordered pair and mirror it."""
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
