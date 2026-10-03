"""Molecular RMSD calculation through the required upstream iRMSD package."""

from typing import Literal

import numpy as np
from irmsd import get_irmsd
from numpy.typing import ArrayLike, NDArray

from ._pairwise import callback_distances

__all__ = ["irmsd_distances"]


def irmsd_distances(
    atomic_numbers: ArrayLike,
    coordinates: ArrayLike,
    *,
    inversion: Literal["auto", "on", "off"] = "off",
) -> NDArray[np.float64]:
    """Pairwise iRMSD in Å via pprcht/irmsd's get_irmsd.

    Coordinates have shape (N_conformers, N_atoms, 3), in Å. Atomic numbers
    have shape (N_atoms,) for shared atom order, or (N_conformers, N_atoms)
    for independently reordered conformers. All must have the same formula.
    Inversion defaults to off to keep enantiomers distinct; auto/on follow
    the upstream backend's policies. Atom selection is the caller's choice.
    """
    if inversion not in ("auto", "on", "off"):
        raise ValueError("inversion must be 'auto', 'on', or 'off'")
    if np.iscomplexobj(coordinates) or np.iscomplexobj(atomic_numbers):
        raise ValueError("coordinates and atomic numbers must be real")
    xyz = np.array(coordinates, dtype=float, copy=True)
    if xyz.ndim != 3 or xyz.shape[2] != 3 or xyz.shape[1] < 1:
        raise ValueError("coordinates must have shape (N_conformers, N_atoms, 3), N_atoms > 0")
    if not np.isfinite(xyz).all():
        raise ValueError("coordinates must be finite")
    z = np.asarray(atomic_numbers, dtype=float)
    if not np.isfinite(z).all() or np.any((z < 1) | (z > 118) | (z != np.floor(z))):
        raise ValueError("atomic numbers must be integers from 1 to 118")
    n, atoms, _ = xyz.shape
    if z.shape == (atoms,):
        z = np.broadcast_to(z, (n, atoms))
    elif z.shape != (n, atoms):
        raise ValueError("atomic_numbers must have shape (N_atoms,) or (N_conformers, N_atoms)")
    if n and np.any(np.sort(z, axis=1) != np.sort(z[0])):
        raise ValueError("all conformers must have the same elemental composition")
    z = np.array(z, dtype=np.int32, copy=True)
    flag = {"auto": 0, "on": 1, "off": 2}[inversion]

    def distance(i: int, j: int) -> float:
        return float(get_irmsd(z[i], xyz[i], z[j], xyz[j], iinversion=flag)[0])

    return callback_distances(range(n), distance)
