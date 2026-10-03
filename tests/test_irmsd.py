"""Integration checks run against the installed upstream Fortran backend."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from conformer_metric.backends import irmsd_distances


@pytest.fixture
def water():
    return np.array([[0, 0, 0], [0.9572, 0, 0], [-0.239987, 0.927297, 0]])


def test_real_backend_rotation_translation_and_permutation(water):
    pytest.importorskip("irmsd")
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    moved = (water @ rotation + [3, -2, 4])[[2, 0, 1]]
    coords = np.array([water, moved])
    numbers = np.array([[8, 1, 1], [1, 8, 1]])
    original = coords.copy()
    result = irmsd_distances(numbers, coords)
    assert_allclose(result, np.zeros((2, 2)), atol=1e-7)
    assert_array_equal(coords, original)
    assert_array_equal(numbers, [[8, 1, 1], [1, 8, 1]])


def test_real_backend_nonidentical_conformers(water):
    backend = pytest.importorskip("irmsd")
    stretched = water.copy()
    stretched[1] *= 1.2
    coords = np.array([water, stretched])
    z = np.array([8, 1, 1])
    result = irmsd_distances(z, coords)
    direct = backend.get_irmsd(z, water, z, stretched, iinversion=2)[0]
    assert result[0, 1] > 0.01
    assert result[0, 1] == pytest.approx(direct)
    assert_array_equal(result, result.T)
    assert_array_equal(result.diagonal(), [0, 0])


def test_real_backend_inversion_is_explicit():
    backend = pytest.importorskip("irmsd")
    xyz = np.array(
        [[0, 0, 0], [0.6, 0.6, 0.6], [0.8, -0.8, -0.8], [-1, 1, -1], [-1.1, -1.1, 1.1]]
    )
    z = [6, 1, 9, 17, 35]
    reflected = xyz * [-1, 1, 1]
    coords = np.array([xyz, reflected])
    assert irmsd_distances(z, coords, inversion="off")[0, 1] > 0.1
    # The adapter forwards policy rather than correcting the upstream algorithm.
    for policy, flag in (("auto", 0), ("on", 1), ("off", 2)):
        expected = backend.get_irmsd(np.array(z), xyz, np.array(z), reflected, iinversion=flag)[
            0
        ]
        assert irmsd_distances(z, coords, inversion=policy)[0, 1] == pytest.approx(expected)


@pytest.mark.xfail(
    strict=True,
    reason="Upstream irmsd 0.1.2 fails forced inversion when every canonical rank is unique",
)
def test_upstream_forced_inversion_all_unique_ranks():
    pytest.importorskip("irmsd")
    xyz = np.array(
        [[0, 0, 0], [0.6, 0.6, 0.6], [0.8, -0.8, -0.8], [-1, 1, -1], [-1.1, -1.1, 1.1]]
    )
    reflected = xyz * [-1, 1, 1]
    assert irmsd_distances([6, 1, 9, 17, 35], [xyz, reflected], inversion="on")[0, 1] < 1e-7


@pytest.mark.parametrize(
    ("numbers", "coords"),
    [
        ([8, 1, 1], np.zeros((3, 3))),
        ([8, 1, 1], np.zeros((2, 3, 2))),
        ([], np.zeros((2, 0, 3))),
        ([8, 1], np.zeros((2, 3, 3))),
        ([8, 1.5, 1], np.zeros((2, 3, 3))),
        ([8, 0, 1], np.zeros((2, 3, 3))),
        ([8, 119, 1], np.zeros((2, 3, 3))),
        ([8, np.inf, 1], np.zeros((2, 3, 3))),
        ([8, 1, 1], np.full((2, 3, 3), np.nan)),
        ([8, 1j, 1], np.zeros((2, 3, 3))),
        ([[8, 1, 1], [6, 1, 1]], np.zeros((2, 3, 3))),
    ],
)
def test_adapter_validation(numbers, coords):
    with pytest.raises(ValueError):
        irmsd_distances(numbers, coords)


def test_invalid_inversion(water):
    with pytest.raises(ValueError):
        irmsd_distances([8, 1, 1], np.array([water]), inversion="invalid")


def test_missing_backend_has_actionable_error(monkeypatch, water):
    import sys

    monkeypatch.setitem(sys.modules, "irmsd", None)
    with pytest.raises(ImportError, match="uv sync --extra irmsd"):
        irmsd_distances([8, 1, 1], np.array([water, water]))
