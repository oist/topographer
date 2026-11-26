"""The sparse bottleneck distance equals the original dense bisection exactly (``==``)."""

from __future__ import annotations

import time

import numpy as np
import polars as pl
import pytest
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

from topographer.topology import diagram_bottleneck_distance


def _reference_bottleneck(pts1: np.ndarray, pts2: np.ndarray) -> float:
    """The original dense algorithm: bisection over all candidates, full augmented matching."""
    n1, n2 = len(pts1), len(pts2)
    if n1 == 0 and n2 == 0:
        return 0.0
    if n1 == 0:
        return float(np.max(np.abs(pts2[:, 1] - pts2[:, 0]) / 2.0))
    if n2 == 0:
        return float(np.max(np.abs(pts1[:, 1] - pts1[:, 0]) / 2.0))
    candidates = [0.0]
    diff_b = np.abs(pts1[:, 0:1] - pts2[:, 0:1].T)
    diff_d = np.abs(pts1[:, 1:2] - pts2[:, 1:2].T)
    pairwise_dist = np.maximum(diff_b, diff_d)
    candidates.extend(pairwise_dist.ravel().tolist())
    diag1 = np.abs(pts1[:, 1] - pts1[:, 0]) / 2.0
    diag2 = np.abs(pts2[:, 1] - pts2[:, 0]) / 2.0
    candidates.extend(diag1.tolist())
    candidates.extend(diag2.tolist())
    unique_cands = np.unique(candidates)
    low, high = 0, len(unique_cands) - 1
    best = float(unique_cands[-1])
    while low <= high:
        mid = (low + high) // 2
        delta = float(unique_cands[mid])
        k = n1 + n2
        adj = np.zeros((k, k), dtype=bool)
        adj[:n1, :n2] = pairwise_dist <= delta
        for i in range(n1):
            if diag1[i] <= delta:
                adj[i, n2 + i] = True
        for j in range(n2):
            if diag2[j] <= delta:
                adj[n1 + j, j] = True
        adj[n1:, n2:] = True
        matching = maximum_bipartite_matching(csr_matrix(adj), perm_type="column")
        if np.all(matching >= 0):
            best = delta
            high = mid - 1
        else:
            low = mid + 1
    return float(best)


def _table(points: np.ndarray) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "birth_scalar": points[:, 0].astype(np.float64),
            "death_scalar": points[:, 1].astype(np.float64),
        },
        schema={"birth_scalar": pl.Float64, "death_scalar": pl.Float64},
    )


def _assert_same(first: np.ndarray, second: np.ndarray) -> None:
    expected = _reference_bottleneck(first, second)
    assert diagram_bottleneck_distance(_table(first), _table(second)) == expected
    assert diagram_bottleneck_distance(_table(second), _table(first)) == expected


def _random_diagram(rng: np.random.Generator, size: int, orientation: float) -> np.ndarray:
    raw = rng.random((size, 2))
    points = np.column_stack((raw.min(axis=1), raw.max(axis=1)))
    return orientation * points


@pytest.mark.parametrize("orientation", [1.0, -1.0])
def test_matches_reference_on_random_diagrams(orientation: float) -> None:
    rng = np.random.default_rng(11)
    for _ in range(120):
        first = _random_diagram(rng, int(rng.integers(0, 41)), orientation)
        second = _random_diagram(rng, int(rng.integers(0, 41)), orientation)
        _assert_same(first, second)


def test_matches_reference_on_tied_integer_grids() -> None:
    rng = np.random.default_rng(12)
    for _ in range(120):
        first = rng.integers(0, 5, size=(int(rng.integers(1, 31)), 2)).astype(np.float64)
        second = rng.integers(0, 5, size=(int(rng.integers(1, 31)), 2)).astype(np.float64)
        _assert_same(first, second)


def test_matches_reference_on_many_near_diagonal_points() -> None:
    rng = np.random.default_rng(13)
    for _ in range(60):
        base = _random_diagram(rng, int(rng.integers(1, 15)), 1.0)
        noise = rng.random(int(rng.integers(1, 25)))
        near = np.column_stack((noise, noise + rng.random(len(noise)) * 1e-3))
        jittered = base + rng.normal(0.0, 1e-3, base.shape)
        _assert_same(base, np.vstack((jittered, near)))


def test_matches_reference_with_an_empty_side_and_unequal_sizes() -> None:
    rng = np.random.default_rng(14)
    empty = np.empty((0, 2))
    full = _random_diagram(rng, 17, 1.0)
    _assert_same(empty, full)
    _assert_same(empty, empty)
    _assert_same(full[:3], full)
    _assert_same(full[:1], full)


def test_matches_reference_on_duplicated_and_identical_points() -> None:
    rng = np.random.default_rng(15)
    base = _random_diagram(rng, 12, 1.0)
    _assert_same(base, base)
    _assert_same(np.vstack((base, base)), base)
    _assert_same(np.vstack((base, base)), np.vstack((base[:5], base[:5], base[:5])))
    zero_persistence = np.column_stack((rng.random(6), np.zeros(6)))
    zero_persistence[:, 1] = zero_persistence[:, 0]
    _assert_same(zero_persistence, base)
    _assert_same(zero_persistence, zero_persistence)


def test_matches_hera_on_larger_diagrams() -> None:
    hera = pytest.importorskip("gudhi.hera")
    rng = np.random.default_rng(16)
    first = _random_diagram(rng, 500, 1.0)
    second = np.vstack(
        (first + rng.normal(0.0, 0.01, first.shape), _random_diagram(rng, 50, 1.0) * 0.05)
    )
    second = np.column_stack(
        (np.minimum(second[:, 0], second[:, 1]), np.maximum(second[:, 0], second[:, 1]))
    )
    expected = float(hera.bottleneck_distance(first, second, 0))
    assert diagram_bottleneck_distance(_table(first), _table(second)) == pytest.approx(
        expected, abs=1e-9
    )
    other = _random_diagram(rng, 500, 1.0)
    assert diagram_bottleneck_distance(_table(first), _table(other)) == pytest.approx(
        float(hera.bottleneck_distance(first, other, 0)), abs=1e-9
    )


@pytest.mark.slow
def test_bottleneck_of_800_point_diagrams_is_fast() -> None:
    rng = np.random.default_rng(17)
    first = _table(_random_diagram(rng, 800, 1.0))
    second = _table(_random_diagram(rng, 800, 1.0))
    start = time.perf_counter()
    diagram_bottleneck_distance(first, second)
    assert time.perf_counter() - start < 5.0


def test_subnormal_costs_terminate_and_stay_exact() -> None:
    tiny = 5e-324
    first = np.array([[0.0, 8 * tiny], [0.0, 2 * tiny]])
    second = np.array([[0.0, 4 * tiny]])
    assert first.max() * 2.0**-40 == 0.0
    _assert_same(first, second)
