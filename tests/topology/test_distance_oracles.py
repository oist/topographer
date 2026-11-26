"""Check the public diagram distances against GUDHI's exact Hera and against hand values.

Conventions matched to the oracle:

* The public functions read ``birth_scalar`` / ``death_scalar`` and use the L-infinity
  ground metric, with the diagonal cost of a point equal to ``|d - b| / 2``. That is
  GUDHI's ``internal_p=inf`` (the default) with the same diagonal projection.
* ``diagram_wasserstein_distance(p=q)`` equals ``gudhi.hera.wasserstein_distance(order=q)``.
* GUDHI expects ``birth <= death`` and no points on the diagonal. Split-tree diagrams have
  ``birth >= death``; negating both coordinates maps one convention onto the other and
  leaves every distance unchanged, so the oracle gets ``orientation * points`` while the
  public functions get the raw points (they use ``abs(d - b)``).
* Hera's bottleneck is run with ``delta=0`` (exact). Its Wasserstein is a ``1 + delta``
  approximation that hangs at ``delta=0``, so ``delta=1e-8`` and a 1e-6 relative tolerance.
* ``pair_kind=None`` (no filter) is used throughout; the diagrams hold finite points only.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from topographer.examples import build_cube_scalar_network
from topographer.topology import (
    ScalarNetworkTopology,
    diagram_bottleneck_distance,
    diagram_wasserstein_distance,
)


def to_table(points: np.ndarray, kind: str = "ordinary") -> pl.DataFrame:
    return pl.DataFrame(
        {
            "birth_scalar": points[:, 0].astype(np.float64),
            "death_scalar": points[:, 1].astype(np.float64),
            "pair_kind": [kind] * len(points),
        },
        schema={
            "birth_scalar": pl.Float64,
            "death_scalar": pl.Float64,
            "pair_kind": pl.String,
        },
    )


def random_diagram(rng: np.random.Generator, size: int, orientation: float) -> np.ndarray:
    """Random (birth, death) points with death above birth (+1) or below it (-1)."""
    births = rng.random(size)
    return np.column_stack([births, births + orientation * rng.random(size) * 0.6])


# Hera's Wasserstein is a (1 + delta)-approximation and does not terminate for delta=0, so a
# tiny delta stands in for exactness and comparisons use a matching relative tolerance.
_HERA_DELTA = 1e-8
_REL = 1e-6


def hera_bottleneck(first: np.ndarray, second: np.ndarray) -> float:
    hera = pytest.importorskip("gudhi.hera")
    return float(hera.bottleneck_distance(first, second, 0))


def hera_wasserstein(first: np.ndarray, second: np.ndarray, order: float) -> float:
    hera = pytest.importorskip("gudhi.hera")
    return float(
        hera.wasserstein_distance(first, second, order=order, internal_p=np.inf, delta=_HERA_DELTA)
    )


# --- oracle comparisons -----------------------------------------------------------------------


@pytest.mark.parametrize("orientation", [1.0, -1.0])
def test_distances_match_hera_on_random_diagrams(orientation: float) -> None:
    pytest.importorskip("gudhi")
    rng = np.random.default_rng(2026)
    sizes = [(0, 0), (0, 3), (4, 0), (1, 1), (2, 5), (7, 3), (6, 6), (12, 9), (1, 8), (60, 45)]
    sizes += [tuple(int(s) for s in rng.integers(0, 14, size=2)) for _ in range(20)]
    for n, m in sizes:
        first = random_diagram(rng, n, orientation)
        second = random_diagram(rng, m, orientation)
        theirs = (orientation * first, orientation * second)
        ours = (to_table(first), to_table(second))

        assert diagram_bottleneck_distance(*ours) == pytest.approx(
            hera_bottleneck(*theirs), abs=1e-9
        )
        for order in (1.0, 2.0):
            assert diagram_wasserstein_distance(*ours, p=order) == pytest.approx(
                hera_wasserstein(*theirs, order), rel=_REL, abs=1e-9
            )


def test_distances_match_hera_on_tied_integer_grid_points() -> None:
    pytest.importorskip("gudhi")
    # Duplicated points and exact ties; hand value of the bottleneck below.
    first = np.array([[0, 2], [3, 4], [3, 4]], dtype=np.float64)
    second = np.array([[5, 6], [4, 6], [4, 6]], dtype=np.float64)
    ours = (to_table(first), to_table(second))

    assert diagram_bottleneck_distance(*ours) == hera_bottleneck(first, second)
    assert diagram_wasserstein_distance(*ours, p=1.0) == pytest.approx(
        hera_wasserstein(first, second, 1.0), rel=_REL
    )
    assert diagram_wasserstein_distance(*ours, p=2.0) == pytest.approx(
        hera_wasserstein(first, second, 2.0), rel=_REL
    )


def test_distances_match_hera_on_cube_component_persistence() -> None:
    pytest.importorskip("gudhi")
    pairs = ScalarNetworkTopology(build_cube_scalar_network()).component_persistence().pairs
    # Compare the real cube diagram with a shifted and a thinned copy of itself.
    points = pairs.select("birth_scalar", "death_scalar").to_numpy()
    shifted = points + np.array([0.05, -0.03])
    thinned = points[:-1]
    for other in (shifted, thinned):
        # Cube pairs mix birth < death (sublevel) and birth > death (relative); the oracle
        # only needs points off the diagonal and the same pairing of coordinates, as the
        # L-infinity diagonal cost |d - b| / 2 is orientation free.
        assert diagram_bottleneck_distance(pairs, to_table(other)) == pytest.approx(
            hera_bottleneck(points, other), abs=1e-9
        )
        assert diagram_wasserstein_distance(pairs, to_table(other), p=1.0) == pytest.approx(
            hera_wasserstein(points, other, 1.0), rel=_REL, abs=1e-9
        )


# --- hand values (no oracle needed) -----------------------------------------------------------


def test_tied_grid_bottleneck_hand_value() -> None:
    first = np.array([[0, 2], [3, 4], [3, 4]], dtype=np.float64)
    second = np.array([[5, 6], [4, 6], [4, 6]], dtype=np.float64)
    # Diagonal costs |d - b| / 2: first = (1.0, 0.5, 0.5), second = (0.5, 1.0, 1.0). The point
    # (0, 2) is at L-infinity distance >= 3 from every second point, so it must go to the
    # diagonal at cost 1.0; hence the bottleneck is at least 1.0. Sending every point to the
    # diagonal costs at most 1.0, so it is exactly 1.0 (matching (3, 4) -> (4, 6) would cost
    # max(1, 2) = 2).
    assert diagram_bottleneck_distance(to_table(first), to_table(second)) == 1.0


def test_distances_by_hand() -> None:
    point = np.array([[0.0, 1.0]])
    empty = np.zeros((0, 2))
    assert diagram_wasserstein_distance(to_table(empty), to_table(empty)) == 0.0
    assert diagram_bottleneck_distance(to_table(empty), to_table(empty)) == 0.0
    # An unmatched point costs its distance to the diagonal, persistence / 2.
    assert diagram_wasserstein_distance(to_table(point), to_table(empty)) == pytest.approx(0.5)
    assert diagram_bottleneck_distance(to_table(empty), to_table(point)) == pytest.approx(0.5)
    assert diagram_wasserstein_distance(to_table(point), to_table(point)) == 0.0
    assert diagram_bottleneck_distance(to_table(point), to_table(point)) == 0.0
    shifted = np.array([[0.1, 1.3]])
    # L-infinity ground metric: max(|0.1|, |0.3|) = 0.3 for both.
    assert diagram_wasserstein_distance(to_table(point), to_table(shifted)) == pytest.approx(0.3)
    assert diagram_bottleneck_distance(to_table(point), to_table(shifted)) == pytest.approx(0.3)
