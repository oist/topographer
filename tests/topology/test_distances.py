"""Unit tests for pairwise persistence diagram distances."""

from __future__ import annotations

import polars as pl
import pytest

from topographer.examples import build_cube_scalar_network
from topographer.topology import (
    PersistenceResult,
    ScalarNetworkTopology,
    diagram_bottleneck_distance,
    diagram_wasserstein_distance,
)


def _make_dummy_persistence_result(
    births: list[float],
    deaths: list[float],
    kinds: list[str] | None = None,
) -> PersistenceResult:
    n = len(births)
    pair_kinds = kinds if kinds is not None else ["ordinary"] * n
    pairs = pl.DataFrame(
        {
            "pair_id": list(range(n)),
            "homology_dimension": [0] * n,
            "pair_kind": pair_kinds,
            "birth_simplex": [[f"v{i}"] for i in range(n)],
            "death_simplex": [[f"v{i}"] for i in range(n)],
            "birth_scalar": births,
            "death_scalar": deaths,
            "persistence": [abs(d - b) for b, d in zip(births, deaths, strict=False)],
        },
        schema={
            "pair_id": pl.Int64,
            "homology_dimension": pl.Int64,
            "pair_kind": pl.String,
            "birth_simplex": pl.List(pl.String),
            "death_simplex": pl.List(pl.String),
            "birth_scalar": pl.Float64,
            "death_scalar": pl.Float64,
            "persistence": pl.Float64,
        },
    )
    return PersistenceResult(pairs=pairs)


# =====================================================================
# Diagram Bottleneck Distance Tests
# =====================================================================


def test_diagram_bottleneck_self_distance_zero() -> None:
    cube_net = build_cube_scalar_network()
    topo = ScalarNetworkTopology(cube_net)
    res = topo.component_persistence()

    assert diagram_bottleneck_distance(res, res) == 0.0
    assert diagram_bottleneck_distance(res.pairs, res.pairs) == 0.0


def test_diagram_bottleneck_symmetry() -> None:
    d1 = pl.DataFrame({"birth_scalar": [0.0, 1.0], "death_scalar": [3.0, 2.0]})
    d2 = pl.DataFrame({"birth_scalar": [0.5, 1.2], "death_scalar": [2.8, 1.8]})

    dist12 = diagram_bottleneck_distance(d1, d2)
    dist21 = diagram_bottleneck_distance(d2, d1)
    assert dist12 == pytest.approx(dist21)
    assert dist12 > 0.0


def test_diagram_bottleneck_triangle_inequality() -> None:
    d1 = pl.DataFrame({"birth_scalar": [0.0, 1.0], "death_scalar": [4.0, 3.0]})
    d2 = pl.DataFrame({"birth_scalar": [0.5, 1.5], "death_scalar": [3.5, 2.5]})
    d3 = pl.DataFrame({"birth_scalar": [1.0, 2.0], "death_scalar": [5.0, 4.0]})

    d12 = diagram_bottleneck_distance(d1, d2)
    d23 = diagram_bottleneck_distance(d2, d3)
    d13 = diagram_bottleneck_distance(d1, d3)

    assert d13 <= d12 + d23 + 1e-12


def test_diagram_bottleneck_scalar_shift_invariance() -> None:
    d1 = pl.DataFrame({"birth_scalar": [0.0, 1.0], "death_scalar": [4.0, 2.5]})
    d2 = pl.DataFrame({"birth_scalar": [0.2, 0.8], "death_scalar": [3.8, 2.0]})

    shift = 42.5
    d1_shifted = d1.with_columns(
        (pl.col("birth_scalar") + shift).alias("birth_scalar"),
        (pl.col("death_scalar") + shift).alias("death_scalar"),
    )
    d2_shifted = d2.with_columns(
        (pl.col("birth_scalar") + shift).alias("birth_scalar"),
        (pl.col("death_scalar") + shift).alias("death_scalar"),
    )

    base_dist = diagram_bottleneck_distance(d1, d2)
    shifted_dist = diagram_bottleneck_distance(d1_shifted, d2_shifted)
    assert base_dist == pytest.approx(shifted_dist)


def test_diagram_bottleneck_empty_and_unequal_pair_counts() -> None:
    empty_df = pl.DataFrame(schema={"birth_scalar": pl.Float64, "death_scalar": pl.Float64})
    assert diagram_bottleneck_distance(empty_df, empty_df) == 0.0

    d1 = pl.DataFrame({"birth_scalar": [1.0, 2.0], "death_scalar": [5.0, 4.0]})
    # Point (1.0, 5.0) has diag cost (5-1)/2 = 2.0; (2.0, 4.0) has diag cost (4-2)/2 = 1.0.
    assert diagram_bottleneck_distance(d1, empty_df) == pytest.approx(2.0)
    assert diagram_bottleneck_distance(empty_df, d1) == pytest.approx(2.0)

    # Unequal counts: d1 has 2 points, d2 has 1 point matching (1.0, 5.0) exactly.
    d2 = pl.DataFrame({"birth_scalar": [1.0], "death_scalar": [5.0]})
    # Remaining point (2.0, 4.0) matches to diagonal with cost 1.0.
    assert diagram_bottleneck_distance(d1, d2) == pytest.approx(1.0)


def test_diagram_bottleneck_pair_kind_filtering() -> None:
    d1 = _make_dummy_persistence_result(
        births=[0.0, 1.0],
        deaths=[4.0, 2.0],
        kinds=["ordinary", "relative"],
    )
    d2 = _make_dummy_persistence_result(
        births=[0.0, 1.0],
        deaths=[4.0, 2.5],
        kinds=["ordinary", "relative"],
    )

    dist_ord = diagram_bottleneck_distance(d1, d2, pair_kind="ordinary")
    assert dist_ord == 0.0

    dist_rel = diagram_bottleneck_distance(d1, d2, pair_kind="relative")
    assert dist_rel == pytest.approx(0.5)


def test_diagram_bottleneck_rejects_invalid_inputs() -> None:
    with pytest.raises(TypeError, match="Expected PersistenceResult or pl.DataFrame"):
        diagram_bottleneck_distance("invalid", pl.DataFrame())  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="Expected PersistenceResult or pl.DataFrame"):
        diagram_bottleneck_distance(pl.DataFrame(), 123)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="pair_kind must be a string or None"):
        diagram_bottleneck_distance(
            pl.DataFrame(),
            pl.DataFrame(),
            pair_kind=42,  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="does not contain a 'pair_kind' column"):
        df_no_kind = pl.DataFrame({"birth_scalar": [1.0], "death_scalar": [2.0]})
        diagram_bottleneck_distance(df_no_kind, df_no_kind, pair_kind="ordinary")

    with pytest.raises(ValueError, match="missing required columns"):
        bad_df = pl.DataFrame({"wrong_col": [1.0]})
        diagram_bottleneck_distance(bad_df, bad_df)

    with pytest.raises(ValueError, match="must be finite"):
        nan_df = pl.DataFrame({"birth_scalar": [float("nan")], "death_scalar": [2.0]})
        diagram_bottleneck_distance(nan_df, nan_df)


# =====================================================================
# Diagram Wasserstein Distance Tests
# =====================================================================


def test_diagram_wasserstein_self_distance_zero() -> None:
    cube_net = build_cube_scalar_network()
    topo = ScalarNetworkTopology(cube_net)
    res = topo.component_persistence()

    assert diagram_wasserstein_distance(res, res, p=1.0) == 0.0
    assert diagram_wasserstein_distance(res.pairs, res.pairs, p=2.0) == 0.0


def test_diagram_wasserstein_symmetry() -> None:
    d1 = pl.DataFrame({"birth_scalar": [0.0, 1.0], "death_scalar": [3.0, 2.0]})
    d2 = pl.DataFrame({"birth_scalar": [0.5, 1.2], "death_scalar": [2.8, 1.8]})

    dist12 = diagram_wasserstein_distance(d1, d2, p=1.0)
    dist21 = diagram_wasserstein_distance(d2, d1, p=1.0)
    assert dist12 == pytest.approx(dist21)
    assert dist12 > 0.0

    dist12_p2 = diagram_wasserstein_distance(d1, d2, p=2.0)
    dist21_p2 = diagram_wasserstein_distance(d2, d1, p=2.0)
    assert dist12_p2 == pytest.approx(dist21_p2)


def test_diagram_wasserstein_triangle_inequality() -> None:
    d1 = pl.DataFrame({"birth_scalar": [0.0, 1.0], "death_scalar": [4.0, 3.0]})
    d2 = pl.DataFrame({"birth_scalar": [0.5, 1.5], "death_scalar": [3.5, 2.5]})
    d3 = pl.DataFrame({"birth_scalar": [1.0, 2.0], "death_scalar": [5.0, 4.0]})

    for p in (1.0, 2.0):
        d12 = diagram_wasserstein_distance(d1, d2, p=p)
        d23 = diagram_wasserstein_distance(d2, d3, p=p)
        d13 = diagram_wasserstein_distance(d1, d3, p=p)
        assert d13 <= d12 + d23 + 1e-12


def test_diagram_wasserstein_scalar_shift_invariance() -> None:
    d1 = pl.DataFrame({"birth_scalar": [0.0, 1.0], "death_scalar": [4.0, 2.5]})
    d2 = pl.DataFrame({"birth_scalar": [0.2, 0.8], "death_scalar": [3.8, 2.0]})

    shift = -17.3
    d1_shifted = d1.with_columns(
        (pl.col("birth_scalar") + shift).alias("birth_scalar"),
        (pl.col("death_scalar") + shift).alias("death_scalar"),
    )
    d2_shifted = d2.with_columns(
        (pl.col("birth_scalar") + shift).alias("birth_scalar"),
        (pl.col("death_scalar") + shift).alias("death_scalar"),
    )

    base_dist = diagram_wasserstein_distance(d1, d2, p=2.0)
    shifted_dist = diagram_wasserstein_distance(d1_shifted, d2_shifted, p=2.0)
    assert base_dist == pytest.approx(shifted_dist)


def test_diagram_wasserstein_empty_and_unequal_pair_counts() -> None:
    empty_df = pl.DataFrame(schema={"birth_scalar": pl.Float64, "death_scalar": pl.Float64})
    assert diagram_wasserstein_distance(empty_df, empty_df, p=1.0) == 0.0

    d1 = pl.DataFrame({"birth_scalar": [1.0, 2.0], "death_scalar": [5.0, 4.0]})
    # Point 1: (5-1)/2 = 2.0; Point 2: (4-2)/2 = 1.0
    # W1 = 2.0 + 1.0 = 3.0
    assert diagram_wasserstein_distance(d1, empty_df, p=1.0) == pytest.approx(3.0)
    # W2 = sqrt(2.0^2 + 1.0^2) = sqrt(5.0)
    assert diagram_wasserstein_distance(empty_df, d1, p=2.0) == pytest.approx(5.0**0.5)

    # Unequal counts: d2 has point matching (1.0, 5.0) exactly.
    d2 = pl.DataFrame({"birth_scalar": [1.0], "death_scalar": [5.0]})
    # Remaining point (2.0, 4.0) matches to diagonal with cost 1.0.
    assert diagram_wasserstein_distance(d1, d2, p=1.0) == pytest.approx(1.0)
    assert diagram_wasserstein_distance(d1, d2, p=2.0) == pytest.approx(1.0)


def test_diagram_wasserstein_pair_kind_filtering() -> None:
    d1 = _make_dummy_persistence_result(
        births=[0.0, 1.0],
        deaths=[4.0, 2.0],
        kinds=["ordinary", "relative"],
    )
    d2 = _make_dummy_persistence_result(
        births=[0.0, 1.0],
        deaths=[4.0, 2.5],
        kinds=["ordinary", "relative"],
    )

    dist_ord = diagram_wasserstein_distance(d1, d2, pair_kind="ordinary")
    assert dist_ord == 0.0

    dist_rel = diagram_wasserstein_distance(d1, d2, pair_kind="relative")
    assert dist_rel == pytest.approx(0.5)


def test_diagram_wasserstein_rejects_invalid_inputs() -> None:
    valid_df = pl.DataFrame({"birth_scalar": [0.0], "death_scalar": [1.0]})

    with pytest.raises(ValueError, match="p must be a real number >= 1.0"):
        diagram_wasserstein_distance(valid_df, valid_df, p=0.5)

    with pytest.raises(ValueError, match="p must be a real number >= 1.0"):
        diagram_wasserstein_distance(valid_df, valid_df, p=float("nan"))

    with pytest.raises(TypeError, match="Expected PersistenceResult or pl.DataFrame"):
        diagram_wasserstein_distance(123, valid_df)  # type: ignore[arg-type]
