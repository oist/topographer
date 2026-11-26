"""Unit tests for biological ruggedness metrics: roughness-to-slope, sign epistasis, accessibility."""

from __future__ import annotations

import math

import polars as pl
import pytest

from topographer.biology import (
    Alphabet,
    FitnessLandscape,
    FitnessLandscapeTopology,
    roughness_to_slope_ratio,
    sign_epistasis_fraction,
    wildtype_accessibility_fraction,
)
from topographer.examples import build_cube_scalar_network
from topographer.graph import HammingGraph
from topographer.topology import compute_topological_summary


def _make_binary_landscape(fitnesses: dict[str, float]) -> FitnessLandscape:
    seqs = list(fitnesses.keys())
    fits = [fitnesses[s] for s in seqs]
    df = pl.DataFrame({"seq": seqs, "fitness": fits})
    alphabet = Alphabet(("0", "1"))
    hg = HammingGraph(
        df, sequence_col="seq", alphabet=alphabet, vertex_attribute_cols=("fitness",)
    )
    return FitnessLandscape(hg, scalar_attr="fitness")


def test_ruggedness_metrics_reject_invalid_type() -> None:
    with pytest.raises(TypeError, match="Expected FitnessLandscape"):
        roughness_to_slope_ratio(42)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="Expected FitnessLandscape"):
        sign_epistasis_fraction("not a landscape")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="Expected FitnessLandscape"):
        wildtype_accessibility_fraction([])  # type: ignore[arg-type]


def test_roughness_to_slope_ratio_additive_landscape() -> None:
    # Pure additive model: f(x) = 1.0 + 2.0*x0 - 3.0*x1
    fits = {
        "00": 1.0,
        "01": -2.0,
        "10": 3.0,
        "11": 0.0,
    }
    fl = _make_binary_landscape(fits)
    ratio = roughness_to_slope_ratio(fl)
    assert math.isclose(ratio, 0.0, abs_tol=1e-10)
    assert math.isclose(fl.roughness_to_slope_ratio(), 0.0, abs_tol=1e-10)


def test_roughness_to_slope_ratio_purely_epistatic_landscape() -> None:
    # XOR landscape: 00: 0, 01: 1, 10: 1, 11: 0
    fits = {
        "00": 0.0,
        "01": 1.0,
        "10": 1.0,
        "11": 0.0,
    }
    fl = _make_binary_landscape(fits)
    ratio = roughness_to_slope_ratio(fl)
    # The additive regression slope for pure XOR is 0.0, residuals > 0 -> inf or high
    assert ratio > 0.0


def test_roughness_to_slope_ratio_flat_and_single_vertex() -> None:
    fits_flat = {
        "00": 1.0,
        "01": 1.0,
        "10": 1.0,
        "11": 1.0,
    }
    fl_flat = _make_binary_landscape(fits_flat)
    assert roughness_to_slope_ratio(fl_flat) == 0.0

    fits_single = {"0": 5.0}
    fl_single = _make_binary_landscape(fits_single)
    assert roughness_to_slope_ratio(fl_single) == 0.0


def test_sign_epistasis_fraction_additive_vs_epistatic() -> None:
    # Additive: no sign changes
    additive_fits = {
        "00": 0.0,
        "01": 1.0,
        "10": 2.0,
        "11": 3.0,
    }
    fl_add = _make_binary_landscape(additive_fits)
    assert sign_epistasis_fraction(fl_add) == 0.0
    assert fl_add.sign_epistasis_fraction() == 0.0

    # Reciprocal sign epistasis on the square:
    # 00: 0.0 -> 01: 2.0 (beneficial)
    # 10: 2.0 -> 11: 1.0 (deleterious)
    epistatic_fits = {
        "00": 0.0,
        "01": 2.0,
        "10": 2.0,
        "11": 1.0,
    }
    fl_epi = _make_binary_landscape(epistatic_fits)
    assert sign_epistasis_fraction(fl_epi) == 1.0
    assert fl_epi.sign_epistasis_fraction() == 1.0


def test_wildtype_accessibility_fraction_accessible_vs_inaccessible() -> None:
    # Monotonic gradient: all paths accessible
    fits_acc = {
        "00": 0.0,
        "01": 1.0,
        "10": 2.0,
        "11": 3.0,
    }
    fl_acc = _make_binary_landscape(fits_acc)
    assert wildtype_accessibility_fraction(fl_acc) == 1.0
    assert fl_acc.wildtype_accessibility_fraction() == 1.0

    # Valley landscape: global min is 000 (0.0), global max is 111 (10.0),
    # but 1-step mutants have fitness 5.0 and 2-step mutants drop to 2.0.
    # Every monotonic path from 000 to 111 suffers a fitness drop (5.0 -> 2.0).
    fits_inacc = {
        "000": 0.0,
        "001": 5.0,
        "010": 5.0,
        "100": 5.0,
        "011": 2.0,
        "101": 2.0,
        "110": 2.0,
        "111": 10.0,
    }
    fl_inacc = _make_binary_landscape(fits_inacc)
    assert wildtype_accessibility_fraction(fl_inacc) == 0.0
    assert fl_inacc.wildtype_accessibility_fraction() == 0.0


def test_wildtype_accessibility_fraction_ties_follow_extended_pair() -> None:
    """Tied global maxima end at the last one swept, the join tree's extended death."""
    fl = _make_binary_landscape({"00": 0.0, "01": 2.0, "10": 1.0, "11": 2.0})

    # The end is "11" (not "01"): 00 -> 10 -> 11 rises, 00 -> 01 -> 11 does not.
    assert wildtype_accessibility_fraction(fl) == 0.5
    assert fl.wildtype_accessibility_fraction() == 0.5

    extended = (
        FitnessLandscapeTopology(fl)
        .join_tree()
        .persistence()
        .filter(pl.col("pair_kind") == "extended")
        .row(0, named=True)
    )
    assert fl.vertex_ids[extended["birth_vertex_index"]] == "00"
    assert fl.vertex_ids[extended["death_vertex_index"]] == "11"


# Hand-derived 3-locus landscape: the additive cube f = x0 + x1 + x2 with the single
# genotype "011" lowered from 2.0 to 0.5.
#
#   000: 0.0   001: 1.0   010: 1.0   100: 1.0
#   011: 0.5   101: 2.0   110: 2.0   111: 3.0
#
# Sign epistasis: a square is sign-epistatic when a mutation has opposite effects on
# the two backgrounds. Only the three squares that contain "011" can deviate from
# additivity. On x0=0 the x2 effect is +1.0 (000 -> 001) versus -0.5 (010 -> 011);
# on x1=1 the x2 effect is -0.5 (010 -> 011) versus +1.0 (110 -> 111); on x2=1 the x1
# effect is -0.5 (001 -> 011) versus +1.0 (101 -> 111). The squares x0=1, x1=0 and
# x2=0 are additive. So 3 of 6 squares: 0.5.
#
# Roughness-to-slope: the design is the full 2^3 factorial with 0/1 dummy coding, so
# OLS is linear in f and splits as additive base + perturbation p = e * delta_011 with
# e = -1.5. The columns are orthogonal, so each beta_j is mean(p | x_j=1) - mean(p |
# x_j=0): beta_0 = 0 - e/4 = 0.375 and beta_1 = beta_2 = e/4 - 0 = -0.375, on top of the
# base slopes (1, 1, 1). Total betas: (1.375, 0.625, 0.625); mean absolute slope =
# 2.625 / 3 = 0.875. The base is fitted exactly, so the residual is that of p alone:
# the fitted part has variance sum(beta_j^2) / 4 = 3 e^2 / 64, Var(p) = e^2/8 - e^2/64
# = 7 e^2 / 64, so the residual variance is 4 e^2 / 64 = e^2 / 16 and the population
# standard deviation (np.std, ddof=0) is |e| / 4 = 0.375. Ratio = 0.375 / 0.875 = 3/7.
#
# Wild-type accessibility: the global minimum is 000 (0.0) and the unique global
# maximum is 111 (3.0), so there are 3! = 6 shortest paths. The two that pass through
# 011 (000 -> 001 -> 011 -> 111 and 000 -> 010 -> 011 -> 111) step 1.0 -> 0.5 and are
# blocked; the other four rise at every step. Fraction = 4 / 6 = 2/3.
_DENTED_CUBE = {
    "000": 0.0,
    "001": 1.0,
    "010": 1.0,
    "100": 1.0,
    "011": 0.5,
    "101": 2.0,
    "110": 2.0,
    "111": 3.0,
}


def test_sign_epistasis_fraction_is_fractional_on_dented_cube() -> None:
    fl = _make_binary_landscape(_DENTED_CUBE)
    assert sign_epistasis_fraction(fl) == pytest.approx(3 / 6)
    assert fl.sign_epistasis_fraction() == pytest.approx(0.5)


def test_roughness_to_slope_ratio_nonzero_on_dented_cube() -> None:
    fl = _make_binary_landscape(_DENTED_CUBE)
    assert roughness_to_slope_ratio(fl) == pytest.approx(0.375 / 0.875)
    assert fl.roughness_to_slope_ratio() == pytest.approx(3 / 7)


def test_wildtype_accessibility_fraction_is_fractional_on_dented_cube() -> None:
    fl = _make_binary_landscape(_DENTED_CUBE)
    assert wildtype_accessibility_fraction(fl) == pytest.approx(4 / 6)
    assert fl.wildtype_accessibility_fraction() == pytest.approx(2 / 3)


def _cube_landscape() -> FitnessLandscape:
    cube_net = build_cube_scalar_network()
    df = pl.DataFrame(
        {
            "seq": cube_net.vertex_ids,
            "height": [cube_net._vertex_scalars[v] for v in cube_net.vertex_ids],
        }
    )
    alphabet = Alphabet(("0", "1"))
    hg = HammingGraph(df, sequence_col="seq", alphabet=alphabet, vertex_attribute_cols=("height",))
    return FitnessLandscape(hg, scalar_attr="height")


def test_ruggedness_metrics_on_cube_landscape() -> None:
    fl = _cube_landscape()

    # On cube, all 6 squares exhibit sign epistasis
    assert math.isclose(sign_epistasis_fraction(fl), 1.0)
    # Path from min (001) to max (011) is direct step of distance 1, accessible
    assert math.isclose(wildtype_accessibility_fraction(fl), 1.0)
    # Roughness to slope ratio > 0
    assert roughness_to_slope_ratio(fl) > 0.0


def test_fitness_landscape_summary_integration() -> None:
    fl = _cube_landscape()
    fl_topo = FitnessLandscapeTopology(fl)

    summary_from_fl = fl.summary()
    summary_from_topo = fl_topo.summary()
    summary_direct = compute_topological_summary(fl)

    assert summary_from_fl.equals(summary_from_topo)
    assert summary_from_fl.equals(summary_direct)
    assert summary_from_fl.height == 1

    expected_cols = [
        "n_vertices",
        "n_edges",
        "n_peaks",
        "n_sinks",
        "n_split_saddles",
        "n_join_saddles",
        "max_persistence",
        "total_persistence",
        "persistence_entropy",
        "tree_asymmetry_depth_ratio",
        "tree_asymmetry_persistence_ratio",
        "roughness_to_slope_ratio",
        "sign_epistasis_fraction",
        "wildtype_accessibility_fraction",
    ]
    assert summary_from_fl.columns == expected_cols

    row = summary_from_fl.to_dicts()[0]
    assert row["n_vertices"] == 8
    assert row["n_edges"] == 12
    # Local maxima 011 (1.0) and 110 (0.9); local minima 001 (0.0) and 100 (0.1).
    assert row["n_peaks"] == 2
    assert row["n_sinks"] == 2
    assert math.isclose(row["sign_epistasis_fraction"], 1.0)
    assert math.isclose(row["wildtype_accessibility_fraction"], 1.0)
