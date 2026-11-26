"""Independent diagram-level oracle validation against GUDHI SimplexTree."""

from __future__ import annotations

import pytest

try:
    import gudhi

    HAS_GUDHI = True
except ImportError:
    HAS_GUDHI = False

import polars as pl

from topographer.examples import build_cube_scalar_network
from topographer.graph import Graph, ScalarNetwork
from topographer.topology import ScalarNetworkTopology

pytestmark = pytest.mark.skipif(not HAS_GUDHI, reason="gudhi is required for oracle tests")


def _run_gudhi_extended_persistence(
    vertex_scalars: dict[int, float],
    edges: list[tuple[int, int]],
) -> dict[str, list[tuple[float, float]]]:
    """Run GUDHI extended persistence on a 1D simplicial complex."""
    st = gudhi.SimplexTree()
    for v_idx, val in vertex_scalars.items():
        st.insert([v_idx], val)

    for u, v in edges:
        val = max(vertex_scalars[u], vertex_scalars[v])
        st.insert([u, v], val)

    st.extend_filtration()
    diag = st.extended_persistence(homology_coeff_field=2, min_persistence=-1)
    names = ["ordinary", "relative", "extended_positive", "extended_negative"]
    result: dict[str, list[tuple[float, float]]] = {}
    for name, fam in zip(names, diag):
        # Extract pairs, allowing for GUDHI's floating point representation
        result[name] = [(p[1][0], p[1][1]) for p in fam]
    return result


def test_four_vertex_cycle_gudhi_oracle() -> None:
    # 4-cycle: v0:0, v1:3, v2:1, v3:2
    v_scalars = {0: 0.0, 1: 3.0, 2: 1.0, 3: 2.0}
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]

    gudhi_diag = _run_gudhi_extended_persistence(v_scalars, edges)

    # Filter out zero-persistence pairs from GUDHI (GUDHI emits diagonal pairs for non-critical steps)
    gudhi_non_zero = {
        k: [(b, d) for b, d in pairs if abs(b - d) > 1e-6] for k, pairs in gudhi_diag.items()
    }

    # Topographer execution
    vertices = pl.DataFrame({"id": ["v0", "v1", "v2", "v3"], "val": [0.0, 3.0, 1.0, 2.0]})
    e_df = pl.DataFrame({"src": ["v0", "v1", "v2", "v3"], "dst": ["v1", "v2", "v3", "v0"]})
    g = Graph.from_tables(
        vertices,
        e_df,
        vertex_id_col="id",
        source_id_col="src",
        target_id_col="dst",
        vertex_attribute_cols=("val",),
    )
    sn = ScalarNetwork(g, scalar_attr="val")
    top = ScalarNetworkTopology(sn)

    comp_pairs = top.component_persistence().pairs

    # Verify each family matches GUDHI non-zero points
    # 1. ordinary
    comp_ord = comp_pairs.filter(pl.col("pair_kind") == "ordinary")
    assert len(comp_ord) == len(gudhi_non_zero["ordinary"]) == 1
    ord_row = comp_ord.row(0, named=True)
    assert ord_row["birth_scalar"] == pytest.approx(gudhi_non_zero["ordinary"][0][0])
    assert ord_row["death_scalar"] == pytest.approx(gudhi_non_zero["ordinary"][0][1])

    # 2. relative
    comp_rel = comp_pairs.filter(pl.col("pair_kind") == "relative")
    assert len(comp_rel) == len(gudhi_non_zero["relative"]) == 1
    rel_row = comp_rel.row(0, named=True)
    assert rel_row["birth_scalar"] == pytest.approx(gudhi_non_zero["relative"][0][0])
    assert rel_row["death_scalar"] == pytest.approx(gudhi_non_zero["relative"][0][1])

    # 3. extended_positive
    comp_ext_pos = comp_pairs.filter(pl.col("pair_kind") == "extended_positive")
    assert len(comp_ext_pos) == len(gudhi_non_zero["extended_positive"]) == 1
    pos_row = comp_ext_pos.row(0, named=True)
    assert pos_row["birth_scalar"] == pytest.approx(gudhi_non_zero["extended_positive"][0][0])
    assert pos_row["death_scalar"] == pytest.approx(gudhi_non_zero["extended_positive"][0][1])


def test_cube_gudhi_oracle() -> None:
    sn = build_cube_scalar_network()
    top = ScalarNetworkTopology(sn)

    v_ids = sn.vertex_ids
    vid_to_idx = {vid: i for i, vid in enumerate(v_ids)}
    v_scalars = {i: sn._vertex_scalars[vid] for vid, i in vid_to_idx.items()}
    edges = [
        (vid_to_idx[sn.vertex_ids[u]], vid_to_idx[sn.vertex_ids[v]])
        for u, v in sn._graph_snapshot.edge_list()
    ]

    gudhi_diag = _run_gudhi_extended_persistence(v_scalars, edges)
    gudhi_non_zero = {
        k: [(b, d) for b, d in pairs if abs(b - d) > 1e-6] for k, pairs in gudhi_diag.items()
    }

    comp_pairs = top.component_persistence().pairs

    # Ordinary H0
    comp_ord = comp_pairs.filter(pl.col("pair_kind") == "ordinary")
    assert len(comp_ord) == len(gudhi_non_zero["ordinary"]) == 1
    ord_row = comp_ord.row(0, named=True)
    assert ord_row["birth_scalar"] == pytest.approx(gudhi_non_zero["ordinary"][0][0])
    assert ord_row["death_scalar"] == pytest.approx(gudhi_non_zero["ordinary"][0][1])

    # Relative H1
    comp_rel = comp_pairs.filter(pl.col("pair_kind") == "relative")
    assert len(comp_rel) == len(gudhi_non_zero["relative"]) == 1
    rel_row = comp_rel.row(0, named=True)
    assert rel_row["birth_scalar"] == pytest.approx(gudhi_non_zero["relative"][0][0])
    assert rel_row["death_scalar"] == pytest.approx(gudhi_non_zero["relative"][0][1])

    # Extended positive H0
    comp_ext_pos = comp_pairs.filter(pl.col("pair_kind") == "extended_positive")
    assert len(comp_ext_pos) == len(gudhi_non_zero["extended_positive"]) == 1
    pos_row = comp_ext_pos.row(0, named=True)
    assert pos_row["birth_scalar"] == pytest.approx(gudhi_non_zero["extended_positive"][0][0])
    assert pos_row["death_scalar"] == pytest.approx(gudhi_non_zero["extended_positive"][0][1])


def test_tied_scalar_graph_fixtures_gudhi_oracle() -> None:
    # Fixture 1: 4-cycle with tied scalar values: v0: 0.0, v1: 2.0, v2: 1.0, v3: 2.0
    v_scalars1 = {0: 0.0, 1: 2.0, 2: 1.0, 3: 2.0}
    edges1 = [(0, 1), (1, 2), (2, 3), (3, 0)]
    gudhi_diag1 = _run_gudhi_extended_persistence(v_scalars1, edges1)
    gudhi_non_zero1 = {
        k: [(b, d) for b, d in pairs if abs(b - d) > 1e-6] for k, pairs in gudhi_diag1.items()
    }

    vertices1 = pl.DataFrame({"id": ["v0", "v1", "v2", "v3"], "val": [0.0, 2.0, 1.0, 2.0]})
    e_df1 = pl.DataFrame({"src": ["v0", "v1", "v2", "v3"], "dst": ["v1", "v2", "v3", "v0"]})
    g1 = Graph.from_tables(
        vertices1,
        e_df1,
        vertex_id_col="id",
        source_id_col="src",
        target_id_col="dst",
        vertex_attribute_cols=("val",),
    )
    top1 = ScalarNetworkTopology(ScalarNetwork(g1, scalar_attr="val"))
    comp_pairs1 = top1.component_persistence().pairs

    for kind in ["ordinary", "relative", "extended_positive"]:
        t_pairs = sorted(
            [
                (round(r["birth_scalar"], 6), round(r["death_scalar"], 6))
                for r in comp_pairs1.filter(pl.col("pair_kind") == kind).iter_rows(named=True)
            ]
        )
        g_pairs = sorted([(round(b, 6), round(d, 6)) for b, d in gudhi_non_zero1[kind]])
        assert t_pairs == g_pairs

    # Fixture 2: Theta graph with tied scalar values: v0: 0.0, v1: 0.5, v2: 0.5, v3: 1.0
    v_scalars2 = {0: 0.0, 1: 0.5, 2: 0.5, 3: 1.0}
    edges2 = [(0, 3), (0, 1), (1, 3), (0, 2), (2, 3)]
    gudhi_diag2 = _run_gudhi_extended_persistence(v_scalars2, edges2)
    gudhi_non_zero2 = {
        k: [(b, d) for b, d in pairs if abs(b - d) > 1e-6] for k, pairs in gudhi_diag2.items()
    }

    vertices2 = pl.DataFrame({"id": ["v0", "v1", "v2", "v3"], "val": [0.0, 0.5, 0.5, 1.0]})
    e_df2 = pl.DataFrame(
        {
            "src": ["v0", "v0", "v1", "v0", "v2"],
            "dst": ["v3", "v1", "v3", "v2", "v3"],
        }
    )
    g2 = Graph.from_tables(
        vertices2,
        e_df2,
        vertex_id_col="id",
        source_id_col="src",
        target_id_col="dst",
        vertex_attribute_cols=("val",),
    )
    top2 = ScalarNetworkTopology(ScalarNetwork(g2, scalar_attr="val"))
    comp_pairs2 = top2.component_persistence().pairs

    for kind in ["ordinary", "relative", "extended_positive"]:
        t_pairs = sorted(
            [
                (round(r["birth_scalar"], 6), round(r["death_scalar"], 6))
                for r in comp_pairs2.filter(pl.col("pair_kind") == kind).iter_rows(named=True)
            ]
        )
        g_pairs = sorted([(round(b, 6), round(d, 6)) for b, d in gudhi_non_zero2[kind]])
        assert t_pairs == g_pairs
