"""Round trips and refusals of ScalarNetwork.write and ScalarNetwork.read."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl
import pytest
import rustworkx as rx

from topographer.examples import build_cube_scalar_network
from topographer.graph import Graph, HammingGraph, ScalarNetwork, SequenceGraph
from topographer.statistics import BottleneckToNominal, bootstrap
from topographer.topology import ScalarNetworkTopology
from topographer.topology._band import band_network_fingerprint

# --- builders ------------------------------------------------------------------------------------


def make_network(
    ids: list[Any],
    scalars: list[Any],
    edges: list[tuple[Any, Any]],
    *,
    extra_vertex: dict[str, list[Any]] | None = None,
    extra_edge: dict[str, list[Any]] | None = None,
    edge_ids: list[Any] | None = None,
    multigraph: bool = False,
    edge_scalar_attr: str | None = None,
) -> ScalarNetwork:
    vertex_table = pl.DataFrame({"vid": ids, "value": scalars, **(extra_vertex or {})})
    edge_data: dict[str, list[Any]] = {
        "src": [e[0] for e in edges],
        "dst": [e[1] for e in edges],
        **(extra_edge or {}),
    }
    if edge_ids is not None:
        edge_data["eid"] = edge_ids
    edge_table = pl.DataFrame(
        edge_data,
        schema_overrides=None
        if edges
        else {"src": vertex_table["vid"].dtype, "dst": vertex_table["vid"].dtype},
    )
    graph = Graph.from_tables(
        vertex_table,
        edge_table,
        vertex_id_col="vid",
        source_id_col="src",
        target_id_col="dst",
        edge_id_col="eid" if edge_ids is not None else None,
        vertex_attribute_cols=["value", *(extra_vertex or {})],
        edge_attribute_cols=list(extra_edge or {}),
        multigraph=multigraph,
    )
    return ScalarNetwork(graph, scalar_attr="value", edge_scalar_attr=edge_scalar_attr)


def with_se(ids: list[Any], edges: list[tuple[Any, Any]]) -> ScalarNetwork:
    return make_network(
        ids,
        [0.1, 0.9, 0.4, 0.7][: len(ids)],
        edges,
        extra_vertex={
            "se": [0.01, 0.02, 0.03, 0.04][: len(ids)],
            "label": ["w", "x", "y", "z"][: len(ids)],
            "count": [3, 1, 4, 1][: len(ids)],
            "flag": [True, False, True, False][: len(ids)],
        },
        extra_edge={
            "weight": [0.5, 0.25, 0.75][: len(edges)],
            "kind": ["a", "b", "c"][: len(edges)],
        },
    )


def string_network() -> ScalarNetwork:
    return with_se(["a", "b", "c", "d"], [("c", "a"), ("a", "b"), ("d", "b")])


def int_network() -> ScalarNetwork:
    return with_se([10, 2, 7, 33], [(10, 2), (7, 2), (33, 7)])


def float_network() -> ScalarNetwork:
    return with_se([0.5, 1.5, 2.25, 3.0], [(0.5, 1.5), (2.25, 1.5), (3.0, 2.25)])


def bool_network() -> ScalarNetwork:
    return make_network([False, True], [0.2, 0.8], [(True, False)])


def edge_scalar_network() -> ScalarNetwork:
    return make_network(
        ["a", "b", "c"],
        [0.3, 0.6, 0.1],
        [("a", "b"), ("c", "b")],
        extra_edge={"cost": [0.25, 0.75]},
        edge_scalar_attr="cost",
    )


def multigraph_network() -> ScalarNetwork:
    return make_network(
        ["a", "b", "c"],
        [0.3, 0.6, 0.1],
        [("a", "b"), ("b", "a"), ("c", "b")],
        extra_edge={"w": [1.0, 2.0, 3.0]},
        edge_ids=["e2", "e0", "e1"],
        multigraph=True,
    )


def single_vertex_network() -> ScalarNetwork:
    return make_network(["only"], [0.5], [])


def single_vertex_edge_scalar_network() -> ScalarNetwork:
    return make_network(["only"], [0.5], [], extra_edge={"cost": []}, edge_scalar_attr="cost")


def cube_network() -> ScalarNetwork:
    return build_cube_scalar_network()


NETWORKS = {
    "string": string_network,
    "int": int_network,
    "float": float_network,
    "bool": bool_network,
    "edge_scalar": edge_scalar_network,
    "multigraph": multigraph_network,
    "single_vertex": single_vertex_network,
    "single_vertex_edge_scalar": single_vertex_edge_scalar_network,
    "cube": cube_network,
}


def payloads(network: ScalarNetwork) -> tuple[list[dict], list[dict]]:
    graph = network.graph
    nodes = [graph.get_node_data(i) for i in graph.node_indices()]
    edges = [graph.get_edge_data_by_index(i) for i in graph.edge_indices()]
    return nodes, edges


def typed(values: list[dict]) -> list[list[tuple[str, type, Any]]]:
    return [sorted((k, type(v), v) for k, v in row.items()) for row in values]


def assert_same_network(original: ScalarNetwork, restored: ScalarNetwork) -> None:
    assert type(restored) is type(original)
    assert restored.vertex_ids == original.vertex_ids
    assert [type(v) for v in restored.vertex_ids] == [type(v) for v in original.vertex_ids]
    assert restored.scalar_attr == original.scalar_attr
    assert restored.edge_scalar_attr == original.edge_scalar_attr
    assert restored.vertex_id_attr == original.vertex_id_attr
    assert restored.edge_id_attr == original.edge_id_attr
    assert restored.graph.multigraph == original.graph.multigraph
    assert restored.graph.edge_list() == original.graph.edge_list()
    node_a, edge_a = payloads(original)
    node_b, edge_b = payloads(restored)
    assert typed(node_b) == typed(node_a)
    assert typed(edge_b) == typed(edge_a)


# --- round trips ---------------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(NETWORKS))
def test_round_trip_preserves_everything(tmp_path: Path, name: str) -> None:
    original = NETWORKS[name]()
    original.write(tmp_path / "net")

    restored = ScalarNetwork.read(tmp_path / "net")

    assert_same_network(original, restored)


@pytest.mark.parametrize("name", ["string", "int", "float", "multigraph", "cube"])
def test_round_trip_is_stable_when_written_again(tmp_path: Path, name: str) -> None:
    original = NETWORKS[name]()
    original.write(tmp_path / "one")
    ScalarNetwork.read(tmp_path / "one").write(tmp_path / "two")

    assert_same_network(original, ScalarNetwork.read(tmp_path / "two"))
    for table in ("vertices.parquet", "edges.parquet"):
        assert pl.read_parquet(tmp_path / "one" / table).equals(
            pl.read_parquet(tmp_path / "two" / table)
        )


@pytest.mark.parametrize("name", ["string", "int", "float"])
def test_round_trip_keeps_both_fingerprints(tmp_path: Path, name: str) -> None:
    original = NETWORKS[name]()
    original.write(tmp_path / "net")
    restored = ScalarNetwork.read(tmp_path / "net")

    statistic = BottleneckToNominal("superlevel")
    assert bootstrap.fingerprint(restored, statistic.prepare(restored)) == bootstrap.fingerprint(
        original, statistic.prepare(original)
    )
    assert band_network_fingerprint(restored, se_attr="se") == band_network_fingerprint(
        original, se_attr="se"
    )


def test_edge_order_survives_a_round_trip(tmp_path: Path) -> None:
    original = string_network()
    original.write(tmp_path / "net")
    restored = ScalarNetwork.read(tmp_path / "net")

    assert original.graph.edge_list() == [(2, 0), (0, 1), (3, 1)]
    assert restored.graph.edge_list() == original.graph.edge_list()


def test_restored_network_is_usable(tmp_path: Path) -> None:
    original = string_network()
    original.write(tmp_path / "net")
    restored = ScalarNetwork.read(tmp_path / "net")

    expected = ScalarNetworkTopology(original).persistence_diagram("superlevel")
    actual = ScalarNetworkTopology(restored).persistence_diagram("superlevel")
    assert actual.pairs.equals(expected.pairs)


def test_manifest_contents(tmp_path: Path) -> None:
    edge_scalar_network().write(tmp_path / "net")

    text = (tmp_path / "net" / "manifest.json").read_text()
    manifest = json.loads(text)
    assert text == json.dumps(manifest, sort_keys=True, indent=2)
    assert manifest["format"] == "topographer.scalar_network"
    assert manifest["format_version"] == 1
    assert isinstance(manifest["topographer_version"], str)
    assert manifest["network_class"] == "ScalarNetwork"
    assert manifest["scalar_attr"] == "value"
    assert manifest["edge_scalar_attr"] == "cost"
    assert manifest["vertex_id_attr"] == "vid"
    assert manifest["edge_id_attr"] is None
    assert manifest["multigraph"] is False
    assert (manifest["vertex_count"], manifest["edge_count"]) == (3, 2)
    assert manifest["vertex_attributes"] == ["value"]
    assert manifest["edge_attributes"] == ["cost"]
    assert sorted(p.name for p in (tmp_path / "net").iterdir()) == [
        "edges.parquet",
        "manifest.json",
        "vertices.parquet",
    ]


def test_tables_hold_native_ids_and_no_generated_edge_id(tmp_path: Path) -> None:
    string_network().write(tmp_path / "net")

    vertices = pl.read_parquet(tmp_path / "net" / "vertices.parquet")
    edges = pl.read_parquet(tmp_path / "net" / "edges.parquet")
    assert vertices["vid"].to_list() == ["a", "b", "c", "d"]
    assert edges.columns[:2] == ["source_id", "target_id"]
    assert "edge_id" not in edges.columns
    assert edges["source_id"].to_list() == ["c", "a", "d"]


def test_declared_edge_id_column_is_stored(tmp_path: Path) -> None:
    multigraph_network().write(tmp_path / "net")

    edges = pl.read_parquet(tmp_path / "net" / "edges.parquet")
    assert edges.columns[:3] == ["source_id", "target_id", "eid"]
    assert edges["eid"].to_list() == ["e2", "e0", "e1"]


# --- new properties ------------------------------------------------------------------------------


def test_identity_attribute_names_from_each_constructor_path() -> None:
    assert string_network().vertex_id_attr == "vid"
    assert string_network().edge_id_attr is None
    assert multigraph_network().edge_id_attr == "eid"

    rx_graph = rx.PyGraph()
    rx_graph.add_node({"name": "a", "value": 0.1})
    rx_graph.add_node({"name": "b", "value": 0.2})
    rx_graph.add_edge(0, 1, {"key": "k", "w": 1.0})
    graph = Graph.from_rustworkx(
        rx_graph,
        vertex_id_attr="name",
        edge_id_attr="key",
        vertex_attribute_cols=["value"],
        edge_attribute_cols=["w"],
    )
    network = ScalarNetwork(graph, scalar_attr="value")
    assert (network.vertex_id_attr, network.edge_id_attr) == ("name", "key")

    table = pl.DataFrame({"seq": ["AA", "AC", "CC"], "fit": [0.1, 0.9, 0.5]})
    from topographer.biology import DNA_ALPHABET

    hamming = HammingGraph(
        table, sequence_col="seq", alphabet=DNA_ALPHABET, vertex_attribute_cols=("fit",)
    )
    assert ScalarNetwork(hamming, scalar_attr="fit").vertex_id_attr == "sequence"
    assert ScalarNetwork(hamming, scalar_attr="fit").edge_id_attr is None

    sequence = SequenceGraph(
        pl.DataFrame({"vid": ["v1", "v2"], "seq": ["AA", "AC"], "fit": [0.1, 0.9]}),
        pl.DataFrame({"s": ["v1"], "t": ["v2"]}),
        vertex_id_col="vid",
        sequence_col="seq",
        source_id_col="s",
        target_id_col="t",
        alphabet=DNA_ALPHABET,
        vertex_attribute_cols=("fit",),
    )
    assert ScalarNetwork(sequence, scalar_attr="fit").vertex_id_attr == "vid"


def test_example_networks_carry_identity_names() -> None:
    cube = build_cube_scalar_network()
    assert cube.vertex_id_attr == "node_id"
    assert cube.edge_id_attr is None


def test_hamming_network_round_trips(tmp_path: Path) -> None:
    from topographer.biology import DNA_ALPHABET

    table = pl.DataFrame({"seq": ["AA", "AC", "CC"], "fit": [0.1, 0.9, 0.5]})
    hamming = HammingGraph(
        table, sequence_col="seq", alphabet=DNA_ALPHABET, vertex_attribute_cols=("fit",)
    )
    original = ScalarNetwork(hamming, scalar_attr="fit")
    original.write(tmp_path / "net")

    assert_same_network(original, ScalarNetwork.read(tmp_path / "net"))


# --- refusals ------------------------------------------------------------------------------------


def test_write_refuses_existing_path_and_changes_nothing(tmp_path: Path) -> None:
    target = tmp_path / "net"
    target.mkdir()
    (target / "keep.txt").write_text("keep")

    with pytest.raises(FileExistsError):
        string_network().write(target)

    assert sorted(p.name for p in target.iterdir()) == ["keep.txt"]
    assert (target / "keep.txt").read_text() == "keep"


def test_write_does_not_create_missing_parents(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        string_network().write(tmp_path / "missing" / "net")


def test_write_rejects_tuple_vertex_ids_before_creating_anything(tmp_path: Path) -> None:
    network = _tuple_id_network()

    with pytest.raises(TypeError, match="vertex identities"):
        network.write(tmp_path / "net")
    assert not (tmp_path / "net").exists()


def _tuple_id_network() -> ScalarNetwork:
    rx_graph = rx.PyGraph()
    rx_graph.add_node({"name": (0, 1), "value": 0.1})
    rx_graph.add_node({"name": (1, 2), "value": 0.2})
    rx_graph.add_edge(0, 1, {})
    graph = Graph.from_rustworkx(rx_graph, vertex_id_attr="name", vertex_attribute_cols=["value"])
    return ScalarNetwork(graph, scalar_attr="value")


def test_write_rejects_non_finite_float_vertex_ids(tmp_path: Path) -> None:
    network = make_network([1.0, float("inf")], [0.1, 0.2], [(1.0, float("inf"))])

    with pytest.raises(TypeError, match="finite"):
        network.write(tmp_path / "net")
    assert not (tmp_path / "net").exists()


def test_write_rejects_non_scalar_edge_ids(tmp_path: Path) -> None:
    network = _tuple_edge_id_network()

    with pytest.raises(TypeError, match="edge identities"):
        network.write(tmp_path / "net")
    assert not (tmp_path / "net").exists()


def _tuple_edge_id_network() -> ScalarNetwork:
    rx_graph = rx.PyGraph()
    rx_graph.add_node({"name": "a", "value": 0.1})
    rx_graph.add_node({"name": "b", "value": 0.2})
    rx_graph.add_edge(0, 1, {"key": ("a", "b")})
    graph = Graph.from_rustworkx(
        rx_graph, vertex_id_attr="name", edge_id_attr="key", vertex_attribute_cols=["value"]
    )
    return ScalarNetwork(graph, scalar_attr="value")


def _object_attr_network(attr_value: Any, *, on_edge: bool = False) -> ScalarNetwork:
    rx_graph = rx.PyGraph()
    rx_graph.add_node({"name": "a", "value": 0.1, "extra": 1 if on_edge else attr_value})
    rx_graph.add_node({"name": "b", "value": 0.2, "extra": 2 if on_edge else attr_value})
    rx_graph.add_edge(0, 1, {"extra": attr_value if on_edge else 1})
    graph = Graph.from_rustworkx(
        rx_graph,
        vertex_id_attr="name",
        vertex_attribute_cols=["value", "extra"],
        edge_attribute_cols=["extra"],
    )
    return ScalarNetwork(graph, scalar_attr="value")


@pytest.mark.parametrize("on_edge", [False, True])
def test_write_rejects_object_typed_attributes(tmp_path: Path, on_edge: bool) -> None:
    network = _object_attr_network(object(), on_edge=on_edge)

    with pytest.raises(TypeError, match="extra"):
        network.write(tmp_path / "net")
    assert not (tmp_path / "net").exists()


def test_write_rejects_mixed_type_attribute_columns(tmp_path: Path) -> None:
    rx_graph = rx.PyGraph()
    rx_graph.add_node({"name": "a", "value": 0.1, "extra": 1})
    rx_graph.add_node({"name": "b", "value": 0.2, "extra": 1.5})
    rx_graph.add_edge(0, 1, {})
    graph = Graph.from_rustworkx(
        rx_graph, vertex_id_attr="name", vertex_attribute_cols=["value", "extra"]
    )

    with pytest.raises(TypeError, match="extra"):
        ScalarNetwork(graph, scalar_attr="value").write(tmp_path / "net")
    assert not (tmp_path / "net").exists()


@pytest.mark.parametrize("name", ["source_id", "target_id"])
def test_write_rejects_reserved_attribute_names(tmp_path: Path, name: str) -> None:
    network = make_network(["a", "b"], [0.1, 0.2], [("a", "b")], extra_edge={name: [1.0]})

    with pytest.raises(TypeError, match=name):
        network.write(tmp_path / "net")
    assert not (tmp_path / "net").exists()


def written(tmp_path: Path) -> Path:
    string_network().write(tmp_path / "net")
    return tmp_path / "net"


def edit_manifest(folder: Path, **changes: Any) -> None:
    path = folder / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest.update(changes)
    path.write_text(json.dumps(manifest, sort_keys=True, indent=2))


def test_read_rejects_a_missing_folder(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not found|does not exist"):
        ScalarNetwork.read(tmp_path / "nope")


def test_read_rejects_a_missing_manifest(tmp_path: Path) -> None:
    folder = written(tmp_path)
    (folder / "manifest.json").unlink()

    with pytest.raises(ValueError, match="manifest"):
        ScalarNetwork.read(folder)


def test_read_rejects_malformed_manifest(tmp_path: Path) -> None:
    folder = written(tmp_path)
    (folder / "manifest.json").write_text("{not json")

    with pytest.raises(ValueError, match="manifest"):
        ScalarNetwork.read(folder)


def test_read_rejects_a_wrong_format_and_names_it(tmp_path: Path) -> None:
    folder = written(tmp_path)
    edit_manifest(folder, format="topographer.workspace")

    with pytest.raises(ValueError, match="topographer.workspace"):
        ScalarNetwork.read(folder)


@pytest.mark.parametrize("version", [2, 0, "1", True, 1.0])
def test_read_rejects_other_format_versions(tmp_path: Path, version: Any) -> None:
    folder = written(tmp_path)
    edit_manifest(folder, format_version=version)

    with pytest.raises(ValueError, match="format_version"):
        ScalarNetwork.read(folder)


@pytest.mark.parametrize("table", ["vertices.parquet", "edges.parquet"])
def test_read_rejects_a_deleted_table(tmp_path: Path, table: str) -> None:
    folder = written(tmp_path)
    (folder / table).unlink()

    with pytest.raises(ValueError, match=table):
        ScalarNetwork.read(folder)


@pytest.mark.parametrize(
    ("table", "column"),
    [("vertices.parquet", "se"), ("vertices.parquet", "vid"), ("edges.parquet", "target_id")],
)
def test_read_rejects_a_dropped_column(tmp_path: Path, table: str, column: str) -> None:
    folder = written(tmp_path)
    pl.read_parquet(folder / table).drop(column).write_parquet(folder / table)

    with pytest.raises(ValueError, match=column):
        ScalarNetwork.read(folder)


@pytest.mark.parametrize("key", ["vertex_count", "edge_count"])
def test_read_rejects_count_disagreement(tmp_path: Path, key: str) -> None:
    folder = written(tmp_path)
    edit_manifest(folder, **{key: 99})

    with pytest.raises(ValueError, match=key.replace("_", " ")):
        ScalarNetwork.read(folder)


def test_read_runs_normal_validation_again(tmp_path: Path) -> None:
    folder = written(tmp_path)
    edges = pl.read_parquet(folder / "edges.parquet").head(1)
    edges.write_parquet(folder / "edges.parquet")
    edit_manifest(folder, edge_count=1)

    with pytest.raises(ValueError, match="connected"):
        ScalarNetwork.read(folder)


def test_failed_write_removes_the_created_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*args: Any, **kwargs: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(pl.DataFrame, "write_parquet", boom)
    with pytest.raises(OSError):
        string_network().write(tmp_path / "net")

    assert not (tmp_path / "net").exists()


# --- null attribute values -----------------------------------------------------------------------


def _null_attr_network() -> ScalarNetwork:
    rx_graph = rx.PyGraph(multigraph=False)
    rx_graph.add_node(
        {"name": "a", "value": 0.1, "s": "x", "i": 1, "f": 0.5, "b": True, "n": None}
    )
    rx_graph.add_node(
        {"name": "b", "value": 0.2, "s": None, "i": None, "f": None, "b": None, "n": None}
    )
    rx_graph.add_node(
        {"name": "c", "value": 0.3, "s": "z", "i": 3, "f": 1.5, "b": False, "n": None}
    )
    rx_graph.add_edge(0, 1, {"w": None, "k": "p"})
    rx_graph.add_edge(1, 2, {"w": 2.0, "k": None})
    graph = Graph.from_rustworkx(
        rx_graph,
        vertex_id_attr="name",
        vertex_attribute_cols=["value", "s", "i", "f", "b", "n"],
        edge_attribute_cols=["w", "k"],
    )
    return ScalarNetwork(graph, scalar_attr="value")


def test_null_attribute_values_round_trip_as_none(tmp_path: Path) -> None:
    original = _null_attr_network()
    original.write(tmp_path / "net")

    restored = ScalarNetwork.read(tmp_path / "net")

    assert_same_network(original, restored)
    nodes, edges = payloads(restored)
    assert [n["s"] for n in nodes] == ["x", None, "z"]
    assert [n["i"] for n in nodes] == [1, None, 3]
    assert [n["n"] for n in nodes] == [None, None, None]
    assert [e["w"] for e in edges] == [None, 2.0]


def test_null_columns_take_their_type_from_non_null_values(tmp_path: Path) -> None:
    _null_attr_network().write(tmp_path / "net")

    vertices = pl.read_parquet(tmp_path / "net" / "vertices.parquet")
    assert vertices["i"].dtype == pl.Int64
    assert vertices["s"].dtype == pl.String
    assert vertices["b"].dtype == pl.Boolean
    assert vertices["i"].null_count() == 1
    assert vertices["n"].null_count() == 3


def test_null_attribute_with_mixed_non_null_types_is_still_refused(tmp_path: Path) -> None:
    rx_graph = rx.PyGraph()
    rx_graph.add_node({"name": "a", "value": 0.1, "extra": 1})
    rx_graph.add_node({"name": "b", "value": 0.2, "extra": 1.5})
    rx_graph.add_node({"name": "c", "value": 0.3, "extra": None})
    rx_graph.add_edge(0, 1, {})
    rx_graph.add_edge(1, 2, {})
    graph = Graph.from_rustworkx(
        rx_graph, vertex_id_attr="name", vertex_attribute_cols=["value", "extra"]
    )

    with pytest.raises(TypeError, match="extra"):
        ScalarNetwork(graph, scalar_attr="value").write(tmp_path / "net")


@pytest.mark.parametrize(
    ("table", "column"),
    [
        ("vertices.parquet", "vid"),
        ("vertices.parquet", "value"),
        ("edges.parquet", "source_id"),
        ("edges.parquet", "cost"),
    ],
)
def test_read_rejects_nulls_in_identity_endpoint_and_scalar_columns(
    tmp_path: Path, table: str, column: str
) -> None:
    edge_scalar_network().write(tmp_path / "net")
    path = tmp_path / "net" / table
    frame = pl.read_parquet(path)
    values = frame[column].to_list()
    values[0] = None
    frame.with_columns(pl.Series(column, values, dtype=frame[column].dtype)).write_parquet(path)

    with pytest.raises(ValueError, match=f"{column}.*null"):
        ScalarNetwork.read(tmp_path / "net")


def test_public_from_tables_still_refuses_nulls() -> None:
    with pytest.raises(ValueError, match="Null values"):
        Graph.from_tables(
            pl.DataFrame({"vid": ["a", "b"], "x": [1, None]}),
            pl.DataFrame({"s": ["a"], "t": ["b"]}),
            vertex_id_col="vid",
            source_id_col="s",
            target_id_col="t",
            vertex_attribute_cols=["x"],
        )
