"""Round trips and refusals of MergeTree.write and MergeTree.read."""

from __future__ import annotations

from collections.abc import Callable
import json
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from topographer.examples import build_cube_scalar_network, build_two_peak_terrain_scalar_network
from topographer.graph import ScalarNetwork
from topographer.topology import MergeTree, ScalarNetworkTopology
import topographer.topology._merge_tree as merge_tree_module

NETWORKS: dict[str, Callable[[], ScalarNetwork]] = {
    "cube": build_cube_scalar_network,
    "terrain": build_two_peak_terrain_scalar_network,
}
KINDS = ("join", "split")


def make_tree(name: str, kind: str) -> MergeTree:
    topology = ScalarNetworkTopology(NETWORKS[name]())
    return topology.join_tree() if kind == "join" else topology.split_tree()


def assert_same_graph(a: Any, b: Any) -> None:
    assert a.nodes() == b.nodes()
    assert a.weighted_edge_list() == b.weighted_edge_list()
    assert typed(a) == typed(b)


def typed(graph: Any) -> list[list[tuple[str, type]]]:
    payloads = [*graph.nodes(), *(data for _, _, data in graph.weighted_edge_list())]
    return [sorted((key, type(value)) for key, value in payload.items()) for payload in payloads]


def assert_same_tree(original: MergeTree, restored: MergeTree) -> None:
    assert restored.kind == original.kind
    assert restored.scalar_attr == original.scalar_attr
    assert_same_graph(restored.graph, original.graph)
    assert_same_graph(restored.augmented_graph, original.augmented_graph)
    assert restored.events.equals(original.events)
    assert restored.mapping.branch_vertices.equals(original.mapping.branch_vertices)
    assert restored.mapping.critical_vertices.equals(original.mapping.critical_vertices)
    assert restored.persistence().equals(original.persistence())
    assert restored.spans().equals(original.spans())


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("name", sorted(NETWORKS))
def test_round_trip_preserves_everything(tmp_path: Path, name: str, kind: str) -> None:
    original = make_tree(name, kind)
    original.write(tmp_path / "tree")

    restored = MergeTree.read(tmp_path / "tree")

    assert_same_tree(original, restored)
    assert restored.original is None
    assert restored.simplifications == {}


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("name", sorted(NETWORKS))
def test_round_trip_with_two_simplifications(tmp_path: Path, name: str, kind: str) -> None:
    original = make_tree(name, kind)
    thresholds = [0.05, 0.15, 0.25]
    for threshold in thresholds[:2]:
        original.simplify(threshold)
    original.write(tmp_path / "tree")

    restored = MergeTree.read(tmp_path / "tree")

    assert sorted(restored.simplifications) == sorted(original.simplifications)
    assert list(restored.simplifications) == list(original.simplifications)
    for threshold, expected in original.simplifications.items():
        variant = restored.simplify(threshold)
        assert variant is restored.simplifications[threshold]
        assert variant.original is restored
        assert variant.kind == kind
        assert_same_graph(variant.graph, expected.graph)
        assert variant.mapping.critical_vertices.equals(expected.mapping.critical_vertices)
        assert variant.persistence().equals(expected.persistence())
        assert variant.augmented_graph.nodes() == expected.augmented_graph.nodes()
    fresh = make_tree(name, kind)
    assert_same_graph(restored.simplify(thresholds[2]).graph, fresh.simplify(thresholds[2]).graph)


def test_simplified_variants_that_actually_prune_round_trip(tmp_path: Path) -> None:
    original = make_tree("cube", "split")
    variant = original.simplify(0.15)
    assert variant.graph.num_nodes() < original.graph.num_nodes()
    original.write(tmp_path / "tree")

    restored = MergeTree.read(tmp_path / "tree")

    assert_same_graph(restored.simplify(0.15).graph, variant.graph)


def test_reading_runs_no_construction_or_reduction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = make_tree("terrain", "join")
    original.simplify(0.1)
    original.write(tmp_path / "tree")

    def boom(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("recomputed")

    for name in ("build_join_tree", "build_split_tree", "reduce_merge_tree"):
        monkeypatch.setattr(merge_tree_module, name, boom)
    monkeypatch.setattr(merge_tree_module, "construct_oriented_merge_tree", boom)

    restored = MergeTree.read(tmp_path / "tree")

    assert_same_tree(original, restored)
    assert 0.1 in restored.simplifications


def test_writing_a_simplified_tree_is_refused(tmp_path: Path) -> None:
    variant = make_tree("cube", "split").simplify(0.15)

    with pytest.raises(ValueError, match="original"):
        variant.write(tmp_path / "tree")
    assert not (tmp_path / "tree").exists()


def test_existing_path_is_refused_and_untouched(tmp_path: Path) -> None:
    (tmp_path / "tree").mkdir()
    (tmp_path / "tree" / "keep.txt").write_text("keep")

    with pytest.raises(FileExistsError):
        make_tree("cube", "join").write(tmp_path / "tree")
    assert [p.name for p in (tmp_path / "tree").iterdir()] == ["keep.txt"]


def test_layout_and_manifest(tmp_path: Path) -> None:
    tree = make_tree("cube", "split")
    tree.simplify(0.15)
    tree.write(tmp_path / "tree")

    folder = tmp_path / "tree"
    assert sorted(p.name for p in folder.iterdir()) == [
        "augmented_edges.parquet",
        "augmented_nodes.parquet",
        "edges.parquet",
        "events.parquet",
        "manifest.json",
        "nodes.parquet",
        "simplifications.parquet",
    ]
    manifest = json.loads((folder / "manifest.json").read_text())
    assert manifest["format"] == "topographer.merge_tree"
    assert manifest["format_version"] == 1
    assert manifest["kind"] == "split"
    assert manifest["scalar_attr"] == tree.scalar_attr
    assert manifest["vertex_count"] == 8
    assert manifest["tree_node_count"] == 4
    assert manifest["simplification_thresholds"] == [0.15]
    nodes = pl.read_parquet(folder / "nodes.parquet")
    assert nodes.columns[0] == "vertex_index"
    assert nodes["vertex_index"].dtype == pl.Int64
    assert pl.read_parquet(folder / "simplifications.parquet").schema == {
        "threshold": pl.Float64,
        "vertex_index": pl.Int64,
    }
    assert "source_vertex_index" in pl.read_parquet(folder / "edges.parquet").columns


def test_no_simplifications_gives_an_empty_table(tmp_path: Path) -> None:
    make_tree("cube", "join").write(tmp_path / "tree")

    assert pl.read_parquet(tmp_path / "tree" / "simplifications.parquet").height == 0


def test_second_write_of_a_restored_tree_is_identical(tmp_path: Path) -> None:
    original = make_tree("terrain", "split")
    original.simplify(0.1)
    original.write(tmp_path / "one")
    MergeTree.read(tmp_path / "one").write(tmp_path / "two")

    for table in (
        "nodes",
        "edges",
        "augmented_nodes",
        "augmented_edges",
        "events",
        "simplifications",
    ):
        assert pl.read_parquet(tmp_path / "one" / f"{table}.parquet").equals(
            pl.read_parquet(tmp_path / "two" / f"{table}.parquet")
        )


# --- validation ----------------------------------------------------------------------------------


def written(tmp_path: Path, *, simplify: bool = False, kind: str = "split") -> Path:
    tree = make_tree("terrain", kind)
    if simplify:
        tree.simplify(0.1)
    tree.write(tmp_path / "tree")
    return tmp_path / "tree"


def edit_manifest(folder: Path, **changes: Any) -> None:
    path = folder / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest.update(changes)
    path.write_text(json.dumps(manifest, sort_keys=True, indent=2))


def edit_table(folder: Path, name: str, edit: Callable[[pl.DataFrame], pl.DataFrame]) -> None:
    path = folder / f"{name}.parquet"
    edit(pl.read_parquet(path)).write_parquet(path)


def test_read_rejects_a_missing_folder_manifest_and_wrong_format(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not found"):
        MergeTree.read(tmp_path / "nope")
    folder = written(tmp_path)
    edit_manifest(folder, format="topographer.scalar_network")
    with pytest.raises(ValueError, match="topographer.scalar_network"):
        MergeTree.read(folder)
    edit_manifest(folder, format="topographer.merge_tree", format_version=2)
    with pytest.raises(ValueError, match="format_version"):
        MergeTree.read(folder)
    (folder / "manifest.json").unlink()
    with pytest.raises(ValueError, match="manifest"):
        MergeTree.read(folder)


@pytest.mark.parametrize(
    "key", ["kind", "scalar_attr", "vertex_count", "tree_node_count", "simplification_thresholds"]
)
def test_read_rejects_missing_manifest_keys(tmp_path: Path, key: str) -> None:
    folder = written(tmp_path)
    manifest = json.loads((folder / "manifest.json").read_text())
    del manifest[key]
    (folder / "manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match=key):
        MergeTree.read(folder)


@pytest.mark.parametrize(
    "table",
    ["nodes", "edges", "augmented_nodes", "augmented_edges", "events", "simplifications"],
)
def test_read_rejects_a_missing_table(tmp_path: Path, table: str) -> None:
    folder = written(tmp_path)
    (folder / f"{table}.parquet").unlink()

    with pytest.raises(ValueError, match=table):
        MergeTree.read(folder)


@pytest.mark.parametrize(
    ("table", "column"),
    [
        ("nodes", "vertex_index"),
        ("nodes", "event"),
        ("edges", "source_vertex_index"),
        ("augmented_edges", "target_vertex_index"),
        ("events", "active_neighbor_components"),
        ("simplifications", "threshold"),
    ],
)
def test_read_rejects_a_dropped_column(tmp_path: Path, table: str, column: str) -> None:
    folder = written(tmp_path)
    edit_table(folder, table, lambda frame: frame.drop(column))

    with pytest.raises(ValueError, match=column):
        MergeTree.read(folder)


def test_read_rejects_count_disagreement(tmp_path: Path) -> None:
    folder = written(tmp_path)
    edit_manifest(folder, tree_node_count=99)
    with pytest.raises(ValueError, match="tree_node_count"):
        MergeTree.read(folder)


def test_read_rejects_vertex_count_disagreement(tmp_path: Path) -> None:
    folder = written(tmp_path)
    edit_manifest(folder, vertex_count=99)

    with pytest.raises(ValueError, match="vertex_count"):
        MergeTree.read(folder)


def test_read_rejects_an_invalid_kind(tmp_path: Path) -> None:
    folder = written(tmp_path)
    edit_manifest(folder, kind="sideways")

    with pytest.raises(ValueError, match="kind"):
        MergeTree.read(folder)


def test_read_rejects_a_wrong_kind_for_the_event_labels(tmp_path: Path) -> None:
    folder = written(tmp_path, kind="split")
    edit_manifest(folder, kind="join")

    with pytest.raises(ValueError, match="event labels"):
        MergeTree.read(folder)


def test_read_rejects_event_labels_that_do_not_fit_the_kind_in_events(tmp_path: Path) -> None:
    folder = written(tmp_path, kind="split")
    edit_table(
        folder,
        "events",
        lambda frame: frame.with_columns(
            pl.when(pl.col("event") == "maximum")
            .then(pl.lit("minimum"))
            .otherwise(pl.col("event"))
            .alias("event")
        ),
    )

    with pytest.raises(ValueError, match="event labels"):
        MergeTree.read(folder)


def test_read_rejects_a_missing_edge(tmp_path: Path) -> None:
    folder = written(tmp_path)
    edit_table(folder, "edges", lambda frame: frame.head(frame.height - 1))

    with pytest.raises(ValueError, match="edges for"):
        MergeTree.read(folder)


def test_read_rejects_an_edge_to_an_unknown_vertex(tmp_path: Path) -> None:
    folder = written(tmp_path)
    edit_table(
        folder,
        "edges",
        lambda frame: frame.with_columns(
            pl.when(pl.int_range(pl.len()) == 0)
            .then(pl.lit(10_000))
            .otherwise(pl.col("target_vertex_index"))
            .alias("target_vertex_index")
        ),
    )

    with pytest.raises(ValueError, match="unknown vertex_index"):
        MergeTree.read(folder)


def test_read_rejects_edges_pointing_toward_the_root(tmp_path: Path) -> None:
    folder = written(tmp_path)
    edit_table(
        folder,
        "edges",
        lambda frame: frame.rename(
            {
                "source_vertex_index": "target_vertex_index",
                "target_vertex_index": "source_vertex_index",
            }
        ),
    )

    with pytest.raises(ValueError, match="one root|point from the root"):
        MergeTree.read(folder)


def test_read_rejects_two_roots_with_the_right_edge_count(tmp_path: Path) -> None:
    folder = written(tmp_path)

    def two_roots(frame: pl.DataFrame) -> pl.DataFrame:
        # Point the last edge at its own source: edge count stays, one node loses its parent.
        last = frame.height - 1
        return frame.with_columns(
            pl.when(pl.int_range(pl.len()) == last)
            .then(pl.col("source_vertex_index"))
            .otherwise(pl.col("target_vertex_index"))
            .alias("target_vertex_index")
        )

    edit_table(folder, "edges", two_roots)

    with pytest.raises(ValueError, match="root|connected|point from"):
        MergeTree.read(folder)


def test_read_rejects_duplicate_vertex_indices_in_the_augmented_nodes(tmp_path: Path) -> None:
    folder = written(tmp_path)
    edit_table(
        folder,
        "augmented_nodes",
        lambda frame: frame.with_columns(
            pl.when(pl.int_range(pl.len()) == 1)
            .then(pl.col("vertex_index").first())
            .otherwise(pl.col("vertex_index"))
            .alias("vertex_index")
        ),
    )

    with pytest.raises(ValueError, match="duplicate"):
        MergeTree.read(folder)


def test_read_rejects_an_augmented_vertex_index_out_of_range(tmp_path: Path) -> None:
    folder = written(tmp_path)
    edit_table(
        folder,
        "augmented_nodes",
        lambda frame: frame.with_columns(
            pl.when(pl.int_range(pl.len()) == 0)
            .then(pl.lit(10_000))
            .otherwise(pl.col("vertex_index"))
            .alias("vertex_index")
        ),
    )
    edit_table(folder, "augmented_edges", lambda frame: frame.clear())

    with pytest.raises(ValueError, match="edge|cover|unknown"):
        MergeTree.read(folder)


def test_read_rejects_events_that_do_not_cover_the_vertices(tmp_path: Path) -> None:
    folder = written(tmp_path)
    edit_table(
        folder,
        "events",
        lambda frame: frame.with_columns(
            pl.when(pl.int_range(pl.len()) == 0)
            .then(pl.col("vertex_index").last())
            .otherwise(pl.col("vertex_index"))
            .alias("vertex_index")
        ),
    )

    with pytest.raises(ValueError, match="events"):
        MergeTree.read(folder)


def test_read_rejects_a_nonpositive_threshold(tmp_path: Path) -> None:
    folder = written(tmp_path, simplify=True)
    edit_table(
        folder,
        "simplifications",
        lambda frame: frame.with_columns(pl.lit(-0.1).alias("threshold")),
    )
    edit_manifest(folder, simplification_thresholds=[-0.1])

    with pytest.raises(ValueError, match="positive"):
        MergeTree.read(folder)


def test_read_rejects_a_simplified_node_outside_the_reduced_tree(tmp_path: Path) -> None:
    folder = written(tmp_path, simplify=True)
    edit_table(
        folder,
        "simplifications",
        lambda frame: frame.with_columns(
            pl.when(pl.int_range(pl.len()) == 0)
            .then(pl.lit(10_000))
            .otherwise(pl.col("vertex_index"))
            .alias("vertex_index")
        ),
    )

    with pytest.raises(ValueError, match="not a node of the reduced tree"):
        MergeTree.read(folder)


def test_read_rejects_thresholds_that_disagree_with_the_manifest(tmp_path: Path) -> None:
    folder = written(tmp_path, simplify=True)
    edit_manifest(folder, simplification_thresholds=[0.1, 0.2])
    with pytest.raises(ValueError, match="simplification_thresholds"):
        MergeTree.read(folder)
    edit_manifest(folder, simplification_thresholds=[0.3])
    with pytest.raises(ValueError, match="simplification_thresholds"):
        MergeTree.read(folder)


def test_read_rejects_a_simplification_that_is_not_a_tree(tmp_path: Path) -> None:
    folder = written(tmp_path, simplify=True)
    nodes = pl.read_parquet(folder / "nodes.parquet")
    assert nodes.height > 2
    # One kept node of many is still a tree, so keep two non-adjacent leaves instead.
    leaves = pl.read_parquet(folder / "edges.parquet")
    kept = sorted(
        set(nodes["vertex_index"].to_list()) - set(leaves["source_vertex_index"].to_list())
    )
    edit_table(
        folder,
        "simplifications",
        lambda frame: pl.DataFrame(
            {"threshold": [0.1] * len(kept), "vertex_index": kept},
            schema={"threshold": pl.Float64, "vertex_index": pl.Int64},
        ),
    )

    with pytest.raises(ValueError, match="not a rooted tree"):
        MergeTree.read(folder)


def test_read_rejects_nulls_in_tables(tmp_path: Path) -> None:
    folder = written(tmp_path)
    edit_table(
        folder,
        "nodes",
        lambda frame: frame.with_columns(
            pl.when(pl.int_range(pl.len()) == 0)
            .then(None)
            .otherwise(pl.col("scalar"))
            .alias("scalar")
        ),
    )

    with pytest.raises(ValueError, match="nulls"):
        MergeTree.read(folder)


def test_failed_write_removes_the_created_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*args: Any, **kwargs: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(pl.DataFrame, "write_parquet", boom)
    with pytest.raises(OSError):
        make_tree("cube", "join").write(tmp_path / "tree")

    assert not (tmp_path / "tree").exists()


@pytest.mark.parametrize("kind", KINDS)
def test_seeded_random_networks_round_trip(tmp_path: Path, kind: str) -> None:
    import random

    from topographer.graph import Graph

    rng = random.Random(7)
    count = 25
    ids = [f"v{i}" for i in range(count)]
    pairs = {(rng.randrange(i), i) for i in range(1, count)}
    pairs |= {tuple(sorted(rng.sample(range(count), 2))) for _ in range(15)}
    graph = Graph.from_tables(
        pl.DataFrame({"vid": ids, "x": [rng.random() for _ in ids]}),
        pl.DataFrame(
            {"s": [ids[a] for a, _ in sorted(pairs)], "t": [ids[b] for _, b in sorted(pairs)]}
        ),
        vertex_id_col="vid",
        source_id_col="s",
        target_id_col="t",
        vertex_attribute_cols=["x"],
    )
    topology = ScalarNetworkTopology(ScalarNetwork(graph, scalar_attr="x"))
    original = topology.join_tree() if kind == "join" else topology.split_tree()
    original.simplify(0.1)
    original.simplify(0.3)
    original.write(tmp_path / "tree")

    restored = MergeTree.read(tmp_path / "tree")

    assert_same_tree(original, restored)
    for threshold, expected in original.simplifications.items():
        assert_same_graph(restored.simplify(threshold).graph, expected.graph)
