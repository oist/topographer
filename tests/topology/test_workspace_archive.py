"""Round trips, the band rules and refusals of ScalarNetworkTopology.write and read."""

from __future__ import annotations

from collections.abc import Callable
import json
from pathlib import Path
import shutil
from typing import Any

import polars as pl
import pytest

from topographer.examples import cube_edge_table, cube_node_table
from topographer.graph import Graph, ScalarNetwork
from topographer.statistics import bootstrap_persistence_band
from topographer.topology import BootstrapPersistenceBand, MergeTree, ScalarNetworkTopology
import topographer.topology._workspace as workspace_module

SE = [0.05, 0.02, 0.08, 0.03, 0.06, 0.04, 0.07, 0.01]


def cube_with_se(*, scalar_shift: float = 0.0, se_shift: float = 0.0) -> ScalarNetwork:
    nodes = cube_node_table().with_columns(
        pl.Series("se", [v + se_shift for v in SE]),
        (
            pl.col("height")
            + pl.when(pl.col("node_id") == "000").then(scalar_shift).otherwise(0.0)
        ).alias("height"),
    )
    graph = Graph.from_tables(
        nodes,
        cube_edge_table(),
        vertex_id_col="node_id",
        source_id_col="source",
        target_id_col="target",
        vertex_attribute_cols=("height", "label", "x", "y", "se"),
        edge_attribute_cols=("distance", "capacity"),
    )
    return ScalarNetwork(graph, scalar_attr="height")


def make_band(network: ScalarNetwork) -> BootstrapPersistenceBand:
    return bootstrap_persistence_band(
        network, filtration="superlevel", n_draws=6, seed=3, se_attr="se"
    )


def assert_same_tree(original: MergeTree, restored: MergeTree) -> None:
    assert restored.kind == original.kind
    assert restored.graph.nodes() == original.graph.nodes()
    assert restored.graph.weighted_edge_list() == original.graph.weighted_edge_list()
    assert restored.augmented_graph.nodes() == original.augmented_graph.nodes()
    assert (
        restored.augmented_graph.weighted_edge_list()
        == original.augmented_graph.weighted_edge_list()
    )
    assert restored.events.equals(original.events)
    assert restored.mapping.branch_vertices.equals(original.mapping.branch_vertices)
    assert restored.mapping.critical_vertices.equals(original.mapping.critical_vertices)
    assert restored.persistence().equals(original.persistence())
    assert restored.spans().equals(original.spans())
    assert sorted(restored.simplifications) == sorted(original.simplifications)


def boom(*args: Any, **kwargs: Any) -> None:
    raise AssertionError("recomputed")


def full_workspace() -> ScalarNetworkTopology:
    network = cube_with_se()
    topology = ScalarNetworkTopology(network)
    topology.join_tree().simplify(0.15)
    topology.split_tree().simplify(0.15)
    topology.set_band(make_band(network))
    return topology


# --- round trips ---------------------------------------------------------------------------------


def test_network_only_round_trip(tmp_path: Path) -> None:
    original = ScalarNetworkTopology(cube_with_se())
    original.write(tmp_path / "ws")

    restored = ScalarNetworkTopology.read(tmp_path / "ws")

    assert type(restored) is ScalarNetworkTopology
    assert type(restored.network) is ScalarNetwork
    assert restored.network.vertex_ids == original.network.vertex_ids
    assert restored.band is None
    assert not (tmp_path / "ws" / "trees").exists()
    assert not (tmp_path / "ws" / "band-distances.parquet").exists()
    assert json.loads((tmp_path / "ws" / "manifest.json").read_text())["trees"] == []


@pytest.mark.parametrize("kind", ["join", "split"])
def test_one_tree_round_trip(tmp_path: Path, kind: str) -> None:
    original = ScalarNetworkTopology(cube_with_se())
    tree = original.join_tree() if kind == "join" else original.split_tree()
    original.write(tmp_path / "ws")

    restored = ScalarNetworkTopology.read(tmp_path / "ws")

    assert sorted(p.name for p in (tmp_path / "ws" / "trees").iterdir()) == [kind]
    assert_same_tree(tree, restored.join_tree() if kind == "join" else restored.split_tree())


def test_both_trees_round_trip_and_the_other_kind_is_still_lazy(tmp_path: Path) -> None:
    original = ScalarNetworkTopology(cube_with_se())
    trees = {"join": original.join_tree(), "split": original.split_tree()}
    original.write(tmp_path / "ws")

    restored = ScalarNetworkTopology.read(tmp_path / "ws")

    assert_same_tree(trees["join"], restored.join_tree())
    assert_same_tree(trees["split"], restored.split_tree())
    assert restored.compute_merge_tree("sublevel") is restored.join_tree()
    assert restored.summary().equals(original.summary())
    assert restored.component_persistence().pairs.equals(original.component_persistence().pairs)


def test_unstored_kind_is_computed_lazily_as_usual(tmp_path: Path) -> None:
    original = ScalarNetworkTopology(cube_with_se())
    original.join_tree()
    original.write(tmp_path / "ws")

    restored = ScalarNetworkTopology.read(tmp_path / "ws")

    assert restored._split_tree is None
    assert_same_tree(original.split_tree(), restored.split_tree())


def test_workspace_with_band_and_simplifications_round_trips(tmp_path: Path) -> None:
    original = full_workspace()
    original.write(tmp_path / "ws")

    restored = ScalarNetworkTopology.read(tmp_path / "ws")

    assert restored.band == original.band
    assert restored.band.distances == original.band.distances  # type: ignore[union-attr]
    assert restored.band.threshold == original.band.threshold  # type: ignore[union-attr]
    for kind in ("join", "split"):
        a = original.join_tree() if kind == "join" else original.split_tree()
        b = restored.join_tree() if kind == "join" else restored.split_tree()
        assert_same_tree(a, b)
        assert b.simplify(0.15).original is b


def test_a_large_seed_is_stored_as_a_string_and_survives(tmp_path: Path) -> None:
    network = cube_with_se()
    topology = ScalarNetworkTopology(network)
    big = 2**70 + 5
    band = bootstrap_persistence_band(
        network, filtration="superlevel", n_draws=3, seed=big, se_attr="se"
    )
    topology.set_band(band)
    topology.write(tmp_path / "ws")

    manifest = json.loads((tmp_path / "ws" / "manifest.json").read_text())
    assert manifest["band"]["seed"] == str(big)
    assert ScalarNetworkTopology.read(tmp_path / "ws").band == band


def test_layout_and_manifest(tmp_path: Path) -> None:
    full_workspace().write(tmp_path / "ws")

    folder = tmp_path / "ws"
    assert sorted(p.name for p in folder.iterdir()) == [
        "band-distances.parquet",
        "manifest.json",
        "network",
        "trees",
    ]
    assert sorted(p.name for p in (folder / "trees").iterdir()) == ["join", "split"]
    manifest = json.loads((folder / "manifest.json").read_text())
    assert manifest["format"] == "topographer.workspace"
    assert manifest["format_version"] == 1
    assert manifest["workspace_class"] == "ScalarNetworkTopology"
    assert manifest["trees"] == ["join", "split"]
    band = manifest["band"]
    assert band["distance"] == "bottleneck_linf_h0_capped_essential"
    assert band["quantile_method"] == "linear"
    assert band["filtration"] == "superlevel"
    assert band["n_draws"] == 6
    assert band["seed"] == "3"
    assert band["se_attr"] == "se"
    assert set(band) == {
        "filtration",
        "quantile",
        "radius",
        "n_draws",
        "seed",
        "se_attr",
        "network_fingerprint",
        "distance",
        "quantile_method",
    }
    table = pl.read_parquet(folder / "band-distances.parquet")
    assert table.schema == {"draw": pl.Int64, "distance": pl.Float64}
    assert table["draw"].to_list() == list(range(6))


def test_nested_folders_are_readable_on_their_own(tmp_path: Path) -> None:
    original = full_workspace()
    original.write(tmp_path / "ws")

    network = ScalarNetwork.read(tmp_path / "ws" / "network")
    tree = MergeTree.read(tmp_path / "ws" / "trees" / "split")

    assert network.vertex_ids == original.network.vertex_ids
    assert_same_tree(original.split_tree(), tree)


def test_existing_path_refused_and_untouched(tmp_path: Path) -> None:
    (tmp_path / "ws").mkdir()
    (tmp_path / "ws" / "keep.txt").write_text("keep")

    with pytest.raises(FileExistsError):
        ScalarNetworkTopology(cube_with_se()).write(tmp_path / "ws")
    assert [p.name for p in (tmp_path / "ws").iterdir()] == ["keep.txt"]


def test_failed_write_leaves_nothing_behind(tmp_path: Path) -> None:
    network = cube_with_se()
    network._graph_snapshot.get_node_data(0)["label"] = object()  # not storable
    topology = ScalarNetworkTopology(network)

    with pytest.raises(TypeError):
        topology.write(tmp_path / "ws")
    assert not (tmp_path / "ws").exists()


# --- no recomputation ----------------------------------------------------------------------------


def test_read_and_compute_merge_tree_run_no_tree_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = full_workspace()
    original.write(tmp_path / "ws")
    monkeypatch.setattr(workspace_module, "construct_oriented_merge_tree", boom)

    restored = ScalarNetworkTopology.read(tmp_path / "ws")
    join = restored.compute_merge_tree("sublevel")
    split = restored.compute_merge_tree("superlevel")

    assert_same_tree(original.join_tree(), join)
    assert_same_tree(original.split_tree(), split)
    assert join is restored.join_tree()
    assert restored.band == original.band


def test_a_kind_that_was_not_stored_does_compute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = ScalarNetworkTopology(cube_with_se())
    original.join_tree()
    original.write(tmp_path / "ws")
    restored = ScalarNetworkTopology.read(tmp_path / "ws")
    monkeypatch.setattr(workspace_module, "construct_oriented_merge_tree", boom)

    restored.compute_merge_tree("sublevel")
    with pytest.raises(AssertionError, match="recomputed"):
        restored.compute_merge_tree("superlevel")


# --- set_band ------------------------------------------------------------------------------------


def test_set_band_accepts_a_matching_band_and_exposes_it() -> None:
    network = cube_with_se()
    topology = ScalarNetworkTopology(network)
    band = make_band(network)

    assert topology.band is None
    topology.set_band(band)

    assert topology.band is band


@pytest.mark.parametrize(
    "changes",
    [{"scalar_shift": 0.01}, {"se_shift": 0.01}],
    ids=["scalar", "standard-error"],
)
def test_set_band_rejects_a_band_from_a_changed_network(changes: dict[str, float]) -> None:
    band = make_band(cube_with_se())
    topology = ScalarNetworkTopology(cube_with_se(**changes))

    with pytest.raises(ValueError, match="different network"):
        topology.set_band(band)
    assert topology.band is None


def test_set_band_rejects_a_second_band() -> None:
    network = cube_with_se()
    topology = ScalarNetworkTopology(network)
    topology.set_band(make_band(network))

    with pytest.raises(ValueError, match="already"):
        topology.set_band(make_band(network).at_quantile(0.5))


@pytest.mark.parametrize("value", [None, 0.5, "band", {"radius": 1.0}])
def test_set_band_rejects_a_non_band(value: Any) -> None:
    with pytest.raises(TypeError, match="BootstrapPersistenceBand"):
        ScalarNetworkTopology(cube_with_se()).set_band(value)


# --- tampering and wrong folder kinds ------------------------------------------------------------


def written(tmp_path: Path) -> Path:
    full_workspace().write(tmp_path / "ws")
    return tmp_path / "ws"


def edit_manifest(folder: Path, edit: Callable[[dict[str, Any]], None]) -> None:
    path = folder / "manifest.json"
    manifest = json.loads(path.read_text())
    edit(manifest)
    path.write_text(json.dumps(manifest, sort_keys=True, indent=2))


def edit_table(path: Path, edit: Callable[[pl.DataFrame], pl.DataFrame]) -> None:
    edit(pl.read_parquet(path)).write_parquet(path)


def set_row(column: str, row: int, value: Any) -> Callable[[pl.DataFrame], pl.DataFrame]:
    def edit(frame: pl.DataFrame) -> pl.DataFrame:
        values = frame[column].to_list()
        values[row] = value
        return frame.with_columns(pl.Series(column, values, dtype=frame[column].dtype))

    return edit


def test_read_rejects_an_edited_network_scalar(tmp_path: Path) -> None:
    folder = written(tmp_path)
    edit_table(folder / "network" / "vertices.parquet", set_row("height", 0, 0.123))

    with pytest.raises(ValueError, match="scalar"):
        ScalarNetworkTopology.read(folder)


def test_read_rejects_an_edited_tree_scalar(tmp_path: Path) -> None:
    folder = written(tmp_path)
    edit_table(folder / "trees" / "join" / "augmented_nodes.parquet", set_row("scalar", 2, 0.77))

    with pytest.raises(ValueError, match="scalar"):
        ScalarNetworkTopology.read(folder)


def test_read_rejects_a_changed_vertex_id_in_a_tree(tmp_path: Path) -> None:
    folder = written(tmp_path)
    edit_table(folder / "trees" / "split" / "nodes.parquet", set_row("vertex_id", 0, "zzz"))

    with pytest.raises(ValueError, match="vertex_id"):
        ScalarNetworkTopology.read(folder)


def test_read_rejects_trees_swapped_between_kinds(tmp_path: Path) -> None:
    folder = written(tmp_path)
    (folder / "trees" / "join").rename(folder / "trees" / "tmp")
    (folder / "trees" / "split").rename(folder / "trees" / "join")
    (folder / "trees" / "tmp").rename(folder / "trees" / "split")

    with pytest.raises(ValueError, match="join tree folder holds a split tree"):
        ScalarNetworkTopology.read(folder)


def test_read_rejects_a_tree_with_the_wrong_scalar_attr(tmp_path: Path) -> None:
    folder = written(tmp_path)
    edit_manifest(folder / "trees" / "join", lambda m: m.update(scalar_attr="x"))

    with pytest.raises(ValueError, match="scalar_attr"):
        ScalarNetworkTopology.read(folder)


def test_read_rejects_a_tree_with_the_wrong_vertex_count(tmp_path: Path) -> None:
    folder = written(tmp_path)
    shutil.rmtree(folder / "trees" / "join")
    other = ScalarNetworkTopology(
        ScalarNetwork(
            Graph.from_tables(
                pl.DataFrame({"v": ["a", "b"], "height": [0.1, 0.9]}),
                pl.DataFrame({"s": ["a"], "t": ["b"]}),
                vertex_id_col="v",
                source_id_col="s",
                target_id_col="t",
                vertex_attribute_cols=["height"],
            ),
            scalar_attr="height",
        )
    )
    other.join_tree().write(folder / "trees" / "join")

    with pytest.raises(ValueError, match="vertices, the network has"):
        ScalarNetworkTopology.read(folder)


def test_read_rejects_a_listed_tree_that_is_missing(tmp_path: Path) -> None:
    folder = written(tmp_path)
    shutil.rmtree(folder / "trees" / "split")

    with pytest.raises(ValueError, match="split"):
        ScalarNetworkTopology.read(folder)


def test_read_rejects_unknown_tree_kinds(tmp_path: Path) -> None:
    folder = written(tmp_path)
    edit_manifest(folder, lambda m: m.update(trees=["join", "sideways"]))

    with pytest.raises(ValueError, match="trees"):
        ScalarNetworkTopology.read(folder)


def test_read_rejects_an_edited_band_distance(tmp_path: Path) -> None:
    folder = written(tmp_path)
    edit_table(folder / "band-distances.parquet", set_row("distance", 0, 9.5))

    with pytest.raises(ValueError, match="quantile"):
        ScalarNetworkTopology.read(folder)


@pytest.mark.parametrize(
    "change",
    [
        {"distance": "bottleneck_l2"},
        {"quantile_method": "lower"},
    ],
)
def test_read_rejects_wrong_convention_strings(tmp_path: Path, change: dict[str, str]) -> None:
    folder = written(tmp_path)
    edit_manifest(folder, lambda m: m["band"].update(change))

    with pytest.raises(ValueError, match=next(iter(change))):
        ScalarNetworkTopology.read(folder)


def test_read_rejects_a_band_for_a_different_network(tmp_path: Path) -> None:
    folder = written(tmp_path)
    edit_manifest(folder, lambda m: m["band"].update(network_fingerprint="0" * 64))

    with pytest.raises(ValueError, match="different network"):
        ScalarNetworkTopology.read(folder)


def test_read_rejects_a_network_changed_after_the_band(tmp_path: Path) -> None:
    folder = written(tmp_path)
    edit_table(folder / "network" / "vertices.parquet", set_row("se", 0, 0.5))

    with pytest.raises(ValueError, match="different network"):
        ScalarNetworkTopology.read(folder)


@pytest.mark.parametrize("seed", [3, "x3", 3.5, "-3"])
def test_read_rejects_a_bad_seed(tmp_path: Path, seed: Any) -> None:
    folder = written(tmp_path)
    edit_manifest(folder, lambda m: m["band"].update(seed=seed))

    with pytest.raises(ValueError, match="seed"):
        ScalarNetworkTopology.read(folder)


def test_read_rejects_missing_band_table_and_keys(tmp_path: Path) -> None:
    folder = written(tmp_path)
    (folder / "band-distances.parquet").unlink()
    with pytest.raises(ValueError, match="band-distances"):
        ScalarNetworkTopology.read(folder)
    edit_manifest(folder, lambda m: m["band"].pop("se_attr"))
    with pytest.raises(ValueError, match="se_attr"):
        ScalarNetworkTopology.read(folder)


def test_read_rejects_misordered_draws(tmp_path: Path) -> None:
    folder = written(tmp_path)
    edit_table(folder / "band-distances.parquet", set_row("draw", 0, 4))

    with pytest.raises(ValueError, match="draws"):
        ScalarNetworkTopology.read(folder)


def test_read_rejects_missing_manifest_keys(tmp_path: Path) -> None:
    folder = written(tmp_path)
    edit_manifest(folder, lambda m: m.pop("band"))

    with pytest.raises(ValueError, match="band"):
        ScalarNetworkTopology.read(folder)


def test_workspace_and_network_folders_name_the_right_reader(tmp_path: Path) -> None:
    folder = written(tmp_path)

    with pytest.raises(ValueError, match=r"ScalarNetworkTopology\.read"):
        ScalarNetwork.read(folder)
    with pytest.raises(ValueError, match=r"ScalarNetwork\.read"):
        ScalarNetworkTopology.read(folder / "network")
    with pytest.raises(ValueError, match=r"MergeTree\.read"):
        ScalarNetworkTopology.read(folder / "trees" / "join")
    with pytest.raises(ValueError, match=r"MergeTree\.read"):
        ScalarNetwork.read(folder / "trees" / "join")
    with pytest.raises(ValueError, match=r"ScalarNetworkTopology\.read"):
        MergeTree.read(folder)


def test_read_rejects_other_format_versions(tmp_path: Path) -> None:
    folder = written(tmp_path)
    edit_manifest(folder, lambda m: m.update(format_version=2))

    with pytest.raises(ValueError, match="format_version"):
        ScalarNetworkTopology.read(folder)
