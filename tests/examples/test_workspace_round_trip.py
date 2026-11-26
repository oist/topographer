"""A written and re-read cube workspace gives identical trees, pairs and figure data."""

from __future__ import annotations

from pathlib import Path

from polars.testing import assert_frame_equal
import pytest

from topographer.examples._cases import CASES
from topographer.examples._presentation import build_example_charts, build_example_tables
from topographer.topology import ScalarNetworkTopology


@pytest.fixture(scope="module", params=["cube", "hypercube"])
def workspaces(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> tuple[str, ScalarNetworkTopology, ScalarNetworkTopology]:
    case = CASES[request.param]
    original = case.build_topology()
    folder = tmp_path_factory.mktemp(request.param) / "workspace"
    original.write(folder)
    return request.param, original, ScalarNetworkTopology.read(folder)


def _options(name: str) -> dict:
    case = CASES[name]
    return {
        "vertices": case.vertices(),
        "edges": case.edges(),
        "simplification_threshold": case.simplification_threshold,
        "tree_node_attrs": case.tree_node_attrs,
    }


def test_restored_trees_and_persistence_pairs_are_identical(workspaces) -> None:
    _, original, restored = workspaces
    for tree_of in (ScalarNetworkTopology.join_tree, ScalarNetworkTopology.split_tree):
        before, after = tree_of(original), tree_of(restored)
        assert_frame_equal(before.events, after.events)
        assert_frame_equal(before.persistence(), after.persistence())
        assert_frame_equal(before.spans(), after.spans())
        assert before.graph.num_nodes() == after.graph.num_nodes()
        assert before.graph.num_edges() == after.graph.num_edges()
    assert_frame_equal(
        original.component_persistence().pairs, restored.component_persistence().pairs
    )


def test_restored_tables_are_identical(workspaces) -> None:
    name, original, restored = workspaces
    before = build_example_tables(original, **_options(name))
    after = build_example_tables(restored, **_options(name))
    assert before.keys() == after.keys()
    for key in before:
        assert_frame_equal(before[key], after[key], check_dtypes=True), key


def test_restored_figures_carry_identical_data(workspaces) -> None:
    name, original, restored = workspaces
    before = build_example_charts(original, **_options(name))
    after = build_example_charts(restored, **_options(name))
    assert before.keys() == after.keys()
    for key in before:
        assert before[key].to_dict() == after[key].to_dict(), key


def test_restored_simplified_trees_match(workspaces) -> None:
    name, original, restored = workspaces
    threshold = CASES[name].simplification_threshold
    for tree_of in (ScalarNetworkTopology.join_tree, ScalarNetworkTopology.split_tree):
        before = tree_of(original).simplify(threshold)
        after = tree_of(restored).simplify(threshold)
        assert_frame_equal(before.events, after.events)
        assert_frame_equal(before.persistence(), after.persistence())


def test_workspace_folder_refuses_overwrite(workspaces, tmp_path: Path) -> None:
    _, original, _ = workspaces
    target = tmp_path / "workspace"
    original.write(target)
    with pytest.raises(FileExistsError):
        original.write(target)
