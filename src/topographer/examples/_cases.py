"""The canonical example cases that the CLI and the docs-asset generator export."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import shutil

import polars as pl

from topographer.examples._presentation import export_example_artifacts
from topographer.examples.cube import (
    CUBE_SIMPLIFICATION_THRESHOLD,
    CUBE_TREE_NODE_ATTRS,
    build_cube_topology,
    cube_edge_table,
    cube_node_table,
)
from topographer.examples.hamming import (
    HAMMING_SIMPLIFICATION_THRESHOLD,
    HAMMING_TREE_NODE_ATTRS,
    build_hamming_topology,
    hypercube_edge_table,
    hypercube_node_table,
)
from topographer.topology import ScalarNetworkTopology

__all__ = ["CASES", "ExampleCase", "export_case", "load_or_build_workspace"]


@dataclass(frozen=True, slots=True)
class ExampleCase:
    """One canonical example: its workspace builder and the tables behind its figures."""

    prefix: str
    build_topology: Callable[[], ScalarNetworkTopology]
    vertices: Callable[[], pl.DataFrame]
    edges: Callable[[], pl.DataFrame]
    tree_node_attrs: tuple[str, ...]
    simplification_threshold: float


CASES: dict[str, ExampleCase] = {
    "cube": ExampleCase(
        prefix="cube",
        build_topology=build_cube_topology,
        vertices=cube_node_table,
        edges=cube_edge_table,
        tree_node_attrs=CUBE_TREE_NODE_ATTRS,
        simplification_threshold=CUBE_SIMPLIFICATION_THRESHOLD,
    ),
    "hypercube": ExampleCase(
        prefix="hypercube",
        build_topology=build_hamming_topology,
        vertices=hypercube_node_table,
        edges=hypercube_edge_table,
        tree_node_attrs=HAMMING_TREE_NODE_ATTRS,
        simplification_threshold=HAMMING_SIMPLIFICATION_THRESHOLD,
    ),
}


def load_or_build_workspace(
    case: ExampleCase, workspace_dir: Path, *, force: bool = False
) -> ScalarNetworkTopology:
    """Read the saved workspace of a case, or build and save it when absent (or forced)."""
    if workspace_dir.exists() and not force:
        return ScalarNetworkTopology.read(workspace_dir)
    if workspace_dir.exists():
        shutil.rmtree(workspace_dir)
    topology = case.build_topology()
    workspace_dir.parent.mkdir(parents=True, exist_ok=True)
    topology.write(workspace_dir)
    return topology


def export_case(
    case: ExampleCase,
    topology: ScalarNetworkTopology,
    output_dir: Path,
    *,
    simplification_threshold: float | None = None,
) -> dict[str, Path]:
    """Export the tables and charts of a case from its (possibly restored) workspace."""
    return export_example_artifacts(
        topology,
        vertices=case.vertices(),
        edges=case.edges(),
        output_dir=output_dir,
        prefix=case.prefix,
        simplification_threshold=(
            case.simplification_threshold
            if simplification_threshold is None
            else simplification_threshold
        ),
        tree_node_attrs=case.tree_node_attrs,
    )
