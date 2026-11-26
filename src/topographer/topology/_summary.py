"""Topological summary invariants for scalar networks and merge trees."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
import rustworkx as rx

from topographer.graph import ScalarNetwork
from topographer.topology._merge_tree import MergeTree
from topographer.topology._workspace import ScalarNetworkTopology

_landscape_summary_columns: Any = None


def _register_landscape_summary(fn: Any) -> None:
    """Register the biology columns appended for a fitness landscape.

    Biology calls this when it is imported. Topology does not import biology.
    """
    global _landscape_summary_columns
    _landscape_summary_columns = fn


def _tree_height(tree: MergeTree) -> int:
    """Return the maximum depth from root to leaf in the reduced merge tree."""
    graph = tree.graph
    if graph.num_nodes() == 0 or graph.num_edges() == 0:
        return 0
    return int(rx.dag_longest_path_length(graph))


def compute_topological_summary(target: Any) -> pl.DataFrame:
    """Compute a single-row Polars DataFrame of topological invariants.

    Parameters
    ----------
    target : ScalarNetwork | ScalarNetworkTopology | FitnessLandscape | FitnessLandscapeTopology
        Input scalar network or topology workspace.

    Returns
    -------
    pl.DataFrame
        A 1-row DataFrame containing:
        - `n_vertices`: Total vertices in the scalar network
        - `n_edges`: Total edges in the scalar network
        - `n_peaks`: Number of local maxima (split tree leaves)
        - `n_sinks`: Number of local minima (join tree leaves)
        - `n_split_saddles`: Number of split saddles
        - `n_join_saddles`: Number of join saddles
        - `max_persistence`: Maximum split tree persistence
        - `total_persistence`: Sum of split tree persistence pairs
        - `persistence_entropy`: Shannon entropy (base 2) of split persistence distribution
        - `tree_asymmetry_depth_ratio`: join tree height / split tree height
        - `tree_asymmetry_persistence_ratio`: join total persistence / split total persistence
        If a FitnessLandscape or FitnessLandscapeTopology is provided, the following
        biological ruggedness metrics are appended:
        - `roughness_to_slope_ratio`: Residual standard deviation over average slope
        - `sign_epistasis_fraction`: Fraction of Hamming squares exhibiting sign epistasis
        - `wildtype_accessibility_fraction`: Fraction of monotonic paths increasing in fitness

    Examples
    --------
    >>> from topographer.examples import build_cube_scalar_network
    >>> from topographer.topology import compute_topological_summary
    >>> summary = compute_topological_summary(build_cube_scalar_network())
    >>> summary.shape
    (1, 11)
    >>> summary.item(0, "n_vertices")
    8
    """
    if isinstance(target, ScalarNetworkTopology):
        topology: ScalarNetworkTopology = target
        network: ScalarNetwork = target.network
    elif isinstance(target, ScalarNetwork):
        network = target
        topology = ScalarNetworkTopology(target)
    else:
        raise TypeError(
            f"Expected ScalarNetwork, ScalarNetworkTopology, FitnessLandscape, "
            f"or FitnessLandscapeTopology, got {type(target).__name__}"
        )

    split_tree = topology.compute_merge_tree("superlevel")
    join_tree = topology.compute_merge_tree("sublevel")

    # Network topology counts
    n_vertices = int(network.vertex_count)
    n_edges = int(network.edge_count)

    n_peaks = (
        split_tree.events.filter(pl.col("event") == "maximum").height
        if not split_tree.events.is_empty()
        else 0
    )
    n_sinks = (
        join_tree.events.filter(pl.col("event") == "minimum").height
        if not join_tree.events.is_empty()
        else 0
    )
    n_split_saddles = (
        split_tree.events.filter(pl.col("event") == "split_saddle").height
        if not split_tree.events.is_empty()
        else 0
    )
    n_join_saddles = (
        join_tree.events.filter(pl.col("event") == "join_saddle").height
        if not join_tree.events.is_empty()
        else 0
    )

    # Persistence metrics on split tree
    split_df = split_tree.persistence()
    if split_df.is_empty():
        max_persistence = 0.0
        total_persistence = 0.0
        persistence_entropy = 0.0
    else:
        pers_vals = split_df["persistence"].to_numpy()
        max_persistence = float(np.max(pers_vals)) if len(pers_vals) > 0 else 0.0
        total_persistence = float(np.sum(pers_vals))
        if total_persistence <= 0.0:
            persistence_entropy = 0.0
        else:
            p_hat = pers_vals / total_persistence
            pos_mask = p_hat > 0
            if np.any(pos_mask):
                persistence_entropy = float(-np.sum(p_hat[pos_mask] * np.log2(p_hat[pos_mask])))
            else:
                persistence_entropy = 0.0

    # Tree asymmetry metrics
    join_height = _tree_height(join_tree)
    split_height = _tree_height(split_tree)
    if split_height == 0:
        depth_ratio = 1.0 if join_height == 0 else float("inf")
    else:
        depth_ratio = float(join_height) / float(split_height)

    join_df = join_tree.persistence()
    join_total = float(join_df["persistence"].sum()) if not join_df.is_empty() else 0.0
    if total_persistence == 0.0:
        pers_ratio = 1.0 if join_total == 0.0 else float("inf")
    else:
        pers_ratio = join_total / total_persistence

    row_data: dict[str, list[Any]] = {
        "n_vertices": [n_vertices],
        "n_edges": [n_edges],
        "n_peaks": [n_peaks],
        "n_sinks": [n_sinks],
        "n_split_saddles": [n_split_saddles],
        "n_join_saddles": [n_join_saddles],
        "max_persistence": [max_persistence],
        "total_persistence": [total_persistence],
        "persistence_entropy": [persistence_entropy],
        "tree_asymmetry_depth_ratio": [depth_ratio],
        "tree_asymmetry_persistence_ratio": [pers_ratio],
    }
    schema: dict[str, Any] = {
        "n_vertices": pl.Int64,
        "n_edges": pl.Int64,
        "n_peaks": pl.Int64,
        "n_sinks": pl.Int64,
        "n_split_saddles": pl.Int64,
        "n_join_saddles": pl.Int64,
        "max_persistence": pl.Float64,
        "total_persistence": pl.Float64,
        "persistence_entropy": pl.Float64,
        "tree_asymmetry_depth_ratio": pl.Float64,
        "tree_asymmetry_persistence_ratio": pl.Float64,
    }

    extras = _landscape_summary_columns(target) if _landscape_summary_columns is not None else None
    if extras:
        for key, value in extras.items():
            row_data[key] = [value]
            schema[key] = pl.Float64

    return pl.DataFrame(row_data, schema=schema)
