"""Deterministic branch-local swarm layout for stop-codon markers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import polars as pl

from topographer.plotting._data import TreePlotData

__all__ = [
    "SwarmConfig",
    "compute_branch_local_swarm",
]


@dataclass(frozen=True, slots=True)
class SwarmConfig:
    """Settings for deterministic branch-local swarm coordinate packing.

    Parameters
    ----------
    step_x
        Horizontal step size for colliding points. If ``None``, defaults to
        a fraction of the minimum distance between distinct branch lanes.
    delta_y
        Vertical collision window in scalar units. If ``None``, defaults to
        2% of the tree's scalar range.

    """

    step_x: float | None = None
    delta_y: float | None = None

    def __post_init__(self) -> None:
        if self.step_x is not None and self.step_x <= 0.0:
            raise ValueError("step_x must be positive.")
        if self.delta_y is not None and self.delta_y <= 0.0:
            raise ValueError("delta_y must be positive.")


def _symmetric_offset_index(index: int) -> int:
    """Return symmetric offset sequence: 0, +1, -1, +2, -2, +3, -3..."""
    if index == 0:
        return 0
    sign = 1 if index % 2 == 1 else -1
    magnitude = (index + 1) // 2
    return sign * magnitude


def compute_branch_local_swarm(
    member_table: pl.DataFrame,
    plot_data: TreePlotData,
    *,
    critical_table: pl.DataFrame | None = None,
    config: SwarmConfig | None = None,
    filter_stop_codons: bool = True,
) -> pl.DataFrame:
    """Compute deterministic branch-local coordinates for stop-bearing sequences.

    Regular stop-codon-bearing sequences are laid out in a symmetric horizontal
    swarm within their assigned branch lane: sparse points remain centered, and
    locally dense points (whose fitness is within ``delta_y``) spread
    horizontally by multiples of ``step_x``. No vertical jitter is applied.
    Stop-bearing critical nodes are placed at their exact tree-node positions.

    Parameters
    ----------
    member_table
        Table of regular vertices, typically from ``build_tree_location_tables``.
    plot_data
        Tree plot data containing node and edge horizontal coordinates.
    critical_table
        Optional table of critical vertices from ``build_tree_location_tables``.
    config
        Optional swarm configuration tuning horizontal step and vertical proximity.
    filter_stop_codons
        If ``True``, filters input tables to rows where ``has_stop_codon == True``.

    Returns
    -------
    pl.DataFrame
        Table with ``x`` and ``y`` coordinates suitable for ``TreeAnnotationLayer``.

    Examples
    --------
    >>> from topographer.examples import build_cube_scalar_network
    >>> from topographer.plotting import build_tree_plot_data, compute_branch_local_swarm
    >>> from topographer.topology import (
    ...     build_join_tree,
    ...     build_tree_location_tables,
    ...     reduce_merge_tree,
    ... )
    >>> network = build_cube_scalar_network()
    >>> reduced = reduce_merge_tree(build_join_tree(network.graph, scalar_attr="height").tree)
    >>> tables = build_tree_location_tables(reduced)
    >>> swarm = compute_branch_local_swarm(
    ...     tables.branch_member_table, build_tree_plot_data(reduced), filter_stop_codons=False
    ... )
    >>> swarm.columns[-2:]
    ['x', 'y']
    """

    resolved_config = config or SwarmConfig()

    # Determine vertical scalar range
    all_scalars = [float(v) for v in plot_data.node_table.get_column("scalar").to_list()]
    if not member_table.is_empty() and "scalar" in member_table.columns:
        all_scalars.extend(float(v) for v in member_table.get_column("scalar").to_list())
    scalar_min = min(all_scalars) if all_scalars else 0.0
    scalar_max = max(all_scalars) if all_scalars else 1.0
    scalar_range = max(scalar_max - scalar_min, 1e-6)

    # Determine default delta_y (2% of range)
    delta_y = (
        resolved_config.delta_y if resolved_config.delta_y is not None else 0.02 * scalar_range
    )

    # Determine default step_x based on distinct x positions
    node_x_values = sorted(set(float(x) for x in plot_data.node_table.get_column("x").to_list()))
    if len(node_x_values) > 1:
        min_lane_dx = min(
            node_x_values[i + 1] - node_x_values[i]
            for i in range(len(node_x_values) - 1)
            if node_x_values[i + 1] - node_x_values[i] > 1e-9
        )
    else:
        min_lane_dx = 1.0
    step_x = resolved_config.step_x if resolved_config.step_x is not None else 0.15 * min_lane_dx

    # Map branch edge to trunk x-coordinate
    branch_x_map: dict[frozenset[int], float] = {}
    for row in plot_data.edge_table.to_dicts():
        src = int(row["source_tree_node_index"])
        tgt = int(row["target_tree_node_index"])
        branch_x_map[frozenset((src, tgt))] = float(row["x_target"])

    node_coords: dict[int, tuple[float, float]] = {
        int(row["tree_node_index"]): (float(row["x"]), float(row["y"]))
        for row in plot_data.node_table.to_dicts()
    }

    markers: list[dict[str, Any]] = []

    # Process critical stops
    if critical_table is not None and not critical_table.is_empty():
        crit = critical_table
        if filter_stop_codons and "has_stop_codon" in crit.columns:
            crit = crit.filter(pl.col("has_stop_codon"))
        for row in crit.to_dicts():
            tree_node_index = int(row["tree_node_index"])
            if tree_node_index in node_coords:
                x_coord, y_coord = node_coords[tree_node_index]
            else:
                x_coord = 0.0
                y_coord = float(row.get("scalar", 0.0))
            marker_row = {
                **row,
                "tree_node_index": tree_node_index,
                "x": x_coord,
                "y": y_coord,
            }
            markers.append(marker_row)

    # Process regular members
    members = member_table
    if filter_stop_codons and "has_stop_codon" in members.columns:
        members = members.filter(pl.col("has_stop_codon"))

    if not members.is_empty():
        # Group by branch
        branches = members.partition_by(
            ["source_tree_node_index", "target_tree_node_index"], as_dict=True
        )
        for (src, tgt), branch_df in sorted(branches.items()):
            src_idx = int(src)
            tgt_idx = int(tgt)
            x_center = branch_x_map.get(
                frozenset((src_idx, tgt_idx)),
                node_coords.get(tgt_idx, (0.0, 0.0))[0],
            )
            # Sort deterministically
            sorted_rows = branch_df.sort(["scalar", "vertex_index"]).to_dicts()

            # Cluster points by vertical collision window delta_y
            clusters: list[list[dict[str, Any]]] = []
            current_cluster: list[dict[str, Any]] = []
            last_y: float | None = None

            for row in sorted_rows:
                y = float(row["scalar"])
                if last_y is None or (y - last_y) < delta_y:
                    current_cluster.append(row)
                else:
                    clusters.append(current_cluster)
                    current_cluster = [row]
                last_y = y
            if current_cluster:
                clusters.append(current_cluster)

            for cluster in clusters:
                for idx, row in enumerate(cluster):
                    offset_mult = _symmetric_offset_index(idx)
                    x_val = x_center + offset_mult * step_x
                    y_val = float(row["scalar"])
                    markers.append(
                        {
                            **row,
                            "tree_node_index": None,
                            "x": x_val,
                            "y": y_val,
                        }
                    )

    schema = {
        "tree_node_index": pl.Int64,
        "vertex_index": pl.Int64,
        "scalar": pl.Float64,
        "x": pl.Float64,
        "y": pl.Float64,
    }
    if not markers:
        return pl.DataFrame(schema=schema)

    df = pl.DataFrame(markers)
    sort_cols = [c for c in ["tree_node_index", "vertex_index", "scalar"] if c in df.columns]
    return df.sort(sort_cols, nulls_last=True)
