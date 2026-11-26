"""Critical-tree reduction for augmented merge trees."""

from __future__ import annotations

from collections.abc import Callable
import json
from typing import Any

import rustworkx as rx

from topographer.topology._tree_construction import _report_progress

__all__ = ["reduce_merge_tree"]


def reduce_merge_tree(
    augmented_tree: Any,
    *,
    progress_callback: Callable[[int, int], None] | None = None,
) -> rx.PyDiGraph:
    """Contract regular-node chains into a backbone tree with edge vertex chains.

    The reduced tree keeps only non-regular vertices as nodes. Each reduced edge carries the
    full sequence of contracted vertex payloads along the corresponding chain, serialized as
    JSON so the result remains GraphML-friendly.

    Examples
    --------
    >>> from topographer.examples import build_cube_scalar_network
    >>> from topographer.topology import build_join_tree, reduce_merge_tree
    >>> network = build_cube_scalar_network()
    >>> augmented = build_join_tree(network.graph, scalar_attr="height").tree
    >>> reduce_merge_tree(augmented).num_nodes()
    4
    """

    node_indices = [int(node_index) for node_index in augmented_tree.node_indices()]
    total = max(1, len(node_indices) * 3)
    _report_progress(progress_callback, 0, total)
    backbone_tree = rx.PyDiGraph()
    backbone_node_by_aug_node: dict[int, int] = {}
    regular_event = "regular"
    preserve_boundary_regular: dict[int, bool] = {}

    for position, aug_node_idx in enumerate(node_indices, start=1):
        preserve_boundary_regular[int(aug_node_idx)] = (
            len(list(augmented_tree.in_edges(aug_node_idx))) == 0
            or len(list(augmented_tree.out_edges(aug_node_idx))) == 0
        )
        _report_progress(progress_callback, position, total)

    for position, aug_node_idx in enumerate(node_indices, start=len(node_indices) + 1):
        node_data = dict(augmented_tree.get_node_data(aug_node_idx))
        if (
            node_data.get("event") == regular_event
            and not preserve_boundary_regular[int(aug_node_idx)]
        ):
            continue
        backbone_node_by_aug_node[int(aug_node_idx)] = backbone_tree.add_node(node_data)
        _report_progress(progress_callback, position, total)

    def _collect_chain(
        source_aug_node: int,
        next_aug_node: int,
        start_edge_data: dict[str, Any],
    ) -> tuple[int, int, list[dict[str, Any]], dict[str, Any]] | None:
        regular_count = 0
        current_node = int(next_aug_node)
        last_edge_data = dict(start_edge_data)
        vertex_chain = [dict(augmented_tree.get_node_data(source_aug_node))]

        while True:
            current_node_data = dict(augmented_tree.get_node_data(current_node))
            if current_node_data.get("event") != regular_event:
                vertex_chain.append(current_node_data)
                return current_node, regular_count, vertex_chain, last_edge_data
            if preserve_boundary_regular.get(int(current_node), False):
                vertex_chain.append(current_node_data)
                return current_node, regular_count, vertex_chain, last_edge_data

            regular_count += 1
            vertex_chain.append(current_node_data)

            out_edges = list(augmented_tree.out_edges(current_node))
            if len(out_edges) != 1:
                return None

            _src_idx, dst_idx, edge_data = out_edges[0]
            current_node = int(dst_idx)
            last_edge_data = dict(edge_data)

    for position, (source_aug_node, source_backbone_node) in enumerate(
        backbone_node_by_aug_node.items(), start=len(node_indices) * 2 + 1
    ):
        for edge in augmented_tree.out_edges(source_aug_node):
            _src_idx, next_aug_node, edge_data = edge
            walk_result = _collect_chain(
                int(source_aug_node),
                int(next_aug_node),
                dict(edge_data),
            )
            if walk_result is None:
                continue

            target_aug_node, contracted_regular_count, vertex_chain, terminal_edge_data = (
                walk_result
            )
            target_backbone_node = backbone_node_by_aug_node.get(target_aug_node)
            if target_backbone_node is None:
                continue

            source_data = backbone_tree.get_node_data(source_backbone_node)
            target_data = backbone_tree.get_node_data(target_backbone_node)
            source_scalar = float(source_data["scalar"])
            target_scalar = float(target_data["scalar"])
            contracted_payload = dict(terminal_edge_data)
            contracted_payload["source_scalar"] = source_scalar
            contracted_payload["target_scalar"] = target_scalar
            contracted_payload["scalar_difference"] = abs(target_scalar - source_scalar)
            contracted_payload["merge_scalar"] = target_scalar
            contracted_payload["from_repr"] = int(source_data["vertex_index"])
            contracted_payload["to_saddle"] = int(target_data["vertex_index"])
            contracted_payload["regular_count"] = int(contracted_regular_count)
            contracted_payload["vertex_chain_json"] = json.dumps(vertex_chain, sort_keys=True)
            contracted_payload["vertex_chain_count"] = len(vertex_chain)
            backbone_tree.add_edge(
                source_backbone_node,
                target_backbone_node,
                contracted_payload,
            )
        _report_progress(progress_callback, min(position, total), total)

    _report_progress(progress_callback, total, total)

    return backbone_tree
