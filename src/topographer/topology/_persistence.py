"""Branch-span and persistence-pair tables for scalar merge trees."""

from __future__ import annotations

from collections.abc import Callable
import math
from typing import Any
import warnings

import polars as pl

__all__ = [
    "build_tree_branch_span_table",
    "build_tree_critical_branch_span_table",
    "build_tree_critical_persistence_table",
    "build_tree_extended_persistence_pair_table",
    "build_tree_ordinary_persistence_pair_table",
    "build_tree_persistence_pair_table",
    "build_tree_persistence_table",
]


def build_tree_branch_span_table(
    tree: Any,
    *,
    extremum_event: str,
    saddle_event: str,
    label_attr: str = "label",
    progress_callback: Callable[[int, int], None] | None = None,
) -> pl.DataFrame:
    """Build deterministic extremum-to-saddle branch spans.

    Each matching edge contributes a row, including the branch that continues
    through a merge. This is a branch-prominence summary, not an elder-rule
    zero-dimensional persistence pairing.


    """

    schema = {
        "extremum_tree_node_index": pl.Int64,
        "saddle_tree_node_index": pl.Int64,
        "extremum_vertex_index": pl.Int64,
        "saddle_vertex_index": pl.Int64,
        "extremum_label": pl.String,
        "saddle_label": pl.String,
        "extremum_scalar": pl.Float64,
        "saddle_scalar": pl.Float64,
        "branch_span": pl.Float64,
        "extremum_event": pl.String,
        "saddle_event": pl.String,
    }
    _require_tree_interface(tree)

    rows: list[dict[str, Any]] = []
    edges = list(tree.edge_list())
    _report_progress(progress_callback, 0, len(edges))
    for position, (source_node_index, target_node_index) in enumerate(edges, start=1):
        source_data = _node_mapping(tree, source_node_index)
        target_data = _node_mapping(tree, target_node_index)
        if str(source_data.get("event", "")) != extremum_event:
            _report_progress(progress_callback, position, len(edges))
            continue
        if str(target_data.get("event", "")) != saddle_event:
            _report_progress(progress_callback, position, len(edges))
            continue

        extremum_scalar = _finite_scalar(source_data, source_node_index)
        saddle_scalar = _finite_scalar(target_data, target_node_index)
        extremum_vertex_index = int(source_data.get("vertex_index", source_node_index))
        saddle_vertex_index = int(target_data.get("vertex_index", target_node_index))
        rows.append(
            {
                "extremum_tree_node_index": int(source_node_index),
                "saddle_tree_node_index": int(target_node_index),
                "extremum_vertex_index": extremum_vertex_index,
                "saddle_vertex_index": saddle_vertex_index,
                "extremum_label": str(source_data.get(label_attr, extremum_vertex_index)),
                "saddle_label": str(target_data.get(label_attr, saddle_vertex_index)),
                "extremum_scalar": extremum_scalar,
                "saddle_scalar": saddle_scalar,
                "branch_span": abs(saddle_scalar - extremum_scalar),
                "extremum_event": str(source_data.get("event", "")),
                "saddle_event": str(target_data.get("event", "")),
            }
        )
        _report_progress(progress_callback, position, len(edges))

    if not rows:
        return pl.DataFrame(schema=schema)
    return pl.DataFrame(rows, schema=schema).sort(
        ["branch_span", "extremum_scalar", "extremum_vertex_index"],
        descending=[True, True, False],
    )


def build_tree_critical_branch_span_table(
    tree: Any,
    *,
    source_events: tuple[str, ...],
    target_events: tuple[str, ...],
    root_node_index: int | None = None,
    root_event: str | None = None,
    label_attr: str = "label",
    progress_callback: Callable[[int, int], None] | None = None,
) -> pl.DataFrame:
    """Build scalar spans for selected directed critical-tree branches."""

    _require_tree_interface(tree)
    if (root_node_index is None) != (root_event is None):
        raise ValueError("root_node_index and root_event must be provided together.")

    schema = {
        "source_tree_node_index": pl.Int64,
        "target_tree_node_index": pl.Int64,
        "source_vertex_index": pl.Int64,
        "target_vertex_index": pl.Int64,
        "source_label": pl.String,
        "target_label": pl.String,
        "source_scalar": pl.Float64,
        "target_scalar": pl.Float64,
        "branch_span": pl.Float64,
        "source_event": pl.String,
        "target_event": pl.String,
    }
    rows: list[dict[str, Any]] = []
    edges = list(tree.edge_list())
    _report_progress(progress_callback, 0, len(edges))
    for position, (source_node_index, target_node_index) in enumerate(edges, start=1):
        source_data = _node_mapping(tree, source_node_index)
        target_data = _node_mapping(tree, target_node_index)
        source_event = str(source_data.get("event", "regular"))
        target_event = (
            str(root_event)
            if target_node_index == root_node_index
            else str(target_data.get("event", "regular"))
        )
        if source_event in source_events and target_event in target_events:
            source_scalar = _finite_scalar(source_data, source_node_index)
            target_scalar = _finite_scalar(target_data, target_node_index)
            source_vertex_index = int(source_data.get("vertex_index", source_node_index))
            target_vertex_index = int(target_data.get("vertex_index", target_node_index))
            rows.append(
                {
                    "source_tree_node_index": int(source_node_index),
                    "target_tree_node_index": int(target_node_index),
                    "source_vertex_index": source_vertex_index,
                    "target_vertex_index": target_vertex_index,
                    "source_label": str(source_data.get(label_attr, source_vertex_index)),
                    "target_label": str(target_data.get(label_attr, target_vertex_index)),
                    "source_scalar": source_scalar,
                    "target_scalar": target_scalar,
                    "branch_span": abs(target_scalar - source_scalar),
                    "source_event": source_event,
                    "target_event": target_event,
                }
            )
        _report_progress(progress_callback, position, len(edges))
    if not rows:
        return pl.DataFrame(schema=schema)
    return pl.DataFrame(rows, schema=schema).sort(
        ["branch_span", "source_scalar", "source_vertex_index"],
        descending=[True, True, False],
    )


def build_tree_ordinary_persistence_pair_table(
    tree: Any,
    *,
    tree_kind: str,
    label_attr: str = "label",
    progress_callback: Callable[[int, int], None] | None = None,
) -> pl.DataFrame:
    """Build finite elder-rule zero-dimensional persistence pairs.

    Each row records the non-elder component that dies at a merge saddle.
    The surviving global component is omitted; use
    :func:`build_tree_extended_persistence_pair_table` to include its finite
    opposite-global-extremum endpoint pair.


    """

    table = _build_tree_persistence_pair_table(
        tree,
        tree_kind=tree_kind,
        label_attr=label_attr,
        include_essential=False,
        progress_callback=progress_callback,
    )
    return _with_pair_kind(table, "ordinary")


def build_tree_extended_persistence_pair_table(
    tree: Any,
    *,
    tree_kind: str,
    label_attr: str = "label",
    progress_callback: Callable[[int, int], None] | None = None,
) -> pl.DataFrame:
    """Build extended H0 persistence pairs for a merge tree or forest.

    This contains the finite ordinary elder-rule rows plus one ``"extended"``
    row per tree component.  The extended row pairs the essential extremum with
    that component's opposite global extremum; it is not a saddle death.


    """

    table = _build_tree_persistence_pair_table(
        tree,
        tree_kind=tree_kind,
        label_attr=label_attr,
        include_essential=True,
        progress_callback=progress_callback,
    )
    ordinary = _with_pair_kind(table.filter(~pl.col("is_essential")), "ordinary")
    endpoint_rows = [
        _extended_endpoint_row(
            tree,
            row,
            label_attr=label_attr,
        )
        for row in table.filter(pl.col("is_essential")).to_dicts()
    ]
    extended = _explicit_persistence_table(endpoint_rows)
    if ordinary.is_empty():
        return extended
    if extended.is_empty():
        return ordinary
    return pl.concat([ordinary, extended]).sort(
        ["persistence", "birth_vertex_index", "death_vertex_index"],
        descending=[True, False, False],
    )


def build_tree_persistence_pair_table(
    tree: Any,
    *,
    tree_kind: str,
    label_attr: str = "label",
    include_essential: bool = False,
    progress_callback: Callable[[int, int], None] | None = None,
) -> pl.DataFrame:
    """Build elder-rule zero-dimensional persistence pairs from a merge forest.

    ``tree_kind`` is ``"join"`` for a sublevel merge tree or ``"split"``
    for a decreasing-superlevel merge tree. Finite rows pair every component
    that loses the elder rule with its merge saddle. If ``include_essential``
    is true, the surviving component of each forest component is represented
    by a row with null death and persistence values.


    """

    return _build_tree_persistence_pair_table(
        tree,
        tree_kind=tree_kind,
        label_attr=label_attr,
        include_essential=include_essential,
        progress_callback=progress_callback,
    )


def _build_tree_persistence_pair_table(
    tree: Any,
    *,
    tree_kind: str,
    label_attr: str,
    include_essential: bool,
    progress_callback: Callable[[int, int], None] | None,
) -> pl.DataFrame:
    if tree_kind not in {"join", "split"}:
        raise ValueError("tree_kind must be 'join' or 'split'.")
    _require_tree_interface(tree, require_nodes=True)

    node_indices = sorted(int(node) for node in tree.node_indices())
    node_data = {node: _node_mapping(tree, node) for node in node_indices}
    scalar = {node: _finite_scalar(node_data[node], node) for node in node_indices}
    vertex = {node: int(node_data[node].get("vertex_index", node)) for node in node_indices}
    predecessors = {node: [] for node in node_indices}
    successors = {node: [] for node in node_indices}
    for source, target in tree.edge_list():
        source = int(source)
        target = int(target)
        if source not in predecessors or target not in predecessors:
            raise ValueError("Tree edges must reference existing nodes.")
        predecessors[target].append(source)
        successors[source].append(target)
        if (tree_kind == "join" and scalar[source] > scalar[target]) or (
            tree_kind == "split" and scalar[source] < scalar[target]
        ):
            raise ValueError(f"{tree_kind} tree edges are not monotone in sweep direction.")
    for node in node_indices:
        predecessors[node].sort()
        successors[node].sort()
        if len(successors[node]) > 1:
            raise ValueError("A merge-tree node must have at most one outgoing edge.")

    order = _topological_order(node_indices, predecessors, successors)
    extremum_event = "minimum" if tree_kind == "join" else "maximum"
    saddle_event = "join_saddle" if tree_kind == "join" else "split_saddle"
    representative: dict[int, int] = {}
    finite_rows: list[dict[str, Any]] = []
    _report_progress(progress_callback, 0, len(order))

    for position, node in enumerate(order, start=1):
        incoming = predecessors[node]
        event = str(node_data[node].get("event", ""))
        if not incoming:
            if event != extremum_event:
                raise ValueError(
                    f"Merge-tree source node {node} must have event {extremum_event!r}."
                )
            representative[node] = node
        elif len(incoming) == 1:
            if event not in {"regular", saddle_event}:
                raise ValueError(
                    f"Single-input merge-tree node {node} has invalid event {event!r}."
                )
            representative[node] = representative[incoming[0]]
        else:
            if event != saddle_event:
                raise ValueError(
                    f"Multi-input merge-tree node {node} must have event {saddle_event!r}."
                )
            births = sorted({representative[parent] for parent in incoming})
            elder = min(
                births, key=lambda birth: _elder_key(tree_kind, scalar[birth], vertex[birth])
            )
            for birth in births:
                if birth == elder:
                    continue
                finite_rows.append(
                    _persistence_row(
                        birth=birth,
                        death=node,
                        node_data=node_data,
                        scalar=scalar,
                        vertex=vertex,
                        label_attr=label_attr,
                    )
                )
            representative[node] = elder
        _report_progress(progress_callback, position, len(order))

    essential_births = sorted(
        {representative[node] for node in node_indices if not successors[node]},
        key=lambda birth: (vertex[birth], birth),
    )
    if include_essential:
        finite_rows.extend(
            _persistence_row(
                birth=birth,
                death=None,
                node_data=node_data,
                scalar=scalar,
                vertex=vertex,
                label_attr=label_attr,
            )
            for birth in essential_births
        )

    schema = {
        "birth_tree_node_index": pl.Int64,
        "death_tree_node_index": pl.Int64,
        "birth_vertex_index": pl.Int64,
        "death_vertex_index": pl.Int64,
        "birth_label": pl.String,
        "death_label": pl.String,
        "birth_scalar": pl.Float64,
        "death_scalar": pl.Float64,
        "persistence": pl.Float64,
        "birth_event": pl.String,
        "death_event": pl.String,
        "is_essential": pl.Boolean,
    }
    if not finite_rows:
        return pl.DataFrame(schema=schema)
    finite_rows.sort(
        key=lambda row: (
            bool(row["is_essential"]),
            -(float(row["persistence"]) if row["persistence"] is not None else 0.0),
            int(row["birth_vertex_index"]),
            int(row["death_vertex_index"] or -1),
        )
    )
    return pl.DataFrame(finite_rows, schema=schema)


def build_tree_persistence_table(
    tree: Any,
    *,
    extremum_event: str,
    saddle_event: str,
    label_attr: str = "label",
    progress_callback: Callable[[int, int], None] | None = None,
) -> pl.DataFrame:
    """Deprecated compatibility wrapper for :func:`build_tree_branch_span_table`."""

    _warn_branch_span_name("build_tree_persistence_table", "build_tree_branch_span_table")
    return build_tree_branch_span_table(
        tree,
        extremum_event=extremum_event,
        saddle_event=saddle_event,
        label_attr=label_attr,
        progress_callback=progress_callback,
    ).rename(
        {
            "extremum_scalar": "birth_scalar",
            "saddle_scalar": "death_scalar",
            "branch_span": "persistence",
        }
    )


def build_tree_critical_persistence_table(
    tree: Any,
    *,
    source_events: tuple[str, ...],
    target_events: tuple[str, ...],
    root_node_index: int | None = None,
    root_event: str | None = None,
    label_attr: str = "label",
    progress_callback: Callable[[int, int], None] | None = None,
) -> pl.DataFrame:
    """Deprecated compatibility wrapper for critical branch spans."""

    _warn_branch_span_name(
        "build_tree_critical_persistence_table", "build_tree_critical_branch_span_table"
    )
    return build_tree_critical_branch_span_table(
        tree,
        source_events=source_events,
        target_events=target_events,
        root_node_index=root_node_index,
        root_event=root_event,
        label_attr=label_attr,
        progress_callback=progress_callback,
    ).rename(
        {
            "source_scalar": "birth_scalar",
            "target_scalar": "death_scalar",
            "branch_span": "persistence",
        }
    )


def _persistence_row(
    *,
    birth: int,
    death: int | None,
    node_data: dict[int, dict[str, Any]],
    scalar: dict[int, float],
    vertex: dict[int, int],
    label_attr: str,
) -> dict[str, Any]:
    birth_data = node_data[birth]
    death_data = node_data[death] if death is not None else None
    return {
        "birth_tree_node_index": birth,
        "death_tree_node_index": death,
        "birth_vertex_index": vertex[birth],
        "death_vertex_index": vertex[death] if death is not None else None,
        "birth_label": str(birth_data.get(label_attr, vertex[birth])),
        "death_label": (
            str(death_data.get(label_attr, vertex[death])) if death_data is not None else None
        ),
        "birth_scalar": scalar[birth],
        "death_scalar": scalar[death] if death is not None else None,
        "persistence": abs(scalar[death] - scalar[birth]) if death is not None else None,
        "birth_event": str(birth_data["event"]),
        "death_event": str(death_data["event"]) if death_data is not None else None,
        "is_essential": death is None,
    }


def _with_pair_kind(table: pl.DataFrame, pair_kind: str) -> pl.DataFrame:
    """Return an explicit persistence table with one row provenance value."""

    return _explicit_persistence_table(
        [
            {key: value for key, value in row.items() if key != "is_essential"}
            | {"pair_kind": pair_kind}
            for row in table.to_dicts()
        ]
    )


def _explicit_persistence_table(rows: list[dict[str, Any]]) -> pl.DataFrame:
    schema = {
        "birth_tree_node_index": pl.Int64,
        "death_tree_node_index": pl.Int64,
        "birth_vertex_index": pl.Int64,
        "death_vertex_index": pl.Int64,
        "birth_label": pl.String,
        "death_label": pl.String,
        "birth_scalar": pl.Float64,
        "death_scalar": pl.Float64,
        "persistence": pl.Float64,
        "birth_event": pl.String,
        "death_event": pl.String,
        "pair_kind": pl.String,
    }
    if not rows:
        return pl.DataFrame(schema=schema)
    return pl.DataFrame(rows, schema=schema).sort(
        ["persistence", "birth_vertex_index", "death_vertex_index"],
        descending=[True, False, False],
    )


def _extended_endpoint_row(
    tree: Any,
    essential_row: dict[str, Any],
    *,
    label_attr: str,
) -> dict[str, Any]:
    """Pair an essential birth with the terminal root of its tree component."""

    birth = int(essential_row["birth_tree_node_index"])
    root = _tree_root_for_node(tree, birth)
    root_data = _node_mapping(tree, root)
    root_scalar = _finite_scalar(root_data, root)
    root_vertex = int(root_data.get("vertex_index", root))
    birth_scalar = float(essential_row["birth_scalar"])
    return {
        "birth_tree_node_index": birth,
        "death_tree_node_index": root,
        "birth_vertex_index": int(essential_row["birth_vertex_index"]),
        "death_vertex_index": root_vertex,
        "birth_label": str(essential_row["birth_label"]),
        "death_label": str(root_data.get(label_attr, root_vertex)),
        "birth_scalar": birth_scalar,
        "death_scalar": root_scalar,
        "persistence": abs(root_scalar - birth_scalar),
        "birth_event": str(essential_row["birth_event"]),
        "death_event": str(root_data.get("event", "regular")),
        "pair_kind": "extended",
    }


def _tree_root_for_node(tree: Any, node: int) -> int:
    successors: dict[int, list[int]] = {int(index): [] for index in tree.node_indices()}
    for source, target in tree.edge_list():
        successors[int(source)].append(int(target))
    current = node
    while successors[current]:
        current = successors[current][0]
    return current


def _elder_key(tree_kind: str, scalar: float, vertex_index: int) -> tuple[float, int]:
    return (scalar, vertex_index) if tree_kind == "join" else (-scalar, vertex_index)


def _topological_order(
    node_indices: list[int],
    predecessors: dict[int, list[int]],
    successors: dict[int, list[int]],
) -> list[int]:
    remaining = {node: len(predecessors[node]) for node in node_indices}
    ready = sorted(node for node in node_indices if remaining[node] == 0)
    order: list[int] = []
    while ready:
        node = ready.pop(0)
        order.append(node)
        for successor in successors[node]:
            remaining[successor] -= 1
            if remaining[successor] == 0:
                ready.append(successor)
                ready.sort()
    if len(order) != len(node_indices):
        raise ValueError("Merge tree must be acyclic.")
    return order


def _require_tree_interface(tree: Any, *, require_nodes: bool = False) -> None:
    required = ["edge_list", "get_node_data"]
    if require_nodes:
        required.append("node_indices")
    if any(not hasattr(tree, attr) for attr in required):
        raise TypeError(f"Unsupported graph type: {type(tree).__name__}")


def _node_mapping(tree: Any, node_index: int) -> dict[str, Any]:
    payload = tree.get_node_data(node_index)
    if not isinstance(payload, dict):
        raise TypeError(f"Tree node {node_index} payload must be a dictionary.")
    return dict(payload)


def _finite_scalar(payload: dict[str, Any], node_index: int) -> float:
    if "scalar" not in payload:
        raise KeyError(f"Tree node {node_index} is missing scalar metadata.")
    try:
        scalar = float(payload["scalar"])
    except (TypeError, ValueError) as error:
        raise TypeError(f"Tree node {node_index} scalar must be a finite real number.") from error
    if not math.isfinite(scalar):
        raise ValueError(f"Tree node {node_index} scalar must be finite.")
    return scalar


def _warn_branch_span_name(old_name: str, new_name: str) -> None:
    warnings.warn(
        f"{old_name}() reports branch spans, not elder-rule persistence pairs; "
        f"use {new_name}(). The old name is still accepted in 0.5.0.",
        DeprecationWarning,
        stacklevel=3,
    )


def _report_progress(
    callback: Callable[[int, int], None] | None, completed: int, total: int
) -> None:
    if callback is None:
        return
    step = max(1, total // 1000)
    if completed == 0 or completed == total or completed % step == 0:
        callback(completed, total)
