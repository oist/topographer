"""Adapters for creating Graph from external graph libraries and data tables."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
import importlib.util
from typing import TYPE_CHECKING, Any

import polars as pl
import rustworkx as rx

from topographer.graph._payloads import _copy_payload

if TYPE_CHECKING:
    from topographer.graph._core import Graph


def _canonical_vertex_ids(vertex_ids: Iterable[Any]) -> list[Any]:
    """Return vertex IDs in native order after checking one orderable type.

    Mixed types are rejected before any comparison, so the caller sees a clear
    message instead of a raw ``<`` error. Types without a native order, such as
    ``complex``, are rejected for the same reason.
    """
    ids = list(vertex_ids)
    types = {type(vid) for vid in ids}
    if len(types) > 1:
        names = sorted(t.__name__ for t in types)
        raise TypeError(f"Mixed vertex identity types are not permitted; got {names}.")
    for vid in ids:
        try:
            hash(vid)
        except TypeError:
            raise TypeError(f"Vertex identity {vid!r} is not hashable.") from None
    try:
        return sorted(ids)
    except TypeError:
        name = next(iter(types)).__name__
        raise TypeError(
            f"Vertex identities of type {name!r} have no native ordering among these values."
        ) from None


def from_rustworkx(
    cls: type[Graph],
    graph: rx.PyGraph | rx.PyDiGraph,
    *,
    vertex_id_attr: str,
    edge_id_attr: str | None = None,
    vertex_attribute_cols: Sequence[str] = (),
    edge_attribute_cols: Sequence[str] = (),
) -> Graph:
    directed = isinstance(graph, rx.PyDiGraph)
    multigraph = getattr(graph, "multigraph", False)
    new_rx = rx.PyDiGraph(multigraph=multigraph) if directed else rx.PyGraph(multigraph=multigraph)

    # Collect vertices and validate stable identities
    node_indices = graph.node_indices()
    raw_nodes = [(idx, graph.get_node_data(idx)) for idx in node_indices]

    vid_to_old_idx: dict[Any, int] = {}
    for idx, data in raw_nodes:
        if not isinstance(data, dict) or vertex_id_attr not in data:
            raise ValueError(f"Node payload missing vertex_id_attr {vertex_id_attr!r}")
        vid = data[vertex_id_attr]
        if vid in vid_to_old_idx:
            raise ValueError(f"Duplicate vertex ID {vid!r} found in rustworkx graph.")
        try:
            hash(vid)
        except TypeError:
            raise TypeError(f"Vertex identity {vid!r} is not hashable.") from None
        vid_to_old_idx[vid] = idx

    canonical_vids = _canonical_vertex_ids(vid_to_old_idx)

    # Validate vertex attribute cols
    for col in vertex_attribute_cols:
        if col == vertex_id_attr:
            raise ValueError(
                f"Identity column {col!r} cannot be selected as an ordinary attribute."
            )

    vertex_id_to_index: dict[Any, int] = {}
    index_to_vertex_id: dict[int, Any] = {}

    for vid in canonical_vids:
        old_idx = vid_to_old_idx[vid]
        old_data = graph.get_node_data(old_idx)
        new_payload = {vertex_id_attr: vid}
        for attr in vertex_attribute_cols:
            if attr not in old_data:
                raise ValueError(f"Vertex attribute {attr!r} missing on vertex {vid!r}")
            new_payload[attr] = _copy_payload(old_data[attr])
        new_idx = new_rx.add_node(new_payload)
        vertex_id_to_index[vid] = new_idx
        index_to_vertex_id[new_idx] = vid

    # Validate edge_id_attr if provided or graph has parallel edges
    if edge_id_attr is not None:
        for col in edge_attribute_cols:
            if col == edge_id_attr:
                raise ValueError(
                    f"Edge identity attribute {col!r} cannot be selected as an ordinary attribute."
                )
    elif graph.has_parallel_edges():
        raise ValueError("Multigraph with parallel edges requires explicit edge_id_attr.")

    # Copy edges
    edge_ids: list[Any] | None = [] if edge_id_attr is not None else None
    edge_id_to_index: dict[Any, int] = {}
    index_to_edge_id: dict[int, Any] = {}

    for e_idx in graph.edge_indices():
        u_old, v_old = graph.get_edge_endpoints_by_index(e_idx)
        edge_data = graph.get_edge_data_by_index(e_idx)
        u_vid = graph.get_node_data(u_old)[vertex_id_attr]
        v_vid = graph.get_node_data(v_old)[vertex_id_attr]
        u_new = vertex_id_to_index[u_vid]
        v_new = vertex_id_to_index[v_vid]

        new_edge_payload = {}
        if isinstance(edge_data, dict):
            for attr in edge_attribute_cols:
                if attr not in edge_data:
                    raise ValueError(f"Edge attribute {attr!r} missing on edge ({u_vid}, {v_vid})")
                new_edge_payload[attr] = _copy_payload(edge_data[attr])
        new_e_idx = new_rx.add_edge(u_new, v_new, new_edge_payload)

        if edge_id_attr is not None:
            if not isinstance(edge_data, dict) or edge_id_attr not in edge_data:
                raise ValueError(f"Edge payload missing edge_id_attr {edge_id_attr!r}")
            eid = edge_data[edge_id_attr]
            try:
                hash(eid)
            except TypeError:
                raise TypeError(f"Edge identity {eid!r} is not hashable.") from None
            if eid in edge_id_to_index:
                raise ValueError(f"Duplicate edge ID {eid!r} found.")
            edge_ids.append(eid)
            edge_id_to_index[eid] = new_e_idx
            index_to_edge_id[new_e_idx] = eid

    return cls(
        new_rx,
        vertex_ids=canonical_vids,
        vertex_id_to_index=vertex_id_to_index,
        index_to_vertex_id=index_to_vertex_id,
        vertex_attr_names=set(vertex_attribute_cols),
        edge_attr_names=set(edge_attribute_cols),
        vertex_id_attr=vertex_id_attr,
        edge_ids=edge_ids,
        edge_id_to_index=edge_id_to_index if edge_ids is not None else None,
        index_to_edge_id=index_to_edge_id if edge_ids is not None else None,
        edge_id_attr=edge_id_attr,
    )


def from_networkx(
    cls: type[Graph],
    graph: Any,
    *,
    vertex_id_attr: str = "id",
    edge_id_attr: str | None = None,
    vertex_attribute_cols: Sequence[str] = (),
    edge_attribute_cols: Sequence[str] = (),
) -> Graph:
    if importlib.util.find_spec("networkx") is None:
        raise ImportError(
            "networkx is required for Graph.from_networkx. Install with `uv add --extra networkx networkx`."
        )

    directed = graph.is_directed()
    multigraph = graph.is_multigraph() if hasattr(graph, "is_multigraph") else False
    new_rx = rx.PyDiGraph(multigraph=multigraph) if directed else rx.PyGraph(multigraph=multigraph)

    canonical_vids = _canonical_vertex_ids(graph.nodes())

    for col in vertex_attribute_cols:
        if col == vertex_id_attr:
            raise ValueError(
                f"Identity column {col!r} cannot be selected as an ordinary attribute."
            )

    vertex_id_to_index: dict[Any, int] = {}
    index_to_vertex_id: dict[int, Any] = {}

    for vid in canonical_vids:
        node_data = graph.nodes[vid]
        payload = {vertex_id_attr: vid}
        for attr in vertex_attribute_cols:
            if attr not in node_data:
                raise ValueError(f"Vertex attribute {attr!r} missing on vertex {vid!r}")
            payload[attr] = _copy_payload(node_data[attr])
        idx = new_rx.add_node(payload)
        vertex_id_to_index[vid] = idx
        index_to_vertex_id[idx] = vid

    if edge_id_attr is not None:
        for col in edge_attribute_cols:
            if col == edge_id_attr:
                raise ValueError(
                    f"Edge identity attribute {col!r} cannot be selected as an ordinary attribute."
                )
    elif multigraph:
        raise ValueError("Multigraph requires explicit edge_id_attr.")

    edge_ids: list[Any] | None = [] if edge_id_attr is not None else None
    edge_id_to_index: dict[Any, int] = {}
    index_to_edge_id: dict[int, Any] = {}

    for edge in graph.edges(data=True):
        u, v, data = edge[0], edge[1], edge[2]
        u_idx = vertex_id_to_index[u]
        v_idx = vertex_id_to_index[v]
        payload = {}
        for attr in edge_attribute_cols:
            if attr not in data:
                raise ValueError(f"Edge attribute {attr!r} missing on edge ({u}, {v})")
            payload[attr] = _copy_payload(data[attr])
        new_e_idx = new_rx.add_edge(u_idx, v_idx, payload)

        if edge_id_attr is not None:
            if edge_id_attr not in data:
                raise ValueError(f"Edge data missing edge_id_attr {edge_id_attr!r}")
            eid = data[edge_id_attr]
            try:
                hash(eid)
            except TypeError:
                raise TypeError(f"Edge identity {eid!r} is not hashable.") from None
            if eid in edge_id_to_index:
                raise ValueError(f"Duplicate edge ID {eid!r} found.")
            edge_ids.append(eid)
            edge_id_to_index[eid] = new_e_idx
            index_to_edge_id[new_e_idx] = eid

    return cls(
        new_rx,
        vertex_ids=canonical_vids,
        vertex_id_to_index=vertex_id_to_index,
        index_to_vertex_id=index_to_vertex_id,
        vertex_attr_names=set(vertex_attribute_cols),
        edge_attr_names=set(edge_attribute_cols),
        vertex_id_attr=vertex_id_attr,
        edge_ids=edge_ids,
        edge_id_to_index=edge_id_to_index if edge_ids is not None else None,
        index_to_edge_id=index_to_edge_id if edge_ids is not None else None,
        edge_id_attr=edge_id_attr,
    )


def from_igraph(
    cls: type[Graph],
    graph: Any,
    *,
    vertex_id_attr: str,
    edge_id_attr: str | None = None,
    vertex_attribute_cols: Sequence[str] = (),
    edge_attribute_cols: Sequence[str] = (),
) -> Graph:
    if importlib.util.find_spec("igraph") is None:
        raise ImportError(
            "igraph is required for Graph.from_igraph. Install with `uv add --extra igraph igraph`."
        )

    directed = graph.is_directed()
    multigraph = graph.has_multiple() if hasattr(graph, "has_multiple") else False
    new_rx = rx.PyDiGraph(multigraph=multigraph) if directed else rx.PyGraph(multigraph=multigraph)

    if vertex_id_attr not in graph.vs.attributes():
        raise ValueError(
            f"Vertex ID attribute {vertex_id_attr!r} not found in igraph vertex attributes."
        )

    raw_vids = graph.vs[vertex_id_attr]
    canonical_vids = _canonical_vertex_ids(raw_vids)
    if len(set(raw_vids)) != len(raw_vids):
        raise ValueError("Duplicate vertex identities found in igraph graph.")

    for col in vertex_attribute_cols:
        if col == vertex_id_attr:
            raise ValueError(
                f"Identity column {col!r} cannot be selected as an ordinary attribute."
            )

    vertex_id_to_index: dict[Any, int] = {}
    index_to_vertex_id: dict[int, Any] = {}

    vid_to_orig_idx = {vid: i for i, vid in enumerate(raw_vids)}

    for vid in canonical_vids:
        orig_idx = vid_to_orig_idx[vid]
        payload = {vertex_id_attr: vid}
        for attr in vertex_attribute_cols:
            if attr not in graph.vs.attributes():
                raise ValueError(f"Vertex attribute {attr!r} missing in igraph vertices.")
            payload[attr] = _copy_payload(graph.vs[orig_idx][attr])
        idx = new_rx.add_node(payload)
        vertex_id_to_index[vid] = idx
        index_to_vertex_id[idx] = vid

    if edge_id_attr is not None:
        if edge_id_attr not in graph.es.attributes():
            raise ValueError(
                f"Edge ID attribute {edge_id_attr!r} not found in igraph edge attributes."
            )
        for col in edge_attribute_cols:
            if col == edge_id_attr:
                raise ValueError(
                    f"Edge identity attribute {col!r} cannot be selected as an ordinary attribute."
                )
    elif multigraph:
        raise ValueError("Multigraph requires explicit edge_id_attr.")

    edge_ids: list[Any] | None = [] if edge_id_attr is not None else None
    edge_id_to_index: dict[Any, int] = {}
    index_to_edge_id: dict[int, Any] = {}

    for edge in graph.es:
        u_vid = raw_vids[edge.source]
        v_vid = raw_vids[edge.target]
        u_idx = vertex_id_to_index[u_vid]
        v_idx = vertex_id_to_index[v_vid]
        payload = {}
        for attr in edge_attribute_cols:
            if attr not in graph.es.attributes():
                raise ValueError(f"Edge attribute {attr!r} missing in igraph edges.")
            payload[attr] = _copy_payload(edge[attr])
        new_e_idx = new_rx.add_edge(u_idx, v_idx, payload)

        if edge_id_attr is not None:
            eid = edge[edge_id_attr]
            try:
                hash(eid)
            except TypeError:
                raise TypeError(f"Edge identity {eid!r} is not hashable.") from None
            if eid in edge_id_to_index:
                raise ValueError(f"Duplicate edge ID {eid!r} found.")
            edge_ids.append(eid)
            edge_id_to_index[eid] = new_e_idx
            index_to_edge_id[new_e_idx] = eid

    return cls(
        new_rx,
        vertex_ids=canonical_vids,
        vertex_id_to_index=vertex_id_to_index,
        index_to_vertex_id=index_to_vertex_id,
        vertex_attr_names=set(vertex_attribute_cols),
        edge_attr_names=set(edge_attribute_cols),
        vertex_id_attr=vertex_id_attr,
        edge_ids=edge_ids,
        edge_id_to_index=edge_id_to_index if edge_ids is not None else None,
        index_to_edge_id=index_to_edge_id if edge_ids is not None else None,
        edge_id_attr=edge_id_attr,
    )


def from_tables(
    cls: type[Graph],
    vertex_table: pl.DataFrame,
    edge_table: pl.DataFrame,
    *,
    vertex_id_col: str,
    source_id_col: str,
    target_id_col: str,
    edge_id_col: str | None = None,
    vertex_attribute_cols: Sequence[str] = (),
    edge_attribute_cols: Sequence[str] = (),
    directed: bool = False,
    multigraph: bool = False,
    allow_null_attributes: bool = False,
) -> Graph:
    if vertex_id_col not in vertex_table.columns:
        raise ValueError(f"vertex_id_col {vertex_id_col!r} not found in vertex_table.")
    if source_id_col not in edge_table.columns or target_id_col not in edge_table.columns:
        raise ValueError("source_id_col or target_id_col not found in edge_table.")

    # Validate non-null and unique vertex IDs
    if vertex_table[vertex_id_col].null_count() > 0:
        raise ValueError(f"Null values in vertex ID column {vertex_id_col!r}.")
    raw_vids = vertex_table[vertex_id_col].to_list()
    if len(set(raw_vids)) != len(raw_vids):
        raise ValueError(f"Duplicate vertex IDs found in {vertex_id_col!r}.")

    canonical_vids = _canonical_vertex_ids(raw_vids)

    # Validate vertex attribute columns
    for col in vertex_attribute_cols:
        if col == vertex_id_col:
            raise ValueError(
                f"Identity column {col!r} cannot be selected as an ordinary attribute."
            )
        if col not in vertex_table.columns:
            raise ValueError(f"Vertex attribute column {col!r} missing from vertex_table.")
        if not allow_null_attributes and vertex_table[col].null_count() > 0:
            raise ValueError(f"Null values in vertex attribute column {col!r}.")

    # Validate edge ID column if supplied or multigraph
    edge_ids: list[Any] | None = None
    if edge_id_col is not None:
        if edge_id_col not in edge_table.columns:
            raise ValueError(f"edge_id_col {edge_id_col!r} not found in edge_table.")
        if edge_table[edge_id_col].null_count() > 0:
            raise ValueError(f"Null values in edge ID column {edge_id_col!r}.")
        raw_eids = edge_table[edge_id_col].to_list()
        if len(set(raw_eids)) != len(raw_eids):
            raise ValueError(f"Duplicate edge IDs found in {edge_id_col!r}.")
        for eid in raw_eids:
            try:
                hash(eid)
            except TypeError:
                raise TypeError(f"Edge identity {eid!r} is not hashable.") from None
        edge_ids = raw_eids
    elif multigraph:
        raise ValueError("Multigraph requires explicit edge_id_col.")

    # Validate edge attribute columns
    for col in edge_attribute_cols:
        if edge_id_col is not None and col == edge_id_col:
            raise ValueError(
                f"Edge identity column {col!r} cannot be selected as an ordinary attribute."
            )
        if col not in edge_table.columns:
            raise ValueError(f"Edge attribute column {col!r} missing from edge_table.")
        if not allow_null_attributes and edge_table[col].null_count() > 0:
            raise ValueError(f"Null values in edge attribute column {col!r}.")

    new_rx = rx.PyDiGraph(multigraph=multigraph) if directed else rx.PyGraph(multigraph=multigraph)

    vertex_id_to_index: dict[Any, int] = {}
    index_to_vertex_id: dict[int, Any] = {}

    v_lookup = {row[vertex_id_col]: row for row in vertex_table.iter_rows(named=True)}

    for vid in canonical_vids:
        row = v_lookup[vid]
        payload = {vertex_id_col: vid}
        for attr in vertex_attribute_cols:
            payload[attr] = row[attr]
        idx = new_rx.add_node(payload)
        vertex_id_to_index[vid] = idx
        index_to_vertex_id[idx] = vid

    # Validate edges
    seen_pairs: set[tuple[Any, Any]] = set()
    edge_id_to_index: dict[Any, int] = {}
    index_to_edge_id: dict[int, Any] = {}

    for row_idx, row in enumerate(edge_table.iter_rows(named=True)):
        src = row[source_id_col]
        dst = row[target_id_col]
        if src not in vertex_id_to_index or dst not in vertex_id_to_index:
            raise ValueError(f"Unknown edge endpoint ({src}, {dst}).")

        pair_key = (src, dst) if directed else ((src, dst) if src <= dst else (dst, src))
        if not multigraph:
            if pair_key in seen_pairs:
                raise ValueError(
                    f"Repeated edge endpoint pair {pair_key} without multigraph=True."
                )
            seen_pairs.add(pair_key)

        src_idx = vertex_id_to_index[src]
        dst_idx = vertex_id_to_index[dst]
        payload = {attr: row[attr] for attr in edge_attribute_cols}
        e_idx = new_rx.add_edge(src_idx, dst_idx, payload)
        if edge_ids is not None:
            eid = edge_ids[row_idx]
            edge_id_to_index[eid] = e_idx
            index_to_edge_id[e_idx] = eid

    return cls(
        new_rx,
        vertex_ids=canonical_vids,
        vertex_id_to_index=vertex_id_to_index,
        index_to_vertex_id=index_to_vertex_id,
        vertex_attr_names=set(vertex_attribute_cols),
        edge_attr_names=set(edge_attribute_cols),
        vertex_id_attr=vertex_id_col,
        edge_ids=edge_ids,
        edge_id_to_index=edge_id_to_index if edge_ids is not None else None,
        index_to_edge_id=index_to_edge_id if edge_ids is not None else None,
        edge_id_attr=edge_id_col,
    )
