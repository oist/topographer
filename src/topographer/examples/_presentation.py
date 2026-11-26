"""Shared presentation primitives and helpers for canonical topology workflows.

Provides:
- Graph / ScalarNetwork plot tables and coordinated Altair charts sharing identical layout coordinates.
- Component (H0) persistence plot tables retaining pair kind.
- Two coordinated layered persistence panels with shared domains and encodings.
- Augmented, reduced, and simplified merge-tree plot tables using established IPlotX geometry.
- Independent merge-tree and persistence panels composed with Altair.
- Marimo tree tabs composition helper.
- Deterministic CSV/SVG artifact writers.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl
import rustworkx as rx

from topographer.graph import Graph, ScalarNetwork
from topographer.plotting import (
    PersistencePlotConfig,
    TreePlotConfig,
    TreePlotData,
    build_tree_plot_data,
    plot_persistence,
    plot_tree,
)
from topographer.topology import MergeTree, PersistenceResult, ScalarNetworkTopology

__all__ = [
    "GraphPlotData",
    "build_canonical_persistence_table",
    "build_example_charts",
    "build_example_tables",
    "build_example_tree_plot_data",
    "build_graph_plot_data",
    "build_tree_persistence_threshold_table",
    "build_tree_stage_plot_data",
    "compose_merge_tree_tabs",
    "export_example_artifacts",
    "export_stage_artifacts",
    "filter_qualifying_tree_pairs",
    "plot_canonical_persistence_diagrams",
    "plot_graph_view",
    "plot_scalar_network_view",
    "plot_tree_with_persistence",
    "scalar_domain_of",
    "summarize_scalar_network",
]


def summarize_scalar_network(network: ScalarNetwork) -> pl.DataFrame:
    """Build compact analytical summary table for a ScalarNetwork."""
    rx_g = network.graph
    v_count = network.vertex_count
    e_count = network.edge_count
    cycle_rank = e_count - v_count + 1 if v_count > 0 else 0
    scalars = [float(network._vertex_scalars[vid]) for vid in network.vertex_ids]
    min_s = min(scalars) if scalars else 0.0
    max_s = max(scalars) if scalars else 0.0
    range_s = max_s - min_s
    unique_levels = len(set(scalars))
    tied_levels = len(scalars) - unique_levels
    connected = rx.is_connected(rx_g) if hasattr(rx, "is_connected") else True

    return pl.DataFrame(
        {
            "metric": [
                "Vertices",
                "Edges",
                "Connected",
                "Cycle Rank (|E| - |V| + 1)",
                "Scalar Min",
                "Scalar Max",
                "Scalar Range",
                "Unique Scalar Levels",
                "Tied Levels",
            ],
            "value": [
                str(v_count),
                str(e_count),
                str(connected),
                str(cycle_rank),
                f"{min_s:.4f}",
                f"{max_s:.4f}",
                f"{range_s:.4f}",
                str(unique_levels),
                str(tied_levels),
            ],
        }
    )


@dataclass(frozen=True, slots=True)
class GraphPlotData:
    """Tabular plot data for graph and scalar network layouts."""

    node_table: pl.DataFrame
    edge_table: pl.DataFrame


def build_graph_plot_data(
    graph: Graph | ScalarNetwork,
    *,
    coordinates: dict[Any, tuple[float, float]],
    scalar_attr: str | None = None,
) -> GraphPlotData:
    """Build inspectable node and edge tables consuming fixed layout coordinates.

    Parameters
    ----------
    graph
        Graph or ScalarNetwork instance.
    coordinates
        Mapping from stable vertex identity to (x, y) float coordinates.
    scalar_attr
        Optional vertex scalar attribute name. If graph is a ScalarNetwork,
        defaults to its declared scalar_attr.
    """
    if isinstance(graph, ScalarNetwork):
        rx_g = graph.graph
        resolved_scalar = scalar_attr or graph.scalar_attr
        v_ids = graph.vertex_ids
        v_scalars = graph._vertex_scalars
    elif isinstance(graph, Graph):
        rx_g = graph.graph
        resolved_scalar = scalar_attr
        v_ids = graph.vertex_ids
        if resolved_scalar is not None:
            v_scalars = {}
            for vid in v_ids:
                idx = graph._vertex_id_to_index[vid]
                data = graph._graph.get_node_data(idx)
                v_scalars[vid] = float(data[resolved_scalar])
        else:
            v_scalars = {}
    else:
        raise TypeError(f"Expected Graph or ScalarNetwork, got {type(graph).__name__}")

    # Build node rows
    node_rows: list[dict[str, Any]] = []
    for vid in v_ids:
        if vid not in coordinates:
            raise KeyError(f"Missing coordinate for vertex {vid!r}")
        x, y = coordinates[vid]
        row: dict[str, Any] = {
            "vertex_id": str(vid),
            "x": float(x),
            "y": float(y),
        }
        if resolved_scalar is not None:
            row[resolved_scalar] = float(v_scalars[vid])
        node_rows.append(row)

    node_table = pl.DataFrame(node_rows)

    # Build edge rows
    edge_rows: list[dict[str, Any]] = []
    for e_idx in rx_g.edge_indices():
        u_idx, v_idx = rx_g.get_edge_endpoints_by_index(e_idx)
        u_id = v_ids[u_idx]
        v_id = v_ids[v_idx]
        x_src, y_src = coordinates[u_id]
        x_dst, y_dst = coordinates[v_id]
        edge_rows.append(
            {
                "edge_id": f"{u_id}-{v_id}",
                "source": str(u_id),
                "target": str(v_id),
                "x_source": float(x_src),
                "y_source": float(y_src),
                "x_target": float(x_dst),
                "y_target": float(y_dst),
            }
        )

    edge_table = pl.DataFrame(edge_rows)
    return GraphPlotData(node_table=node_table, edge_table=edge_table)


def plot_graph_view(
    plot_data: GraphPlotData,
    *,
    width: int = 400,
    height: int = 300,
    node_size: int = 120,
    edge_width: float = 1.5,
) -> Any:
    """Render topology-only Graph view without scalar color encoding."""
    import altair as alt

    edges = (
        alt.Chart(plot_data.edge_table)
        .mark_rule(color="#94a3b8", strokeWidth=edge_width)
        .encode(
            x=alt.X("x_source:Q", axis=None),
            y=alt.Y("y_source:Q", axis=None),
            x2="x_target:Q",
            y2="y_target:Q",
        )
    )

    nodes = (
        alt.Chart(plot_data.node_table)
        .mark_circle(size=node_size, color="#0284c7", stroke="#0f172a", strokeWidth=1)
        .encode(
            x=alt.X("x:Q", axis=None),
            y=alt.Y("y:Q", axis=None),
            tooltip=[alt.Tooltip("vertex_id:N", title="Vertex ID")],
        )
    )

    labels = (
        alt.Chart(plot_data.node_table)
        .mark_text(dy=-12, fontSize=11, color="#0f172a")
        .encode(
            x=alt.X("x:Q", axis=None),
            y=alt.Y("y:Q", axis=None),
            text="vertex_id:N",
        )
    )

    return (
        (edges + nodes + labels)
        .properties(width=width, height=height)
        .configure_view(strokeWidth=0)
    )


def plot_scalar_network_view(
    plot_data: GraphPlotData,
    *,
    scalar_attr: str,
    width: int = 400,
    height: int = 300,
    node_size: int = 120,
    edge_width: float = 1.5,
    highlight_edges: Sequence[tuple[Any, Any]] | None = None,
) -> Any:
    """Render ScalarNetwork view adding scalar color encoding and optional edge highlighting."""
    import altair as alt

    edge_df = plot_data.edge_table
    if highlight_edges is not None:
        norm_hl = {
            (str(u), str(v)) if str(u) <= str(v) else (str(v), str(u)) for u, v in highlight_edges
        }

        def is_hl(row: dict[str, Any]) -> bool:
            u, v = str(row["source"]), str(row["target"])
            key = (u, v) if u <= v else (v, u)
            return key in norm_hl

        edge_df = edge_df.with_columns(
            pl.struct(["source", "target"])
            .map_elements(is_hl, return_dtype=pl.Boolean)
            .alias("is_highlighted")
        )
    else:
        edge_df = edge_df.with_columns(pl.lit(False).alias("is_highlighted"))

    edges = (
        alt.Chart(edge_df)
        .mark_rule()
        .encode(
            x=alt.X("x_source:Q", axis=None),
            y=alt.Y("y_source:Q", axis=None),
            x2="x_target:Q",
            y2="y_target:Q",
            color=alt.condition(
                "datum.is_highlighted === true",
                alt.value("#dc2626"),
                alt.value("#cbd5e1" if highlight_edges is not None else "#94a3b8"),
            ),
            strokeWidth=alt.condition(
                "datum.is_highlighted === true",
                alt.value(edge_width * 2.5),
                alt.value(edge_width),
            ),
        )
    )

    nodes = (
        alt.Chart(plot_data.node_table)
        .mark_circle(size=node_size, stroke="#0f172a", strokeWidth=1)
        .encode(
            x=alt.X("x:Q", axis=None),
            y=alt.Y("y:Q", axis=None),
            color=alt.Color(
                f"{scalar_attr}:Q",
                scale=alt.Scale(scheme="plasma"),
                title=scalar_attr.replace("_", " ").title(),
            ),
            tooltip=[
                alt.Tooltip("vertex_id:N", title="Vertex ID"),
                alt.Tooltip(f"{scalar_attr}:Q", title="Scalar", format=".3f"),
            ],
        )
    )

    labels = (
        alt.Chart(plot_data.node_table)
        .mark_text(dy=-12, fontSize=11, color="#0f172a")
        .encode(
            x=alt.X("x:Q", axis=None),
            y=alt.Y("y:Q", axis=None),
            text="vertex_id:N",
        )
    )

    return (
        (edges + nodes + labels)
        .properties(width=width, height=height)
        .configure_view(strokeWidth=0)
    )


def build_canonical_persistence_table(component_result: PersistenceResult) -> pl.DataFrame:
    """Build the component (H0) persistence plot table with joined endpoint labels."""

    return component_result.pairs.with_columns(
        pl.col("birth_simplex").list.join(", ").alias("birth_endpoint"),
        pl.col("death_simplex").list.join(", ").alias("death_endpoint"),
    )


def plot_canonical_persistence_diagrams(
    persistence_table: pl.DataFrame,
    *,
    scalar_domain: tuple[float, float],
    width: int = 350,
    height: int = 350,
) -> Any:
    """Render the component (H0) persistence diagram with ``plot_persistence``."""
    from topographer.plotting import plot_persistence

    return plot_persistence(
        persistence_table,
        config=PersistencePlotConfig(width=width, height=height, domain=scalar_domain),
        color_by="pair_kind",
        shape_by="pair_kind",
        tooltip_attrs=("pair_id", "pair_kind", "persistence"),
        event_legend_title="Pair Kind",
        semantic_event_colors={
            "ordinary": "#2563eb",
            "relative": "#7c3aed",
            "extended_positive": "#059669",
        },
        semantic_event_shapes={
            "ordinary": "triangle-down",
            "relative": "triangle-up",
            "extended_positive": "diamond",
        },
    )


def build_tree_stage_plot_data(
    tree_graph: rx.PyDiGraph,
    *,
    tree_kind: str,
    scalar_attr: str = "scalar",
) -> TreePlotData:
    """Build TreePlotData for augmented, reduced, or simplified merge trees using IPlotX."""
    root_kind = "maximum" if tree_kind == "join" else "minimum"
    resolved_attr = scalar_attr
    if tree_graph.node_indices():
        sample_node = tree_graph.get_node_data(tree_graph.node_indices()[0])
        if isinstance(sample_node, dict):
            if resolved_attr not in sample_node and "scalar" in sample_node:
                resolved_attr = "scalar"
    return build_tree_plot_data(
        tree_graph,
        scalar_attr=resolved_attr,
        root_kind=root_kind,
        node_attrs=("vertex_id", "label", "node_id", "genotype_id"),
        horizontal_layout="iplotx",
    )


def filter_qualifying_tree_pairs(
    persistence_table: pl.DataFrame,
    threshold: float,
    *,
    tree_kind: str = "join",
) -> pl.DataFrame:
    """Filter ordinary persistence pairs with persistence <= threshold."""
    if threshold <= 0:
        return persistence_table.filter(pl.lit(False))
    return persistence_table.filter(
        (pl.col("pair_kind") == "ordinary") & (pl.col("persistence") <= threshold)
    )


def build_tree_persistence_threshold_table(
    scalar_domain: tuple[float, float], *, threshold: float | None, tree_kind: str
) -> pl.DataFrame:
    """Return a clipped diagonal guide for the ordinary-persistence threshold."""
    if threshold is None or threshold <= 0:
        return pl.DataFrame(schema={"x": pl.Float64, "y": pl.Float64})
    lower, upper = scalar_domain
    if threshold > upper - lower:
        return pl.DataFrame(schema={"x": pl.Float64, "y": pl.Float64})
    if tree_kind == "join":
        return pl.DataFrame({"x": [lower + threshold, upper], "y": [lower, upper - threshold]})
    return pl.DataFrame({"x": [lower, upper - threshold], "y": [lower + threshold, upper]})


def compose_merge_tree_tabs(
    join_content: Any,
    split_content: Any,
) -> Any:
    """Create a Marimo tabs interface presenting Join tree and Split tree at equal depth."""
    import marimo as mo

    return mo.ui.tabs(
        {
            "Join tree": join_content,
            "Split tree": split_content,
        }
    )


def export_stage_artifacts(
    tables: dict[str, pl.DataFrame],
    charts: dict[str, Any],
    output_dir: Path | str,
    *,
    prefix: str,
) -> dict[str, Path]:
    """Deterministically write computed Polars tables and Altair charts as CSV, HTML, and SVG."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    for name, df in tables.items():
        csv_path = out / f"{prefix}_{name}.csv"
        # CSV sink in Polars does not support nested List columns directly; format list columns as strings
        export_df = df
        list_cols = [
            col
            for col, dtype in zip(df.columns, df.dtypes, strict=False)
            if isinstance(dtype, pl.List)
        ]
        if list_cols:
            export_df = df.with_columns([pl.col(c).list.join(", ") for c in list_cols])
        export_df.write_csv(csv_path)
        paths[f"table_{name}"] = csv_path

    for name, chart in charts.items():
        svg_path = out / f"{prefix}_{name}.svg"
        html_path = out / f"{prefix}_{name}.html"
        chart.save(svg_path)
        chart.save(html_path, inline=True)
        paths[f"chart_{name}_svg"] = svg_path
        paths[f"chart_{name}_html"] = html_path

    return paths


def build_example_tree_plot_data(tree: MergeTree, *, node_attrs: Sequence[str]) -> TreePlotData:
    """Build plot data for a merge tree (reduced or simplified) of an example workspace."""
    return build_tree_plot_data(
        tree.graph,
        scalar_attr="scalar",
        root_kind="maximum" if tree.kind == "join" else "minimum",
        node_attrs=tuple(node_attrs),
        horizontal_layout="iplotx",
    )


def plot_tree_with_persistence(
    tree: MergeTree,
    plot_data: TreePlotData,
    *,
    domain: tuple[float, float],
    title: str,
    threshold: float | None = None,
) -> Any:
    """Compose a merge tree beside its persistence diagram, with an optional threshold guide."""
    import altair as alt

    persistence = plot_persistence(
        tree.persistence(),
        config=PersistencePlotConfig(width=280, height=300, domain=domain),
        color_by="pair_kind",
        shape_by="pair_kind",
        include_config=False,
        event_legend_title="Pair Kind",
        semantic_event_colors={"ordinary": "#2563eb", "extended": "#059669"},
        semantic_event_shapes={
            "ordinary": "triangle-down" if tree.kind == "join" else "triangle-up",
            "extended": "diamond",
        },
    )
    if threshold is not None:
        guide = (
            alt.Chart(
                build_tree_persistence_threshold_table(
                    domain, threshold=threshold, tree_kind=tree.kind
                )
            )
            .mark_line(color="#94a3b8", strokeDash=[4, 4])
            .encode(x="x:Q", y="y:Q")
        )
        persistence = alt.layer(persistence, guide)
    return (
        alt.hconcat(
            plot_tree(
                plot_data,
                config=TreePlotConfig(
                    width=280, height=300, scalar_domain=domain, edge_color="#475569"
                ),
                interactive=False,
                include_config=False,
            ).properties(title=title),
            persistence,
        )
        .resolve_scale(x="independent", y="shared", color="independent", shape="independent")
        .configure_view(strokeWidth=0)
    )


def scalar_domain_of(network: ScalarNetwork) -> tuple[float, float]:
    """Return the (minimum, maximum) scalar value of a network."""
    scalars = [float(network._vertex_scalars[vid]) for vid in network.vertex_ids]
    return (min(scalars), max(scalars)) if scalars else (0.0, 1.0)


def build_example_tables(
    topology: ScalarNetworkTopology,
    *,
    vertices: pl.DataFrame,
    edges: pl.DataFrame,
    simplification_threshold: float,
    tree_node_attrs: Sequence[str],
) -> dict[str, pl.DataFrame]:
    """Derive the example tables from a workspace (computed or restored from disk)."""
    network = topology.network
    low, high = scalar_domain_of(network)
    summary = pl.DataFrame(
        [
            {"metric": "vertex_count", "value": network.vertex_count},
            {"metric": "edge_count", "value": network.edge_count},
            {"metric": "scalar_min", "value": low},
            {"metric": "scalar_max", "value": high},
        ]
    )
    tables: dict[str, pl.DataFrame] = {
        "vertices": vertices,
        "edges": edges,
        "summary": summary,
        "persistence": build_canonical_persistence_table(topology.component_persistence()),
    }
    for kind, tree in (("join", topology.join_tree()), ("split", topology.split_tree())):
        plot_data = build_example_tree_plot_data(tree, node_attrs=tree_node_attrs)
        persistence = tree.persistence()
        tables[f"{kind}_events"] = plot_data.node_table.select(
            "tree_node_index", "event", "display_event", "scalar"
        )
        tables[f"{kind}_persistence"] = persistence
        tables[f"{kind}_qualifying"] = filter_qualifying_tree_pairs(
            persistence, simplification_threshold, tree_kind=kind
        )
    return tables


def build_example_charts(
    topology: ScalarNetworkTopology,
    *,
    vertices: pl.DataFrame,
    edges: pl.DataFrame,
    simplification_threshold: float,
    tree_node_attrs: Sequence[str],
) -> dict[str, Any]:
    """Derive the example charts from a workspace (computed or restored from disk)."""
    network = topology.network
    scalar_attr = network.scalar_attr
    domain = scalar_domain_of(network)
    vertex_col = (
        "node_id"
        if "node_id" in vertices.columns
        else ("binary_genotype" if "binary_genotype" in vertices.columns else "vertex_id")
    )
    graph = Graph.from_tables(
        vertices,
        edges,
        vertex_id_col=vertex_col,
        source_id_col="source",
        target_id_col="target",
        vertex_attribute_cols=tuple(
            c for c in (scalar_attr, "x", "y") if c in vertices.columns and c != vertex_col
        ),
    )
    coordinates = {row[vertex_col]: (row["x"], row["y"]) for row in vertices.to_dicts()}
    network_data = build_graph_plot_data(graph, coordinates=coordinates, scalar_attr=scalar_attr)
    charts: dict[str, Any] = {
        "scalar_network": plot_scalar_network_view(network_data, scalar_attr=scalar_attr),
        "canonical_persistence": plot_canonical_persistence_diagrams(
            build_canonical_persistence_table(topology.component_persistence()),
            scalar_domain=domain,
        ),
    }
    for kind, tree in (("join", topology.join_tree()), ("split", topology.split_tree())):
        reduced = build_example_tree_plot_data(tree, node_attrs=tree_node_attrs)
        simplified = tree.simplify(simplification_threshold)
        simplified_data = build_example_tree_plot_data(simplified, node_attrs=tree_node_attrs)
        name = kind.capitalize()
        charts[f"{kind}_tree_reduced"] = plot_tree_with_persistence(
            tree, reduced, domain=domain, title=f"Reduced {name} Tree"
        )
        charts[f"{kind}_tree_augmented"] = plot_tree_with_persistence(
            tree, reduced, domain=domain, title=f"Augmented {name} Tree"
        )
        charts[f"{kind}_tree_simplified"] = plot_tree_with_persistence(
            tree,
            simplified_data,
            domain=domain,
            title=f"Simplified {name} Tree",
            threshold=simplification_threshold,
        )
    return charts


def export_example_artifacts(
    topology: ScalarNetworkTopology,
    *,
    vertices: pl.DataFrame,
    edges: pl.DataFrame,
    output_dir: Path | str,
    prefix: str,
    simplification_threshold: float,
    tree_node_attrs: Sequence[str],
) -> dict[str, Path]:
    """Write the example tables (CSV) and charts (SVG and HTML) derived from a workspace."""
    options = {
        "vertices": vertices,
        "edges": edges,
        "simplification_threshold": simplification_threshold,
        "tree_node_attrs": tree_node_attrs,
    }
    return export_stage_artifacts(
        build_example_tables(topology, **options),
        build_example_charts(topology, **options),
        output_dir=output_dir,
        prefix=prefix,
    )
