import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import marimo as mo

    from topographer.examples import hamming
    from topographer.examples._cases import CASES, export_case
    from topographer.examples.hamming import HAMMING_TREE_NODE_ATTRS as TREE_NODE_ATTRS
    from topographer.examples.hamming import build_hamming_topology
    from topographer.examples._presentation import (
        build_canonical_persistence_table,
        build_example_tree_plot_data,
        build_graph_plot_data,
        compose_merge_tree_tabs,
        filter_qualifying_tree_pairs,
        plot_canonical_persistence_diagrams,
        plot_scalar_network_view,
        plot_tree_with_persistence,
        summarize_scalar_network,
    )

    return (
        CASES,
        Path,
        TREE_NODE_ATTRS,
        build_hamming_topology,
        build_canonical_persistence_table,
        build_example_tree_plot_data,
        build_graph_plot_data,
        compose_merge_tree_tabs,
        export_case,
        filter_qualifying_tree_pairs,
        hamming,
        mo,
        plot_canonical_persistence_diagrams,
        plot_scalar_network_view,
        plot_tree_with_persistence,
        summarize_scalar_network,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Canonical 4-Hypercube Topology Workflow

    This browser-native notebook demonstrates the end-to-end topological data analysis workflow on the canonical
    4-dimensional hypercube graph. Sixteen binary sequences form the vertices, connected by 32 Hamming-radius-1
    edges, endowed with a normalized scalar function.
    """)
    return


@app.cell
def _(build_hamming_topology, hamming):
    # Stage 1: Build the hypercube workspace (network plus join and split trees)
    topology = build_hamming_topology()
    raw_nodes = hamming.hypercube_node_table()
    raw_edges = hamming.hypercube_edge_table()
    coordinates = hamming.hypercube_coordinates()
    sn = topology.network
    graph = sn.graph
    return topology, coordinates, graph, raw_edges, raw_nodes, sn


@app.cell(hide_code=True)
def _(mo, raw_edges, raw_nodes):
    mo.md(f"""
    ## 1. Raw Input Tables

    The network comprises **{raw_nodes.height}** vertices and **{raw_edges.height}** edges.
    """)
    return


@app.cell
def _(mo, raw_nodes):
    mo.ui.table(raw_nodes)
    return


@app.cell
def _(topology):
    # Stage 2: Read the network and both trees from the workspace
    network = topology.network
    join_tree = topology.join_tree()
    split_tree = topology.split_tree()
    return join_tree, network, split_tree


@app.cell
def _(build_graph_plot_data, coordinates, network):
    # Stage 3: Render scalar-encoded network view at layout coordinates
    sn_plot_data = build_graph_plot_data(network, coordinates=coordinates)
    return (sn_plot_data,)


@app.cell
def _(mo, plot_scalar_network_view, sn_plot_data):
    mo.ui.altair_chart(plot_scalar_network_view(sn_plot_data, scalar_attr="scalar"))
    return


@app.cell
def _(network, summarize_scalar_network):
    # Stage 4: Compact analytical summary of the ScalarNetwork
    summary_table = summarize_scalar_network(network)
    return (summary_table,)


@app.cell
def _(mo, summary_table):
    mo.ui.table(summary_table)
    return


@app.cell
def _(build_canonical_persistence_table, topology):
    # Stage 5: Derive the canonical component persistence table
    canonical_persistence_table = build_canonical_persistence_table(
        topology.component_persistence()
    )
    return (canonical_persistence_table,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Component Persistence Diagram

    Component (H0) persistence pairs are drawn on one coordinate plane:
    - **Vertical axis ($Y$)**: Birth scalar, reinforcing the physical mental model of sweeping a level plane vertically across the network.
    - **Horizontal axis ($X$)**: Death scalar, relative to the diagonal reference line $Y = X$.
    - **Above diagonal ($Y > X$)**: Features with $\text{birth} > \text{death}$, comprising relative superlevel component merges (triangles up $\vartriangle$).
    - **Below diagonal ($Y < X$)**: Features with $\text{birth} < \text{death}$, comprising ordinary sublevel component merges (triangles down $\triangledown$ pointing to local minima) and the global component (diamond $\lozenge$ pointing to the global maximum).

    """)
    return


@app.cell
def _(canonical_persistence_table, mo, plot_canonical_persistence_diagrams):
    mo.ui.altair_chart(
        plot_canonical_persistence_diagrams(
            canonical_persistence_table, scalar_domain=(0.0, 1.0)
        ),
        chart_selection=False,
    )
    return


@app.cell
def _(canonical_persistence_table, mo):
    mo.ui.table(canonical_persistence_table)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3. Join and Split Merge Trees

    Join and split trees summarize sublevel and superlevel component connectivity. Both tabs below
    present the reduced graph, augmented graph (with all input vertices), and ordinary-persistence
    simplification at equal depth with independent threshold controls.
    """)
    return


@app.cell
def _(mo):
    # Independent threshold sliders for Join and Split tree simplification
    # Default is 10% of scalar range: 0.10 * (0.941 - 0.059) = ~0.09
    join_threshold_slider = mo.ui.slider(
        start=0.01,
        stop=0.5,
        step=0.01,
        value=0.09,
        label="Join Simplification Threshold",
    )
    split_threshold_slider = mo.ui.slider(
        start=0.01,
        stop=0.5,
        step=0.01,
        value=0.09,
        label="Split Simplification Threshold",
    )
    return join_threshold_slider, split_threshold_slider


@app.cell
def _(
    TREE_NODE_ATTRS,
    build_example_tree_plot_data,
    compose_merge_tree_tabs,
    filter_qualifying_tree_pairs,
    join_threshold_slider,
    mo,
    plot_tree_with_persistence,
    split_threshold_slider,
    topology,
):
    # Stage 7: Render mirrored Marimo tabs with independent simplification controls
    # Join tab content
    _j_thresh = join_threshold_slider.value
    _j_tree = topology.join_tree()
    _j_plot_data = build_example_tree_plot_data(_j_tree, node_attrs=TREE_NODE_ATTRS)
    _j_simplified = build_example_tree_plot_data(
        _j_tree.simplify(_j_thresh), node_attrs=TREE_NODE_ATTRS
    )
    _j_qualifying = filter_qualifying_tree_pairs(
        _j_tree.persistence(), _j_thresh, tree_kind="join"
    )

    _join_red_chart = plot_tree_with_persistence(
        _j_tree, _j_plot_data, domain=(0.0, 1.0), title="Reduced Join Tree"
    )
    _join_aug_chart = plot_tree_with_persistence(
        _j_tree,
        _j_plot_data,
        domain=(0.0, 1.0),
        title="Augmented Join Tree (All Vertices Retained)",
    )
    _join_simp_chart = plot_tree_with_persistence(
        _j_tree,
        _j_simplified,
        domain=(0.0, 1.0),
        title=f"Simplified Join Tree (thr={_j_thresh:.2f})",
        threshold=_j_thresh,
    )

    _join_content = mo.vstack(
        [
            mo.md("### Primary: Reduced Join Tree & Sublevel Persistence"),
            _join_red_chart,
            mo.accordion({"Augmented Join Tree (Secondary)": _join_aug_chart}),
            mo.md(f"### Join Tree Simplification (threshold = {_j_thresh:.2f})"),
            join_threshold_slider,
            mo.md(
                "**Qualifying Ordinary Persistence Pairs (persistence <= threshold):**"
            ),
            mo.ui.table(_j_qualifying),
            _join_simp_chart,
        ]
    )

    # Split tab content
    _s_thresh = split_threshold_slider.value
    _s_tree = topology.split_tree()
    _s_plot_data = build_example_tree_plot_data(_s_tree, node_attrs=TREE_NODE_ATTRS)
    _s_simplified = build_example_tree_plot_data(
        _s_tree.simplify(_s_thresh), node_attrs=TREE_NODE_ATTRS
    )
    _s_qualifying = filter_qualifying_tree_pairs(
        _s_tree.persistence(), _s_thresh, tree_kind="split"
    )

    _split_red_chart = plot_tree_with_persistence(
        _s_tree, _s_plot_data, domain=(0.0, 1.0), title="Reduced Split Tree"
    )
    _split_aug_chart = plot_tree_with_persistence(
        _s_tree,
        _s_plot_data,
        domain=(0.0, 1.0),
        title="Augmented Split Tree (All Vertices Retained)",
    )
    _split_simp_chart = plot_tree_with_persistence(
        _s_tree,
        _s_simplified,
        domain=(0.0, 1.0),
        title=f"Simplified Split Tree (thr={_s_thresh:.2f})",
        threshold=_s_thresh,
    )

    _split_content = mo.vstack(
        [
            mo.md("### Primary: Reduced Split Tree & Superlevel Persistence"),
            _split_red_chart,
            mo.accordion({"Augmented Split Tree (Secondary)": _split_aug_chart}),
            mo.md(f"### Split Tree Simplification (threshold = {_s_thresh:.2f})"),
            split_threshold_slider,
            mo.md(
                "**Qualifying Ordinary Persistence Pairs (persistence <= threshold):**"
            ),
            mo.ui.table(_s_qualifying),
            _split_simp_chart,
        ]
    )

    tree_tabs = compose_merge_tree_tabs(_join_content, _split_content)
    tree_tabs
    return


@app.cell
def _(CASES, Path, export_case, topology):
    # Stage 8: Deterministic artifact export callable
    def export_hypercube_artifacts(
        output_dir: Path | str = "output/hypercube",
    ) -> dict[str, Path]:
        """Export deterministic Hamming tables and charts derived from the workspace."""
        return export_case(CASES["hypercube"], topology, Path(output_dir))

    return (export_hypercube_artifacts,)


if __name__ == "__main__":
    app.run()
