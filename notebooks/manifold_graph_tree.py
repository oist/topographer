import marimo

__generated_with = "0.23.8"
app = marimo.App(width="full", auto_download=["png", "svg"])


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Manifold · Graph · Tree

    This figure is a conceptual construction of a split tree. A smooth two-peak
    fitness manifold supplies ground truth; a graph samples that manifold in three
    dimensions; and the reduced split tree records decreasing-superlevel components.

    The manifold is deliberately illustrative. It is not a claim that sequence
    space is a smooth two-dimensional object.
    """)


@app.cell
def _():
    import anywidget
    import marimo as mo
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np
    import rustworkx as rx
    import traitlets

    from topographer.plotting._layout import (
        build_iplotx_tree_layout,
        extract_iplotx_tree_coordinates,
    )
    from topographer.topology import build_split_tree

    return (
        build_iplotx_tree_layout,
        build_split_tree,
        extract_iplotx_tree_coordinates,
        anywidget,
        mpl,
        mo,
        np,
        nx,
        plt,
        rx,
        traitlets,
    )


@app.cell
def _(
    build_iplotx_tree_layout,
    build_split_tree,
    extract_iplotx_tree_coordinates,
    mpl,
    np,
    nx,
    plt,
    rx,
):
    from itertools import pairwise
    from pathlib import Path

    ASSET_STEM = "manifold_graph_tree"
    PALETTES = {
        "light": dict(
            background="#ffffff",
            text="#111111",
            neutral="#4D4D4D",
            inactive="#D9D9D9",
            surface="#E9E9E9",
            plane="#4D4D4D",
            edge="#111111",
        ),
        "dark": dict(
            background="#0d1117",
            text="#f0f6fc",
            neutral="#f0f6fc",
            inactive="#2d333b",
            surface="#2d333b",
            plane="#ffffff",
            edge="#f0f6fc",
        ),
    }
    BLUE, ORANGE, GREEN = "#0072B2", "#D55E00", "#009E73"

    GRID_COLUMNS = 6
    GRID_ROWS = 6
    LEFT_PEAK_INDEX = 7
    SADDLE_INDEX = 15
    RIGHT_PEAK_INDEX = 28
    GLOBAL_MINIMUM_INDEX = 5
    HIGH_PEAK_THRESHOLD = 0.99
    LOW_PEAK_THRESHOLD = 0.70
    FULL_GRAPH_THRESHOLD = 0.00

    def sample_landscape_grid():
        """Return a regular 6×6 cell-centre lattice over the unit square."""
        coordinates = (np.arange(GRID_COLUMNS) + 0.5) / GRID_COLUMNS
        return np.array(
            [
                (x_coordinate, y_coordinate)
                for y_coordinate in coordinates
                for x_coordinate in coordinates
            ]
        )

    LEFT_GAUSSIAN_CENTER = np.array([0.25, 0.25])
    RIGHT_GAUSSIAN_CENTER = np.array([0.75, 0.75])
    GLOBAL_MINIMUM = np.array([1.0, 0.0])

    def raw_fitness(x, y):
        """Return a two-Gaussian landscape whose right-hand peak is higher."""
        left_peak = 0.72 * np.exp(
            -(
                (x - LEFT_GAUSSIAN_CENTER[0]) ** 2 / (2 * 0.20**2)
                + (y - LEFT_GAUSSIAN_CENTER[1]) ** 2 / (2 * 0.20**2)
            )
        )
        right_peak = np.exp(
            -(
                (x - RIGHT_GAUSSIAN_CENTER[0]) ** 2 / (2 * 0.20**2)
                + (y - RIGHT_GAUSSIAN_CENTER[1]) ** 2 / (2 * 0.20**2)
            )
        )
        return left_peak + right_peak + 0.04

    def fitness_gradient_and_hessian(point):
        """Return analytic derivatives of the two-Gaussian fitness manifold."""
        centers = (LEFT_GAUSSIAN_CENTER, RIGHT_GAUSSIAN_CENTER)
        amplitudes = (0.72, 1.0)
        width = 0.20
        gradient = np.zeros(2)
        hessian = np.zeros((2, 2))
        for center, amplitude in zip(centers, amplitudes):
            displacement = point - center
            value = amplitude * np.exp(-np.dot(displacement, displacement) / (2 * width**2))
            gradient -= value * displacement / width**2
            hessian += value * (
                np.outer(displacement, displacement) / width**4 - np.eye(2) / width**2
            )
        return gradient, hessian

    def stationary_point(seed):
        """Find a stationary point with deterministic Newton iterations."""
        point = np.array(seed, dtype=float)
        for _ in range(20):
            gradient, hessian = fitness_gradient_and_hessian(point)
            point -= np.linalg.solve(hessian, gradient)
        return point

    LEFT_PEAK = stationary_point(LEFT_GAUSSIAN_CENTER)
    SADDLE = stationary_point((0.5, 0.5))
    RIGHT_PEAK = stationary_point(RIGHT_GAUSSIAN_CENTER)
    SAMPLE_POINTS = sample_landscape_grid()
    SAMPLE_POINTS[LEFT_PEAK_INDEX] = LEFT_PEAK
    SAMPLE_POINTS[SADDLE_INDEX] = SADDLE
    SAMPLE_POINTS[RIGHT_PEAK_INDEX] = RIGHT_PEAK
    SAMPLE_POINTS[GLOBAL_MINIMUM_INDEX] = GLOBAL_MINIMUM

    def landscape_data():
        axis = np.linspace(0.0, 1.0, 121)
        x_grid, y_grid = np.meshgrid(axis, axis)
        raw_grid = raw_fitness(x_grid, y_grid)
        raw_min = float(raw_grid.min())
        raw_max = max(float(raw_grid.max()), float(raw_fitness(RIGHT_PEAK[0], RIGHT_PEAK[1])))
        normalized_grid = (raw_grid - raw_min) / (raw_max - raw_min)
        point_values = (raw_fitness(SAMPLE_POINTS[:, 0], SAMPLE_POINTS[:, 1]) - raw_min) / (
            raw_max - raw_min
        )
        return x_grid, y_grid, normalized_grid, point_values

    def grid_edges():
        """Connect the perturbed lattice through its horizontal and vertical neighbours."""
        edges = []
        for row in range(GRID_ROWS):
            for column in range(GRID_COLUMNS):
                node = row * GRID_COLUMNS + column
                if column + 1 < GRID_COLUMNS:
                    edges.append((node, node + 1))
                if row + 1 < GRID_ROWS:
                    edges.append((node, node + GRID_COLUMNS))
        return edges

    def build_graph(points, values):
        """Build the scalar graph and its exact augmented split tree."""
        topology = nx.Graph()
        topology.add_nodes_from(range(len(points)))
        topology.add_edges_from(grid_edges())
        graph = rx.PyGraph()
        for index, value in enumerate(values):
            graph.add_node({"scalar": float(value), "source_index": int(index)})
        graph.add_edges_from_no_data(grid_edges())
        split_tree = build_split_tree(graph, scalar_attr="scalar", node_attrs=("source_index",))
        return topology, split_tree

    def active_components(graph, values, threshold):
        active_nodes = {
            index for index, value in enumerate(values) if is_active_at_threshold(value, threshold)
        }
        components = [
            set(component) for component in nx.connected_components(graph.subgraph(active_nodes))
        ]
        return active_nodes, components

    def is_active_at_threshold(value, threshold):
        """Reserve scalar value one as the unactivated opening sweep position."""
        return threshold < 1.0 and value >= threshold

    def sweep_thresholds(saddle_fitness):
        """Return the four displayed superlevel thresholds."""
        return (
            HIGH_PEAK_THRESHOLD,
            LOW_PEAK_THRESHOLD,
            float(saddle_fitness),
            FULL_GRAPH_THRESHOLD,
        )

    def node_lineage(index, value, threshold, saddle_threshold):
        """Color peak basins consistently; reserve green for the saddle and its descendants."""
        if index == SADDLE_INDEX and threshold <= saddle_threshold:
            return GREEN
        if threshold <= saddle_threshold and value <= saddle_threshold:
            return GREEN
        return BLUE if SAMPLE_POINTS[index].sum() > 1.0 else ORANGE

    def assert_construction(graph, values, split_tree, thresholds):
        counts = [len(active_components(graph, values, threshold)[1]) for threshold in thresholds]
        assert values.max() > thresholds[0], "The highest sample must exceed the first threshold."
        assert np.all((0.0 <= values) & (values <= 1.0)), "Fitness values must lie in [0, 1]."
        assert all(a > b for a, b in pairwise(thresholds)), "Thresholds must strictly decrease."
        assert thresholds[2] == values[SADDLE_INDEX], (
            "Row three must use the sampled saddle fitness."
        )
        assert counts == [1, 2, 1, 1], f"Unexpected component progression: {counts}."
        assert not active_components(graph, values, 1.0)[0], (
            "A threshold of one must begin before every sampled vertex enters."
        )
        assert active_components(graph, values, 0.0)[0] == set(graph.nodes), (
            "A threshold of zero must reveal the complete sampled graph."
        )
        assert nx.is_connected(graph), "The full sampled graph must be connected."
        assert graph.number_of_nodes() == GRID_ROWS * GRID_COLUMNS, (
            "The graph must retain the full lattice."
        )
        assert np.allclose(SAMPLE_POINTS[LEFT_PEAK_INDEX], LEFT_PEAK)
        assert np.allclose(SAMPLE_POINTS[RIGHT_PEAK_INDEX], RIGHT_PEAK)
        assert np.allclose(SAMPLE_POINTS[SADDLE_INDEX], SADDLE)
        assert np.allclose(SAMPLE_POINTS[GLOBAL_MINIMUM_INDEX], GLOBAL_MINIMUM)
        for critical_point in (LEFT_PEAK, SADDLE, RIGHT_PEAK):
            gradient, _ = fitness_gradient_and_hessian(critical_point)
            assert np.linalg.norm(gradient) < 1e-10, "Critical points must be stationary."
        events = {
            int(row["vertex_index"]): row["event"] for row in split_tree.event_table.to_dicts()
        }
        assert events[LEFT_PEAK_INDEX] == "maximum"
        assert events[RIGHT_PEAK_INDEX] == "maximum"
        assert events[SADDLE_INDEX] == "split_saddle"
        assert (
            min(range(len(values)), key=lambda index: (values[index], index))
            == GLOBAL_MINIMUM_INDEX
        )

    def surface_colors(x_grid, y_grid, values, threshold, saddle_threshold, theme="light"):
        palette = PALETTES[theme]
        colors = np.empty(values.shape + (4,))
        colors[...] = mpl.colors.to_rgba(palette["surface"], alpha=0.76)
        active = values >= threshold if threshold < 1.0 else np.zeros_like(values, dtype=bool)
        blue_region = (x_grid + y_grid) > 1.0
        colors[active & blue_region] = mpl.colors.to_rgba(BLUE, alpha=0.86)
        colors[active & ~blue_region] = mpl.colors.to_rgba(ORANGE, alpha=0.86)
        if threshold <= saddle_threshold:
            colors[active & (values <= saddle_threshold)] = mpl.colors.to_rgba(GREEN, alpha=0.82)
        return colors

    def assert_surface_sweep_coloring(saddle_threshold):
        """Lock the manifold's post-saddle shared structure to green."""
        epsilon = 1e-6
        values = np.array([[saddle_threshold - epsilon, saddle_threshold + epsilon]])
        x_grid = np.array([[0.25, 0.75]])
        y_grid = np.array([[0.25, 0.75]])
        colors = surface_colors(
            x_grid,
            y_grid,
            values,
            saddle_threshold - epsilon,
            saddle_threshold,
        )
        assert np.allclose(colors[0, 0], mpl.colors.to_rgba(GREEN, alpha=0.82)), (
            "Newly entered post-saddle manifold structure must be green."
        )
        assert np.allclose(colors[0, 1], mpl.colors.to_rgba(BLUE, alpha=0.86)), (
            "The blue peak basin must retain its lineage above the saddle."
        )

    def style_3d_axis(ax, theme="light"):
        palette = PALETTES[theme]
        ax.set_facecolor(palette["background"])
        ax.computed_zorder = False
        ax.view_init(elev=29, azim=-57)
        ax.set(xlim=(0.0, 1.0), ylim=(0.0, 1.0), zlim=(0.0, 1.05))
        ax.set_axis_off()
        ax.set_box_aspect((1.0, 1.0, 0.58))

    def draw_critical_markers(ax, values, threshold, theme="light"):
        palette = PALETTES[theme]
        for index, marker, color in (
            (LEFT_PEAK_INDEX, "^", ORANGE),
            (RIGHT_PEAK_INDEX, "^", BLUE),
            (SADDLE_INDEX, "D", GREEN),
            (GLOBAL_MINIMUM_INDEX, "v", GREEN),
        ):
            active = is_active_at_threshold(values[index], threshold)
            ax.scatter(
                SAMPLE_POINTS[index, 0],
                SAMPLE_POINTS[index, 1],
                min(1.04, values[index] + 0.015),
                marker=marker,
                s=126 if marker in {"^", "v"} else 84,
                c=color if active else palette["inactive"],
                edgecolors=palette["text"],
                linewidths=0.9,
                zorder=10,
            )

    def draw_manifold(
        ax,
        x_grid,
        y_grid,
        values,
        points,
        point_values,
        threshold,
        saddle_threshold,
        theme="light",
    ):
        ax.plot_surface(
            x_grid,
            y_grid,
            values,
            facecolors=surface_colors(x_grid, y_grid, values, threshold, saddle_threshold, theme),
            rcount=48,
            ccount=48,
            linewidth=0,
            antialiased=True,
            shade=True,
        )
        plane_x, plane_y = np.meshgrid(np.array([0.0, 1.0]), np.array([0.0, 1.0]))
        ax.plot_surface(
            plane_x,
            plane_y,
            np.full_like(plane_x, threshold),
            color=PALETTES[theme]["plane"],
            alpha=0.13,
            linewidth=0,
            shade=False,
        )
        draw_critical_markers(ax, point_values, threshold, theme)
        style_3d_axis(ax, theme)

    def draw_graph(
        ax,
        graph,
        points,
        values,
        threshold,
        saddle_threshold,
        x_grid,
        y_grid,
        surface,
        theme="light",
    ):
        active_nodes, _ = active_components(graph, values, threshold)
        ax.plot_surface(
            x_grid,
            y_grid,
            surface,
            color=PALETTES[theme]["surface"],
            alpha=0.16,
            rcount=40,
            ccount=40,
            linewidth=0,
            shade=True,
        )
        plane_x, plane_y = np.meshgrid(np.array([0.0, 1.0]), np.array([0.0, 1.0]))
        ax.plot_surface(
            plane_x,
            plane_y,
            np.full_like(plane_x, threshold),
            color=PALETTES[theme]["plane"],
            alpha=0.13,
            linewidth=0,
            shade=False,
        )

        def node_color(node):
            if node not in active_nodes:
                return PALETTES[theme]["inactive"]
            return node_lineage(node, values[node], threshold, saddle_threshold)

        def node_opacity(node):
            if node not in active_nodes:
                return 0.24
            return 0.35 + 0.65 * values[node]

        def edge_opacity(source, target):
            if source not in active_nodes or target not in active_nodes:
                return 0.20
            return min(node_opacity(source), node_opacity(target))

        for source, target in sorted(
            graph.edges, key=lambda edge: max(values[index] for index in edge)
        ):
            is_active = source in active_nodes and target in active_nodes
            edge_color = PALETTES[theme]["inactive"]
            if is_active:
                source_color = node_color(source)
                target_color = node_color(target)
                edge_color = source_color if source_color == target_color else GREEN
            ax.plot(
                points[[source, target], 0],
                points[[source, target], 1],
                values[[source, target]],
                color=mpl.colors.to_rgba(edge_color, edge_opacity(source, target)),
                linewidth=2.1 if is_active else 0.7,
                zorder=1,
            )
        regular_nodes = [
            node
            for node in graph.nodes
            if node not in {LEFT_PEAK_INDEX, RIGHT_PEAK_INDEX, SADDLE_INDEX, GLOBAL_MINIMUM_INDEX}
        ]
        regular_nodes.sort(key=lambda node: values[node])
        ax.scatter(
            points[regular_nodes, 0],
            points[regular_nodes, 1],
            values[regular_nodes],
            s=44,
            c=[mpl.colors.to_rgba(node_color(node), node_opacity(node)) for node in regular_nodes],
            edgecolors=PALETTES[theme]["text"],
            linewidths=0.8,
            zorder=2,
        )
        draw_critical_markers(ax, values, threshold, theme)
        style_3d_axis(ax, theme)

    def draw_tree(ax, split_tree, threshold, saddle_threshold, theme="light"):
        palette = PALETTES[theme]
        layout = build_iplotx_tree_layout(
            split_tree.tree,
            scalar_attr="scalar",
            root_kind="minimum",
            node_attrs=("source_index",),
        )
        coordinate_table = extract_iplotx_tree_coordinates(layout)
        x_positions = {
            int(row["tree_node_index"]): float(row["x"]) for row in coordinate_table.to_dicts()
        }
        tree_nodes = list(split_tree.tree.node_indices())
        payloads = {node: dict(split_tree.tree.get_node_data(node)) for node in tree_nodes}
        scalars = {node: float(payloads[node]["scalar"]) for node in tree_nodes}
        assert set(x_positions) == set(tree_nodes), (
            "IPlotX must position every augmented-tree node."
        )
        tree_edges = [
            split_tree.tree.get_edge_endpoints_by_index(edge_index)
            for edge_index in split_tree.tree.edge_indices()
        ]
        adjacency = {node: [] for node in tree_nodes}
        for first, second in tree_edges:
            adjacency[first].append(second)
            adjacency[second].append(first)
        saddle_node = next(
            node for node in tree_nodes if payloads[node]["source_index"] == SADDLE_INDEX
        )
        root_node = next(
            node for node in tree_nodes if payloads[node]["source_index"] == GLOBAL_MINIMUM_INDEX
        )
        predecessors = {saddle_node: None}
        queue = [saddle_node]
        while queue:
            node = queue.pop(0)
            for neighbour in adjacency[node]:
                if neighbour not in predecessors:
                    predecessors[neighbour] = node
                    queue.append(neighbour)
        trunk_nodes = {root_node}
        node = root_node
        while node != saddle_node:
            node = predecessors[node]
            trunk_nodes.add(node)
        for node in trunk_nodes:
            x_positions[node] = x_positions[saddle_node]
        assert all(x_positions[node] == x_positions[saddle_node] for node in trunk_nodes), (
            "The post-saddle trunk must descend vertically."
        )
        active = {
            node for node, scalar in scalars.items() if is_active_at_threshold(scalar, threshold)
        }
        ax.axhline(threshold, color=palette["neutral"], alpha=0.45, linewidth=1.0, zorder=0)
        for first, second in tree_edges:
            parent, child = (
                (first, second) if scalars[first] < scalars[second] else (second, first)
            )
            if child in active:
                source_index = int(payloads[child]["source_index"])
                color = (
                    GREEN
                    if parent in trunk_nodes and child in trunk_nodes
                    else node_lineage(source_index, scalars[child], threshold, saddle_threshold)
                )
                if parent not in active:
                    coordinates = (
                        [x_positions[child], x_positions[child]],
                        [scalars[child], threshold],
                    )
                elif parent in trunk_nodes and child in trunk_nodes:
                    coordinates = (
                        [x_positions[parent], x_positions[child]],
                        [scalars[parent], scalars[child]],
                    )
                else:
                    coordinates = (
                        [x_positions[child], x_positions[child], x_positions[parent]],
                        [scalars[child], scalars[parent], scalars[parent]],
                    )
                ax.plot(
                    coordinates[0],
                    coordinates[1],
                    color=color,
                    linewidth=2.2,
                    zorder=1,
                )
        regular_nodes = [
            node
            for node in active
            if payloads[node]["event"] == "regular"
            and payloads[node]["source_index"] != GLOBAL_MINIMUM_INDEX
        ]
        for node in regular_nodes:
            source_index = int(payloads[node]["source_index"])
            ax.scatter(
                x_positions[node],
                scalars[node],
                marker="o",
                s=22,
                color=node_lineage(source_index, scalars[node], threshold, saddle_threshold),
                edgecolors=palette["text"],
                linewidths=0.7,
                zorder=2,
            )
        for node in active:
            event = payloads[node]["event"]
            if event == "regular" and payloads[node]["source_index"] != GLOBAL_MINIMUM_INDEX:
                continue
            source_index = int(payloads[node]["source_index"])
            marker, size = (
                ("v", 126)
                if source_index == GLOBAL_MINIMUM_INDEX
                else (("^", 126) if event == "maximum" else ("D", 84))
            )
            ax.scatter(
                x_positions[node],
                scalars[node],
                marker=marker,
                s=size,
                color=node_lineage(source_index, scalars[node], threshold, saddle_threshold),
                edgecolors=palette["text"],
                linewidths=1.0,
                zorder=10,
            )
        ax.set(
            xlim=(min(x_positions.values()) - 0.6, max(x_positions.values()) + 0.6),
            ylim=(-0.03, 1.05),
            xticks=[],
            yticks=[],
        )
        for spine in ("top", "right", "bottom", "left"):
            ax.spines[spine].set_visible(False)

    def construction_data():
        """Build the deterministic landscape, sampled graph, tree, and checkpoints."""
        x_grid, y_grid, values, point_values = landscape_data()
        graph, split_tree = build_graph(SAMPLE_POINTS, point_values)
        thresholds = sweep_thresholds(point_values[SADDLE_INDEX])
        assert_construction(graph, point_values, split_tree, thresholds)
        assert_surface_sweep_coloring(thresholds[2])
        return x_grid, y_grid, values, point_values, graph, split_tree, thresholds

    def build_figure(theme="light"):
        x_grid, y_grid, values, point_values, graph, split_tree, thresholds = construction_data()
        figure = plt.figure(figsize=(10.7, 11.5), layout="constrained")
        grid = figure.add_gridspec(5, 3, height_ratios=(0.22, 1, 1, 1, 1))
        for column, title in enumerate(("Manifold", "Graph", "Tree")):
            header = figure.add_subplot(grid[0, column])
            header.set_facecolor(PALETTES[theme]["background"])
            header.text(
                0.5,
                0.3,
                title,
                color=PALETTES[theme]["text"],
                ha="center",
                va="center",
                fontsize=15,
                fontweight="bold",
            )
            header.set_axis_off()
        for row, threshold in enumerate(thresholds):
            manifold = figure.add_subplot(grid[row + 1, 0], projection="3d")
            graph_axis = figure.add_subplot(grid[row + 1, 1], projection="3d")
            tree_axis = figure.add_subplot(grid[row + 1, 2])
            draw_manifold(
                manifold,
                x_grid,
                y_grid,
                values,
                SAMPLE_POINTS,
                point_values,
                threshold,
                thresholds[2],
                theme,
            )
            draw_graph(
                graph_axis,
                graph,
                SAMPLE_POINTS,
                point_values,
                threshold,
                thresholds[2],
                x_grid,
                y_grid,
                values,
                theme,
            )
            tree_axis.set_facecolor(PALETTES[theme]["background"])
            draw_tree(tree_axis, split_tree, threshold, thresholds[2], theme)
            manifold.text2D(
                0.02,
                0.88,
                str(row + 1),
                transform=manifold.transAxes,
                va="top",
                ha="left",
                fontsize=24,
                fontweight="bold",
                fontfamily="Arial",
                color=PALETTES[theme]["text"],
            )
        return figure

    def build_interactive_figure(threshold, theme="light", figure=None):
        """Return the three linked panels at one continuously chosen threshold."""
        x_grid, y_grid, values, point_values, graph, split_tree, thresholds = construction_data()
        if figure is None:
            figure = plt.figure(figsize=(10.7, 3.35), layout="constrained")
        else:
            figure.clear()
        grid = figure.add_gridspec(2, 3, height_ratios=(0.18, 1))
        for column, title in enumerate(("Manifold", "Graph", "Tree")):
            header = figure.add_subplot(grid[0, column])
            header.set_facecolor(PALETTES[theme]["background"])
            header.text(
                0.5,
                0.3,
                title,
                color=PALETTES[theme]["text"],
                ha="center",
                va="center",
                fontsize=15,
                fontweight="bold",
            )
            header.set_axis_off()
        manifold = figure.add_subplot(grid[1, 0], projection="3d")
        graph_axis = figure.add_subplot(grid[1, 1], projection="3d")
        tree_axis = figure.add_subplot(grid[1, 2])
        draw_manifold(
            manifold,
            x_grid,
            y_grid,
            values,
            SAMPLE_POINTS,
            point_values,
            threshold,
            thresholds[2],
            theme,
        )
        draw_graph(
            graph_axis,
            graph,
            SAMPLE_POINTS,
            point_values,
            threshold,
            thresholds[2],
            x_grid,
            y_grid,
            values,
            theme,
        )
        tree_axis.set_facecolor(PALETTES[theme]["background"])
        draw_tree(tree_axis, split_tree, threshold, thresholds[2], theme)
        return figure

    def sweep_status(threshold):
        """Describe the topological state at a scalar threshold."""
        _, _, _, point_values, graph, _, thresholds = construction_data()
        component_count = len(active_components(graph, point_values, threshold)[1])
        if threshold > thresholds[1]:
            phase = "Stage 1: higher peak"
        elif threshold > thresholds[2]:
            phase = "Stage 2: two components"
        elif threshold > thresholds[3]:
            phase = "Stage 3: merged at the saddle"
        else:
            phase = "Stage 4: full graph"
        return (
            f"{phase} — {component_count} active component{'s' if component_count != 1 else ''}."
        )

    def write_assets(figure=None):
        directory = Path("docs") / "figures" / "manifold_graph_tree"
        directory.mkdir(parents=True, exist_ok=True)
        for theme in PALETTES:
            current = figure if figure is not None and theme == "light" else build_figure(theme)
            current.patch.set_facecolor(PALETTES[theme]["background"])
            for suffix in ("png", "svg", "pdf"):
                asset_path = directory / f"{ASSET_STEM}_{theme}.{suffix}"
                current.savefig(
                    asset_path, dpi=300, bbox_inches="tight", facecolor=current.get_facecolor()
                )
                if suffix == "svg":
                    normalized_lines = [
                        line.rstrip() for line in asset_path.read_text().splitlines()
                    ]
                    asset_path.write_text("\n".join(normalized_lines) + "\n")

    def write_animation_assets():
        """Export a smooth, deterministic high-to-low presentation sweep."""
        from matplotlib import animation

        frame_rate = 24
        frame_count = 145
        thresholds = np.linspace(1.0, 0.0, frame_count)
        directory = Path("docs") / "figures" / "manifold_graph_tree"
        directory.mkdir(parents=True, exist_ok=True)
        paths = []
        for theme in PALETTES:
            figure = plt.figure(figsize=(10.7, 3.35), layout="constrained")
            writers = (
                ("gif", animation.PillowWriter(fps=frame_rate)),
                ("mp4", animation.FFMpegWriter(fps=frame_rate, bitrate=2400)),
            )
            try:
                for suffix, writer in writers:
                    asset_path = directory / f"{ASSET_STEM}_{theme}.{suffix}"
                    with writer.saving(figure, str(asset_path), dpi=180):
                        for threshold in thresholds:
                            build_interactive_figure(float(threshold), theme, figure)
                            writer.grab_frame()
                    paths.append(asset_path)
            finally:
                plt.close(figure)
        return tuple(paths)

    return (
        ASSET_STEM,
        build_figure,
        build_interactive_figure,
        construction_data,
        sweep_status,
        write_animation_assets,
        write_assets,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Figure and export

    The four rows implement a decreasing superlevel sweep. Orange and blue retain
    the identities of the two peak lineages after their saddle merger; green marks
    newly admitted shared, lower-fitness structure and the tree trunk. The tree
    retains every sampled vertex so the construction is visible continuously.

    Suggested caption: *A conceptual split-tree construction. A two-peak fitness
    manifold (left) is sampled as a local-neighborhood graph in three dimensions
    (middle). As the fitness threshold decreases, connected superlevel components
    are recorded by an augmented split tree (right). Orange and blue retain peak
    lineage identity after merger, while green shows shared lower-fitness growth.
    The manifold is conceptual ground truth; the graph is a generic discretization
    rather than a literal model of sequence-space geometry.*
    """)


@app.cell
def _(build_figure, write_assets):
    figure = build_figure()
    write_assets(figure)
    return (figure,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Interactive construction

    Start at scalar value 1, then scrub or play downward. Each crossing admits
    more of the superlevel construction; at 0 the complete augmented tree is
    visible. The graph retains faint future structure for context.
    """)


@app.cell
def _(anywidget, construction_data, mo, traitlets):
    *_, thresholds = construction_data()

    class SweepPlayer(anywidget.AnyWidget):
        """A small High-to-Low slider with browser-side playback."""

        threshold = traitlets.Float(1.0).tag(sync=True)
        minimum = traitlets.Float(0.0).tag(sync=True)
        maximum = traitlets.Float(1.0).tag(sync=True)
        reset_threshold = traitlets.Float(1.0).tag(sync=True)
        checkpoints = traitlets.List(traitlets.Float(), list(thresholds)).tag(sync=True)
        checkpoint_labels = traitlets.List(traitlets.Unicode(), ["1", "2", "3", "4"]).tag(
            sync=True
        )

        _esm = r"""
        function render({ model, el }) {
          const root = document.createElement("div");
          root.style.cssText = "display:flex;flex-direction:column;align-items:stretch;gap:0.55rem;font-family:system-ui,sans-serif;min-width:9rem;max-width:11rem";
          const actions = document.createElement("div");
          actions.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:0.45rem";
          const buttonStyle = "appearance:none;border:1px solid #94a3b8;border-radius:0.4rem;background:#f8fafc;color:#0f172a;padding:0.5rem 0.65rem;font:600 0.9rem system-ui,sans-serif;cursor:pointer;box-shadow:0 1px 2px rgb(15 23 42 / 0.12)";
          const play = document.createElement("button");
          play.type = "button";
          play.textContent = "Play";
          play.style.cssText = buttonStyle;
          play.setAttribute("aria-label", "Play or pause sweep");
          const reset = document.createElement("button");
          reset.type = "button";
          reset.textContent = "Reset";
          reset.style.cssText = buttonStyle;
          reset.setAttribute("aria-label", "Reset sweep to scalar value one");
          const slider = document.createElement("input");
          slider.type = "range";
          slider.min = "0";
          slider.max = "1000";
          slider.step = "1";
          slider.style.cssText = "appearance:slider-vertical;writing-mode:vertical-lr;width:1.55rem;height:13rem;margin:0.1rem auto";
          slider.setAttribute("aria-label", "Scalar threshold, high to low");
          const value = document.createElement("output");
          value.style.cssText = "font:600 0.82rem system-ui,sans-serif;text-align:center;color:#334155";
          const sliderGroup = document.createElement("div");
          sliderGroup.style.cssText = "display:grid;grid-template-columns:auto 1fr;align-items:center;gap:0.45rem";
          const endpointLabels = document.createElement("div");
          endpointLabels.style.cssText = "display:flex;flex-direction:column;justify-content:space-between;height:13rem;font-size:0.75rem;color:#475569";
          endpointLabels.innerHTML = "<span>1 High</span><span>0 Low</span>";
          const checkpoints = document.createElement("div");
          checkpoints.style.cssText = "display:grid;gap:0.4rem";
          const minimum = () => model.get("minimum");
          const maximum = () => model.get("maximum");
          const positionFor = (threshold) => 1000 * (maximum() - threshold) / (maximum() - minimum());
          const thresholdFor = (position) => maximum() - position / 1000 * (maximum() - minimum());
          let timer = null;
          let lastSent = null;
          function send(threshold) {
            const bounded = Math.max(minimum(), Math.min(maximum(), threshold));
            if (lastSent !== null && Math.abs(lastSent - bounded) < 0.0005) return;
            lastSent = bounded;
            model.set("threshold", bounded);
            model.save_changes();
          }
          function update() {
            const threshold = model.get("threshold");
            slider.value = String(positionFor(threshold));
            value.textContent = `threshold ${threshold.toFixed(3)}`;
          }
          slider.addEventListener("input", () => send(thresholdFor(Number(slider.value))));
          reset.addEventListener("click", () => {
            stop();
            send(model.get("reset_threshold"));
          });
          function stop() {
            if (timer !== null) clearInterval(timer);
            timer = null;
            play.textContent = "Play";
          }
          play.addEventListener("click", () => {
            if (timer !== null) { stop(); return; }
            play.textContent = "Pause";
            timer = setInterval(() => {
              const next = model.get("threshold") - 0.012;
              if (next <= minimum()) { send(minimum()); stop(); return; }
              send(next);
            }, 80);
          });
          for (const [index, checkpoint] of model.get("checkpoints").entries()) {
            const button = document.createElement("button");
            button.type = "button";
            button.textContent = `Stage ${model.get("checkpoint_labels")[index]}`;
            button.style.cssText = buttonStyle;
            button.addEventListener("click", () => { stop(); send(checkpoint); });
            checkpoints.appendChild(button);
          }
          actions.append(play, reset);
          sliderGroup.append(endpointLabels, slider);
          root.append(actions, sliderGroup, value, checkpoints);
          el.append(root);
          update();
          model.on("change:threshold", update);
          return () => { stop(); model.off("change:threshold", update); root.remove(); };
        }
        export default { render };
        """

    sweep_player = mo.ui.anywidget(SweepPlayer())
    return (sweep_player,)


@app.cell(hide_code=True)
def _(mo):
    theme = mo.ui.radio(options=["light", "dark"], value="light", label="Theme")
    export_animation = mo.ui.run_button(
        label="Generate MP4 + GIF", kind="success", tooltip="Export the full high-to-low sweep"
    )
    return (export_animation, theme)


@app.cell(hide_code=True)
def _(export_animation, mo, theme):
    mo.vstack(
        [
            mo.md(
                "### Presentation animation\n\n"
                "Generate a 6-second MP4 and looping GIF in `docs/figures/manifold_graph_tree/`."
            ),
            theme,
            export_animation,
        ]
    )


@app.cell(hide_code=True)
def _(export_animation, write_animation_assets):
    if export_animation.value:
        write_animation_assets()


@app.cell(hide_code=True)
def _(build_interactive_figure, mo, sweep_player, sweep_status, theme):
    current_threshold = float(sweep_player.threshold)
    mo.hstack(
        [
            sweep_player,
            mo.vstack(
                [
                    mo.md(f"**{sweep_status(current_threshold)}**"),
                    build_interactive_figure(current_threshold, theme.value),
                ]
            ),
        ],
        align="start",
        wrap=True,
        gap=1.0,
        widths=[0, 1],
    )


if __name__ == "__main__":
    app.run()
