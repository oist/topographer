from __future__ import annotations

import pytest


def _build_three_node_tree():
    import rustworkx as rx

    tree = rx.PyDiGraph()
    tree.add_node(
        {
            "vertex_index": 0,
            "scalar": 0.0,
            "event": "minimum",
            "aa_sequence": "AAA",
        }
    )
    tree.add_node(
        {
            "vertex_index": 1,
            "scalar": 0.5,
            "event": "regular",
            "aa_sequence": "AAA",
        }
    )
    tree.add_node(
        {
            "vertex_index": 2,
            "scalar": 1.0,
            "event": "maximum",
            "aa_sequence": "A*A",
        }
    )
    tree.add_edge(0, 1, {"regular_count": 2})
    tree.add_edge(1, 2, {"regular_count": 3})
    return tree


def _build_branching_tree():
    import rustworkx as rx

    tree = rx.PyDiGraph()
    tree.add_node({"vertex_index": 0, "scalar": 0.0, "event": "minimum"})
    tree.add_node({"vertex_index": 1, "scalar": 0.4, "event": "regular"})
    tree.add_node({"vertex_index": 2, "scalar": 0.8, "event": "maximum"})
    tree.add_node({"vertex_index": 3, "scalar": 1.0, "event": "maximum"})
    tree.add_edge(0, 1, {"regular_count": 1})
    tree.add_edge(1, 2, {"regular_count": 1})
    tree.add_edge(1, 3, {"regular_count": 1})
    return tree


@pytest.mark.parametrize(
    ("root_kind", "expected_root", "expected_edges"),
    [
        ("minimum", 0, [("0", "1"), ("1", "2")]),
        ("maximum", 2, [("2", "1"), ("1", "0")]),
    ],
)
def test_build_iplotx_tree_layout_supports_both_root_orientations(
    root_kind: str,
    expected_root: int,
    expected_edges: list[tuple[str, str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import topographer.plotting._layout as layout_module

    tree = _build_three_node_tree()
    graph_edge_list_call_count = 0
    original_graph_edge_list = layout_module._graph_edge_list

    def counted_graph_edge_list(graph):
        nonlocal graph_edge_list_call_count
        graph_edge_list_call_count += 1
        return original_graph_edge_list(graph)

    monkeypatch.setattr(layout_module, "_graph_edge_list", counted_graph_edge_list)

    layout_result = layout_module.build_iplotx_tree_layout(
        tree, scalar_attr="scalar", root_kind=root_kind
    )

    assert graph_edge_list_call_count == 1
    assert layout_result.root_node_index == expected_root
    assert layout_result.tree.name == str(expected_root)
    assert layout_result.scalar_min == pytest.approx(0.0)
    assert layout_result.scalar_max == pytest.approx(1.0)
    assert set(layout_result.regular_count_by_edge_name) == set(expected_edges)
    assert sorted(layout_result.regular_count_by_edge_name.values()) == [2.0, 3.0]


def test_build_iplotx_tree_layout_preserves_branch_lengths() -> None:
    from topographer.plotting._layout import build_iplotx_tree_layout

    tree = _build_three_node_tree()
    layout_result = build_iplotx_tree_layout(tree, scalar_attr="scalar", root_kind="minimum")

    first_child = layout_result.tree.children[0]
    assert first_child.name == "1"
    assert first_child.branch_length == pytest.approx(0.5)
    assert first_child.children[0].name == "2"
    assert first_child.children[0].branch_length == pytest.approx(0.5)


def test_extract_iplotx_tree_coordinates_preserves_scalar_y_axis() -> None:
    from topographer.plotting._layout import (
        build_iplotx_tree_layout,
        extract_iplotx_tree_coordinates,
    )

    tree = _build_three_node_tree()
    layout = build_iplotx_tree_layout(tree, scalar_attr="scalar", root_kind="minimum")
    coordinates = extract_iplotx_tree_coordinates(layout)

    assert coordinates.columns == ["tree_node_index", "x", "y"]
    assert coordinates["tree_node_index"].to_list() == [0, 1, 2]
    assert coordinates["y"].to_list() == [0.0, 0.5, 1.0]
    assert coordinates["x"].min() >= 0.0


def test_build_iplotx_tree_layout_copies_optional_metadata_only_when_requested() -> None:
    from topographer.plotting._layout import build_iplotx_tree_layout

    tree = _build_three_node_tree()

    default_layout = build_iplotx_tree_layout(tree, scalar_attr="scalar")
    requested_layout = build_iplotx_tree_layout(
        tree,
        scalar_attr="scalar",
        node_attrs=("aa_sequence",),
    )

    assert not hasattr(default_layout.tree, "aa_sequence")
    assert requested_layout.tree.aa_sequence == "AAA"


def test_hierarchical_tree_layout_assigns_distinct_leaf_x_positions() -> None:
    pytest.importorskip("networkx")
    from topographer.plotting._layout import hierarchical_tree_layout

    tree = _build_branching_tree()
    node_data_by_idx = {
        int(node_index): dict(tree.get_node_data(node_index)) for node_index in tree.node_indices()
    }

    positions = hierarchical_tree_layout(tree, node_data_by_idx)

    assert positions[0][0] == pytest.approx(5.0)
    assert positions[2][0] == pytest.approx(0.0)
    assert positions[3][0] == pytest.approx(10.0)
    assert positions[1][0] == pytest.approx(5.0)


def test_build_iplotx_tree_layout_rejects_disconnected_graphs() -> None:
    import rustworkx as rx

    from topographer.plotting._layout import build_iplotx_tree_layout

    tree = rx.PyDiGraph()
    tree.add_node({"vertex_index": 0, "scalar": 0.0, "event": "minimum"})
    tree.add_node({"vertex_index": 1, "scalar": 1.0, "event": "maximum"})

    with pytest.raises(ValueError, match="connected|reach all nodes"):
        build_iplotx_tree_layout(tree, scalar_attr="scalar", root_kind="minimum")


def _build_tied_extrema_merge_tree(tree_kind: str):
    import rustworkx as rx

    from topographer.topology._tree_construction import build_join_tree, build_split_tree

    graph = rx.PyGraph()
    for value in [1.0, 0.5, 1.0, 0.0, 0.0]:
        graph.add_node({"scalar": value})
    for source, target in [(0, 1), (1, 2), (3, 1), (4, 1)]:
        graph.add_edge(source, target, None)
    build_tree = build_join_tree if tree_kind == "join" else build_split_tree
    return build_tree(graph, scalar_attr="scalar")


def _root_vertex_index(tree, root_node_index: int) -> int:
    return tree.get_node_data(root_node_index)["vertex_index"]


@pytest.mark.parametrize(("tree_kind", "root_kind"), [("join", "maximum"), ("split", "minimum")])
def test_make_tree_layout_iplotx_root_matches_merge_tree_root_for_tied_extrema(
    tree_kind: str, root_kind: str
) -> None:
    from topographer.plotting import make_tree_layout

    result = _build_tied_extrema_merge_tree(tree_kind)

    plot_data = make_tree_layout(
        result.tree,
        scalar_attr="scalar",
        root_kind=root_kind,
        horizontal_layout="iplotx",
    )

    assert _root_vertex_index(result.tree, plot_data.root_node_index) == _root_vertex_index(
        result.tree, result.root_node_index
    )


@pytest.mark.parametrize(("tree_kind", "root_kind"), [("join", "maximum"), ("split", "minimum")])
def test_build_iplotx_tree_layout_root_matches_merge_tree_root_for_tied_extrema(
    tree_kind: str, root_kind: str
) -> None:
    from topographer.plotting._layout import build_iplotx_tree_layout

    result = _build_tied_extrema_merge_tree(tree_kind)

    layout_result = build_iplotx_tree_layout(
        result.tree, scalar_attr="scalar", root_kind=root_kind
    )

    assert _root_vertex_index(result.tree, layout_result.root_node_index) == _root_vertex_index(
        result.tree, result.root_node_index
    )


def test_extract_iplotx_tree_coordinates_handles_deep_chain_without_recursion() -> None:
    import sys

    import rustworkx as rx

    from topographer.plotting._layout import (
        build_iplotx_tree_layout,
        extract_iplotx_tree_coordinates,
    )

    node_count = 400
    tree = rx.PyDiGraph()
    for index in range(node_count):
        tree.add_node(
            {"vertex_index": index, "scalar": index / (node_count - 1), "event": "regular"}
        )
    for index in range(node_count - 1):
        tree.add_edge(index, index + 1, {"regular_count": 1})

    old_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(200)
    try:
        layout = build_iplotx_tree_layout(tree, scalar_attr="scalar", root_kind="minimum")
        coordinates = extract_iplotx_tree_coordinates(layout)
    finally:
        sys.setrecursionlimit(old_limit)

    assert coordinates.height == node_count


def test_extract_iplotx_tree_coordinates_matches_iplotx_vertical_layout() -> None:
    pytest.importorskip("iplotx")
    plt = pytest.importorskip("matplotlib.pyplot")
    import iplotx as ipx
    import rustworkx as rx

    from topographer.plotting._layout import (
        build_iplotx_tree_layout,
        extract_iplotx_tree_coordinates,
    )

    scalars = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    edges = [(0, 1), (1, 2), (1, 3), (1, 4), (3, 5), (3, 6), (6, 7), (4, 8), (2, 9)]
    tree = rx.PyDiGraph()
    for index, scalar in enumerate(scalars):
        tree.add_node({"vertex_index": index, "scalar": scalar, "event": "regular"})
    for source, target in edges:
        tree.add_edge(source, target, {"regular_count": 1})

    layout = build_iplotx_tree_layout(tree, scalar_attr="scalar", root_kind="minimum")
    coordinates = extract_iplotx_tree_coordinates(layout)

    figure, axis = plt.subplots()
    try:
        artist = ipx.tree(layout.tree, ax=axis, layout="vertical")
        expected = {
            int(node.name): float(row["_ipx_layout_0"])
            for node, row in artist.get_layout("vertex").iterrows()
        }
    finally:
        plt.close(figure)

    assert coordinates.height == len(scalars)
    for row in coordinates.iter_rows(named=True):
        assert row["x"] == pytest.approx(expected[row["tree_node_index"]], abs=1e-12)


# A split tree whose tallest branch hangs off the lowest saddle. Each vertex is
# ``(vertex_index, scalar)``; the global maximum is vertex 7 and the lowest
# saddle is vertex 1. Ordering children by their own scalar puts the tall branch
# rightmost at every fork, so these trees separate the layouts.
_TALL_BRANCH_VERTICES = [
    (0, 0.0),  # root, global minimum
    (1, 0.2),  # lowest saddle
    (2, 0.35),  # short leaf off the lowest saddle
    (3, 0.4),  # saddle that leads to the tall branch
    (4, 0.5),  # short leaf
    (5, 0.6),  # saddle
    (6, 0.7),  # leaf
    (7, 1.0),  # global maximum
]
_TALL_BRANCH_EDGES = [(0, 1), (1, 2), (1, 3), (3, 4), (3, 5), (5, 6), (5, 7)]
_TALL_BRANCH_LEAVES = {2, 4, 6, 7}
_TALL_BRANCH_FORKS = {
    # fork vertex: children from left to right under the "subtree_extremum" rule
    1: [3, 2],
    3: [5, 4],
    5: [7, 6],
}


def _mirror(vertices):
    """Flip every scalar, ``s -> 1 - s``, turning a split-tree fixture into a join-tree one.

    The mirrored tree is rooted at its global maximum and its deepest valley takes
    the place of the tall peak, so the ``subtree_extremum`` rule gives the same
    left-to-right order as on the original.
    """

    return [(vertex, 1.0 - scalar) for vertex, scalar in vertices]


def _build_rooted_tree(vertices, edges, insertion_order=None, regular_counts=None):
    """Build a rooted tree whose node indices follow ``insertion_order``.

    ``regular_counts`` maps ``(parent, child)`` to the ``regular_count`` edge attribute.
    """

    import rustworkx as rx

    by_vertex = dict(vertices)
    order = insertion_order if insertion_order is not None else [v for v, _ in vertices]
    tree = rx.PyDiGraph()
    node_by_vertex = {
        vertex: tree.add_node(
            {"vertex_index": vertex, "scalar": by_vertex[vertex], "event": "regular"}
        )
        for vertex in order
    }
    for parent, child in edges:
        regular_count = (regular_counts or {}).get((parent, child), 0)
        tree.add_edge(
            node_by_vertex[parent], node_by_vertex[child], {"regular_count": regular_count}
        )
    return tree


def _x_by_vertex_from_make_tree_layout(tree, horizontal_layout, root_kind="minimum"):
    from topographer.plotting import make_tree_layout

    plot_data = make_tree_layout(
        tree, scalar_attr="scalar", root_kind=root_kind, horizontal_layout=horizontal_layout
    )
    return {row["vertex_index"]: row["x"] for row in plot_data.node_table.to_dicts()}


def _x_by_vertex_from_iplotx_layout(tree, horizontal_layout, root_kind="minimum"):
    from topographer.plotting._layout import (
        build_iplotx_tree_layout,
        extract_iplotx_tree_coordinates,
    )

    layout = build_iplotx_tree_layout(
        tree, scalar_attr="scalar", root_kind=root_kind, horizontal_layout=horizontal_layout
    )
    coordinates = extract_iplotx_tree_coordinates(layout)
    return {
        layout.node_data_by_index[row["tree_node_index"]]["vertex_index"]: row["x"]
        for row in coordinates.to_dicts()
    }


_SUBTREE_EXTREMUM_LAYOUTS = [
    pytest.param(_x_by_vertex_from_make_tree_layout, id="make_tree_layout"),
    pytest.param(_x_by_vertex_from_iplotx_layout, id="build_iplotx_tree_layout"),
]


# A split tree where, at every fork that has an internal child, the branch to the
# tall peak starts LOWER than its sibling leaf (saddle 3 at 0.3 against leaf 2 at
# 0.35, and so on). Ordering children by their own scalar, highest first, puts the
# tall branch rightmost here, so this tree separates ``subtree_extremum`` from that
# naive key. The last fork (vertex 7) joins two leaves, where the own scalar and the
# subtree maximum necessarily agree.
_LOW_START_VERTICES = [
    (0, 0.0),  # root, global minimum
    (1, 0.1),  # lowest saddle
    (2, 0.35),  # leaf, taller than the start of the tall branch below
    (3, 0.3),  # saddle that leads to the tall branch
    (4, 0.65),  # leaf, taller than the saddle below
    (5, 0.6),  # saddle
    (6, 0.85),  # leaf, taller than the saddle below
    (7, 0.7),  # saddle
    (8, 1.0),  # global maximum
    (9, 0.75),  # leaf
]
_LOW_START_EDGES = [(0, 1), (1, 2), (1, 3), (3, 4), (3, 5), (5, 6), (5, 7), (7, 8), (7, 9)]
_LOW_START_FORKS = {1: [3, 2], 3: [5, 4], 5: [7, 6], 7: [8, 9]}
_LOW_START_LEAVES = {2, 4, 6, 8, 9}

# The join-tree mirror of the fixture above, rooted at the global maximum: the
# deepest valley (vertex 8) hangs off the HIGHEST saddle (vertex 1), and at every
# fork with an internal child the branch to the valley starts HIGHER than its
# sibling leaf (saddle 3 at 0.7 against leaf 2 at 0.65, and so on). Ordering
# children by their own scalar, lowest first, puts the valley branch rightmost.
_VALLEY_START_VERTICES = _mirror(_LOW_START_VERTICES)

# (root_kind, vertices, edges, forks, leaves, extremum leaf): the split fixtures
# must put their global maximum leftmost, the join fixtures their global minimum.
_FARTHEST_BRANCH_CASES = [
    pytest.param(
        "minimum",
        _TALL_BRANCH_VERTICES,
        _TALL_BRANCH_EDGES,
        _TALL_BRANCH_FORKS,
        _TALL_BRANCH_LEAVES,
        7,
        id="split-tall-branch-starts-higher",
    ),
    pytest.param(
        "minimum",
        _LOW_START_VERTICES,
        _LOW_START_EDGES,
        _LOW_START_FORKS,
        _LOW_START_LEAVES,
        8,
        id="split-tall-branch-starts-lower",
    ),
    pytest.param(
        "maximum",
        _VALLEY_START_VERTICES,
        _LOW_START_EDGES,
        _LOW_START_FORKS,
        _LOW_START_LEAVES,
        8,
        id="join-deep-branch-starts-higher",
    ),
    pytest.param(
        "maximum",
        _mirror(_TALL_BRANCH_VERTICES),
        _TALL_BRANCH_EDGES,
        _TALL_BRANCH_FORKS,
        _TALL_BRANCH_LEAVES,
        7,
        id="join-deep-branch-starts-lower",
    ),
]


@pytest.mark.parametrize("x_by_vertex_of", _SUBTREE_EXTREMUM_LAYOUTS)
@pytest.mark.parametrize(
    ("root_kind", "vertices", "edges", "forks", "leaves", "extremum"), _FARTHEST_BRANCH_CASES
)
def test_subtree_extremum_puts_the_farthest_branch_leftmost_at_every_fork(
    x_by_vertex_of, root_kind, vertices, edges, forks, leaves, extremum
) -> None:
    pytest.importorskip("iplotx")
    tree = _build_rooted_tree(vertices, edges)

    x_by_vertex = x_by_vertex_of(tree, "subtree_extremum", root_kind=root_kind)

    for fork, expected_left_to_right in forks.items():
        children = [child for parent, child in edges if parent == fork]
        assert sorted(children, key=x_by_vertex.__getitem__) == expected_left_to_right
    assert min(leaves, key=x_by_vertex.__getitem__) == extremum
    assert x_by_vertex[extremum] == min(x_by_vertex.values())


@pytest.mark.parametrize("horizontal_layout", ["leaf_order", "iplotx"])
def test_existing_layouts_still_put_the_tallest_branch_rightmost(horizontal_layout: str) -> None:
    pytest.importorskip("iplotx")
    tree = _build_rooted_tree(_TALL_BRANCH_VERTICES, _TALL_BRANCH_EDGES)

    x_by_vertex = _x_by_vertex_from_make_tree_layout(tree, horizontal_layout)

    assert x_by_vertex[7] == max(x_by_vertex.values())


@pytest.mark.parametrize("horizontal_layout", ["leaf_order", "iplotx"])
def test_existing_layouts_do_not_put_the_deepest_valley_leftmost_on_a_join_tree(
    horizontal_layout: str,
) -> None:
    pytest.importorskip("iplotx")
    tree = _build_rooted_tree(_VALLEY_START_VERTICES, _LOW_START_EDGES)

    x_by_vertex = _x_by_vertex_from_make_tree_layout(tree, horizontal_layout, "maximum")

    assert min(_LOW_START_LEAVES, key=x_by_vertex.__getitem__) != 8


@pytest.mark.parametrize("x_by_vertex_of", _SUBTREE_EXTREMUM_LAYOUTS)
@pytest.mark.parametrize("root_kind", ["minimum", "maximum"])
def test_subtree_extremum_breaks_ties_by_subtree_size_then_node_index(
    x_by_vertex_of, root_kind: str
) -> None:
    pytest.importorskip("iplotx")
    # Both children of the saddle reach 0.8 (0.2 when mirrored into a join tree). Leaf
    # 2 has the lower node index but the branch through vertex 3 holds more vertices,
    # so it goes first. Leaves 5 and 6 tie on scalar and size, so the lower node index
    # (vertex 5) goes first.
    vertices = [(0, 0.0), (1, 0.2), (2, 0.8), (3, 0.5), (4, 0.8), (5, 0.9), (6, 0.9)]
    if root_kind == "maximum":
        vertices = _mirror(vertices)
    edges = [(0, 1), (1, 2), (1, 3), (3, 4), (0, 5), (0, 6)]
    tree = _build_rooted_tree(vertices, edges)

    x_by_vertex = x_by_vertex_of(tree, "subtree_extremum", root_kind=root_kind)

    assert x_by_vertex[3] < x_by_vertex[2]
    assert x_by_vertex[5] < x_by_vertex[6]


@pytest.mark.parametrize("x_by_vertex_of", _SUBTREE_EXTREMUM_LAYOUTS)
@pytest.mark.parametrize("root_kind", ["minimum", "maximum"])
def test_subtree_extremum_size_tie_break_ignores_regular_count(
    x_by_vertex_of, root_kind: str
) -> None:
    pytest.importorskip("iplotx")
    # Both children of vertex 1 reach 0.8 (0.2 when mirrored into a join tree). Leaf 2
    # has one tree node, the branch through vertex 3 has two. Leaf 2 carries 1000
    # regular vertices on its edge, the branch none: it still loses the tie, and it
    # would also win on the lower node index if sizes were equal.
    vertices = [(0, 0.0), (1, 0.2), (2, 0.8), (3, 0.5), (4, 0.8)]
    if root_kind == "maximum":
        vertices = _mirror(vertices)
    edges = [(0, 1), (1, 2), (1, 3), (3, 4)]
    tree = _build_rooted_tree(vertices, edges, regular_counts={(1, 2): 1000})

    x_by_vertex = x_by_vertex_of(tree, "subtree_extremum", root_kind=root_kind)

    assert x_by_vertex[3] < x_by_vertex[2]


@pytest.mark.parametrize("x_by_vertex_of", _SUBTREE_EXTREMUM_LAYOUTS)
@pytest.mark.parametrize(
    ("root_kind", "mirror"), [("minimum", False), ("maximum", True)], ids=["split", "join"]
)
def test_subtree_extremum_layout_is_deterministic(
    x_by_vertex_of, root_kind: str, mirror: bool
) -> None:
    pytest.importorskip("iplotx")
    vertices = _mirror(_TALL_BRANCH_VERTICES) if mirror else _TALL_BRANCH_VERTICES
    tree = _build_rooted_tree(vertices, _TALL_BRANCH_EDGES)
    shuffled = _build_rooted_tree(
        vertices, _TALL_BRANCH_EDGES, insertion_order=[5, 2, 7, 0, 4, 1, 6, 3]
    )

    first = x_by_vertex_of(tree, "subtree_extremum", root_kind=root_kind)
    second = x_by_vertex_of(tree, "subtree_extremum", root_kind=root_kind)
    permuted = x_by_vertex_of(shuffled, "subtree_extremum", root_kind=root_kind)

    assert first == second
    assert first == permuted


@pytest.mark.parametrize("root_kind", ["minimum", "maximum"])
def test_make_tree_layout_subtree_extremum_tables_are_identical_across_runs(
    root_kind: str,
) -> None:
    pytest.importorskip("iplotx")
    from topographer.plotting import make_tree_layout

    vertices = _mirror(_TALL_BRANCH_VERTICES) if root_kind == "maximum" else _TALL_BRANCH_VERTICES
    tree = _build_rooted_tree(vertices, _TALL_BRANCH_EDGES)

    first = make_tree_layout(tree, root_kind=root_kind, horizontal_layout="subtree_extremum")
    second = make_tree_layout(tree, root_kind=root_kind, horizontal_layout="subtree_extremum")

    assert first.node_table.equals(second.node_table)
    assert first.edge_table.equals(second.edge_table)
    assert first.edge_segment_table.equals(second.edge_segment_table)


# A path network with scalars in [0, 1]. Under the default ``iplotx`` order neither
# its global maximum (vertex 2) nor its global minimum (vertex 7) is a leftmost
# leaf, so the computed trees below check the rule rather than a lucky layout.
_PATH_SCALARS = [0.375, 0.25, 1.0, 0.75, 0.875, 0.125, 0.5, 0.0, 0.625]


def _compute_path_merge_tree(tree_kind: str):
    """Compute a split or join tree through the public API on ``_PATH_SCALARS``."""

    import polars as pl

    from topographer import Graph, ScalarNetwork, ScalarNetworkTopology

    ids = [f"v{index}" for index in range(len(_PATH_SCALARS))]
    graph = Graph.from_tables(
        pl.DataFrame({"id": ids, "val": _PATH_SCALARS}),
        pl.DataFrame({"src": ids[:-1], "dst": ids[1:]}),
        vertex_id_col="id",
        source_id_col="src",
        target_id_col="dst",
        vertex_attribute_cols=("val",),
    )
    topology = ScalarNetworkTopology(ScalarNetwork(graph, scalar_attr="val"))
    if tree_kind == "split":
        return topology.split_tree()
    return topology.join_tree()


_COMPUTED_TREE_CASES = [
    # tree kind, root kind, vertex of the global extremum farthest from the root
    pytest.param("split", "minimum", _PATH_SCALARS.index(max(_PATH_SCALARS)), id="split"),
    pytest.param("join", "maximum", _PATH_SCALARS.index(min(_PATH_SCALARS)), id="join"),
]


@pytest.mark.parametrize("x_by_vertex_of", _SUBTREE_EXTREMUM_LAYOUTS)
@pytest.mark.parametrize(("tree_kind", "root_kind", "extremum"), _COMPUTED_TREE_CASES)
def test_subtree_extremum_puts_the_global_extremum_leftmost_on_computed_merge_trees(
    x_by_vertex_of, tree_kind: str, root_kind: str, extremum: int
) -> None:
    pytest.importorskip("iplotx")
    graph = _compute_path_merge_tree(tree_kind).graph

    default_x = x_by_vertex_of(graph, "iplotx", root_kind=root_kind)
    x_by_vertex = x_by_vertex_of(graph, "subtree_extremum", root_kind=root_kind)

    assert default_x[extremum] != min(default_x.values())
    assert x_by_vertex[extremum] == min(x_by_vertex.values()) == 0.0


@pytest.mark.parametrize(("tree_kind", "root_kind", "extremum"), _COMPUTED_TREE_CASES)
def test_make_tree_layout_infers_the_root_kind_of_a_merge_tree_for_subtree_extremum(
    tree_kind: str, root_kind: str, extremum: int
) -> None:
    pytest.importorskip("iplotx")
    from topographer.plotting import make_tree_layout

    merge_tree = _compute_path_merge_tree(tree_kind)

    plot_data = make_tree_layout(merge_tree, horizontal_layout="subtree_extremum")

    assert plot_data.root_kind == root_kind
    x_by_vertex = {row["vertex_index"]: row["x"] for row in plot_data.node_table.to_dicts()}
    assert x_by_vertex[extremum] == min(x_by_vertex.values()) == 0.0


@pytest.mark.parametrize("horizontal_layout", ["bogus", "subtree_max"])
def test_horizontal_layout_rejects_unknown_values(horizontal_layout: str) -> None:
    from topographer.plotting import make_tree_layout
    from topographer.plotting._layout import build_iplotx_tree_layout

    tree = _build_rooted_tree(_TALL_BRANCH_VERTICES, _TALL_BRANCH_EDGES)

    with pytest.raises(ValueError, match="subtree_extremum"):
        make_tree_layout(tree, root_kind="minimum", horizontal_layout=horizontal_layout)
    with pytest.raises(ValueError, match="subtree_extremum"):
        build_iplotx_tree_layout(tree, scalar_attr="scalar", horizontal_layout=horizontal_layout)


@pytest.mark.parametrize(("root_kind", "expected_root"), [("minimum", 0), ("maximum", 2)])
def test_select_tree_root_node_index_is_public_and_picks_extremum(
    root_kind: str, expected_root: int
) -> None:
    from topographer.plotting import select_tree_root_node_index

    tree = _build_three_node_tree()
    assert select_tree_root_node_index(tree, "scalar", root_kind) == expected_root


def test_select_tree_root_node_index_rejects_unknown_root_kind() -> None:
    from topographer.plotting import select_tree_root_node_index

    with pytest.raises(ValueError, match="root_kind"):
        select_tree_root_node_index(_build_three_node_tree(), "scalar", "middle")
