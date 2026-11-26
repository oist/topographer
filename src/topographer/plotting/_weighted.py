"""Leaf-count-weighted horizontal positions for rooted trees."""

from __future__ import annotations


def weighted_tree_positions(
    children: dict[int, list[int]],
    root: int,
    scalars: dict[int, float],
    root_kind: str,
) -> dict[int, float]:
    """Place a rooted tree so every subtree owns an x interval as wide as its leaf count.

    Children are ordered by the subtree extremum farthest from the root (persistence order),
    then by leaf count, then by node index; the first child gets the smallest x. The root owns
    ``[0, L)`` for ``L`` leaves and each child a contiguous sub-interval of width equal to its
    leaf count. A leaf sits at its interval centre, so leaves are exactly one unit apart. An
    internal node sits at the leaf-weighted mean of its children's x, which always lies between
    its first and last child, so edges never cross. Only x is computed: the vertical coordinate
    stays the scalar value. The traversal is iterative, so deep trees do not recurse.

    Examples
    --------
    >>> from topographer.plotting._weighted import weighted_tree_positions
    >>> x = weighted_tree_positions({0: [1, 2], 1: [], 2: []}, 0, {0: 0.0, 1: 1.0, 2: 2.0}, "minimum")
    >>> sorted(x.items())
    [(0, 1.0), (1, 1.5), (2, 0.5)]
    """

    from topographer.plotting._layout import _order_children_by_subtree_extremum

    ordered = {node: list(kids) for node, kids in children.items()}
    _order_children_by_subtree_extremum(ordered, root, scalars, root_kind, size_key="leaves")

    preorder: list[int] = []
    stack = [root]
    while stack:
        node = stack.pop()
        preorder.append(node)
        stack.extend(reversed(ordered[node]))

    leaves: dict[int, int] = {}
    for node in reversed(preorder):
        kids = ordered[node]
        leaves[node] = sum(leaves[c] for c in kids) if kids else 1

    lower: dict[int, float] = {root: 0.0}
    for node in preorder:
        cursor = lower[node]
        for child in ordered[node]:
            lower[child] = cursor
            cursor += leaves[child]

    positions: dict[int, float] = {}
    for node in reversed(preorder):
        kids = ordered[node]
        if not kids:
            positions[node] = lower[node] + 0.5
        else:
            positions[node] = sum(leaves[c] * positions[c] for c in kids) / leaves[node]
    return positions
