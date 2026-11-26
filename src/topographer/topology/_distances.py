"""Pairwise persistence diagram distance metrics."""

from __future__ import annotations

import math

import numpy as np
import polars as pl
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from scipy.spatial import cKDTree

from topographer.topology._network_persistence import PersistenceResult


def _extract_diagram_points(
    target: PersistenceResult | pl.DataFrame,
    pair_kind: str | None = None,
) -> np.ndarray:
    """Extract (birth, death) coordinate pairs from PersistenceResult or DataFrame."""
    if isinstance(target, PersistenceResult):
        df = target.pairs
    elif isinstance(target, pl.DataFrame):
        df = target
    else:
        raise TypeError(
            f"Expected PersistenceResult or pl.DataFrame, got {type(target).__name__}."
        )

    if pair_kind is not None:
        if not isinstance(pair_kind, str):
            raise TypeError(f"pair_kind must be a string or None, got {type(pair_kind).__name__}.")
        if "pair_kind" not in df.columns:
            raise ValueError(
                f"pair_kind={pair_kind!r} was specified, but DataFrame does not contain a 'pair_kind' column."
            )
        df = df.filter(pl.col("pair_kind") == pair_kind)

    if df.is_empty():
        return np.empty((0, 2), dtype=np.float64)

    required_cols = {"birth_scalar", "death_scalar"}
    if not required_cols.issubset(set(df.columns)):
        missing = sorted(required_cols - set(df.columns))
        raise ValueError(f"Persistence table is missing required columns: {missing}.")

    b = df["birth_scalar"].to_numpy().astype(np.float64)
    d = df["death_scalar"].to_numpy().astype(np.float64)

    if not (np.all(np.isfinite(b)) and np.all(np.isfinite(d))):
        raise ValueError("Persistence coordinates must be finite real numbers.")

    return np.column_stack((b, d))


def _pairs_within(
    tree1: cKDTree, tree2: cKDTree, pts1: np.ndarray, pts2: np.ndarray, radius: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(i, j, dist)`` for all cross pairs with L-infinity distance ``<= radius``.

    The tree query uses a slightly inflated radius and the distances are then recomputed and
    filtered exactly, so boundary ties never depend on the tree's internal comparison.
    """
    query_radius = float(np.nextafter(radius * (1.0 + 1e-9), np.inf))
    found = tree1.sparse_distance_matrix(tree2, query_radius, p=np.inf, output_type="ndarray")
    i, j = found["i"], found["j"]
    dist = np.maximum(np.abs(pts1[i, 0] - pts2[j, 0]), np.abs(pts1[i, 1] - pts2[j, 1]))
    keep = dist <= radius
    return i[keep], j[keep], dist[keep]


def _covers_forced(rows: np.ndarray, cols: np.ndarray, forced: np.ndarray, n_cols: int) -> bool:
    """Whether the edges ``(rows, cols)`` admit a matching covering every forced row."""
    n_forced = int(forced.sum())
    if n_forced == 0:
        return True
    if n_forced > n_cols:
        return False
    keep = forced[rows]
    compact = np.cumsum(forced) - 1
    forced_rows = compact[rows[keep]]
    if np.bincount(forced_rows, minlength=n_forced).min() == 0:
        return False
    graph = csr_matrix(
        (np.ones(forced_rows.size, dtype=np.int8), (forced_rows, cols[keep])),
        shape=(n_forced, n_cols),
    )
    return bool(np.all(maximum_bipartite_matching(graph, perm_type="column") >= 0))


def _bottleneck_exact(pts1: np.ndarray, pts2: np.ndarray) -> float:
    """Exact bottleneck distance between two non-empty ``(k, 2)`` diagrams.

    The answer is the smallest candidate (``0``, a cross-pair L-infinity distance or a diagonal
    cost) at which the augmented bipartite graph has a perfect matching. At threshold ``t`` the
    diagonal copies match one another freely and any point with diagonal cost ``<= t`` can be
    parked on the diagonal, so a perfect matching exists exactly when one point-to-point matching
    covers every "forced" point (diagonal cost ``> t``) of both diagrams. By the
    Mendelsohn-Dulmage theorem that holds iff each side's forced points can be covered
    separately: two sparse bipartite matchings.

    Feasibility is monotone in ``t`` and, for real ``t``, equals feasibility at the largest
    candidate ``<= t``. So the threshold is bracketed by doubling over real values, narrowed
    a few times, and only then are the candidates inside the small final bracket enumerated
    and bisected exactly. Edges are enumerated sparsely with a KD-tree (``p=inf``), never as a
    dense matrix.
    """
    n1, n2 = len(pts1), len(pts2)
    diag1 = np.abs(pts1[:, 1] - pts1[:, 0]) / 2.0
    diag2 = np.abs(pts2[:, 1] - pts2[:, 0]) / 2.0
    tree1, tree2 = cKDTree(pts1), cKDTree(pts2)

    def feasible_with(t: float, i: np.ndarray, j: np.ndarray) -> bool:
        return _covers_forced(i, j, diag1 > t, n2) and _covers_forced(j, i, diag2 > t, n1)

    def feasible(t: float) -> bool:
        forced1, forced2 = diag1 > t, diag2 > t
        if not (forced1.any() or forced2.any()):
            return True
        if forced1.sum() > n2 or forced2.sum() > n1:
            return False
        i, j, _ = _pairs_within(tree1, tree2, pts1, pts2, t)
        return feasible_with(t, i, j)

    upper = float(max(diag1.max(), diag2.max()))  # parking every point is always feasible
    if feasible(0.0):
        return 0.0

    low, high = 0.0, max(upper * 2.0**-40, float(np.nextafter(0.0, 1.0)))
    while not feasible(high):
        low = high
        high = min(high * 2.0, upper)
    for _ in range(10):
        if low == 0.0 or high <= low * 1.05:
            break
        middle = 0.5 * (low + high)
        if feasible(middle):
            high = middle
        else:
            low = middle

    i, j, dist = _pairs_within(tree1, tree2, pts1, pts2, high)
    candidates = np.unique(
        np.concatenate(
            [
                dist[dist > low],
                diag1[(diag1 > low) & (diag1 <= high)],
                diag2[(diag2 > low) & (diag2 <= high)],
            ]
        )
    )
    first, last = 0, candidates.size - 1  # the largest candidate <= high is feasible
    while first < last:
        middle = (first + last) // 2
        t = float(candidates[middle])
        keep = dist <= t
        if feasible_with(t, i[keep], j[keep]):
            last = middle
        else:
            first = middle + 1
    return float(candidates[first])


def diagram_bottleneck_distance(
    result1: PersistenceResult | pl.DataFrame,
    result2: PersistenceResult | pl.DataFrame,
    *,
    pair_kind: str | None = None,
) -> float:
    r"""Compute the exact Bottleneck distance ($L_\infty$) between two persistence pair sets.

    The Bottleneck distance is the infimum over all bijections between diagrams
    (augmented with diagonal points) of the supremum $L_\infty$ distance between matched points.
    Points $p = (b, d)$ can be matched to diagonal points $\bar{p} = ((b+d)/2, (b+d)/2)$
    with diagonal cost $|d - b| / 2$.

    Parameters
    ----------
    result1 : PersistenceResult | pl.DataFrame
        First persistence diagram.
    result2 : PersistenceResult | pl.DataFrame
        Second persistence diagram.
    pair_kind : str | None, optional
        Optional filter by pair kind (e.g., 'ordinary', 'relative', 'extended_positive', 'essential').

    Returns
    -------
    float
        The exact Bottleneck distance.

    Examples
    --------
    >>> import polars as pl
    >>> from topographer.topology import diagram_bottleneck_distance
    >>> df1 = pl.DataFrame({"birth_scalar": [0.0, 1.0], "death_scalar": [2.0, 4.0]})
    >>> df2 = pl.DataFrame({"birth_scalar": [0.0], "death_scalar": [2.0]})
    >>> diagram_bottleneck_distance(df1, df2)
    1.5
    """
    pts1 = _extract_diagram_points(result1, pair_kind=pair_kind)
    pts2 = _extract_diagram_points(result2, pair_kind=pair_kind)

    n1 = len(pts1)
    n2 = len(pts2)

    if n1 == 0 and n2 == 0:
        return 0.0
    if n1 == 0:
        return float(np.max(np.abs(pts2[:, 1] - pts2[:, 0]) / 2.0))
    if n2 == 0:
        return float(np.max(np.abs(pts1[:, 1] - pts1[:, 0]) / 2.0))

    return _bottleneck_exact(pts1, pts2)


def diagram_wasserstein_distance(
    result1: PersistenceResult | pl.DataFrame,
    result2: PersistenceResult | pl.DataFrame,
    *,
    pair_kind: str | None = None,
    p: float = 1.0,
) -> float:
    r"""Compute the $p$-Wasserstein distance ($p \ge 1.0$) between two persistence pair sets.

    Uses `scipy.optimize.linear_sum_assignment` to find the minimum-cost matching
    between persistence diagrams, with matching to diagonal points
    $\bar{p} = ((b+d)/2, (b+d)/2)$ at cost $((d - b) / 2)^p$.

    Parameters
    ----------
    result1 : PersistenceResult | pl.DataFrame
        First persistence diagram.
    result2 : PersistenceResult | pl.DataFrame
        Second persistence diagram.
    pair_kind : str | None, optional
        Optional filter by pair kind (e.g., 'ordinary', 'relative', 'extended_positive', 'essential').
    p : float, default=1.0
        The Wasserstein power parameter ($p \ge 1.0$).

    Returns
    -------
    float
        The $p$-Wasserstein distance $(\min \text{cost})^{1/p}$.

    Examples
    --------
    >>> import polars as pl
    >>> from topographer.topology import diagram_wasserstein_distance
    >>> df1 = pl.DataFrame({"birth_scalar": [0.0], "death_scalar": [2.0]})
    >>> df2 = pl.DataFrame({"birth_scalar": [0.0], "death_scalar": [2.0]})
    >>> diagram_wasserstein_distance(df1, df2)
    0.0
    """
    if not isinstance(p, (int, float)) or math.isnan(p) or p < 1.0:
        raise ValueError(f"p must be a real number >= 1.0, got {p}.")
    p_val = float(p)

    pts1 = _extract_diagram_points(result1, pair_kind=pair_kind)
    pts2 = _extract_diagram_points(result2, pair_kind=pair_kind)

    n1 = len(pts1)
    n2 = len(pts2)

    if n1 == 0 and n2 == 0:
        return 0.0

    diag1 = np.abs(pts1[:, 1] - pts1[:, 0]) / 2.0 if n1 > 0 else np.empty(0, dtype=np.float64)
    diag2 = np.abs(pts2[:, 1] - pts2[:, 0]) / 2.0 if n2 > 0 else np.empty(0, dtype=np.float64)

    if n1 == 0:
        total = float(np.sum(diag2**p_val))
        return float(total ** (1.0 / p_val))
    if n2 == 0:
        total = float(np.sum(diag1**p_val))
        return float(total ** (1.0 / p_val))

    k = n1 + n2
    cost_matrix = np.full((k, k), np.inf, dtype=np.float64)

    # Top-left n1 x n2: pairwise L_inf distances raised to p
    diff_b = np.abs(pts1[:, 0:1] - pts2[:, 0:1].T)
    diff_d = np.abs(pts1[:, 1:2] - pts2[:, 1:2].T)
    pair_costs = np.maximum(diff_b, diff_d) ** p_val
    cost_matrix[:n1, :n2] = pair_costs

    # Top-right n1 x n1: diagonal entries for pts1
    for i in range(n1):
        cost_matrix[i, n2 + i] = float(diag1[i] ** p_val)

    # Bottom-left n2 x n2: diagonal entries for pts2
    for j in range(n2):
        cost_matrix[n1 + j, j] = float(diag2[j] ** p_val)

    # Bottom-right n2 x n1: diagonal dummy to dummy = 0.0
    cost_matrix[n1:, n2:] = 0.0

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    min_cost = float(cost_matrix[row_ind, col_ind].sum())
    return float(min_cost ** (1.0 / p_val))


__all__ = [
    "diagram_bottleneck_distance",
    "diagram_wasserstein_distance",
]
