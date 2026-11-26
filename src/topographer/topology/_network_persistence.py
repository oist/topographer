"""Network persistence container and component (H0) persistence algorithms."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Any

import polars as pl

from topographer.topology._scalar_view import vertex_scalar_values

if TYPE_CHECKING:
    from topographer.topology._workspace import ScalarNetworkTopology

PERSISTENCE_PAIR_KINDS = (
    "ordinary",
    "relative",
    "extended_positive",
    "essential",
)

PAIR_KIND_ORDER = {kind: i for i, kind in enumerate(PERSISTENCE_PAIR_KINDS)}


@dataclass(frozen=True, slots=True)
class PersistenceResult:
    """Immutable result containing component persistence pairs.

    Examples
    --------
    >>> from topographer.examples import build_cube_scalar_network
    >>> from topographer.topology import ScalarNetworkTopology
    >>> result = ScalarNetworkTopology(build_cube_scalar_network()).component_persistence()
    >>> result.pairs.height > 0
    True
    """

    pairs: pl.DataFrame

    def __post_init__(self) -> None:
        required_columns = {
            "pair_id",
            "homology_dimension",
            "pair_kind",
            "birth_simplex",
            "death_simplex",
            "birth_scalar",
            "death_scalar",
            "persistence",
        }
        if not required_columns.issubset(set(self.pairs.columns)):
            missing = required_columns - set(self.pairs.columns)
            raise ValueError(f"Persistence pairs DataFrame missing required columns: {missing}")


def sort_and_assign_pair_ids(
    rows: list[dict[str, Any]],
    canonical_vertex_ranks: dict[Any, int],
) -> pl.DataFrame:
    """Sort persistence rows deterministically and assign sequential pair_id values.

    Deterministic sort order:
    1. homology_dimension
    2. pair_kind order: ordinary, relative, extended_positive, essential
    3. native birth_scalar
    4. native death_scalar
    5. canonical birth_simplex rank tuple
    6. canonical death_simplex rank tuple
    """

    def _simplex_ranks(simplex: tuple[Any, ...] | list[Any]) -> tuple[int, ...]:
        return tuple(sorted(canonical_vertex_ranks[v] for v in simplex))

    def _sort_key(item: tuple[int, dict[str, Any]]) -> tuple[Any, ...]:
        _, row = item
        dim = int(row["homology_dimension"])
        kind_rank = PAIR_KIND_ORDER.get(str(row["pair_kind"]), 999)
        b_scalar = float(row["birth_scalar"])
        d_scalar = float(row["death_scalar"])
        b_ranks = _simplex_ranks(row["birth_simplex"])
        d_ranks = _simplex_ranks(row["death_simplex"])
        return (
            dim,
            kind_rank,
            b_scalar,
            math.isnan(d_scalar),
            0.0 if math.isnan(d_scalar) else d_scalar,
            b_ranks,
            d_ranks,
        )

    indexed_rows = list(enumerate(rows))
    indexed_rows.sort(key=_sort_key)

    new_rows: list[dict[str, Any]] = []

    for new_id, (_, row) in enumerate(indexed_rows):
        row_copy = dict(row)
        row_copy["pair_id"] = new_id
        # Ensure simplices are lists for Polars compatibility
        row_copy["birth_simplex"] = list(row["birth_simplex"])
        row_copy["death_simplex"] = list(row["death_simplex"])
        new_rows.append(row_copy)

    schema = {
        "pair_id": pl.Int64,
        "homology_dimension": pl.Int64,
        "pair_kind": pl.String,
        "birth_simplex": pl.List(pl.String),
        "death_simplex": pl.List(pl.String),
        "birth_scalar": pl.Float64,
        "death_scalar": pl.Float64,
        "persistence": pl.Float64,
    }

    if not new_rows:
        df = pl.DataFrame(schema=schema)
    else:
        df = pl.DataFrame(new_rows)
        df = df.with_columns(
            [
                pl.col("pair_id").cast(pl.Int64),
                pl.col("homology_dimension").cast(pl.Int64),
                pl.col("pair_kind").cast(pl.String),
                pl.col("birth_scalar").cast(pl.Float64),
                pl.col("death_scalar").cast(pl.Float64),
                pl.col("persistence").cast(pl.Float64),
            ]
        )

    return df


def _merge_tree_h0_rows(
    tree: Any, filtration: str, index_to_vid: dict[int, Any]
) -> list[dict[str, Any]]:
    rows = []
    for row in tree.persistence().to_dicts():
        if row["pair_kind"] != "ordinary":
            continue
        birth = float(row["birth_scalar"])
        death = float(row["death_scalar"])
        assert birth <= death if filtration == "sublevel" else birth >= death
        rows.append(
            {
                "homology_dimension": 0,
                "pair_kind": "ordinary",
                "birth_simplex": (index_to_vid[int(row["birth_vertex_index"])],),
                "death_simplex": (index_to_vid[int(row["death_vertex_index"])],),
                "birth_scalar": birth,
                "death_scalar": death,
                "persistence": abs(death - birth),
            }
        )
    return rows


def compute_persistence_diagram(
    topology: ScalarNetworkTopology, filtration: str, *, include_essential: str
) -> PersistenceResult:
    """Compute H0 persistence for one filtration in native scalar units."""
    vertex_ids = topology.network.vertex_ids
    ranks = {vid: i for i, vid in enumerate(vertex_ids)}
    index_to_vid = dict(enumerate(vertex_ids))
    tree = topology.compute_merge_tree(filtration)
    rows = _merge_tree_h0_rows(tree, filtration, index_to_vid)
    extended = next(row for row in tree.persistence().to_dicts() if row["pair_kind"] != "ordinary")
    birth = float(extended["birth_scalar"])
    death = float(extended["death_scalar"])
    rows.append(
        {
            "homology_dimension": 0,
            "pair_kind": "essential",
            "birth_simplex": (index_to_vid[int(extended["birth_vertex_index"])],),
            "death_simplex": (index_to_vid[int(extended["death_vertex_index"])],)
            if include_essential == "capped"
            else (),
            "birth_scalar": birth,
            "death_scalar": death if include_essential == "capped" else math.nan,
            "persistence": abs(death - birth) if include_essential == "capped" else math.inf,
        }
    )
    return PersistenceResult(pairs=sort_and_assign_pair_ids(rows, ranks))


def compute_component_persistence(topology: ScalarNetworkTopology) -> PersistenceResult:
    """Compute extended component persistence from lazy join and split trees.

    The global extended pair breaks exact ties in the join sweep's order: it is born
    at the first tied global minimum swept and dies at the last tied global maximum
    swept, matching the join tree's extended row.
    """
    network = topology.network
    vertex_ids = network.vertex_ids
    canonical_ranks = {vid: i for i, vid in enumerate(vertex_ids)}
    index_to_vid = {i: vid for i, vid in enumerate(vertex_ids)}

    rows: list[dict[str, Any]] = []

    rows.extend(
        _merge_tree_h0_rows(topology.compute_merge_tree("sublevel"), "sublevel", index_to_vid)
    )
    relative_rows = _merge_tree_h0_rows(
        topology.compute_merge_tree("superlevel"), "superlevel", index_to_vid
    )
    for row in relative_rows:
        row["homology_dimension"] = 1
        row["pair_kind"] = "relative"
    rows.extend(relative_rows)

    # 3. Global pair: global minimum to global maximum (extended_positive H0).
    # Exact ties follow the ascending (join) sweep's symbolic order (scalar, vertex
    # index): the pair is born at the first tied global minimum swept and dies at the
    # last tied global maximum swept, the join-tree root. It therefore equals the join
    # tree's extended row. Only this global pair follows the join sweep; the relative
    # rows above come from the split sweep.
    scalars = vertex_scalar_values(network)
    global_min_vid = min(
        vertex_ids,
        key=lambda vid: (scalars[vid], canonical_ranks[vid]),
    )
    global_max_vid = max(
        vertex_ids,
        key=lambda vid: (scalars[vid], canonical_ranks[vid]),
    )

    min_scalar = float(scalars[global_min_vid])
    max_scalar = float(scalars[global_max_vid])

    rows.append(
        {
            "homology_dimension": 0,
            "pair_kind": "extended_positive",
            "birth_simplex": (global_min_vid,),
            "death_simplex": (global_max_vid,),
            "birth_scalar": min_scalar,
            "death_scalar": max_scalar,
            "persistence": abs(max_scalar - min_scalar),
        }
    )

    df = sort_and_assign_pair_ids(rows, canonical_vertex_ranks=canonical_ranks)
    return PersistenceResult(pairs=df)
