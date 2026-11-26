"""topographer.topology module."""

from __future__ import annotations

from topographer.topology._band import BootstrapPersistenceBand
from topographer.topology._band import band_network_fingerprint as band_network_fingerprint
from topographer.topology._distances import (
    diagram_bottleneck_distance,
    diagram_wasserstein_distance,
)
from topographer.topology._mapping import MergeTreeMapping
from topographer.topology._merge_tree import MergeTree
from topographer.topology._network_persistence import PersistenceResult
from topographer.topology._persistence import (
    build_tree_critical_branch_span_table as build_tree_critical_branch_span_table,
)
from topographer.topology._persistence import (
    build_tree_extended_persistence_pair_table as build_tree_extended_persistence_pair_table,
)
from topographer.topology._summary import (
    _register_landscape_summary as _register_landscape_summary,
)
from topographer.topology._summary import compute_topological_summary
from topographer.topology._tree_construction import MergeTreeResult as MergeTreeResult
from topographer.topology._tree_construction import build_join_tree, build_split_tree
from topographer.topology._tree_construction import select_extremum_root as select_extremum_root
from topographer.topology._tree_location import build_tree_location_tables
from topographer.topology._tree_metrics import TreeMetrics, build_tree_metrics
from topographer.topology._tree_metrics import _fold_tree as _fold_tree
from topographer.topology._tree_metrics import _losing_children as _losing_children
from topographer.topology._tree_metrics import _pair_value as _pair_value
from topographer.topology._tree_metrics import _rooted_merge_tree as _rooted_merge_tree
from topographer.topology._tree_metrics import _rooted_tree as _rooted_tree
from topographer.topology._tree_metrics import _RootedTree as _RootedTree
from topographer.topology._tree_reduction import reduce_merge_tree
from topographer.topology._workspace import ScalarNetworkTopology

__all__ = [
    "BootstrapPersistenceBand",
    "MergeTree",
    "MergeTreeMapping",
    "PersistenceResult",
    "ScalarNetworkTopology",
    "TreeMetrics",
    "build_join_tree",
    "build_split_tree",
    "build_tree_location_tables",
    "build_tree_metrics",
    "compute_topological_summary",
    "diagram_bottleneck_distance",
    "diagram_wasserstein_distance",
    "reduce_merge_tree",
]
