"""Altair-first interactive plotting for generic scalar networks."""

from __future__ import annotations

from topographer.plotting._config import (
    ColorScaleConfig,
    PersistencePlotConfig,
    ReferenceLineConfig,
    TreeAnnotationLayer,
    TreeMarkerConfig,
    TreePlotConfig,
    TreePresentationConfig,
    build_scalar_plot_domain,
    count_color_scale,
)
from topographer.plotting._data import (
    TreePlotData,
    build_tree_plot_data,
    make_tree_layout,
    relayout_tree_plot_data,
)
from topographer.plotting._layout import select_tree_root_node_index
from topographer.plotting._marginal import (
    compute_fitness_histogram_table,
    plot_fitness_histogram,
)
from topographer.plotting._persistence import (
    plot_branch_span,
    plot_branch_spans,
    plot_extended_persistence,
    plot_ordinary_persistence,
    plot_persistence,
)
from topographer.plotting._swarm import compute_branch_local_swarm
from topographer.plotting._tree import plot_tree

__all__ = [
    "ColorScaleConfig",
    "PersistencePlotConfig",
    "ReferenceLineConfig",
    "TreeAnnotationLayer",
    "TreeMarkerConfig",
    "TreePlotConfig",
    "TreePlotData",
    "TreePresentationConfig",
    "build_scalar_plot_domain",
    "build_tree_plot_data",
    "compute_branch_local_swarm",
    "compute_fitness_histogram_table",
    "count_color_scale",
    "make_tree_layout",
    "plot_branch_span",
    "plot_branch_spans",
    "plot_extended_persistence",
    "plot_fitness_histogram",
    "plot_ordinary_persistence",
    "plot_persistence",
    "plot_tree",
    "relayout_tree_plot_data",
    "select_tree_root_node_index",
]
