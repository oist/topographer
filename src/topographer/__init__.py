"""Public package exports for topographer."""

from __future__ import annotations

from importlib.metadata import version as distribution_version

import topographer.biology  # noqa: F401
from topographer.graph import (
    Graph,
    HammingGraph,
    ScalarNetwork,
    SequenceGraph,
)
from topographer.topology import (
    MergeTree,
    MergeTreeMapping,
    PersistenceResult,
    ScalarNetworkTopology,
)

__all__ = [
    "Graph",
    "HammingGraph",
    "MergeTree",
    "MergeTreeMapping",
    "PersistenceResult",
    "ScalarNetwork",
    "ScalarNetworkTopology",
    "SequenceGraph",
    "__version__",
]

__version__ = distribution_version("topographer")
