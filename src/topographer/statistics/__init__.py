"""topographer.statistics package."""

from __future__ import annotations

from topographer.statistics import bootstrap
from topographer.statistics._multiple_testing import compute_bh_qvalues
from topographer.statistics._persistence_band import (
    BottleneckToNominal,
    bootstrap_persistence_band,
)
from topographer.statistics._standard_errors import validate_se_attr

__all__ = [
    "BottleneckToNominal",
    "bootstrap",
    "bootstrap_persistence_band",
    "compute_bh_qvalues",
    "validate_se_attr",
]
