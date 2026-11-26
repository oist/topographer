"""Every public name in topology and plotting must be listed in great-docs.yml."""

from __future__ import annotations

import importlib
from pathlib import Path
import re

import pytest

CONFIG = Path(__file__).resolve().parents[1] / "great-docs.yml"


def _listed_entries() -> set[str]:
    entries: set[str] = set()
    for match in re.finditer(r"contents:\s*\[([^\]]*)\]", CONFIG.read_text()):
        entries.update(item.strip() for item in match.group(1).split(","))
    return entries


@pytest.mark.parametrize("subpackage", ["topology", "plotting"])
def test_public_names_are_listed_in_great_docs(subpackage: str) -> None:
    module = importlib.import_module(f"topographer.{subpackage}")
    entries = _listed_entries()
    missing = [
        name
        for name in module.__all__
        if f"{subpackage}.{name}" not in entries and name not in entries
    ]
    assert not missing, f"missing from great-docs.yml: {missing}"
