"""Mark the example suite for selective test runs."""

from pathlib import Path

import pytest


def pytest_collection_modifyitems(items):
    """Mark all tests collected under this directory as examples."""
    for item in items:
        if Path(__file__).parent in item.path.parents:
            item.add_marker(pytest.mark.examples)
