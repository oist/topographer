"""Run the doctests of every module under ``topographer``, public and private."""

from __future__ import annotations

import doctest
import importlib
import pkgutil

import pytest

import topographer


def _module_names() -> list[str]:
    return sorted(
        info.name
        for info in pkgutil.walk_packages(topographer.__path__, prefix="topographer.")
        if "__main__" not in info.name
    )


@pytest.mark.parametrize("module_name", _module_names())
def test_module_doctests_pass(module_name: str) -> None:
    try:
        module = importlib.import_module(module_name)
    except ImportError as error:  # optional extras (e.g. plotting) not installed
        pytest.skip(f"optional dependency missing: {error}")

    result = doctest.testmod(module, optionflags=doctest.ELLIPSIS, verbose=False)

    assert result.failed == 0
