"""Keep implementation modules private to their own packages."""

import ast
from importlib.util import resolve_name
from pathlib import Path

import pytest


def imported_modules(node, current_package, source):
    """Resolve static imports, including imports of modules from a package."""
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if not isinstance(node, ast.ImportFrom):
        return []
    module = node.module or ""
    if node.level:
        module = resolve_name("." * node.level + module, current_package)
    modules = [module]
    parts = module.split(".")
    if len(parts) >= 2 and parts[0] == "topographer":
        for alias in node.names:
            candidate = source.joinpath(*parts[1:], alias.name)
            if candidate.with_suffix(".py").is_file() or (candidate / "__init__.py").is_file():
                modules.append(f"{module}.{alias.name}")
    return modules


@pytest.mark.parametrize(
    ("statement", "current_package", "expected"),
    [
        (
            "from ..topology._merge_tree import MergeTree",
            "topographer.plotting",
            "topographer.topology._merge_tree",
        ),
        (
            "from ..topology import _merge_tree",
            "topographer.plotting",
            "topographer.topology._merge_tree",
        ),
        (
            "from topographer.topology import _merge_tree",
            "topographer.plotting",
            "topographer.topology._merge_tree",
        ),
        (
            "import topographer.topology._merge_tree",
            "topographer.plotting",
            "topographer.topology._merge_tree",
        ),
    ],
)
def test_imported_modules_resolves_private_modules(statement, current_package, expected):
    source = Path(__file__).resolve().parents[1] / "src" / "topographer"
    assert expected in imported_modules(ast.parse(statement).body[0], current_package, source)


def test_no_cross_package_private_module_imports():
    source = Path(__file__).resolve().parents[1] / "src" / "topographer"
    violations = []
    for path in source.rglob("*.py"):
        package = path.relative_to(source).parts[0]
        relative = path.relative_to(source)
        current_package = ".".join(("topographer", *relative.parts[:-1]))
        for node in ast.walk(ast.parse(path.read_text())):
            for module in imported_modules(node, current_package, source):
                parts = module.split(".")
                if module == "topographer.graph._archive":
                    # Shared folder helpers. They stay off ``topographer.graph``.
                    continue
                if (
                    len(parts) > 2
                    and parts[0] == "topographer"
                    and parts[1] != package
                    and any(part.startswith("_") for part in parts[2:])
                ):
                    violations.append(f"{path}:{node.lineno}: {module}")
    assert not violations, "\n".join(violations)
