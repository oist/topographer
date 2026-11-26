# Contributing to Topographer

Topographer is a small, reusable scientific Python library for topological analysis of scalar networks.  Contributions should keep the topology core independent of any particular biological study.

## Local setup

Install the development and notebook dependencies with:

```bash
uv sync --extra dev --extra docs
```

Use these checks before opening a pull request:

```bash
uv run ruff format --check .
uv run ruff check .
uv run pytest -q
uv build
```

`just check` runs the same local quality gate.  The project requires Python 3.11 or newer.

## Pull requests

Keep each pull request focused.  Explain the scientific or API motivation, state any compatibility impact, and include deterministic tests for new behavior.  New generic examples should use a compact graph fixture; Dataset-specific data and helpers belong in `topographer.biology` or an example notebook.

Read `agents/API.md`, `agents/TESTING.md`, and `agents/REVIEW.md` before making a substantial change. Update `CHANGELOG.md` whenever a user-visible change is made.

## Design principles

- Prefer Rustworkx graphs and Polars tables at public boundaries.
- Declare vertex and edge scalar attributes explicitly.
- Keep algorithms deterministic and easy to inspect.
- Use Altair plot helpers and reusable immutable plot configuration rather than embedding styling choices throughout notebooks.
