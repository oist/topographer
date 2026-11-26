# TESTING.md

Instructions for code testing.

Run the suite with `uv run --extra dev pytest -q`, or run the full local gate with `just check`. Notebook checks use the `docs` extra (`uv sync --extra dev --extra docs`).

Develop new modules with test-driven development (TDD).

Test layout:
- `tests/graph/`: tests for `Graph`, adapters, `HammingGraph`, `SequenceGraph`, and `ScalarNetwork`
- `tests/topology/`: tests for `ScalarNetworkTopology`, `MergeTree`, mapping, spans, persistence, lineages, and simplification
- `tests/biology/`: tests for alphabets, `FitnessLandscape`, and `FitnessLandscapeTopology`
- `tests/plotting/`: tests for plotting specifications and inspectable data tables
- `tests/examples/`: tests for datasets and example workflows

Test expectations:
- Use small graphs with normalized [0, 1] scalar attributes where practical.
- Keep tests deterministic; avoid randomness unless seeded.
- Assert structural or tabular results, not incidental Rustworkx traversal order.
- Never assert Rustworkx integer node indices as public vertex identities.
- Keep chart tests focused on data and Vega-Lite specifications.
- Maintain passing state with `uv run pytest -q`.
