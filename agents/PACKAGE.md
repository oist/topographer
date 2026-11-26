# PACKAGE.md

Topographer is a merge-tree toolkit for graphs and networks with scalar functions on vertices, edges, or both. The package is based on Topological Data Analysis (TDA).

## Core contract

- The generic core does not assume biological metadata, a particular scalar name, or a particular graph-construction method.
- The entry boundary is `topographer.graph.Graph` (and constrained subclasses `HammingGraph` and `SequenceGraph`), wrapping an internal Rustworkx graph by composition.
- Positional integer node indices from Rustworkx or igraph are never used as public vertex identities.
- Vertex and edge identities are structural state, not ordinary attributes or candidate scalars.
- Every edge has stable public identity: simple graphs use deterministic ordered endpoint pairs; multigraphs require declared edge identities.
- Attribute updates require explicit columns and apply atomically. Graph transformations operate on current state and preserve retained attributes.
- `ScalarNetwork` accepts a concrete `Graph`, `HammingGraph`, or `SequenceGraph` with an explicit `scalar_attr` (and optional `edge_scalar_attr`).
- `ScalarNetworkTopology` derives concrete immutable `MergeTree` objects (`join` and `split`).
- Tree edges point structurally from root toward leaves (global max to leaves for join trees, global min to leaves for split trees).
- `MergeTreeMapping` assigns every input graph vertex to either its displayed reduced-tree branch or its retained critical vertex.
- `topographer.biology` provides biological alphabets (`Alphabet`), sequence landscape structures (`FitnessLandscape`), and topological annotations (`FitnessLandscapeTopology`), ruggedness measures, stop-codon helpers, and sequence-aware tree lineage metrics (`build_tree_lineage_metrics`, `parse_vertex_chain_json`).

## Style

- Use small functions, human-readable docstrings, and clear comments.
- Use Polars for data frames and Rustworkx for internal graph storage.
- Use Altair for public plotting. IPlotX supplies rooted-tree coordinates only.
- Keep reusable visual settings in immutable plot configuration dataclasses.
- Plot filtering and metadata overlays operate on Polars plot tables and never mutate topology objects.

## Structure

```text
src/topographer/
├── __init__.py
├── graph/
│   ├── __init__.py
│   ├── _core.py
│   ├── _adapters.py
│   ├── _payloads.py
│   ├── _archive.py
│   ├── _hamming.py
│   ├── _sequence.py
│   └── _scalar.py
├── topology/
│   ├── __init__.py
│   ├── _workspace.py
│   ├── _tree_construction.py
│   ├── _tree_reduction.py
│   ├── _merge_tree.py
│   ├── _mapping.py
│   ├── _tree_location.py
│   ├── _persistence.py
│   ├── _network_persistence.py
│   ├── _tree_metrics.py
│   ├── _simplification.py
│   ├── _distances.py
│   ├── _summary.py
│   ├── _band.py
│   ├── _scalar_view.py
│   └── _tree_archive.py
├── biology/
│   ├── __init__.py
│   ├── _alphabet.py
│   ├── _faces.py
│   ├── _fitness_landscape.py
│   ├── _lineage.py
│   └── _sequences.py
├── statistics/
│   ├── __init__.py
│   ├── _multiple_testing.py
│   ├── _standard_errors.py
│   ├── _persistence_band.py
│   └── bootstrap.py
├── plotting/
│   ├── __init__.py
│   ├── _config.py
│   ├── _data.py
│   ├── _layout.py
│   ├── _marginal.py
│   ├── _persistence.py
│   ├── _polar.py
│   ├── _shared.py
│   ├── _swarm.py
│   ├── _tree.py
│   └── _weighted.py
└── examples/
    ├── __init__.py
    ├── __main__.py
    ├── _docs.py
    ├── _presentation.py
    ├── cube/
    └── hamming/
```
