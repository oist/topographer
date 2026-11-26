# ARCHITECTURE.md

## Dependency direction

```text
topographer.graph
        ^
        |
topographer.topology
        ^
        |
topographer.biology
        ^
        |
topographer.plotting and topographer.examples
```

- `topographer.graph` implements concrete graph types (`Graph`, `HammingGraph`, `SequenceGraph`) and `ScalarNetwork` via composition over internal Rustworkx graphs. It imports no topology, biology, plotting, or examples.
- `topographer.topology` imports `topographer.graph` and implements topological objects (`ScalarNetworkTopology`, `MergeTree`, `MergeTreeMapping`, spans, persistence, lineages, and simplification). It never imports biology. Fitness-landscape columns on `compute_topological_summary` are registered by `topographer.biology` when that package is imported.
- `topographer.statistics` imports `topographer.graph` and, for the bootstrap persistence band, `topographer.topology` (tree construction, persistence diagrams, distances and the `BootstrapPersistenceBand` record). `topographer.topology` never imports `topographer.statistics`. `statistics.bootstrap` itself has no topology knowledge.
- `topographer.biology` specializes graph and topology concepts (`Alphabet`, `FitnessLandscape`, `FitnessLandscapeTopology`, sequence annotations).
- `topographer.plotting` renders interactive Altair visualizations from inspectable Polars tables.
- `topographer.examples` provides canonical workflow datasets and figure assemblies.

Cross-package code imports from package interfaces rather than underscore modules.
Non-underscore modules are public import paths when documented; underscore
modules are implementation details. The public path `topographer.statistics.bootstrap` retains that status. A package may
also expose selected implementation names through its own interface.
The topology package also exposes construction and reduction helpers
(`MergeTreeResult`, `build_join_tree`, `build_split_tree`, `reduce_merge_tree`,
`select_extremum_root`), and table builders
(`build_tree_critical_branch_span_table`,
`build_tree_extended_persistence_pair_table`, `build_tree_location_tables`).
Internal collaborators also use `_RootedTree`, `_fold_tree`, `_losing_children`,
`_pair_value`, `_rooted_merge_tree`, and `_rooted_tree` through that interface;
their underscore names indicate that they are not stable public contracts.

## Module split assessment

The private biology sequence module holds only helpers with in-repository
consumers (stop codons, Hamming distance, entropy, vertex-chain parsing and
categorical Hamming faces). Persistence plots share validation,
configuration and rendering helpers; merge-tree sweep and construction share trace and
ordering invariants. Deleting any of these modules would redistribute that
coordination to callers or sibling modules. Preserve their current seams until
an independently useful interface can own the extracted behavior.

## Encapsulation rules

- Rustworkx internal node/edge indices are private implementation details and are never exposed as public vertex or edge identities.
- Vertex and edge identities are structural state, not ordinary attributes or candidate scalars.
- External graph representations (Rustworkx, NetworkX, igraph, tables) enter through explicit `Graph.from_*` adapters and funnel through shared authoritative Graph construction.
- Attribute updates and graph transformations operate from current authoritative state.
- Attribute updates and graph transformations do not propagate across snapshots.
- Public tabular interfaces use Polars.
