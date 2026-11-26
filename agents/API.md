# API.md

Public-library rules for `topographer`.

## Stable import paths

- Export primary core types and functions from `topographer`: `Graph`, `HammingGraph`, `SequenceGraph`, `ScalarNetwork`, `ScalarNetworkTopology`, `MergeTree`, `MergeTreeMapping`, `PersistenceResult`.
- Full domain modules live under `topographer.graph`, `topographer.topology`, and `topographer.biology`.
- No backward compatibility modules or pass-through shims exist for retired module paths (`topographer.network`, `topographer.builders`, `topographer.filtration`, `topographer.persistence`, `topographer.tree`, `topographer.reeb`, `topographer.analysis`).
- Avoid exposing internal helpers. Move reusable behavior into named public modules.

## Graph and Scalar Network Contract

- External graphs enter through explicit adapters: `Graph.from_rustworkx`, `Graph.from_networkx`, `Graph.from_igraph`, `Graph.from_tables`.
- Rustworkx node indices are never public vertex identities; explicit, stable, hashable vertex identities are structural state and cannot be selected as scalar attributes.
- Every edge has stable public identity: simple graphs use deterministic ordered endpoint pairs; multigraphs require declared edge identities.
- Attribute updates require explicit attribute columns, reject replacements, and apply atomically.
- Graph transformations operate on current authoritative state and preserve retained attributes and identities.
- `ScalarNetwork(graph, *, scalar_attr, edge_scalar_attr=None)` binds exactly one vertex scalar attribute to a Topographer graph. Raw backend objects are rejected.
- Graphs and scalar networks snapshot internal state to prevent propagation of subsequent attribute updates.
- Topology reads a network through `ScalarNetwork.graph`, `vertex_ids`, `vertex_id_attr` and `scalar_attr`. It does not use private network attributes. The save and read helpers in `topographer.graph._archive` are private: they are not exported from `topographer.graph`.

## Saving and reading

- `ScalarNetwork.write(path)` and `ScalarNetwork.read(path)` (and `FitnessLandscape`, which adds its `sequence_attr` and alphabet symbols) save a network to a new folder and restore it. `read` returns an instance of the class it is called on; `ScalarNetwork.read` of a landscape folder returns a plain `ScalarNetwork`, `FitnessLandscape.read` of a plain folder raises `ValueError`. `vertex_id_attr` and `edge_id_attr` are read-only properties recording the source graph's identity attribute names.
- Folder layout: `vertices.parquet` (row `i` is the vertex at canonical position `i`; the `vertex_id_attr` column holds the native identities, then every vertex attribute), `edges.parquet` (row `j` is edge `j`: `source_id`, `target_id`, the declared `edge_id_attr` column when there is one, then every edge attribute; generated `edge_id` tuples are regenerated, not stored) and `manifest.json` (written last, sorted keys). Manifest keys: `format` (`"topographer.scalar_network"`), `format_version` (`1`), `topographer_version` (informational), `network_class`, `scalar_attr`, `edge_scalar_attr`, `vertex_id_attr`, `edge_id_attr`, `multigraph`, `vertex_count`, `edge_count`, `vertex_attributes`, `edge_attributes` and, for landscapes, `fitness_landscape`.
- `write` refuses before creating anything (`TypeError`) when identities are not finite `str`/`int`/`float`/`bool` of one exact type, when an attribute is not one of those types (all values of one exact type), or when an attribute is named `source_id` or `target_id`; an existing path raises `FileExistsError` and is not modified. Parents are not created.
- `read` raises `ValueError` for a missing folder, manifest or table, a `format` other than exactly the expected one (the message names the format found), a `format_version` other than exactly `1`, missing columns, or counts that disagree with the manifest. It then rebuilds through `Graph.from_tables` and the normal constructors, so connectedness, finite scalars and unique identities are validated again. Vertex order, edge order, value types, `bootstrap.fingerprint` and `band_network_fingerprint` survive a round trip. No pickles, and no Rustworkx indices are stored as identities.
- `MergeTree.write(path)` and `MergeTree.read(path)` save and restore one tree (format `"topographer.merge_tree"`, version 1). Folder: `nodes.parquet` (row `k` is reduced-tree node `k`; the node payload columns in payload key order, starting with `vertex_index`), `edges.parquet` (row `j` is edge `j`: `source_vertex_index`, `target_vertex_index`, then the edge payload columns), `augmented_nodes.parquet` and `augmented_edges.parquet` (the same for the augmented tree with all `n` vertices), `events.parquet` (the event table as is, in sweep order), `simplifications.parquet` (`threshold` Float64, `vertex_index` Int64: the surviving reduced-tree nodes of each cached simplification, zero rows when there is none) and `manifest.json` (`kind`, `scalar_attr`, `vertex_count`, `tree_node_count`, `simplification_thresholds`). Tables never hold Rustworkx indices: nodes are keyed by `vertex_index` (unique within a tree) and row order rebuilds the same node and edge indices.
- `write` on a simplified tree raises `ValueError` (write `tree.original`); an original tree is written with every simplification cached at that moment. `read` restores the trees, events and simplifications without a sweep or a reduction: simplifications are rebuilt from their surviving nodes through the same private code path as `simplify`, and the vertex mapping is recomputed from the restored tree. After `read`, `simplify(t)` for a stored `t` returns the restored variant and `variant.original is restored`.
- `read` raises `ValueError` for a bad manifest, missing table or column, nulls, a node count other than the manifest's, a tree without `n - 1` edges, exactly one root and root-to-leaf edges, augmented or event `vertex_index` values that do not cover `0..n-1` once, event labels that do not fit `kind`, a simplification node outside the reduced tree or a threshold that is not positive, thresholds that differ from the manifest, or a stored simplification that is not a rooted tree.
- `ScalarNetworkTopology.write(path)` and `ScalarNetworkTopology.read(path)` save and restore a workspace (format `"topographer.workspace"`, version 1): `network/` (exactly a `ScalarNetwork` folder), `trees/join/` and `trees/split/` (exactly `MergeTree` folders, only for trees that were already computed; `write` computes nothing), `band-distances.parquet` (`draw` Int64, `distance` Float64; only with a band) and `manifest.json` (`workspace_class`, `trees`, and `band`: null or `filtration`, `quantile`, `radius`, `n_draws`, `seed` as a decimal string, `se_attr`, `network_fingerprint`, `distance` `"bottleneck_linf_h0_capped_essential"`, `quantile_method` `"linear"`). The nested folders are readable on their own with `ScalarNetwork.read` and `MergeTree.read`. A failed `write` removes the folder it created.
- `read` restores the network as the class attribute `_network_type` (`ScalarNetwork`; `FitnessLandscape` for `FitnessLandscapeTopology`, so `ScalarNetworkTopology.read` of a landscape workspace returns the plain classes, intended: `read` returns the class it is called on and drops the landscape extras; `FitnessLandscapeTopology.read` of a plain workspace raises `ValueError` pointing to `ScalarNetworkTopology.read`), the stored trees as the cached trees (`compute_merge_tree` returns them without construction; kinds that were not stored compute lazily) and the band. It checks for each tree that `kind` matches its folder, `scalar_attr` and the vertex count match the network, and that every node's `scalar` and `vertex_id` match the network at its `vertex_index` (O(n), no sweep). A folder of another format fails on `format` with a message naming the `read` to call.
- `ScalarNetworkTopology.band` is the attached `BootstrapPersistenceBand` or `None`; `set_band(band)` raises `TypeError` for a non-band, `ValueError` when a band is already attached (never replaced) or when `band_network_fingerprint(network, se_attr=band.se_attr)` differs from `band.network_fingerprint`. `read` requires the two convention strings to match exactly, rebuilds the band through its constructor (which re-checks the radius against the quantile of the stored distances) and attaches it through the same check as `set_band`.

## Topology Contract

- `ScalarNetworkTopology(network)` derives `MergeTree` objects lazily and deterministically via `join_tree()` and `split_tree()` (or `compute_merge_tree("sublevel"|"superlevel")`).
- `ScalarNetworkTopology` exposes `join_tree()`, `split_tree()`, `component_persistence()` and `persistence_diagram(...)`. Persistence requires a simple graph.
- `ScalarNetworkTopology` derives extended component persistence via `component_persistence()`, returning an immutable `PersistenceResult` with a `pairs` table. Cycle persistence is not provided.
- `MergeTree` is immutable, with edges pointing from root to leaves. It exposes `spans()`, `persistence()`, `lineages()`, `simplify(...)`, and `mapping`.
- `build_tree_metrics(tree)` returns `TreeMetrics` with sequence-free branch, critical-node, and persistence tables of `branch_vertex_count`, `lineage_vertex_count`, and `horton_strahler_order`. A branch counts its start vertex plus regular vertices and excludes its ending saddle; the root vertex appears only in the root critical-node row.
- `MergeTree.simplify(...)` variants are owned and cached by the original tree.

## Biology Contract

- `topographer.biology` exports alphabets (`Alphabet` and presets), `FitnessLandscape`, `FitnessLandscapeTopology`, the ruggedness measures, `has_stop_codon`, `annotate_stop_codons`, `parse_vertex_chain_json`, `TreeLineageMetrics` and `build_tree_lineage_metrics`. Helpers in `biology/_sequences.py` that are not in `topographer.biology.__all__` stay private.
- Every public export carries a runnable `Examples` docstring section.

## Statistics Contract

- `topographer.statistics.bootstrap` is the chunked parametric bootstrap and has no tree knowledge: `run(network, statistic, *, se_attr="se", n_draws, seed, draws=None, n_jobs=1, progress=None) -> BootstrapChunk`, `combine(chunks, statistic, **finish_options)`, `fingerprint(network, statistic, *, se_attr="se")` (the setup fingerprint stored in chunks; the statistic must be bound), `BootstrapChunk.save(path)` / `BootstrapChunk.load(path)` and the `DrawStatistic` protocol (`prepare`, `measure`, `add`, `finish`, `fingerprint`). Draw `d` uses `SeedSequence(seed, spawn_key=(d,))`; combined chunks equal a single run bit for bit; the chunk `.npz` is versioned and unknown versions are rejected rather than misread. Saved chunks have mode `0o666 & ~umask`; process-pool workers ignore SIGTERM, SIGUSR1 and SIGINT.

- `topographer.statistics` exports `BottleneckToNominal` (a `DrawStatistic`: `BottleneckToNominal(filtration, *, se_attr="se")`, bind with `prepare(network)`; one stats column, no counts) and `bootstrap_persistence_band(network, *, filtration, n_draws, seed, quantile=0.95, se_attr="se", n_jobs=1, progress=None) -> BootstrapPersistenceBand`. `bootstrap.combine(chunks, statistic, quantile=...)` returns the band. The statistic and the function must receive the same `se_attr`. The diagram is the ordinary H0 pairs plus the capped global pair.
- `topographer.topology` exports `BootstrapPersistenceBand` (frozen record: `filtration`, `quantile`, `radius`, `n_draws`, `seed`, `se_attr`, `network_fingerprint`, `distances`, `threshold == 2 * radius`, `from_distances`, `at_quantile`). It is not exported from the top-level package. A peak survives iff its ordinary H0 persistence is greater than `threshold`; apply it with `MergeTree.simplify(band.threshold)`. The band covers scalar perturbations on a fixed graph only and gives neither calibrated peak-level coverage nor a false-discovery rate.

## Data and Plotting APIs

- Prefer Polars tables at package boundaries for tabular input and output.
- Return inspectable data before rendering a visualization.
- Lazy-import optional visualization dependencies so importing `topographer` keeps the core install small.
