# Changelog

All notable changes to this public package are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Zenodo DOI badge and concept DOI in README and CITATION.cff.

## [0.5.1] - 2026-11-26

Release metadata only. No change to the package's behaviour.

### Changed

- The author's affiliation is spelled in full (Okinawa Institute of Science and Technology Graduate University) in the citation and Zenodo metadata.
- `CLAUDE.md` is added next to the existing guidance for coding agents.
- Branch naming in `AGENTS.md` is tool-neutral.

### Removed

- The dependabot configuration and the documentation-site workflow are not part of the public tree.

## [0.5.0] - 2026-11-19

First public release. Versions before 0.5.0 were developed privately and are not part of this changelog.

### Added

- `ScalarNetwork`, `MergeTree` and `ScalarNetworkTopology` can be written to a folder and read back (`write` / `read`), including one bootstrap persistence band on a workspace.
- Bootstrap persistence band: `bootstrap_persistence_band`, `BottleneckToNominal`, and `BootstrapPersistenceBand`, applied by passing `band.threshold` to `MergeTree.simplify`.
- A continuous integration workflow for Linux, macOS and Windows (Python 3.11–3.13) with a coverage report and no coverage gate is included in `.github/workflows/ci.yml`. It is not enabled on the public repository yet.
- Zenodo deposit metadata (`.zenodo.json`).
- Declared public API: `compute_branch_local_swarm` and `select_tree_root_node_index` in `topographer.plotting`; `build_join_tree`, `build_split_tree`, `build_tree_location_tables` and `reduce_merge_tree` in `topographer.topology`.

### Changed

- Development status is Beta. The supported runtimes are Python 3.11, 3.12 and 3.13 on Linux, macOS and Windows.
- The public API is the merge-tree workflow: scalar networks, join and split trees, component persistence, simplification, diagram distances, fitness landscapes and sequence annotation.

[Unreleased]: https://github.com/oist/topographer/compare/v0.5.1...HEAD
[0.5.1]: https://github.com/oist/topographer/releases/tag/v0.5.1
[0.5.0]: https://github.com/oist/topographer/releases/tag/v0.5.0
