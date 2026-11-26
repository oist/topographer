# Topographer

Merge-tree toolkit for networks carrying a scalar function. This glossary fixes the counting vocabulary shared by the topology, biology and plotting layers.

## Language

### Persistence

**Persistence diagram**:
The H0 birth and death pairs for a chosen sublevel or superlevel filtration,
available from `ScalarNetworkTopology.persistence_diagram`. Values retain their
original units and follow the sweep direction.

**Essential pair**:
The pair representing the connected component that survives the entire sweep.
Its death is either null (NaN, with infinite persistence) or capped.

**Capped**:
An essential pair whose death is placed at the global extremum opposite its birth.

### Tree counting

**Branch**:
One edge of a reduced merge tree: the run of vertices from a critical vertex, in scan direction, up to but excluding the next critical vertex.
_Avoid_: Edge, segment, arc

**Branch vertex count**:
The number of vertices on one branch: its starting critical vertex plus its regular vertices. A leaf branch has count one. The saddle it ends at is not counted.
_Avoid_: Branch member count, regular count, branch size

**Lineage**:
The connected component below a branch, immediately before the branch's next critical event: the branch itself together with every branch that feeds into it.

**Lineage vertex count**:
The number of vertices in a branch's lineage. Equals the branch vertex count plus the lineage vertex counts of its child branches. On a simplified tree, counts follow the tree as given, so the root total can be below the network size.
_Avoid_: Lineage sequence count, cumulative count, subtree size

**Root vertex**:
The final critical vertex in scan direction. It starts no branch, so no branch or lineage count ever includes it; the critical-node record for the root event carries the whole-network total.

**Scan direction**:
The order in which vertices are added when sweeping the scalar function: descending for a split tree, ascending for a join tree. Counts and lineages are always read in scan direction.

### Bootstrap persistence band

**Bootstrap draw**:
One resampled scalar network and the merge tree rebuilt from it.

**Bootstrap chunk**:
The result of running a subset of the bootstrap draws (`bootstrap.run` with `draws=`): summed counts and one stats row per draw, tagged with the setup fingerprint. Chunks of one bootstrap, run in any order or process, combine to the same result as a single run.
_Avoid_: Batch, shard, partial run

**Nominal diagram**:
The H0 persistence diagram, with the essential pair capped, of the input network at its declared scalars, before any resampling. Every bootstrap draw's diagram is compared against it.
_Avoid_: Reference diagram, observed diagram, original diagram

**Per-draw bottleneck distance**:
The bottleneck distance (L-infinity) between one bootstrap draw's H0 diagram and the nominal diagram. The radius is a quantile of these distances over all draws.
_Avoid_: Draw error, draw score

**Radius**:
The bottleneck distance quantile c, at the declared band quantile, between bootstrap persistence diagrams and the nominal diagram. Twice the radius (2c) defines the bootstrapped persistence threshold.
_Avoid_: c (in labels), radius as the threshold

**Bootstrapped persistence threshold**:
The persistence boundary 2c that a peak must exceed to survive band simplification, where c is the radius.
_Avoid_: Cone, radius as the threshold, significance of a peak

**Bootstrap persistence band**:
A diagonal strip of vertical height 2c in the persistence diagram, where c is the radius (Fasy et al. 2014). A peak survives if and only if its ordinary H0 persistence is greater than 2c. Its interpretation is limited to scalar perturbations on a fixed graph, and it does not provide calibrated peak-level coverage or a false-discovery rate.
_Avoid_: Cone, confidence band (unqualified), significance of a peak, p-value

**Band-simplified tree**:
The merge tree formed by direct absorption of terminal branches whose ordinary H0 persistence does not exceed the bootstrapped persistence threshold.

**Workspace**:
Informal name for the single object that holds one scalar network together with everything derived from it: merge trees, their simplified trees and the bootstrap persistence band.
