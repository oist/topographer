# Topographer Context

Topographer is a domain-independent toolkit for deriving and inspecting
merge-tree summaries of scalar-valued networks. Its release-defining workflow
is scalar network to join or split tree to branch spans or persistence to
persistence-based simplification. This glossary defines the project language
used by its documentation, APIs, and contributors.

## Language

### Input data

**Raw data**:
The source measurement records and declared relationships from which a graph
and its scalar schema are explicitly constructed. Topographer does not infer
adjacency or scalar meaning from raw data.
_Avoid_: scalar network (before its graph topology and scalar schema are declared)

**Graph**:
A directly usable Topographer graph representing vertices, edges, and optional
arbitrary attributes independently of whether a scalar has been declared. It
owns a private Rustworkx graph by composition and provides a stable public
identity, validation, attribute, transformation, and defensive-export interface
over that implementation. It does not inherit Rustworkx's unrestricted mutation
interface. Generic graphs have explicit `from_rustworkx(...)`,
`from_networkx(...)`, `from_igraph(...)`, and `from_tables(...)` entry points;
each copies and normalizes its input into the same `Graph` contract rather than
retaining a live link to the source object. NetworkX and igraph remain optional
interoperability dependencies. `HammingGraph` and `SequenceGraph` are constrained forms of the
same graph interface rather than inputs requiring conversion before use as a
scalar network.
_Avoid_: network (when the scalar schema is not part of the meaning)

**Graph schema**:
The declared labels and roles by which a graph identifies its vertices, edges,
and attributes. The schema retains caller-declared identity labels such as
`variant_id`, `sample`, or `sequence`; adapters do not replace them with an
invented generic label. Identity fields are structural graph state and are not
ordinary attributes or candidate scalar attributes, even when defensive exports
include them so that the exported payloads are self-describing.

**Vertex identity**:
The stable public value identifying exactly one graph vertex. It is structural
graph state, distinct from the vertex's ordinary attributes and from a private
Rustworkx vertex index.

**Edge identity**:
The stable public value identifying exactly one graph edge. Every edge has an
edge identity. A caller may declare it through an edge-identity field. For a
simple graph without a declared edge identity, Topographer deterministically
uses the ordered endpoint pair. A multigraph requires declared edge identities
because an endpoint pair cannot distinguish parallel edges. Distinct parallel
edges have distinct edge identities even when they share the same endpoint
pair. Edge identities remain stable through attribute updates, defensive
exports, scalar-network snapshots, and graph transformations that retain those
edges.

**Adjacency matrix**:
A square matrix $A$ indexed by graph vertices, where $A_{ij}$ records whether
an edge connects vertices $i$ and $j$ (or, when explicitly stated, an edge
weight). It is one inspectable representation of declared graph topology in
raw data; it does not itself declare a scalar function.

**Neighborhood operator**:
A named, deterministic rule that uses declared vertex attributes to return the
neighbors of a vertex. It generates an adjacency relation that must be made
explicit before constructing a scalar network; it may be symmetric or directed.

**Scalar network**:
An undirected connected graph together with exactly one explicitly selected,
finite numeric scalar attribute on its vertices and at most one explicitly
selected finite numeric scalar attribute on its edges. The vertex scalar is
never inferred, even when the input graph exposes only one candidate, and
construction rejects an absent selection or an attribute missing from any
vertex. A scalar is selected by the string name of a graph attribute. Its values
must be finite real numbers; integers are accepted, while booleans, complex
numbers, nulls, infinities, NaNs, and nonnumeric values are rejected.
Construction also rejects directed or disconnected graphs before any
topology is requested and reports disconnected components. Direction belongs to
derived rooted trees, not the input network.
It is represented by `ScalarNetwork`, which owns a private validated graph
snapshot and exposes only defensive copies. A different scalar or changed graph
is represented by another scalar-network instance. Later attribute updates or
graph transformations of a source `Graph`, `HammingGraph`, or `SequenceGraph`
never propagate into an already constructed scalar network; using newly added
data requires a new scalar-network instance.
_Avoid_: graph (when the declared scalar schema matters)

**Attribute update**:
An operation that adds vertex or edge attributes without changing vertex
identity, edge identity, or adjacency. The caller explicitly names every
attribute column to add. Existing attribute names are rejected rather than
replaced, and the update adds all requested attributes or none of them. It is
distinct from a biological mutation and from a graph transformation.
_Avoid_: mutation

**Graph transformation**:
An operation that produces a new graph with changed vertices or adjacency while
leaving its input graph unchanged. Attribute preservation is part of the
transformation's declared semantics.
_Avoid_: mutation

**Sequence mutation**:
A biological change from one sequence to another. In a sequence graph, an edge
may represent an allowed sequence mutation according to the declared adjacency.
_Avoid_: mutation (when attribute update or graph transformation is meant)

**Scalar function**:
A function assigning an ordered finite numeric value to every vertex, every
edge, or both in a scalar network.
_Avoid_: scalar attribute (when referring to the mathematical function)

**Injective vertex scalar**:
A finite vertex scalar function whose value is distinct at every vertex. It
avoids plateau level sets but is not required by merge-tree or Reeb-graph
construction, which resolve tied values through the symbolic tie-break. It does
not, by itself, assert the stronger smooth or PL Morse conditions.
_Avoid_: Morse function (unless its critical-point conditions have also been
specified)

**Scalar attribute**:
A named graph-payload field whose values represent a scalar function on
vertices or edges.
_Avoid_: scalar function (when referring to storage or schema)

**Scalar simplicial complex**:
Topographer's compact input representation of an explicit finite simplicial
complex together with declared finite numeric vertex scalar attributes. A
selected scalar is understood as the standard piecewise-linear function
obtained by linear extension over each simplex. It is the input domain for an
exact Reeb graph; a scalar graph alone is not silently promoted to this domain.
_Avoid_: scalar network (when filled simplices are part of the meaning)

**Triangulation**:
A simplicial decomposition of a non-simplicial cell complex, such as a
square-faced hypercube 2-skeleton. It specifies the piecewise-linear domain on
which a scalar field and its level sets are evaluated.
_Avoid_: graph drawing (when referring to topological input)

**Face completion**:
The explicit transformation from a scalar network's 1-skeleton to a
two-dimensional cell complex by selecting cycles as polygonal 2-cells. Its
face-selection policy is part of the input semantics, not an inference made by
Reeb-graph construction.

**Face selector**:
A named, deterministic policy that selects the ordered simple cycles to use as
polygonal 2-cells during face completion. Automatic selectors derive faces
only from a validated declared combinatorial domain; an arbitrary scalar
network supplies its faces explicitly.

**Categorical Hamming domain**:
A complete sequence graph with declared per-position alphabets, one vertex for
every sequence in their Cartesian product, and edges exactly between sequences
that differ at one position. Its validated product structure determines
elementary triangular faces within multi-state positions and square faces from
independent changes at two positions.

**Triangulation strategy**:
The deterministic rule that converts each selected polygonal 2-cell into
triangles. The initial strategy uses a fan rooted at the lowest stable vertex
identifier and records every derived diagonal with its source face; it is part
of the scalar-complex provenance.

**Vertex**:
An element of a graph to which zero or more scalar attributes may belong.
_Avoid_: node (except when directly referring to Rustworkx or a compatibility API)

### Component topology

**Filtration**:
A nested family of graphs obtained by thresholding a scalar function. A merge
tree is constructed by tracking connected components through the vertex
sublevel or superlevel filtration of a scalar network.

**Sweep**:
The ordered traversal of a filtration used to construct a merge tree.
_Avoid_: filtration (when referring to the construction process)

**Connected component**:
A maximal connected subgraph. During a sweep, an active component is a
connected component of the graph currently present at that threshold.

**Weak adjacency**:
Connectivity that treats the endpoints of a directed input edge as adjacent
regardless of its orientation. Merge-tree construction uses weak adjacency;
the directed input graph remains available for direction-sensitive analyses.

**Sweep-tree direction**:
The direction of constructed merge-tree edges toward a root. It is distinct
from the orientation of original input edges.

**Ancestor**:
A merge-tree vertex nearer to the root than another vertex.

**Descendant**:
A merge-tree vertex farther from the root than another vertex. A parent or
child is an immediate ancestor or descendant, respectively.

**Threshold**:
A chosen numeric cutoff, usually denoted by $\tau$, against which values of a
named vertex or edge measurement are compared. Its purpose must be qualified,
such as a filtration, persistence, simplification, or significance threshold.

**Filtration threshold**:
A threshold selecting a sublevel or superlevel graph in a scalar-network
filtration.

**Level set**:
The points in the geometric realization of a scalar network or simplicial
complex whose scalar-function value equals a chosen threshold. Vertex values
are extended linearly along edges and, when present, affinely across simplices.

**Reeb graph**:
The quotient of a scalar simplicial complex that identifies points in the same
connected component of the same level set. It may contain cycles. When the
input domain is simply connected, the Reeb graph is a contour tree.
_Avoid_: contour tree (unless the domain is simply connected)

**1-Skeleton Reeb graph**:
A Reeb graph computed directly from a 1-skeleton simplicial complex ($k=1$),
so alternative mutational pathways remain separate unless their level-set
components meet in the network. No filled faces are inferred from cycles.

**Reeb workflow**:
An inspectable pipeline from a scalar network to its 1-skeleton Reeb graph, or
from an explicitly supplied scalar simplicial complex to its Reeb graph. Any
optional faces and derived simplices remain auditable domain inputs rather than
structure inferred by Reeb construction.

**Level-set connectivity oracle**:
A correctness-first Reeb construction that intersects linearly interpolated
edges, and any explicitly supplied simplices, at one regular value in each
interval between vertex scalar values; computes the connected components of
each slice; and stitches them through adjacent critical levels. It is the
reference against which an optimized dynamic-connectivity implementation is
validated.

**Contour tree**:
A Reeb graph whose input domain is simply connected, and therefore a tree. It
is not an arbitrary cycle-pruned Reeb graph. It is a documented Reeb-graph
special case rather than a separate Topographer output.
_Avoid_: Reeb graph (when the tree guarantee matters)

**Sublevel graph**:
The graph induced by vertices whose scalar value is at most a threshold.

**Superlevel graph**:
The graph induced by vertices whose scalar value is at least a threshold.

**Persistence threshold**:
A threshold for deciding which persistence features are retained after a
persistence diagram has been constructed.

**Simplification threshold**:
A threshold used as the criterion for simplifying a graph or merge tree.

**Merge tree**:
A tree or forest recording the history of connected components across a scalar
filtration. Its construction is parameterized by a sweep direction, while its
stored directed edges point structurally from each root toward its leaves.

**Join tree**:
A merge tree constructed by the increasing sublevel sweep of a vertex scalar
function.
_Avoid_: merge tree (when the sweep direction matters)

**Split tree**:
A merge tree constructed by the decreasing superlevel sweep of a vertex scalar
function.
_Avoid_: merge tree (when the sweep direction matters)

For release documentation, the split tree is the primary illustrative sweep;
the join tree is its supported sublevel dual.

**Monotone-ascent path**:
A path of adjacent vertices whose scalar-function values are nondecreasing in
the direction of traversal. In a fitness landscape, it represents mutational
moves that do not decrease fitness.

**Split-tree accessibility**:
The peak lineage or lineages reachable from a vertex by monotone-ascent paths,
as summarized by a split tree. Distinct lineages first become connected at a
split saddle as the superlevel threshold descends.

**Merge forest**:
The collection of merge trees produced from a disconnected scalar network, one
for each connected component.
_Avoid_: merge tree (when connectedness or root count matters)

**Root**:
The directed source and common ancestor of one merge-tree component: the global
maximum of a join tree or the global minimum of a split tree. It is distinct
from filtration direction, which is increasing for a join tree and decreasing
for a split tree.

**Symbolic tie-break**:
The fixed secondary ordering used for equal scalar values during a sweep. The
canonical order uses stable public vertex identity without changing stored
scalar values or zero-persistence features. In grouped trees it applies to whole
groups: each group is one critical point, equal group activations are ordered
by stable group identity, and only one group can form a multiway saddle. Directed
relationships between equal-valued vertices in merge-tree and Reeb-graph construction are
deterministic bookkeeping and do not establish lineage ancestry. Plateau-aware
critical objects containing multiple vertices, including their use for neutral
networks, are reserved for later work; a group selected by edge grouping is not
such a plateau object.

In ordinary merge trees ties are ordered by vertex index in sweep direction. The
vertex index is the native identity order for a ScalarNetwork, whose graph sorts
identities, and the insertion order for a bare Rustworkx graph. A join sweep
treats the smaller index as symbolically lower; a split sweep processes it first,
so it is symbolically higher. A tree root is therefore the last tied global
extremum swept: the highest-index tied maximum of a join tree and the
highest-index tied minimum of a split tree (`select_extremum_root`). The global
extended H0 pair follows the join sweep: it is born at the first tied global
minimum and dies at the last tied global maximum. This does not make the whole
persistence result one symbolic filtration, because its relative H1 rows come
from the split sweep.

**Vertex scalar**:
A scalar function or scalar attribute defined on graph vertices. Merge-tree
construction uses a vertex scalar.

**Edge scalar**:
A scalar function or scalar attribute defined on graph edges. Scalar networks
may retain and validate one edge scalar, but edge-based topology semantics are
reserved for later work.

**Branch**:
One selected directed edge of a reduced merge tree.
_Avoid_: path

### Tree summaries

**Augmented merge tree**:
A merge tree retaining every input vertex and classifying each as an extremum,
regular vertex, or saddle event.

**Reduced merge tree**:
An augmented merge tree after maximal regular-vertex chains have been
contracted while retaining their vertex chains as edge metadata.

**Reduction**:
The contraction of regular-vertex chains that produces a reduced merge tree
without changing its represented component topology.
_Avoid_: simplification

**Simplification**:
An explicit, rule- or threshold-based transformation that removes or smooths
low-prominence merge-tree structure. `MergeTree.simplify` starts only from an
original merge tree. The original retains each resulting variant under its
immutable simplification specification, and each variant points back to the
original; simplified variants are never simplified again. Graph-level smoothing
and pruning (`smooth_merge_tree`, `prune_leaf_branches`) are uncached operations
on a merge-tree graph such as `MergeTree.graph`: they return a graph and
statistics, never a `MergeTree` or a merge-tree mapping, retain nothing on any
original tree, and can be applied to their own results.
_Avoid_: reduction

**Pruning**:
A simplification that removes selected pendant branches from a non-root leaf to
its first branching saddle.

**Smoothing**:
A simplification that contracts selected merge-tree edges. Each contracted
component is represented by its minimum-scalar member. Ties at the minimum
scalar go to the lowest `vertex_index`, which is the sweep's symbolic minimum
for join trees but not for split trees.

**Critical event**:
An extremum birth or a saddle where active components meet during a merge-tree
sweep.

**Critical vertex**:
A vertex at which a critical event occurs. Use “critical node” only in
programmatic or Rustworkx-specific language.
_Avoid_: critical point (the continuous-setting term)

**Regular vertex**:
An activated vertex that extends exactly one active component during a
merge-tree sweep.
_Avoid_: critical event

**Extremum**:
A component-birth event: a minimum in a join tree or a maximum in a split tree.

**Saddle**:
A component-meeting event: a join saddle in a join tree or a split saddle in a
split tree.

**Multi-saddle**:
A Reeb-graph critical vertex where multiple active components split, merge, or
both simultaneously, represented structurally with diamond markers.

**Branch span**:
The absolute scalar difference across a selected directed branch of a reduced
merge tree. Branch spans describe displayed tree geometry, including surviving
and terminal branches.
_Avoid_: persistence

**Lineage**:
The history of one active connected component through a merge-tree sweep. A
lineage begins at an extremum, may continue through saddles under the elder
rule, and terminates at an ordinary death or the tree root.

**Lineage aggregate**:
A cumulative summary of the input vertices owned by one active component during
a merge-tree sweep. At a saddle, each losing component's aggregate is captured
on its finite persistence pair before the components are combined. The saddle
sequence is then added once to the combined post-merge aggregate carried
forward by the elder component. At the root, this aggregate contains every
input vertex in its connected graph component and is captured on the extended
persistence pair.
_Avoid_: branch span, persistence

**Sequence count**:
The number of input sequence vertices owned by a lineage aggregate. It is the
denominator for proportions defined on that lineage.

**Feature count**:
The number of sequence vertices owned by a lineage aggregate that satisfy one
named application-level predicate, such as containing a stop codon. It is
reported alongside the sequence count rather than as an ambiguous count.

**Stop-codon sequence count**:
A feature count in which each sequence contributes one when it contains at
least one in-frame stop codon and otherwise contributes zero. A sequence with
multiple stop codons still contributes one.

**Persistence pair**:
A paired birth and death event recording the lifetime of a topological feature across
a filtration. In component persistence ($H_0$), it is an elder-rule pairing of component
birth and saddle merger or global extrema. In cycle persistence ($H_1$), it is the pairing
of an independent cycle born at an original edge with a relative continuation event
(cone triangle) that contracts it.
_Avoid_: branch

**Persistence**:
A quantitative lifetime summary of topological features across an ascending or
relative continuation filtration. Its variant must be stated as ordinary, relative,
extended positive, or extended negative.
_Avoid_: branch span

**Ordinary persistence**:
The finite elder-rule pairs whose non-elder components die at merge saddles
during the original filtration. The surviving essential component is omitted.
_Avoid_: extended persistence

**Extended H0 persistence**:
Ordinary zero-dimensional persistence supplemented by a finite pair from the
essential global extremum to the opposite global extremum through the relative
continuation of the filtration. When computed from a merge tree, it is the
tree-restricted $H_0$ part of extended persistence and contains no information
about cycles in the source scalar network. Its global endpoint is not a
component-meeting death.
_Avoid_: branch span

**Network extended persistence**:
The component-persistence and cycle-persistence products of the
absolute-to-relative filtration of an entire scalar network viewed as a
one-dimensional complex. It includes finite pairings for connectivity events
and independent network cycles without replacing the network by a tree.
Relative $H_1$ rows encode superlevel component connectivity and are not ordinary
graph loops.
_Avoid_: merge-tree extended persistence

**Component persistence**:
The connectivity part of network extended persistence: ordinary sublevel
component pairs, relative superlevel component pairs, and the global extended
component pair. It is represented by join- and split-tree topology and contains
no source-network cycle classes.

**Cycle persistence**:
The extended negative pairings of independent cycles in the original scalar network.
These cycles receive finite scalar pairings via the descending relative continuation
(cone construction) without physically deleting or modifying edges from the original graph.
It is separate from component persistence and may include requested representative
edge chains.
_Avoid_: relative persistence (when referring to original graph cycles)

**Extended H1 persistence**:
The finite scalar pairings assigned to independent cycles by network extended
persistence. They summarize cycle significance across the scalar range without
filling the cycles with faces or replacing the network by a tree.
_Avoid_: ordinary H1 persistence

**Persistence endpoint**:
The vertex or edge simplex responsible for one end of a persistence pair,
identified through stable public graph identities rather than private
Rustworkx indices. Its filtration value may be induced by a vertex scalar.
_Avoid_: birth vertex (when the responsible simplex may be an edge)

**Representative cycle**:
One deterministic $\mathbb{F}_2$ edge chain representing an extended $H_1$
class. It is an inspectable representative selected by the calculation, not a
unique cycle determined by the homology class.
_Avoid_: canonical cycle

**Persistence pair kind**:
The provenance family of an extended-persistence pair: `ordinary`, `relative`,
`extended_positive`, or `extended_negative`. Pair kind is independent of the
pair's nonnegative homology dimension so the vocabulary extends beyond graphs
to higher-dimensional complexes.
_Avoid_: event type

**Elder rule**:
The rule that, at a saddle, retains the component with the more extreme birth:
the lowest minimum in a join tree or the highest maximum in a split tree. Every
other active component receives a finite death at that saddle.

**Birth**:
The extremum where a connected component appears in a merge-tree sweep.

**Death**:
The saddle where a non-elder connected component merges into an elder one.

**Essential component**:
The elder-rule representative that survives to the root of one merge-tree
component and therefore has no finite ordinary-persistence death. Extended H0
persistence pairs it with the opposite global extremum.
_Avoid_: finite ordinary persistence pair

**Topological feature**:
A generic structure of interest in the topology of a scalar network, such as a
connected component, persistence pair, or merge-tree branch. Its specific kind
must be stated where interpretation matters.

**Topological noise**:
Topology treated as insignificant relative to a chosen simpler representation,
such as one with fewer retained components or features.

**Measurement noise**:
Uncertainty in the measured values of a scalar function or other graph
attribute. It may inform an analysis threshold but is not assumed to be present.

**Significance threshold**:
A threshold on a statistical quantity, such as a p-value, q-value, or test
statistic, used to retain or annotate graph elements. A significance comparison
is commonly stated explicitly as, for example, $q \leq \tau$.

**Finite persistence pair**:
A persistence pair with both a birth extremum and a death saddle.

### Grouped trees

**Group**:
A connected component of the vertices joined by the selected edges of an edge
grouping, treated as one atomic unit in a grouped tree and represented by one
vertex of the quotient graph. Its members are linked stepwise by selected edges
and need not be pairwise similar.
_Avoid_: supernode, cluster, component (when a group is meant)

**Activation**:
The scalar height at which a group enters a grouped-tree sweep: its maximum
member value in a descending sweep and its minimum in an ascending sweep. All
members of a group enter together.
_Avoid_: birth (when no new branch is created)

**Group range**:
The closed interval from the minimum to the maximum source scalar over a
group's members. It describes the group's measurements; it is not a branch,
a persistence interval, or evidence of a monotonic path through its members.
_Avoid_: measured range, amplitude (when the interval is meant), branch span

**Group amplitude**:
The width of a group range: its maximum minus its minimum member value. A
constant group has zero amplitude.
_Avoid_: measured amplitude, persistence, branch span

### Architecture

**Topology core**:
The domain-independent graph, scalar-network, filtration, derived-topology, and
analysis functionality of Topographer. Its public interface is concentrated in
the deep `topographer.graph` and `topographer.topology` modules.

**Graph module**:
`topographer.graph`, the public input-model module containing `Graph`,
`HammingGraph`, `SequenceGraph`, `ScalarNetwork`, and the generic `Alphabet`
value type. It hides Rustworkx composition, external graph conversion, identity
bookkeeping, table validation, attributes, and defensive export behind one
cohesive graph interface. Internal implementation files do not create separate
public namespaces.

**Topology module**:
`topographer.topology`, the public derivation module containing
`ScalarNetworkTopology`, concrete merge-tree types and mappings, branch spans,
persistence, lineages, simplification, and eventually the separately validated
Reeb-graph workflow. These objects and computations remain together because
they form one selective topology-derivation workflow.

**Application module**:
A scientific specialization built on the topology core. The current
`topographer.biology` module contains `FitnessLandscape`,
`FitnessLandscapeTopology`, biological alphabet presets, and sequence-aware
annotations in one public namespace; it does not introduce a premature nested
`biology.topology` namespace or duplicate generic topology implementations.

**Example**:
A reproducible application of Topographer to a named mathematical or scientific
subject. An example may own subject-specific analysis and figure-generation
logic while reusing domain-independent topology and application modules.

**Canonical example analysis**:
The shared staged analysis applied to each maintained example after its input
graph has been constructed. It retains the scalar network, component and cycle
persistence, and join and split merge trees as separate scientific products;
presentation composition and artifact generation do not change its scientific
meaning.

**Example input**:
The validated bundle of a ready Topographer graph, a declared vertex scalar
attribute name, and presentation values returned by a subject-specific adapter.
It carries input configuration into canonical example analysis without executing
topology or rendering downstream artifacts.

**Example presentation**:
The presentation-specific metadata carried beside an example input, consisting
of fixed vertex layout coordinates, the natural scalar domain, and the scalar
axis label. It is consumed by downstream rendering rather than read by scientific
topology analysis.

**Analysis checkpoint**:
A persisted, derived analysis result sufficient to reproduce its figures without
repeating scientific computation. It retains the scientific configuration and
the plotting-ready results, and remains distinct from raw measurement data.

**Figure identity**:
The stable name of a figure within an example landscape and tree kind,
independent of its export format.

**Figure output**:
A concrete exported representation of a figure identity in a selected format.
Several outputs can represent the same figure.

**Gallery entry**:
The public presentation of a named example, such as Cube, Hypercube, or
Weinreich 2006. The entry is labeled by subject rather than by content type.

**Gallery**:
The single public collection of curated example presentations. Gallery entries
are named for their subject, rather than labeled as tutorials, examples, demos,
or case studies. Its index is visually light; an entry may be a concise visual
explanation or a detailed reproducible page. Gallery replaces separate public
Tutorials and Case Studies sections. An embedded interactive notebook may be a
peer gallery entry rather than a separate documentation section.

**Landing visual**:
The wordless, above-the-fold animation that introduces Topographer on the Home
page and README hero, depicting paper Figure 1's geometric split-tree
construction intuition. It is generated by `notebooks/manifold-graph-tree.py`
and stored under `docs/figures/manifold-graph-tree/`.

**Documentation data**:
A curated, version-controlled input or result table under `docs/data/`. It is
inspectable through native documentation table components and supports
reproducible documentation regressions.

**Documentation figure**:
A curated, version-controlled static visual under `docs/figures/`. It is used
when a native documentation component or gallery notebook is not the clearer
presentation.

**Disposable output artifact**:
A temporary, local run artifact produced in `output/` by a demo or exploratory
script that is excluded from version control.

### Representation

**Topology**:
The connectivity and event structure of a graph or merge tree.

**Layout**:
Coordinates used to draw a graph or tree. A layout may change visual geometry
but does not change topology. Declaring or changing a scalar network's selected
scalar does not determine graph coordinates; paired graph and scalar-network
views retain the same layout unless a separate layout rule is explicitly
applied.

**Source scalar**:
The original declared scalar measurement used by an algorithm.

**Display representation**:
Formatting, axis rounding, normalization, or visual scaling used to present
data without changing its source scalar, topology, or persistence.

**Tree marker clearance**:
Vertical scalar-domain space reserved so tree markers at the global extrema or
other displayed events are not clipped by the plot boundary. It is a default
rendering invariant for standalone and composed tree views, though a caller may
explicitly override it for a compact figure. The standard clearance is five
percent of the displayed scalar span above and below the data.

**Tree presentation mode**:
The visual role of a tree chart. Standalone baseline, annotation, and metric
views show a subtle structural critical-event layer so branches remain
dominant. Annotation views add requested sequence, count, or other application
markers above it. A tree panel does not change scientific meaning merely
because it is composed with other panels; its requested presentation mode
remains explicit.

**Reeb presentation mode**:
The visual presentation of a Reeb graph using semantic critical event markers
(upward triangles for maxima, downward triangles for minima, and diamonds for
split, join, and multi-saddles) and interactive hover tooltips, omitting static
text label boxes to avoid visual crowding across cycle-rich graphs.

**Hamming MDS layout**:
The horizontal coordinate layout for Reeb graphs over sequence spaces obtained
via 1D Classical Multidimensional Scaling on pairwise Hamming distance
matrices, ensuring symmetric separation of mutational cycles and consistent
vertex placement across augmented and reduced graphs.

**Baseline critical-event marker**:
A subtle filled black marker: an upward triangle for a maximum, a downward
triangle for a minimum, and a diamond for either split or join saddle. Extremum
triangles are smaller than saddle marks so they do not dominate the tree
branches. Grouped trees omit these markers.

**Tree summary view**:
A standalone or composed display of a branch-level quantitative summary.
Branch span and persistence remain distinct summaries, but use the same
presentation family for composition, metric encodings, and tooltips.

**Scalar-aligned panel**:
A plot panel with an explicitly fixed scalar domain, tick values, height, and
font size so it can be composed in any order with other panels using the same
scalar-axis presentation. Panel composition carries no additional scientific
meaning.
_Avoid_: linked tree summary

**Metric-colour correspondence**:
One selected branch metric uses one shared colour scale on every tree branch
and summary mark to which that metric is unambiguously attached. A mark with no
such correspondence is not assigned a fabricated tree colour.

**Metric-inapplicable endpoint**:
A topological extremum for which a selected branch metric is not meaningful,
such as a one-sequence minimum for an entropy metric. It remains visible in a
summary as a neutral filled black mark, distinguishing undefined from a
measured zero.

**Tree annotation layer**:
A marker layer over a baseline tree that locates application-level observations
such as regular sequence vertices, stop codons, or the wild type. Layers are
ordered by visual priority; the wild-type marker is topmost and must remain
visible when it overlaps a stop-codon marker.

**Marker-free analysis tree**:
A tree presentation for scientific analysis that omits structural maximum,
minimum, and saddle markers to make room for biological annotation layers.
It is the standard analysis presentation; compact teaching demonstrations may
retain structural-event markers.

**Tree layer order**:
The fixed rendering order for a tree chart: branches first; enabled baseline
critical-event markers next; requested sequence, count, and stop-codon
annotations next; and the wild-type annotation last. A presentation mode may
omit a layer but does not reorder the layers it retains.

**Payload**:
The complete mapping attached by Rustworkx to one vertex or edge.

**Attribute**:
A named field in a vertex or edge payload. For example, a vertex payload may
contain `fitness` and `sequence`; an edge payload may contain
`hamming_distance` and `q_value`.

**Annotation**:
An attribute used as descriptive or application metadata rather than as an
input to a topology algorithm. For example, `sequence` is ordinarily an
annotation, while `q_value` becomes a filtering criterion only when explicitly
compared with a significance threshold.

### Derived topology and analyses

**Scalar-network topology**:
A domain-independent derivation workspace initialized from one validated scalar
network whose vertex scalar is already declared. Initialization captures a
fixed snapshot but computes no topological objects. Callers selectively populate
parallel join-tree and split-tree results through explicit computations; later
mutation of the source graph does not change completed results. Constructing a
tree also constructs its validated merge-tree mapping, while branch spans,
persistence, and lineage analysis remain separately requested computations on
that tree. Future Reeb graphs remain reserved for their own exact construction
workflow and are not exposed before that workflow is implemented.

**Fitness-landscape topology**:
A biological topology workspace initialized from one validated fitness
landscape. It contains scalar-network topology for the landscape's underlying
network and adds sequence-specific meaning and annotations without duplicating
the generic topology implementation.

**Topological analysis**:
A derived quantitative or relational summary of a selected topological object,
such as branch spans, persistence, or lineage summaries. These analyses belong
to the corresponding join-tree or split-tree result within scalar-network
topology, are computed only when explicitly requested, and must identify the
tree from which they were derived.

**Papkou 2023 analysis**:
The application of fitness-landscape topology and its analyses to the Papkou et
al. dataset.
Papkou-specific raw-data normalization and adjacency construction precede the
generic topology derivation.
_Avoid_: Papkou tree analysis (when referring to the broader study result)

### Sequence and branch annotations

**Horton–Strahler order**:
A rooted-tree branching descriptor, originating in hydrology for river
networks, assigned to a lineage or displayed branch. A lineage begins with order
one; at a merge, its elder continuation acquires the largest incoming order and
increases by one when two or more incoming lineages share that largest order.
The order captured at a lineage's ordinary death, or at the root for an
extended-persistence pair, is a persistence-pair annotation. It is independent
of scalar magnitude and can change after tree simplification.

**Hamming distance**:
The number of differing positions between two equal-length sequences over a
common alphabet. Qualify it as DNA or amino-acid Hamming distance when needed.

**Persistence Hamming distance**:
The Hamming distance between the input sequences at the endpoints of one
persistence pair: an extremum and its death saddle for ordinary persistence,
or the two opposite global extrema for extended persistence. It is a pair-local
annotation, not a lineage aggregate. When both DNA and amino-acid sequences
are available, report the two distances separately.

**Codon**:
A three-nucleotide DNA unit.

**Translation**:
The mapping from the codons of a DNA sequence to its amino-acid sequence,
including Topographer's three distinct stop-codon states.

**Aligned sequence set**:
A sequence set whose positions are comparable across every sequence. Equal
length is necessary but does not by itself establish biological alignment.

**DNA sequence**:
The nucleotide representation of a sequence.

**Binary genotype encoding**:
An ordered string of zeroes and ones representing the states of a fixed set of
mutations. Documentation must define the mutation order and the meaning of zero
and one before using the encoding as a label.

**Amino-acid sequence**:
The translated representation of a DNA sequence. Use “AA sequence” only in
compact programmatic labels.

**Sequence set**:
Aligned sequences considered together for an analysis. Repeated observations
are retained because they affect observed symbol frequencies and entropy; an
implementation may represent a sequence set as a tuple or list.

**Sequence entropy**:
The sum of per-site Shannon entropies, in bits, across an aligned sequence set.
A lineage aggregate represents its sequence ensemble with cumulative per-site
symbol counts; these counts are combined when lineages merge. Entropy may be
reported as that total or as the per-site vector. When both DNA and amino-acid
sequences are available, report separate entropy summaries for each. Standard
results use total entropy; per-site entropy is an opt-in detailed result rather
than a default plot encoding. It is an annotation, not a topological invariant.

**Synonymous mutation**:
A DNA mutation that changes a nucleotide sequence without changing its
translated amino-acid sequence.

**Synonymous endpoints**:
A branch annotation stating that its two endpoint DNA sequences translate to
the same amino-acid sequence. It does not establish that every intervening
mutation was synonymous.

**Stop codon**:
One of the DNA triplets `TAA`, `TAG`, or `TGA`, which signals translation
termination. Topographer encodes these distinct stop states as `%`, `^`, and
`&`, respectively.

**Stop-codon sequence**:
An amino-acid sequence containing at least one encoded stop-codon state. It is
a binary biological annotation: each qualifying sequence is represented once,
regardless of how many stop codons it contains. It is the analysis unit for
the current stop-codon overlays and fitness distributions.

**Has-stop-codon annotation**:
The Boolean vertex annotation stating whether an amino-acid sequence contains
one or more encoded stop states. It is computed from the biological vertex
table before scalar-network or tree construction, then propagated as vertex
payload; it is not a property calculated by the merge tree.

**Stop-codon count**:
Avoid this unqualified term. Use **lineage stop-codon sequence count** for a
cumulative filtration-component aggregate or **branch stop-codon member count**
for regular vertices located on one displayed branch. In either case, a
sequence with one or several stop codons contributes one.

**Critical stop-codon summary**:
A topological-summary annotation for a critical vertex with a
has-stop-codon annotation. Persistence summaries annotate stop-bearing maxima
or minima only. Branch-span summaries annotate those extrema and stop-bearing
saddles. The same sequence may occur in both summaries at different plotted
coordinates.

**Fitness marginal**:
A distribution of a chosen sequence cohort over the scalar fitness dimension.
For a tree display it shares the tree's vertical fitness axis. Its relative
density normalizes each displayed cohort independently and must state each
cohort's total sequence count. It is the standard stop-codon tree
presentation; the cohort totals preserve prevalence information.

**Canonical amino-acid alphabet**:
The 20 canonical one-letter amino-acid symbols, with no stop or alignment-gap
state. It is exposed as an immutable named alphabet preset.

**Amino-acid-with-stop alphabet**:
The 20 canonical amino acids together with the generic `*` stop symbol: 21
symbols in total.

**Codon-resolved amino-acid alphabet**:
Topographer's 23-symbol convention containing the 20 canonical amino acids and
three distinct one-character stop symbols: `TAA` maps to `%`, `TAG` maps to
`^`, and `TGA` maps to `&`. This is a generic package convention rather than a
Papkou-specific encoding. The generic `*` symbol represents a stop whose codon
identity is unresolved and is not part of this 23-symbol alphabet; a collection
containing both it and all three resolved stops would contain 24 symbols.

**Fitness landscape**:
A biological specialization of scalar network that adds one declared sequence
attribute and alphabet. Its vertices represent sequences, its edges represent allowed
mutational relationships, and its already-declared vertex scalar supplies the
landscape value; no separate `fitness_attr` alias exists. It can be constructed
directly from a graph or tabular sequence-scalar dataset. Construction from a
`HammingGraph` or `SequenceGraph` reuses that graph's validated sequence identity
and declared alphabet, while still requiring an explicit `scalar_attr`.
Construction from a generic `Graph` additionally requires explicit
`sequence_attr` and `alphabet` arguments. Raw Rustworkx, NetworkX, or igraph
objects first enter through the corresponding `Graph.from_*(...)` interface; a
fitness-landscape constructor does not maintain parallel backend-specific input
paths. No constructor ever infers the scalar selection. A
`from_scalar_network` adapter may extract an existing scalar network's protected
graph snapshot and scalar attribute, but it adds no distinct domain state and
can succeed only when the graph already carries the requested sequence field.
_Avoid_: scalar network (when the biological meaning matters)

**Asexual one-mutation setting**:
An interpretation of a fitness landscape in which reproduction is asexual and
an edge represents a single mutational change; sexual recombination is outside
the modeled neighborhood. A split tree then summarizes connectivity among
high-fitness genotypes as the fitness threshold descends.
_Avoid_: evolutionary simulation (Topographer summarizes a declared network; it
does not simulate population dynamics)

**Sequence space**:
The set of possible sequences over a specified alphabet.

**Hamming graph**:
An undirected graph induced by the equal-length sequences present in one vertex
table represented as a `polars.DataFrame`, with an edge between distinct
sequences whose Hamming distance is no
greater than a positive `hamming_radius`, which defaults to one. Radius one
recovers the ordinary observed Hamming graph. The supplied sequences define the complete
vertex domain: construction does not generate absent sequences, but every pair
of supplied sequences within the declared radius is necessarily adjacent.
Adjacency does not require an edge-attribute value; edge annotations are
optional data attached after the Hamming edges exist. A Hamming graph may contain several connected
components; connectivity becomes mandatory only when a final selected graph is
passed to `ScalarNetwork`. Construction requires a declared ordered
alphabet of unique one-character symbols and rejects every sequence containing
an undeclared symbol; the alphabet is never inferred. The unique sequence is the
canonical vertex identity: `sequence_col` identifies the constructor input used
to create that identity but is not copied into the ordinary vertex-attribute
collection. A separate vertex-ID column is neither required nor stored as a
second identity. Public Hamming-graph operations identify vertices by these
sequence values; Rustworkx integer node indices remain an implementation detail.
`neighbors(sequence)` requires that the supplied sequence identify an existing
vertex and otherwise fails explicitly; it returns only adjacent sequences already
present in the graph and never generates absent sequences from the alphabet.
Hamming-graph vertices are canonically ordered lexicographically according to
the declared alphabet rather than by input-table row order. Public results such
as `neighbors(sequence)` follow this canonical order, so equivalent reordered
tables produce the same graph ordering and deterministic Rustworkx bookkeeping.
The `.graph` property returns a defensive Rustworkx copy whose node payloads
are self-describing dictionaries containing `"sequence"` as the representation
of the canonical identity together with every selected vertex attribute. This
exported `"sequence"` entry does not create a second internal identity. Callers
may use node indices within that exported copy through Rustworkx, but those
indices are never accepted by the stable `HammingGraph` interface, and changes
to the copy do not propagate back to the Hamming graph.
Every edge payload in the exported Rustworkx copy is likewise a dictionary of
its selected edge attributes, or an empty dictionary when no edge annotations
were selected. Source and target identities remain encoded by Rustworkx
adjacency and are not duplicated in the edge payload.
Construction preserves vertex-table data only through
explicitly selected attribute columns; those attributes
may use arbitrary data types and unrelated columns remain outside the graph.
The constructor names this selection `vertex_attribute_cols`; it does not call
the columns scalars before a scalar network selects one.
Scalar columns are optional because adjacency-only graph analysis requires no
scalar function. Candidate vertex attributes may also be added later with
`add_vertex_attributes(...)` from a vertex table by matching its sequence
column. Every graph vertex must have exactly one matching table row; missing
sequences, extra sequences, duplicate sequences,
nulls fail immediately. One call may select several named columns and adds all
of them atomically or none of them. Attribute values may have arbitrary data
types; scalar type validation belongs to `ScalarNetwork`. This attribute update
occurs only after validating the complete input; a failure leaves the graph
unchanged. Adding a vertex attribute whose name already exists is an error;
`add_vertex_attributes(...)` never replaces data implicitly. A
Hamming graph may likewise receive attributes for its already-derived edges
through `add_edge_attributes(...)`; this operation never creates adjacency and
requires exactly one matching row for every existing edge. Missing edges, extra
or duplicate undirected endpoint pairs, and nulls fail atomically. One call may
select several named columns and adds all of them or none of them. Adding an edge
attribute whose name already exists is an error;
`add_edge_attributes(...)` never replaces data implicitly. A
later scalar-network instance
explicitly selects and binds exactly one available vertex scalar; it never
infers the selection. Duplicate sequences fail with all conflicting rows
reported.

A Hamming graph is a completed mathematical construction, not a workspace for
biological edge selection. Connectivity and other read-only graph analyses remain
valid, but expansions, contractions, edge removals, and subgraph materialization
are not Hamming-graph operations. Graph transformations for biological workflows
begin with a `SequenceGraph`; no public `HammingGraph`-to-`SequenceGraph` adapter
is introduced without a concrete use case. A completed `HammingGraph` may still
be passed directly to `ScalarNetwork` when its adjacency is used unchanged.

**Sequence-neighborhood operator**:
The neighborhood operator on sequence space that returns the neighboring
sequences of a given sequence according to a specified sequence-neighborhood
relation. It may be symmetric or asymmetric.

**Mutational neighborhood**:
The set of sequences returned by a sequence-neighborhood operator for one
sequence.

**Sequence graph**:
A biological graph whose vertices are selected sequences and whose adjacency is
declared by an authoritative edge table. Vertex and edge tables are accepted as
`polars.DataFrame` instances;
other dataframe representations must be converted by the caller. It requires
the same explicit alphabet validation as a Hamming graph.
It preserves edge annotations and does not replace explicit adjacency
with Hamming-derived edges. The current interface requires undirected adjacency;
source and target table columns identify endpoints without implying direction.
Directed biological networks are reserved for later work. For a nontrivial
connected instance, source and target IDs are
subsets of the unique vertex IDs and their union equals the vertex-ID set.
Construction rejects unknown endpoints, isolated variants, self-loops,
duplicate undirected edges, and conflicting duplicate annotations with precise
diagnostics rather than repairing them silently. A sequence graph may contain
several connected components so that graph-level inspection and explicit
preselection remain possible.
Each sequence identifies exactly one vertex; duplicate sequences fail with all
conflicting vertex IDs or rows reported rather than being aggregated. The
explicit variant ID is the sole public identity accepted by sequence-graph
adjacency operations such as `neighbors(vertex_id)`. Such operations never
interpret the same argument as either a variant ID or a sequence; Rustworkx node
indices likewise remain implementation details. The separate explicit
`vertex_id_for_sequence(sequence)` lookup returns the one corresponding variant
ID and fails for an unknown sequence; sequence uniqueness makes this mapping
one-to-one. The variant-ID column must have one consistent sortable Polars data
type; mixed identity types fail rather than being coerced. Vertices are
canonically ordered by variant ID instead of input-table row order, making
public result ordering, exported Rustworkx bookkeeping, and topology
tie-breaking deterministic across equivalent reordered tables. The
vertex table contributes only unique variant identity, sequence, and explicitly
selected attribute columns; unrelated vertex columns are not copied.
Construction likewise preserves only explicitly selected edge attribute
columns. The constructor names these selections `vertex_attribute_cols` and
`edge_attribute_cols`. Selected graph attributes may use arbitrary data types; scalar type
validation belongs to `ScalarNetwork`. Both vertex and edge scalar selections
are optional at this graph
stage. Candidate vertex attributes may also be added later with
`add_vertex_attributes(...)` from a vertex table using explicit variant identity
matching and strict coverage validation. Every graph vertex must have exactly
one matching table row; missing vertices, extra identities, duplicate identities,
and nulls fail immediately. One call may select several
named columns and adds all of them atomically or none of them. This method mutates
the graph, but only after validating the complete input; a failure leaves the
graph unchanged. Adding a vertex attribute whose name already exists is an error;
`add_vertex_attributes(...)` never replaces data implicitly. A
sequence graph may likewise supplement its authoritative existing edges through
`add_edge_attributes(...)`; this operation never creates adjacency and rejects
unknown endpoint pairs and requires exactly one matching row for every existing
edge. Missing edges, extra or duplicate undirected endpoint pairs, and nulls fail
atomically. One call may select several named columns and adds all of them or
none of them. Adding an edge attribute whose name
already exists is an error; `add_edge_attributes(...)` never replaces data
implicitly. A
later scalar-network instance
explicitly selects and binds exactly one available vertex scalar and may bind at
most one available edge scalar; it never infers the selection, and topology
never operates across several scalar choices simultaneously.

The `.graph` property returns a defensive Rustworkx copy. Each node payload is a
self-describing dictionary containing the canonical variant ID, its sequence,
and every selected vertex attribute. Each edge payload contains its selected
edge attributes, or an empty dictionary when none were selected; endpoint
identities remain encoded by adjacency. Rustworkx node indices are valid only
inside the exported copy, are not accepted by `SequenceGraph` methods, and
changes to the copy do not propagate back to the sequence graph.

Scalar-derived orientation of sequence edges, such as Papkou's lower-to-higher
fitness direction for adaptive walks, is reserved for later work and is not part
of the current graph interface. The present undirected graph retains stable
vertex identity, adjacency, and scalar-candidate attributes so that a future
orientation implementation can derive direction without changing this contract.
Merge-tree root orientation remains a property of derived topology, not of the
input sequence graph.

Before constructing a scalar network, callers may select vertex identities,
filter edges, inspect connected components, and create a new
attribute-preserving induced `SequenceGraph`. These are
graph-preprocessing operations: they never mutate the original graph and never
occur inside `ScalarNetwork`. Every graph-changing sequence-graph operation,
including induced or incident subgraphs, filtering, and component selection,
returns a new `SequenceGraph` and leaves its input unchanged. The result
preserves the alphabet, identity schema, sequence data, and every selected
vertex or edge attribute applicable to the retained vertices and edges;
attributes are neither silently dropped nor recomputed. The Papkou 2023 example first removes every
nonfunctional-to-nonfunctional Hamming edge and then explicitly selects the
largest connected component; selecting functional vertices plus their neighbors
and taking an induced subgraph is not equivalent because it can restore excluded
nonfunctional-to-nonfunctional edges. Exactly one final connected graph is passed
to each scalar-network constructor; that constructor rejects it if it is still
disconnected and never selects a component implicitly.

**Incident subgraph**:
Given selected core vertex identities, the new graph containing every edge of
the original graph incident to at least one core vertex and both endpoints of
those retained edges. It preserves core-to-core and core-to-fringe edges but
excludes fringe-to-fringe edges, retains the corresponding attributes, and does
not mutate the original graph. For Papkou 2023, functional variants form the
core and their one-mutation nonfunctional neighbors form the fringe. This is the
concise reusable `SequenceGraph` operation behind functional-core neighborhood
expansion; it is not exposed by `HammingGraph`.
For the Papkou construction, component selection occurs after this expansion,
not on the functional core: one fringe vertex may connect functional vertices
that otherwise lie in separate functional-core components. The example therefore
constructs a `SequenceGraph` directly from the variant table and authoritative
mutation-edge table, constructs the functional induced subgraph for inspection,
expands it against that original sequence graph through
`incident_subgraph(...)`, and only then keeps the largest connected component.
It does not construct or convert a `HammingGraph` along this path.
`largest_connected_component()` returns a new
graph only when the maximum-size component is unique. If several components tie
for maximum size, it fails with their vertex identities rather than applying an
arbitrary ordering; the caller must inspect `connected_components()` and select
one explicitly. `connected_components()` returns deterministic collections of
variant IDs, ordered first by decreasing component size and then by canonical
variant-ID order.

**Vertex chain**:
The ordered construction-provenance sequence of augmented-tree vertices on one
reduced-tree edge, including its endpoints. It records the regular vertices
contracted between displayed critical endpoints; it is neither a raw-graph path
nor the full membership of the active filtration component.

**Merge-tree mapping**:
The mapping from every original vertex represented by one merge tree to either
its displayed reduced-tree branch or its retained critical vertex. It records
branch belonging and criticality without assigning drawing coordinates; a new
mapping is required whenever `MergeTree.simplify` changes the tree state. Its
public normalized views are `branch_vertices` and `critical_vertices`. Public
domain interfaces use "vertex"; "node" is reserved for Rustworkx implementation
details such as node indices.
_Avoid_: location map, tree coordinates

**Branch membership**:
The original sequence vertices assigned to one displayed reduced-tree branch
during its scalar interval. It is the regular-vertex part of a merge-tree
mapping, not a cumulative lineage aggregate: sequences are not repeated on
continuing branches after a merge. A regular vertex joins the first active
component it encounters; multiple graph connections into that same component,
including through cycles, do not change its branch assignment. Every regular
input vertex has exactly one branch membership; critical vertices map once to
the retained critical-vertex table.

**Branch-local swarm**:
A display of individual branch members at their exact scalar positions, spread
horizontally and symmetrically within the branch lane when their scalar values
are locally dense. It visualizes local membership without changing the scalar
coordinate or converting the members into a lineage aggregate.

**Lineage sequence count**:
The cumulative number of sequence vertices owned by an active filtration
component. It is a lineage aggregate and must not be interpreted as the number
of members located on one displayed branch interval.

**Branch member count**:
The number of regular sequence vertices in one branch membership. It is a
local display count, distinct from a lineage sequence count.

**Lineage stop-codon sequence count**:
The cumulative number of stop-codon sequences in a lineage aggregate.

**Branch stop-codon member count**:
The number of stop-codon sequences in one branch membership. It is a local
display count, distinct from a lineage stop-codon sequence count.

**Wild type**:
The designated biological reference sequence used for comparison. Wild-type
status does not imply a merge-tree event or root role.

### Statistical inference and null models

**Scalar permutation null**:
A statistical null model on a scalar network that uniformly permutes the
declared vertex scalar values across the existing vertices while keeping vertex
identities, edge identities, and graph adjacency unchanged. It tests whether
merge-tree structure and branch persistence depend on the spatial arrangement of
the scalar function across the graph rather than the scalar value distribution
and graph topology alone.
_Avoid_: graph randomization (when adjacency is held fixed)

**Degree-preserving rewiring null**:
A statistical null model on a scalar network that randomizes graph adjacency
through connected double-edge swaps (Maslov–Sneppen Markov chain) while
preserving every vertex's degree, identity, and declared scalar value, and
rejecting swaps that create self-loops, parallel edges, or disconnected
components. It tests whether merge-tree structure is driven by specific local
neighborhood adjacency rather than vertex degree and scalar distributions.

**Scalar uncertainty bootstrap**:
A resampling procedure that perturbs each vertex's declared scalar value using
its declared vertex measurement uncertainty attribute (such as a finite
non-negative standard error `se >= 0`) over a fixed graph adjacency to
propagate experimental measurement noise into merge-tree branch persistence and
branch membership stability. Missing, negative, NaN, or infinite uncertainty
values are rejected.
_Avoid_: experimental p-value (which tests single-vertex scalar departure from a
reference value rather than topological persistence stability)

**Branch membership stability**:
The requirement under uncertainty resampling that a reconstructed merge-tree
branch matches an observed branch only when both its critical endpoints and its
assigned branch membership (the set of vertices associated with that branch)
are identical to the original branch.

**Persistence confidence band**:
A scalar persistence threshold $\tau_\alpha$ at significance level $\alpha$
derived from a null model or scalar uncertainty bootstrap, separating
topological features above the diagonal band ($\text{persistence} > \tau_\alpha$)
from features indistinguishable from random arrangement or measurement noise
($\text{persistence} \le \tau_\alpha$). It provides a statistically calibrated
simplification threshold for `MergeTree.simplify(...)` and a diagonal noise band
for persistence diagrams.


