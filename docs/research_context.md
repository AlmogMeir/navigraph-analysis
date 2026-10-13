# Graph-based behavioural analysis for reconfigurable mazes

### A source document for literature search

**What this is.** A self-contained technical and conceptual summary of (a) a
software contribution — extending **NaviGraph**, the toolchain that converts
video tracking into graph-level behavioural data, so that it supports a
*modular maze whose physical layout and camera registration change between
sessions*; (b) the analysis layer built on its output; and (c) the rodent
foraging experiment currently running through both. It is written to be read
alongside the **NaviGraph methods paper** and an **earlier thesis** that
established the physical maze and the DeepLabCut (DLC) tracking pipeline. The
present work upgrades that earlier pipeline by replacing pixel-space trajectory
analysis with **graph-space analysis**, in which the animal's position is a node
or edge of a known maze graph rather than an (x, y) coordinate.

**A framing note, stated up front.** The maze currently in use is a dual-root
binary tree, and concrete numbers below refer to it. **That topology is an
instantiation, not the subject.** The point of this work is that neither the
tooling nor most of the analysis is committed to it: the graph structure is a
configured, pluggable object, and the setup tooling was specifically rebuilt so
that a maze which is *physically rearranged between sessions* can be re-mapped
in minutes rather than redrawn from scratch. Section 3 says exactly which parts
are structure-agnostic and which are still specialised. Literature searches
should treat "dual-root binary tree" as an example, and weight generality —
reconfigurable environments, arbitrary graph topologies, environment-change
designs — accordingly.

**What it is for.** Finding relevant literature. Sections 8 and 9 state the
claims, constructs and methods that need grounding in prior work, and give
keyword clusters organised by research area. Section 10 proposes analyses the
graph representation makes possible but that have not been run yet; those need
literature too.

Everything in sections 1–7 is implemented and running on real data unless marked
otherwise. Section 10 is prospective. Verified design details are stated as
fact; interpretations are marked as such.

---

## 1. The problem the software work solves

A graph-based analysis of behaviour requires three things to be simultaneously
true for every session:

1. A **graph** describing the environment's topology.
2. A **mapping** from image regions ("tiles") to nodes and edges of that graph.
3. A **homography** registering that session's camera frame to the mapping's
   coordinate space.

If any of the three is wrong or mismatched, position is not merely noisy — it is
assigned to the **wrong discrete node**, which is a categorical error that
propagates silently into every downstream measure. This is qualitatively
different from pixel-space tracking error, where a small registration drift
produces a small coordinate error.

The difficulty in this experiment is that **all three change**. The maze is
modular: segments are physically rearranged between sessions, reward ports are
re-wired, and the apparatus is moved relative to the camera. A pipeline that
assumes one fixed mapping and one fixed calibration cannot express this, and —
worse — will not complain. Most of the software work described here exists to
make the per-session configuration explicit, cheap to produce, and *checkable*.

---

## 2. NaviGraph and the graph-builder abstraction

NaviGraph converts per-frame tracked coordinates into per-frame **graph
locations** — the node or edge whose mapped tile contains the tracked point —
and emits a frame-level table in which position is categorical.

The environment topology is supplied by a **graph builder**, selected by name
from a registry. Registered builders:

| Builder | Produces |
|---|---|
| `binary_tree` | Complete binary tree of configurable height |
| `dual_root_binary_tree` | Two binary trees joined at their roots — *the maze currently in use* |
| `star` | Star / radial-arm topology |
| `fully_connected` | Complete graph |
| `random` | NetworkX random-graph models |
| `file_loader` | **Any graph**, loaded from GraphML / GEXF / GML / adjacency or edge list |

All implement one abstract interface (`build_graph`, plus configuration and
metadata hooks), and new topologies register with a decorator. The `file_loader`
builder in particular means an arbitrary, hand-specified graph is a first-class
option — a radial-arm maze, a honeycomb, an irregular open field discretised into
regions, or a maze whose topology *itself* changes between sessions.

The mapping file records which builder and configuration produced it, so a
mapping is self-describing and a mismatch against the configured graph is
detectable rather than silent.

---

## 3. What is structure-agnostic and what is not

Stated precisely, because it bounds how general the contribution is and should
steer how broadly literature is sought.

**Independent of maze topology** (works on any connected graph):

- Per-frame graph-location assignment.
- Traversal reconstruction's core logic: the augmented state graph, shortest-path
  legality checking, noise-vs-travel discrimination, frame-conserving
  interpolation, validation (§5.3).
- **Path directness** — excess nodes over the graph shortest path.
- **Commitment apex** — the shortest-path-suffix definition (§6.1).
- **Topological distance to goal**, node occupancy, node-level firing-rate maps.
- Traversal-quality metrics and graph-legality violation counts.
- Clock alignment by reward-port coincidence (§5.1) — needs reward ports, not a
  particular topology.
- Most of the latent-state feature frame (§6.6).

**Requires an embedding** (node positions, not a particular topology):

- Turn classification from node-triple geometry.

**Currently specialised to the dual-root tree**, and the honest limit of the
present generality:

- Node-naming conventions (`[L|R][level][index]`) used by several helpers.
- The **virtual-edge rule** — level-4↔5 leaf links have no physical corridor.
  This is a real property of *this* apparatus; a general version would read it
  from the mapping (which tiles are cul-de-sacs) rather than from node names.
- `level_of` / `patch_of`, which need designated roots and patch-root nodes.
- Decision-point importance via child sub-trees — tree-shaped, though it
  generalises naturally to "distinct branch sets leading to distinct goals".

So: the **architecture and tooling are topology-general**; a subset of
**analysis primitives are tree-specialised** and would need modest generalisation
for a different maze. This is a fair description of the current state and a
reasonable thing to say in a thesis.

---

## 4. The setup tooling: mapping and calibrating a maze that moves

This is the part of the work that makes a reconfigurable apparatus tractable. It
spans an interactive mapping GUI and a calibration workflow.

### 4.1 The mapping GUI: four modes, and why "Adjust" is the important one

A PyQt-based dual-view interface (session image on one side, maze schema on the
other) with four modes:

- **Grid mode** — auto-generate tiles from the graph structure, for a regular
  maze that matches a generated layout.
- **Manual mode** — draw tile contours by hand.
- **Adjust mode** — **load an existing mapping and modify it.**
- **Test mode** — verify the mapping by checking where sample points land.

**Adjust mode is the answer to "the maze moved".** Rather than redrawing ~250
tiles per session, the previous session's mapping is loaded and only the tiles
that actually moved are corrected. Its capabilities, and the reason each exists:

| Capability | Why a modular maze needs it |
|---|---|
| Click-select a tile; drag to move | A maze segment shifted between sessions |
| **Shift+drag to rotate**, with angle snapping | A segment was re-seated at an angle — the single most common change, and the one a rectangle-based format cannot express |
| Rubber-band multi-select | A whole arm or patch moved as a unit |
| **Copy / paste tiles keeping their node/edge identity** | One graph element can be covered by *several* tiles, so an irregular or bent corridor is covered by tiling it rather than stretching one oversized rectangle over it |
| Delete tiles | A segment was removed from the configuration |
| **Restore-pristine (non-destructive reset)** | Editing is exploratory; every contour can be returned to the geometry it was loaded with, including after resizes and rotations |

Two consequences worth noting because they changed the data format and the
trust model:

- **Rotation broke the storage assumption.** A rotated tile is no longer an
  axis-aligned rectangle, so it cannot be stored as x/y/width/height. The mapping
  format moved to **arbitrary polygons** (format 3.0), which is what makes
  non-rectilinear and re-seated maze geometry representable at all.
- **Saves are verified.** Because the saved file is the only thing the analysis
  ever sees, and regions are rewritten as contours on the way through it, the
  mapping is **read back after saving and compared element-by-element against
  what was on screen**, reporting any element whose geometry did not survive the
  round trip.

### 4.2 Calibration provenance: binding a mapping to the homography it was drawn on

Tiles are positioned on a session image *after* it has been warped by that
session's calibration, so **a mapping is only meaningful together with the
homography it was drawn against**. The analysis loads its calibration
separately, which creates the opportunity for a silent mismatch that shifts
every node.

The mapping file therefore records the calibration's **path, MD5 and the matrix
itself**, so a mismatch is detectable. A mapping made with no calibration applied
is recorded as such with a warning.

### 4.3 The calibration workflow

Rebuilt around the observation that **only half of a calibration changes between
sessions**:

- **Side-by-side point selection.** Video frame on the left, maze schema on the
  right; clicks alternate video → map → video → map. Both images are scaled to
  matching height with resolution-aware markers; right-click undoes the last
  click or the last pair; coordinates are stored in **original image space, not
  display space** (so window size cannot contaminate the homography); ≥4 pairs
  required. This replaced a sequential "click all source points, then all target
  points" flow that was error-prone to the point of being the main source of bad
  calibrations.
- **Reusable map point sets.** The map side of a calibration is the *same
  picture every session*, so its points are the same every session. A named point
  set (e.g. `Classic16`, 16 labelled points, recorded with the map image and its
  dimensions) is saved once and reused, so a per-session calibration requires
  clicking **only the video side**. This both halves the work and removes
  per-session jitter in the map-side reference, which previously meant every
  session was registered against a slightly different approximation of the same
  points.
- **Overlay preview before saving.** After computing the homography, the video
  frame is warped onto the map and blended (40/60) so alignment is judged
  visually — accept, redo the points, or cancel — instead of discovering a bad
  calibration downstream.
- **IR burn-frame avoidance.** The first ~10 s of video is skipped when grabbing
  the calibration frame, because IR calibration burn frames appear at recording
  start; falls back to frame 0 if the seek fails.
- **Batch, shared and overwrite modes.** `--batch` discovers every session and
  opens the GUI for each, saving a per-session matrix; `--shared` calibrates once
  and copies to all sessions (correct when the rig did not move); `--overwrite`
  forces recalibration. Progress is reported and unusable sessions are skipped
  with a message.
- **Dated per-session calibration files.** Calibration used to land on a single
  `transform_matrix.npy` that the next session overwrote — the mechanism by which
  a mapping and its analysis ended up on different homographies. Each session's
  matrix is now addressable by date, with no fallback, because guessing a date
  risks writing one session's calibration under another's name.

### 4.4 A correctness fix worth recording

The graph-location and map-location plugins each auto-detected "bodyparts" from
column names, and were detecting **columns created by each other** (e.g. a
`nose_map_x` column produced by one being read as a bodypart `nose_map` by the
other), applying the calibration **twice** and producing garbage coordinates.
Derived columns are now excluded from bodypart detection. This is the class of
bug that a discrete-position pipeline hides well: doubly-transformed coordinates
still land on *some* node.

### 4.5 Session selection and output addressing

- A run **names the session it wants** by date (with subject disambiguation)
  instead of analysing whichever session folder happened to sit in the
  experiment directory. Ambiguous or missing dates stop the run and report what
  is available. Dates are accepted in any common ordering/separator.
- Analysis outputs, previously written only to a timestamped folder not
  addressable by session, are **mirrored to `<root>/<subject>/<DD_MM_YYYY>/`**,
  which is where downstream analysis looks them up.
- A standalone QC tool (`mapping_qc.py`) validates a session's mapping *and*
  calibration against BPOD ground truth: visually, by warping a video frame into
  mapping space and drawing the node polygons on it; and numerically, by asking
  for every reward event whether the tracked position at that moment maps to the
  port BPOD says was rewarded. An `--all-matrices` mode scores every calibration
  on disk, which identifies sessions using the wrong one. A companion tool
  renders every session's mapping over its own session image for bulk review.

---

## 5. Data-quality analyses (prerequisites, and results in their own right)

### 5.1 Video/behaviour clock alignment by reward-on-port fraction

The frame-to-trial table is built by cumulatively summing trial durations from
the behavioural control system, which assumes video frame 0 is the start of
trial 0. When a camera starts late this fails silently, and downstream
corrections then inject phantom port visits, shortening measured paths *below
the graph shortest path* and inflating the direct/indirect ratio.

The obvious diagnostic — comparing frames claimed against frames held ("frame
budget") — was tested across 21 sessions of one animal and found **nearly
orthogonal to actual alignment**:

| Session | Frame-budget overshoot | Rewards on the paying port | Truth |
|---|---|---|---|
| 26/05 | +7616 (190 s) | 100.0% | fine — camera stopped early |
| 24/05 | −5007 | 99.3% | fine |
| 14/05 | 0 | 76.0% | poor |
| 01/06 | +5814 | 13.3% | broken — camera started 145 s late |

A budget test would have rejected a perfect session and passed a poor one. The
implemented test instead **measures alignment against behaviour**: the fraction
of reward events that find the animal on the port that just paid it (counting the
port's parent node as "at the port", since leaf/parent flicker is pervasive).
This separates 13% from ≥76% cleanly. Below 60%, a constant camera-start offset
is searched for and applied, preferring the offset that also makes the two clocks
end together when it lies inside the broad likelihood plateau. On the broken
session this recovered alignment from 13.3% to 98.1%, retaining 263 of 269
trials. Sessions that cannot be aligned raise rather than being analysed.

### 5.2 Two systematic location corrections

- **Leaf/parent zigzag smoothing.** A port tile and its parent are adjacent, so a
  nearly stationary head flickers between them within one or two frames; each
  flicker reads downstream as two spurious node visits and a round trip that
  never happened. Zigzag counts were measured across gap thresholds and collapse
  past one frame, so only single-frame excursions are smoothed.
- **Reward-period correction.** The control system knows which port paid and
  when, so for the ~0.25 s consumption window the animal is at that port by
  definition. This is the one place an external ground truth overrides tracking,
  and it is sound only after clock alignment has passed.

### 5.3 Traversal reconstruction

Converting per-frame graph locations into a *sequence of visits* is where
tracking noise becomes spurious behaviour. The reconstructor:

1. Builds an **augmented state graph** in which edges become first-class
   intermediate states, so "is this a legal one-step move?" is one adjacency
   query over nodes *and* edges.
2. **Distinguishes noise from travel.** Short runs unreachable from their
   neighbours are absorbed (three rules: forward, sandwich, long-jump); but two
   *long* runs separated by a large graph distance are a genuine traversal with a
   mid-path dropout and are interpolated instead.
3. **Conserves the frame budget** — inferred states take real frames from
   neighbouring runs, so durations stay meaningful.
4. **Rejects impossible states** — an "edge" whose endpoints are not adjacent is
   treated as an unknown frame, not a location.
5. **Validates and reports** — every consecutive pair is proven graph-legal, and
   counters record what each repair rule did, so a heavily repaired session is
   visible rather than silently different.

---

## 6. The experiment and the behavioural analyses

### 6.0 Apparatus, task, subjects

**Apparatus (current instantiation).** A physical binary tree with two roots:
126 nodes, 125 edges, depth 0–5. Two roots joined to each other; 4 level-1 nodes;
64 leaves. Removing the two roots partitions the maze into **four sub-trees of 31
nodes each** — the **patches** — with the root-root edge as the **corridor**.
Only **8 of the 64 leaves carry reward ports**, so the maze affords a large space
of legal-but-wrong paths. Leaf↔parent links are virtual (cul-de-sacs entered
directly from the parent); all other edges are corridors with measurable transit
time.

**Task** (verified from the control-system exports). The 8 ports form **4 fixed
pairs**, one pair per patch. **Blocks** make one pair active at a time, so patch
switching is imposed by the task. Reward **depletes with repeated sampling of the
same port** and is replenished by switching to its partner. Over one session (298
reward events):

| Previous port | Mean volume | Median | P(no reward) |
|---|---|---|---|
| Different port (alternation) | 0.017 | 0.030 | 43% |
| Same port (repeat) | 0.003 | 0.000 | 77% |

Volumes are quantised (0.03 / 0.015 / 0) and a per-event depletion counter is
logged. Licks are recorded per event, giving a consumption readout independent of
position.

*Interpretation:* a **patch-foraging task with local depletion**, whose optimal
within-patch policy is alternation and whose between-patch policy is set by block
structure. In the session examined the animal repeated (184) more often than it
alternated (114) — substantially sub-optimal, making the learning trajectory
itself an object of study.

**Subjects and recording.** Mice (`NPC1`, `NPC2`, `NPC4`), ~20 sessions each,
2025-08 → 2026-06. Overhead video at 40 fps. DeepLabCut pose with bodyparts
`headstage`, `nose`, `ear_L`, `ear_R`; `headstage` drives graph location with
automatic fallback to `nose`. Neuropixels at 30 kHz, spike-sorted, on a subset of
sessions (the neural layer is optional). BPOD supplies trial structure, reward
times and volumes, port identities, block boundaries and licks.

**The port map is a per-session parameter.** Ports were re-wired and the maze
rebuilt for short windows, so *port number → maze leaf* is **date-ranged per
subject**: NPC1 has four periods (including a change-of-maze window in which one
port moved, and a port swap taking effect partway through it); NPC4 has three
(including a window in which three ports moved, after which the previous layout
was restored). An analysis assuming a fixed port map silently mislabels every
trial in those windows — which are exactly the most interesting sessions. This
is the behavioural-analysis counterpart of the reconfigurability that §4 handles
at the tooling level.

### 6.1 Path directness and change of mind

- **Indirect-path detection**: port-to-port segments exceeding the graph
  shortest path by ≥ k nodes.
- **Commitment apex**: the *earliest node visit from which the remainder of the
  trajectory is exactly the shortest path to the destination port* — a
  graph-defined, parameter-free moment of commitment, marking where a wandering
  trajectory becomes a direct approach. Used as a neural alignment event and as
  a weak label for state inference.
- Associated geometry: path excess, distance-to-goal trajectories, and
  in-patch vs out-of-patch classification of each detour (did it stay inside one
  patch sub-tree or cross the corridor).

### 6.2 Turn classification from graph geometry

Each node visit with a predecessor and successor is classified **LEFT / RIGHT /
STRAIGHT / UTURN** from node-triple geometry (|angle| < 20° straight, > 160°
U-turn; handedness from the cross-product sign). The walk is taken through the
*graph*, so turns are defined continuously across trial and reward boundaries —
the path is physically continuous even where the task's trial structure is not.
Two further topological labels:

- **Inbound / outbound / neutral** — whether depth-from-root decreases, increases
  or is preserved.
- **Toward- / away-from-reward** — relative to currently active ports.

### 6.3 Decision points, including context-dependent ones

A node is an **important decision point** when at least two of its child
sub-trees each contain a *currently active* port. Because port activity changes
with block and with the date-ranged port map, importance is
**context-dependent**: the same junction is a real decision in one block and a
forced corridor in another. The **"sometimes-important"** node set is a
within-subject control holding physical location constant while varying decision
relevance.

Per visit: dwell time; and speed decomposed along two axes — **direction**
(entering vs exiting) and **scope** (strictly the approach/departure edge, vs
including time inside the node up to/from the dwell-window midpoint). This
separates *"arrives slowly"* from *"slows only after committing"*, which a
symmetric window conflates. Paired within-visit and across-class tests.

### 6.4 Speed and kinematics

Per-frame speed from smoothed coordinates, aggregated per node visit, per path,
by **tree level** (reward-agnostic topological grouping) and by **topological
distance to the goal port**.

### 6.5 Cross-session learning curves

Per-session directness, indirect-path counts and mean path excess in date order
per animal. Figures are annotated with configured spans of special sessions
(pre/post surgery, change-of-maze windows) and Sunday sessions are marked, so
schedule and intervention effects are legible on every curve.

### 6.6 Latent behavioural states (hidden Markov models)

Per-node-visit features — *continuous*: mean speed, log dwell, node depth,
topological distance to goal, path excess, log frames since reward, local reward
rate, absolute graph turn angle; *binary*: is-port-node, reward-during-visit,
is-indirect, is-alternation; *categorical emission*: turn type and previous turn
type.

Because there is **no ground-truth state label**, three **weak-label anchors**
encode behavioural expectations and are used for *validation, not fitting*:
`anchor_exploit` (direct within-patch alternation), `anchor_explore` (nodes on
long indirect paths), `anchor_change_of_mind` (the commitment apex).

Four families are fitted on the *same* frame and scored on a *shared* schema
(log-likelihood, AIC, BIC, held-out log-likelihood under a trial-blocked split,
best over seeds): **input-driven IO-HMM**; **sticky Categorical HMM** on turns
alone; **Gaussian HMM** (plain and sticky); **sticky GLM-HMM**. Stickiness is
implemented identically across libraries (κ pseudo-counts on the transition
diagonal), so sticky variants are on equal footing.

Validation uses anchor enrichment (odds ratio, Fisher test, per state per
anchor), state-conditioned feature means, stickiness and run-length
distributions, occupancy, alternation-segment purity, the state trace around
commitment apexes, and **held-out turn-prediction uplift** — the gain from
covariates *plus* one-hot state over covariates alone, which is family-agnostic
and so keeps families with different emissions comparable.

### 6.7 Neural analyses (optional layer)

Spike/video clock stitching; PSTHs aligned to reward, turns and the commitment
apex, split by turn direction × reward outcome; population rate matrices and PCA
with trajectory-in-PC-space analyses around events; **graph-node place fields**
(firing rate per node, spikes over occupancy time, with a spatial-information
score) — a *topological* rate map rather than a 2-D one; and reward-aligned PSTHs
as an independent alignment check.

---

## 7. What is methodologically distinctive

1. **Topology is configuration, not code.** The environment graph comes from a
   pluggable builder, including an arbitrary-graph file loader.
2. **The apparatus is treated as mutable.** Mapping, calibration and port
   assignment are per-session, dated, and explicitly versioned — the tooling
   assumes the maze changes rather than assuming it does not.
3. **Re-mapping is editing, not redrawing.** Adjust mode with rotation, multi-tile
   elements and non-destructive reset makes a physically rearranged maze a
   minutes-long task; polygon storage is what makes re-seated geometry
   expressible.
4. **Configuration is bound to its provenance and verified.** Mappings carry the
   calibration they were drawn against (path + hash + matrix); saved mappings are
   read back and compared against what was on screen.
5. **Position is categorical and topological.** Distances are shortest-path
   lengths; "excess nodes over the topological optimum" is a unit-free measure of
   sub-optimality with no clean pixel-space equivalent.
6. **The state space has designed, discoverable structure.** Patches are graph
   components; the corridor is the single maximal-betweenness edge.
7. **Decision relevance is separable from physical location**, giving a
   within-subject control.
8. **Commitment is definable without free parameters** (shortest-path suffix).
9. **Reconstruction is validated against the graph**; repairs are counted, not
   hidden.
10. **Alignment is validated against behaviour**, not against timing invariants.
11. **Model comparison is enforced across families** on a shared metric schema
    with held-out data.

---

## 8. Claims and constructs that need literature grounding

Each line is something this work asserts, assumes or measures, for which prior
art should be found — supporting, contradicting, or offering a better-established
alternative.

**Apparatus, tooling and reproducibility**

- Modular / reconfigurable maze apparatus for rodents; environment-change and
  environment-geometry-manipulation designs.
- Interactive annotation and region-labelling tools for behavioural video;
  human-in-the-loop spatial annotation.
- Camera-to-arena homography calibration; registration validation for overhead
  behavioural video; reusable fiducial/reference point sets.
- Provenance, versioning and reproducibility standards for behavioural pipelines;
  detecting configuration/data mismatch.
- Discretising continuous space into regions or graphs for behavioural analysis;
  ROI-based versus coordinate-based behavioural scoring.

**Task and behaviour**

- Patch foraging with local depletion in rodents; marginal value theorem;
  patch-leaving decisions; giving-up time.
- Spontaneous and rewarded alternation; its relation to working memory and to
  depletion.
- Binary-tree, labyrinth and multi-arm maze paradigms for sequential spatial
  choice.
- Path stereotypy, route learning, emergence of direct trajectories across
  sessions.
- Vicarious trial and error (VTE), deliberation, pausing at choice points.
- Change of mind / mid-trajectory decision reversal in freely moving animals.

**Representation and analysis method**

- Graph- and topology-based representations of environments in behavioural
  neuroscience; topological versus metric spatial coding.
- Shortest-path / excess-path measures of navigational efficiency.
- Sequence analysis of discrete behavioural states; n-gram and Markov-order
  analysis of trajectories.
- HMMs and GLM-HMMs for latent behavioural state; sticky and HDP-HMM priors;
  model selection without ground truth.
- Weak supervision / anchor-based validation of unsupervised state models.
- Held-out predictive uplift as a latent-state model-comparison metric.

**Neural**

- Place fields in topologically structured, compartmentalised and maze-like
  environments; place coding on graphs.
- Splitter cells; prospective and retrospective coding; trajectory-dependent
  firing at choice points.
- Successor representation and predictive maps.
- Remapping under environmental reconfiguration — directly relevant, since this
  apparatus is deliberately rearranged.
- Reward-prediction-error signalling under depletion; dopamine and foraging.
- Population-geometry / state-space trajectory analyses around behavioural
  events.

**Data quality**

- Synchronisation of video, behavioural-control and electrophysiology clocks;
  offset and drift detection in multi-stream experiments.
- DeepLabCut post-processing, likelihood filtering, and error propagation into
  derived *discrete* measures.
- Standards for reporting tracking quality and data exclusion in freely moving
  recordings.

---

## 9. Keyword clusters for search

**Apparatus and tooling**
modular maze · reconfigurable maze · dynamic maze apparatus · modular behavioural
arena · maze geometry manipulation · ROI annotation behavioural video ·
interactive labelling tool neuroscience · homography calibration arena ·
camera-arena registration · fiducial points calibration · open-source behavioural
analysis pipeline · reproducibility behavioural neuroscience · provenance
tracking analysis pipeline

**Graph representation**
graph-based spatial analysis · topological representation environment ·
maze as graph · node-edge occupancy · discretised spatial analysis · state-space
graph navigation · shortest path behaviour · navigational efficiency index ·
detour behaviour · graph topology behaviour

**Foraging and value**
patch foraging · marginal value theorem · patch leaving decision · reward
depletion · depleting reward schedule · giving-up time · optimal foraging theory
rodent · exploration exploitation trade-off · local reward rate · block structure
reward switching · spontaneous alternation

**Deliberation and choice dynamics**
vicarious trial and error · VTE hippocampus · deliberation freely moving ·
change of mind decision · motor commitment point · pause at choice point ·
head sweeps decision

**Latent state modelling**
GLM-HMM behavioural state · hidden Markov model animal behaviour · sticky HMM ·
HDP-HMM behaviour segmentation · input-output HMM · unsupervised behavioural
state discovery · behavioural motif segmentation · cross-validated log likelihood
behaviour

**Neural coding**
place cells maze topology · topological place field · graph place cell ·
successor representation hippocampus · predictive map · splitter cell ·
prospective coding choice point · trajectory dependent firing · hippocampal
remapping environment change · population geometry decision · reward prediction
error depletion · dopamine foraging

**Hierarchical structure**
hierarchical reinforcement learning options · subgoal discovery graph ·
bottleneck states options framework · betweenness centrality behaviour ·
graph Laplacian state abstraction

**Methods**
DeepLabCut post-processing · pose estimation error propagation · markerless
tracking validation · video behavioural synchronisation · clock drift alignment
electrophysiology video · camera trigger offset · occupancy normalised firing
rate map · spatial information score

---

## 10. Recommended future analyses

Ordered roughly by scientific return per unit of implementation cost. Each uses
graph information already computed but not yet exploited. Literature for these is
also wanted.

### 10.1 Formal patch-leaving / MVT modelling — highest value, lowest cost

Everything needed is logged. The depletion counter and reward volume give
instantaneous patch yield; the graph gives **exact travel cost** between patches
as a shortest-path length and as measured transit time. This permits a direct
marginal-value-theorem test: does the animal leave when local intake falls to the
environment-wide average, and does the leaving threshold shift with travel cost?
Because travel cost is *topologically known and varies across patches*, this is
an unusually clean MVT test. Also compare a leaky-accumulator or hazard-rate
model of leaving, and test whether leaving is better predicted by consecutive
unrewarded visits than by integrated volume.

### 10.2 Encoding models of single units on graph covariates

Replace PSTHs with a **per-unit GLM** over the graph quantities already computed:
node identity or depth, topological distance to the active goal, turn type,
inbound/outbound, toward/away, speed, time since reward, local reward rate,
depletion step, patch identity, decision-point importance. This resolves what
PSTHs cannot — whether apparent reward or turn tuning survives control for speed
and position. The single biggest available upgrade to the neural side.

### 10.3 Successor representation and predictive maps

The environment is a graph with known adjacency and a behaviourally estimable
transition policy — exactly where SR predictions are sharpest. Compare measured
node rate maps against SR eigenvectors / diffusion components of the maze graph,
and test whether fields skew backward along habitual routes. The
four-patch/corridor structure predicts specific low-frequency graph-Laplacian
components: a strong, falsifiable prediction.

### 10.4 Prospective coding and pre-choice decoding

Using the **sometimes-important** node set as a within-subject control, ask
whether population activity at a junction decodes the *upcoming* branch before
the turn is executed, and whether decoding strength tracks decision relevance
rather than physical location. Align on the commitment apex to ask whether a
decodable goal signal appears at or before commitment.

### 10.5 Exploit the reconfiguration itself — remapping and transfer

**The most under-used asset in the dataset.** The change-of-maze windows are a
natural experiment: physical layout and port assignment changed for a few
sessions, then reverted. This affords tests of **remapping** (do node fields
change when the same physical location takes a different task role?), of
**behavioural transfer** (is re-learning on reversion faster than original
learning?), and of **topological vs metric coding** (does the representation
follow the graph or the room?). The tooling in §4 exists precisely to make these
sessions analysable; they are currently treated only as figure annotation.
Deliberately extending this — mapping a genuinely different topology onto the
same apparatus using a different graph builder — would turn incidental
reconfiguration into designed manipulation.

### 10.6 Quantify VTE topologically

The standard VTE measure (integrated absolute angular velocity, zIdPhi) is
pixel-space. Complement it with graph-space measures: dwell at
importance-weighted nodes, micro-reversals across an edge, and head direction
(from `nose`, `ear_L`, `ear_R`) relative to the *graph* angle of each available
branch. Test whether junction VTE predicts subsequent path directness and
declines with learning.

### 10.7 Trajectory sequence statistics and null models

Missing and cheap: fit Markov models of increasing order to the node sequence to
find the memory depth of the policy; measure path entropy and stereotypy per
session as a learning index; and build **graph-legal shuffles** (random walks
matched for occupancy and step count) as null models. Several current measures —
directness, excess nodes, alternation — are reported without a null, which
weakens them.

### 10.8 Hierarchical RL / options framing

The maze has explicit bottlenecks (the corridor; each patch root), identifiable
by betweenness centrality without hand-labelling. Test whether behaviour segments
at these bottlenecks — whether the policy is better described as a sequence of
sub-goals (reach patch → reach port) than as a flat policy — and whether inferred
HMM states align with bottleneck crossings. This connects §6.6 to a normative
framework, and is topology-general.

### 10.9 Lick microstructure as a valuation readout

Licks are logged but used only as a location check. Bout length and latency at a
depleting port are a standard consumption/valuation readout and would give a
second, tracking-independent estimate of subjective patch value, validating 10.1.

### 10.10 Statistical framing across sessions and animals

Most current comparisons pool visits or sessions. Moving to **hierarchical /
mixed-effects models** with animal and session as random effects, and reporting
effect sizes rather than only p-values, would materially strengthen every
behavioural claim — pseudo-replication risk is high given the number of node
visits per session.

---

## 11. Summary for the search

The contribution is a **topology-general, graph-based behavioural analysis
pipeline for reconfigurable maze apparatus**, together with the setup tooling
that makes per-session re-mapping and re-calibration tractable and checkable, and
its application to a rodent patch-foraging task. The maze currently in use is a
dual-root binary tree; the graph structure is configuration, supplied by a
pluggable builder that includes an arbitrary-graph loader.

Specific contributions: an editing-based mapping workflow (rotation, multi-tile
elements, non-destructive reset, polygon geometry) for apparatus that physically
changes between sessions; calibration provenance binding a mapping to its
homography, with reusable map-side point sets and visual pre-save verification;
validated traversal reconstruction from noisy pose tracking; a behaviour-anchored
method for detecting video/behaviour clock misalignment; parameter-free
definitions of path directness and decision commitment; context-dependent
decision-point classification; multi-family latent-state modelling with
anchor-based validation; and graph-node place-field analysis.

Literature is wanted on: **(i)** reconfigurable/modular apparatus and
environment-change designs, and the tooling and reproducibility practices around
them; **(ii)** graph and topological representations of space in behaviour and in
hippocampal coding, including remapping under reconfiguration; **(iii)** patch
foraging and depletion in rodents; **(iv)** deliberation and change of mind at
choice points; **(v)** latent behavioural-state models and their validation; and
**(vi)** methods for tracking quality, synchronisation and registration in freely
moving recordings.
