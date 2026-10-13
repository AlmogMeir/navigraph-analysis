# Validation against the notebook pipeline

Result of running the new pipeline end-to-end on two sessions and comparing
every table against the outputs the `session_pipeline.ipynb` notebook had
already produced. The existing outputs were **not** touched: the run used a
config whose `intermediate` path pointed at a scratch directory, and all 30
existing files were verified byte-size- and mtime-identical afterwards.

Sessions: **NPC1 08/09/2025** (501 trials; inside the change-of-maze window, so
it also exercises the date-ranged port layout) and **NPC4 10/05/2026**
(34 trials).

## Result

| Table | Agreement |
|---|---|
| `traversal_df` | **identical** — all 10 columns, 100%, both sessions |
| `node_edge_df` | **identical** — all 16 columns, 100%, both sessions |
| `node_df` | **identical** to the reference `build_node_df` on the old inputs — 15/17 columns at 100%, same row count; the 2 exceptions are the lick columns (see D1) |
| `summary_df` | identical on trial/frame structure and graph location (100%); differs on lick (D1), low-likelihood pose (D2), 2 reward frames (D3), `fixed_node` (D4) |
| commitment apexes | 18 of 19 land within ±2 frames of an old apex |

Three bugs were found and fixed while doing this; they are listed at the bottom.

## Known differences (new vs the notebook's saved outputs)

### D1. `lick` was empty in every old output — now populated

The notebook's saved `summary_df` has **zero** lick frames for both sessions,
so `lick_during_node` and `lick_count_during_node` were meaningless in all
existing `node_df` files. The new ingest recovers them:

| Session | lick events in `lick_events.csv` | old lick frames | new lick frames |
|---|---|---|---|
| NPC1 08/09/2025 | 7029 | 0 | 6946 |
| NPC4 10/05/2026 | 67 | 0 | 68 |

New is right. (NPC4 reports 68 frames for 67 events because a lick falling
exactly between two frames is within tolerance of both — the notebook's own
rule, kept verbatim; see `test_lick_mask_marks_every_frame_within_tolerance`.)

**Anything computed from licks in the old outputs should be recomputed.**

### D2. Pose is NaN below likelihood 0.3

The notebook read pose from the DeepLabCut `.h5` directly; the new pipeline
reads it from the NaviGraph output, which NaNs out x/y below a 0.3 likelihood
threshold (likelihood itself matches 100%). Affects 645 / 133329 frames (0.48%)
for NPC1 and 359 / 98714 (0.36%) for NPC4 — every one of them at
likelihood ≤ 0.300.

This is NaviGraph's filter, not a regression, and it slightly changes speed
where those frames fall. Left as-is: taking filtered coordinates from the
pipeline's own upstream stage is the more defensible choice. If exact parity
with the notebook matters for a specific figure, read pose from the `.h5`
instead.

### D3. Two reward events differ by one frame (NPC1)

2 of 501 reward frames land one frame (25 ms) later than the notebook's, from
rounding at a trial boundary. NPC4: 0 of 33. Not worth chasing.

### D4. `fixed_node` is not reproduced

The notebook copied a `fixed_node` column out of NaviGraph's own
`traversal_sequence`. Nothing in this repo reads it, and the new pipeline
derives its traversal independently, so it is not carried over. Say so if you
want it back.

### D5. `node_df` row count vs the *saved* `node_df.csv`

The new `node_df` matches the reference implementation exactly (5128 rows for
NPC1, 1274 for NPC4). The saved `node_df.csv` has more (5414 / 1299) because
the notebook applied a further **node-level** reward fix after calling
`build_node_df` and saved the result under the same name. The new pipeline
applies the equivalent correction at the **frame level before** traversal
reconstruction (`corrections=True`), which is a different ordering with a
different — and more defensible — result, since correcting frames first lets
the traversal reconstructor see the corrected sequence.

Run with `corrections=False` to reproduce the reference `build_node_df` exactly.

### D6. Consecutive same-node merging also merges out-and-back excursions

`build_node_df` merges consecutive visits to the same node. After edges are
filtered out, a `node → edge → same node` excursion looks like two consecutive
visits to that node and is merged into one, however long the excursion was:
249 such excursions in NPC1 (median 12 frames, max 857) and 420 in NPC4
(median 18, max 4879 — 122 s).

This is inherited verbatim from `navigraph.analysis.traversal_builder`, so it
is **not** changed here — both pipelines agree. But it erases exactly the
micro-reversals the change-of-mind and VTE analyses care about, and it is worth
a decision: a stay could instead be broken whenever an edge row intervenes.
Flagged, not fixed, because it would silently change every existing number.

## Bugs found and fixed during validation

1. **Trial durations were read from the wrong field** (`io/bpod.trial_end_times`).
   It used `TrialEndTimestamp - TrialStartTimestamp`; the notebook uses the
   *onset of the `TrialEnd` state* in `RawEvents`. The two differ by the length
   of that state — 0.05 s, exactly 2 frames at 40 fps — and because the frame
   table is built by cumulatively summing durations, the error compounds: over
   NPC1's 501 trials it displaced the session end by 1002 frames (25 s).

   The symptom was invisible per trial and catastrophic in aggregate: reward
   events drifted progressively off the ports that paid them, and the alignment
   check rejected nearly every session. Before the fix, 0/19 NPC1 and 3/28 NPC4
   sessions passed; after it, **19/19 and 27/28**, with median on-port fractions
   of 98.8% and 99.9%.

   The one remaining failure is NPC4 01/06/2026 at 13.3% with a +5814 frame
   overshoot — which independently reproduces the exact figures in the
   `a2cdcb5` commit message, i.e. the genuinely broken session the check exists
   to catch. Guarded by `test_trial_duration_comes_from_the_trial_end_state_onset`
   and `test_accumulated_drift_moves_reward_events_off_port`.

2. **Raw NaviGraph pickles could not be read** (`io/navigraph_output._coerce_frame`).
   Current output nests the frame table one level under the session name
   (`{"session_<subject>_<date>": {"dataframe": ...}}`); the loader only looked
   at the top level and raised `TypeError`. It now recurses, and refuses to
   guess when a pickle holds more than one session.

3. **The ingest stage did not exist.** `build_tables` assumed `summary_df`
   already had `frame_idx_global` and `trial_idx`, which raw NaviGraph output
   does not carry — they come from BPOD. `prep/ingest.py` now derives them,
   verifies the clocks, and annotates reward and licks; `_load_summary` calls it
   whenever the frame table lacks those columns.

Two smaller fixes: `reward_df` now carries the BPOD `reward_port` through (the
reward-based correction was silently skipping for want of it), and bodypart
case is canonicalised in the ingest (`ear_r` → `ear_R`) so NPC4 matches NPC1.
