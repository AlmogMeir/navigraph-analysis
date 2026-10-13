# Notebook → module migration map

Where the logic from each original NaviGraph notebook / module now lives. The
original notebooks under `NaviGraph/examples/dual_root_tree/` are untouched and
remain the reference of record.

**Last synced with NaviGraph:** 2026-09-27, against `New_Analysis` @ `a2cdcb5`
plus its uncommitted working tree. See [Sync log](#sync-log) at the bottom for
what each sync brought over.

## Data prep (raw → canonical tables)

| Original | New home |
|---|---|
| `session_pipeline.ipynb` (BPOD → frame table, video/BPOD clock check) | `prep/frame_alignment.py` (`build_frames_df`, `align_frames_to_video`, `on_port_fraction`) |
| `session_pipeline.ipynb` (zigzag detection + smoothing, reward-based location correction) | `prep/corrections.py` (`smooth_leaf_parent_zigzags`, `apply_reward_based_correction`, `detect_zigzags`) |
| `align_frames_trials.ipynb` (frame/trial alignment) | `prep/align.py` (`prepare_summary_df`, `trial_windows`) |
| `improved_node_df.ipynb`, `traversal_pipeline.ipynb` (traversal reconstruction) | `prep/traversal.py` (`reconstruct_traversal`) |
| `navigraph/analysis/traversal_fixer.py` (`ImprovedMazeTraversalFixer`) | `prep/traversal_fixer.py` (`build_traversal_df`, `make_improved_traversal`) — **the default** |
| `navigraph/analysis/traversal_builder.py` (verbatim, de-coupled) | `prep/node_tables.py` (`build_node_df`, `build_node_edge_df`, `fix_edge_bounces`) |
| reward annotation in `align_frames_trials.ipynb` | `prep/reward.py` (`build_reward_table`) |

## Behavioral analyses

| Original | New home |
|---|---|
| `navigraph/analysis/speed_analysis.py` (verbatim) | `behavior/speed.py` |
| `turn_analysis.ipynb` (`classify_turn`, structural turn typing) | `behavior/turns.py` |
| `decision_point_velocity_analysis.ipynb` | `behavior/decision_points.py` |
| `change_of_mind.ipynb` (indirect-path + commitment-apex geometry) | `behavior/change_of_mind.py` |
| `navigraph/analysis/state_features.py` (de-coupled) + `state_hmm.ipynb` features | `behavior/states.py` |
| `navigraph/analysis/state_models.py`, `state_model_comparison.ipynb` (IO-HMM / Gaussian / Categorical / GLM-HMM + shared scoring) | `behavior/state_models.py` |
| `change_of_mind.ipynb` patch-split cells (`classify_patch`, tree levels) | `io/graph.py` (`level_of`, `patch_of`, `classify_patch`) |
| `decision_point_velocity_analysis.ipynb` entering/exiting × incl./excl. node speed | `behavior/decision_points.py` (`visit_speed_metrics`, `paired_wilcoxon`, `speed_breakdown`) |
| `traversal_quality_analysis.ipynb`, `traversal_correction_audit.ipynb` | `behavior/quality.py` |

## Neural layer (optional)

| Original | New home |
|---|---|
| spike/video clock stitching (`change_of_mind`, `turn_analysis`, `state_hmm`) | `neural/clock.py` |
| PSTH bin-and-smooth (`change_of_mind`, `direct_path_analysis`, `turn_analysis`) | `neural/psth.py` |
| `build_population_rate_matrix` / `run_pca` (`change_of_mind`, `speed_unit_correlation`) | `neural/population.py` |
| place-cell analysis (`align_frames_trials`, `direct_path_analysis`) | `neural/place_cells.py` |

## Visualization

| Original | New home |
|---|---|
| shared palettes / PSTH styling (all notebooks) | `viz/style.py` |
| maze/graph overlays + trajectory (`trial_context_viewer`, `turn_analysis`) | `viz/maze.py` |
| `raster_plot.ipynb` (`extract_raster_data`, `did_receive_reward`) | `viz/raster.py` |
| `trial_context_viewer.ipynb` (4-panel figure) | `viz/trial_context.py` |
| `change_of_mind.ipynb` cross-session date annotations + Sunday ticks | `viz/sessions.py` (`annotate_date_ranges`, `mark_sunday_ticks`) + `date_annotations` in `config/project.yaml` |

## Cross-cutting boilerplate (de-duplicated)

| Duplicated across notebooks | Now single-sourced in |
|---|---|
| session/date parsing (both DD_MM_YYYY and YYYY_MM_DD) | `session.py` (`parse_date`, `Session`) |
| graph construction from `FullMazeGrid.pkl` | `io/graph.py` (`load_maze_graph`) + `behavior/states.build_graph_from_full_maze` alias |
| per-subject port layouts (`PORT_LAYOUTS`/`resolve_port_layout`, `align_frames_trials.ipynb`), date-ranged periods, R55/R59 & NPC4 3/4 swaps, port pairs, grandparents | `config/ports.yaml` + `ports.py` (`PortConfig.port_map(subject, date)`) |
| subject-nested paths (`outputs/<subject>/<date>/`) + per-session calibration (`transform_matrix/<YYYY_MM_DD>.npy`) | `config.py` + `session.py` (`output_dir`, `navigraph_dir`, `calibration_npy`, `graph_mapping_pkl`) |
| spike loading (`loadmat`, `cluster_info`, `_unwrap_mat`, good/mua units) | `io/spikes.py` (`load_spikes`) |
| `summary_df` MultiIndex column accessors | `columns.py` |
| hard-coded absolute paths | `config/project.yaml` + `config.py` (`ProjectConfig`) |
| BPOD protocol discovery (several protocols per subject, non-maze recordings sharing a date) | `io/bpod.py` (`find_bpod_mat`, `load_reward_csv`, `trial_end_times`) |
| tracking-column normalization + bodypart auto-pick (`normalize_tracking_columns`, `PRIMARY_BODYPART`) | `io/dlc.py` (`normalize_bodypart_case`, `resolve_bodypart`) |

## Excluded (not analyses / duplicates)

- `align_frames_trials copy.ipynb`, `align_frames_trials (Copy).ipynb`,
  `improved_node_df copy.ipynb`, `improved_node_df copy 2.ipynb`,
  `change_of_mind_prev.ipynb` — working-copy duplicates.
- `UnSupervisedIOHMM.ipynb` — vendored upstream library tutorial.
- root `test_head_direction.py` / `test_working_vs_navigraph.py` / `validate_first_50.py` /
  `svg_export_utility.py` — head-direction/GUI validation scripts, not analyses.
- `navigraph/core/**` (calibration GUI, point selector, `map_point_sets.py`,
  `setup_gui_qt.py`, `file_discovery.py`) — acquisition/setup tooling. This repo
  consumes NaviGraph *output*; it does not produce mappings or calibrations.
- `create_session_folders.py`, `render_all_mappings.py`,
  `extract_missing_session_images.py` — resource-marshalling scripts that
  operate on the NaviGraph resources tree, not on analysis products.
- `analyze_jump_results.ipynb` — a one-off audit of the tracking-jump metric.

## Sync log

### 2026-09-27 — against `New_Analysis` @ `a2cdcb5` (+ working tree)

Covers everything upstream since `a8ed509` (2026-06-29), the commit this repo was
forked from.

**Corrected**

- `config/ports.yaml`: NPC1 gained the 2025-09-02 → 09-08 change-of-maze periods
  (port 6 on `L518`/`L49`, with the 7/8 swap landing mid-window on 09-07), and
  NPC4 now opens on 2026-03-01 (not 04-01) and has its 2026-06-17 → 06-18
  change-of-maze window (ports 1/5/6 moved). Four sessions previously resolved to
  the wrong leaf nodes. `grandparents` gained `L48`/`L49`/`L415`/`R410`, read off
  `FullMazeGrid.pkl` using the same `[sibling leaf, level-3 parent]` rule as the
  existing entries.

**Added**

- `prep/frame_alignment.py`, `prep/corrections.py`, `prep/traversal_fixer.py` —
  the prep chain described in the README. None of this existed here before; the
  pipeline started from an already-aligned, uncorrected `summary_df`.
- `behavior/state_models.py`, `viz/sessions.py`, `io/graph.level_of` /
  `patch_of` / `classify_patch`,
  `behavior/decision_points.visit_speed_metrics` and friends,
  `io/bpod.find_bpod_mat` + CSV loaders, `io/dlc.resolve_bodypart`.

**Deviations from upstream** (deliberate, and better here)

- `apply_reward_based_correction` indexes `summary_df` by `frame_idx_global`
  rather than by row position. The notebook's positional indexing is correct only
  while the frame table starts at frame 0 — which stops being true as soon as the
  alignment step shifts or trims it, precisely the sessions the correction
  matters most for.
- `behavior/state_models.py` probes `hmmlearn` / `IOHMM` / `ssm` lazily, so the
  module imports with none of them installed and each fitter raises a message
  naming what to install. Upstream imports them at module scope.
- `behavior/states.py` keeps its `pd.to_numeric` guard on `path_excess`
  (`build_node_df` leaves `pd.NA`, which breaks a direct `.astype(float)`).
  Upstream's `state_features.py` still has the unguarded cast.

**Not ported** (viz-heavy, listed so the next sync knows they are outstanding)

- `turn_analysis.ipynb`: the toward/away × rewarded/unrewarded PSTH grid, the
  population-level PSTH figure, the turn-direction patch gallery (graph + maze
  image insets) and the inbound/outbound/neutral schematic.
- `change_of_mind.ipynb`: the four new figure cells built on `classify_patch`
  (in-patch vs out-of-patch speed/duration/node-count, speed vs distance from
  reward, speed by tree level). The primitives they need are ported; the figures
  are not.
- `session_pipeline.ipynb`: the side-by-side and neuron-overlay video writers,
  and the second place-cell method (session-wide raw mean FR) with its
  method-1-vs-2 comparison.
