# Notebook → module migration map

Where the logic from each original NaviGraph notebook / module now lives. The
original notebooks under `NaviGraph/examples/dual_root_tree/` are untouched and
remain the reference of record.

## Data prep (raw → canonical tables)

| Original | New home |
|---|---|
| `align_frames_trials.ipynb` (frame/trial alignment) | `prep/align.py` (`prepare_summary_df`, `trial_windows`) |
| `improved_node_df.ipynb`, `traversal_pipeline.ipynb` (traversal reconstruction) | `prep/traversal.py` (`reconstruct_traversal`) |
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

## Cross-cutting boilerplate (de-duplicated)

| Duplicated across notebooks | Now single-sourced in |
|---|---|
| session/date parsing (both DD_MM_YYYY and YYYY_MM_DD) | `session.py` (`parse_date`, `Session`) |
| graph construction from `FullMazeGrid.pkl` | `io/graph.py` (`load_maze_graph`) + `behavior/states.build_graph_from_full_maze` alias |
| reward-port map + 2025-09-07 R55/R59 swap + patch pairs | `config/ports.yaml` + `ports.py` (`PortConfig`) |
| spike loading (`loadmat`, `cluster_info`, `_unwrap_mat`, good/mua units) | `io/spikes.py` (`load_spikes`) |
| `summary_df` MultiIndex column accessors | `columns.py` |
| hard-coded absolute paths | `config/project.yaml` + `config.py` (`ProjectConfig`) |

## Excluded (not analyses / duplicates)

- `align_frames_trials copy.ipynb`, `align_frames_trials (Copy).ipynb`,
  `improved_node_df copy.ipynb`, `improved_node_df copy 2.ipynb`,
  `change_of_mind_prev.ipynb` — working-copy duplicates.
- `UnSupervisedIOHMM.ipynb` — vendored upstream library tutorial.
- root `test_head_direction.py` / `test_working_vs_navigraph.py` / `validate_first_50.py` /
  `svg_export_utility.py` — head-direction/GUI validation scripts, not analyses.
