# navigraph-analysis

Standalone, config-driven analysis pipelines for dual-root-tree maze experiments.

This project consumes **NaviGraph output** (the integrated frame-level DataFrame),
**DeepLabCut pose**, a **graph-mapping pickle**, and — optionally — **Neuropixels
spikes** and **BPOD** behavior, and turns them into ordered per-subject /
per-session analyses. It is **fully independent of the `navigraph` package**: the
portable analysis code was copied in and de-coupled, and the heavy notebooks that
used to hold this logic were distilled into reusable functions.

## Why this exists

The analyses previously lived in ~20 Jupyter notebooks with copy-pasted graph
loaders, reward-port maps, spike loaders, PSTH/PCA code, date parsing, column
accessors and hard-coded absolute paths. Here every one of those is a single,
tested function, and **all paths live in one config file**.

## Install

```bash
cd navigraph-analysis
python -m venv .venv && source .venv/bin/activate
pip install -e .            # behavioral core
pip install -e ".[neural]"  # + HMM state modelling (hmmlearn / IOHMM / sklearn)
```

Requires Python >= 3.9. Core deps: pandas, numpy, networkx, scipy, matplotlib,
seaborn, h5py, tables, pyyaml.

## Configure

Copy `config/project.yaml` to `config/project.local.yaml` (gitignored) and edit
the paths for your machine. That single file replaces every hard-coded path the
notebooks used:

```yaml
subjects: [NPC1, NPC4]
constants: { fps: 40, fs: 30000, maze_start: 30, maze_end: 767 }
paths:
  navigraph_output: ~/.../Navigraph_Raw_Data   # <subject>/<DD_MM_YYYY>/raw_session_data_*.pkl
  graph_pkl:        ~/.../FullMazeGrid.pkl
  intermediate:     ./outputs                   # <subject>/<YYYY-MM-DD>/ — all products
  maze_mappings:    ~/.../maze_mappings         # <subject>_<YYYY_MM_DD>.pkl
  calibration:      ~/.../transform_matrix      # per-session <YYYY_MM_DD>.npy
  spikes:           ~/.../SpikesData            # optional (neural)
  bpod:             ~/.../BPODData              # optional
ports_config: ./config/ports.yaml               # per-subject, date-ranged port layouts
```

Paths are **subject-nested**: raw output, spikes and processed tables live under
a `<subject>/` subfolder (a legacy flat layout still works as a fallback). The
reward-port map is a **parameter** resolved per `(subject, date)` from
`config/ports.yaml` — each subject has date-ranged layout periods, so re-wirings
(NPC1's 2025-09-07 R55/R59 swap) and different mice (NPC4's ports 3/4 swapped)
are handled automatically.

## Run

```bash
nga list                          # discover sessions from the config
nga prep --session 21_08_2025     # build canonical tables (summary/traversal/node/reward)
nga run  --session 21_08_2025     # full behavioral pipeline
nga run  --all                    # every session, chronologically
nga run  --all --neural           # include the spike layer
nga cross                         # cross-session learning curves
```

Outputs are written under `outputs/<subject>/<YYYY-MM-DD>/` (per session) and
`outputs/<subject>/cross_session/` (aggregates), so results are naturally ordered
by session date.

### From Python

```python
from navigraph_analysis import ProjectConfig, Session
from navigraph_analysis.pipelines.session_pipeline import run_session

cfg = ProjectConfig.load()
s = Session.parse("NPC1", "21_08_2025", cfg)
products = run_session(s, cfg)
products["node_df"], products["turn_df"], products["state_features"]
```

## Package layout

```
navigraph_analysis/
  config.py / session.py / columns.py / ports.py   # config, session id/dates, accessors, port maps
  io/        graph, navigraph_output, dlc, spikes, bpod
  prep/      align -> traversal -> node_tables -> reward   (raw -> canonical tables)
  behavior/  speed, turns, decision_points, change_of_mind, states, quality
  neural/    clock, psth, population, place_cells          (optional layer)
  viz/       style, maze, raster, trial_context
  pipelines/ session_pipeline, cross_session, cli
config/      project.yaml, ports.yaml
notebooks/   thin examples that only call the library
tests/       unit tests for the de-duplicated pure functions
```

## Adding a new session or subject

1. Drop the session's `raw_session_data_*.pkl` under
   `<navigraph_output>/<DD_MM_YYYY>/` (and spikes under `<spikes>/<DD_MM_YYYY>/`).
2. Add the subject to `subjects:` in the config if new.
3. `nga run --session <date>` (or `--all`).

Dates are accepted in any ordering/separator (`21_08_2025`, `2025-08-21`, …);
the reward-port map is selected automatically by date via `config/ports.yaml`.

## Relationship to the notebooks

The original NaviGraph notebooks are untouched and remain the reference of record.
Each module's docstring names the notebook(s) it was distilled from. See
`MIGRATION.md` for the notebook → module map.
