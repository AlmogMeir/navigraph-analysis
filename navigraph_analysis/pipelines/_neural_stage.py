"""Neural stage of the per-session pipeline (attached only when --neural is set).

Loads the session's spikes, builds the trial clock, and produces per-good-unit
summaries: reward-aligned PSTH peak and a graph-node place-field summary
(peak node + spatial information). Kept deliberately compact — the detailed,
per-unit figure generation stays in the analysis notebooks that call the
:mod:`navigraph_analysis.neural` primitives.
"""

from __future__ import annotations

import warnings
from typing import Dict

import numpy as np
import pandas as pd

from ..config import ProjectConfig
from ..io import navigraph_output as ngo
from ..io.spikes import load_spikes
from ..neural import clock as _clock
from ..neural import place_cells as _pc
from ..neural import psth as _psth
from ..session import Session


def run(
    session: Session,
    cfg: ProjectConfig,
    tables: Dict[str, pd.DataFrame],
    save: bool = True,
) -> Dict[str, pd.DataFrame]:
    """Compute neural summaries for *session*; returns {} if no spikes available."""
    spikes_dir = session.spikes_dir
    if spikes_dir is None or not spikes_dir.exists():
        warnings.warn(f"No spikes dir for {session}; skipping neural stage.")
        return {}

    sp = load_spikes(spikes_dir, fs=cfg.fs)
    summary_df = tables["summary_df"]
    node_df = tables["node_df"]
    reward_df = tables["reward_df"]

    trial_clock = _clock.build_trial_clock(summary_df, fps=cfg.fps)
    reward_events = _reward_event_times(reward_df, cfg.fps)

    rows = []
    for unit in sp.good_units:
        st = sp.times_for(unit)
        rmap = _pc.node_rate_map(st, node_df, fps=cfg.fps)
        si = _pc.spatial_information(rmap)
        peak_node = rmap.iloc[0]["node_name"] if len(rmap) else None
        peak_rate = float(rmap.iloc[0]["rate_hz"]) if len(rmap) else np.nan

        reward_psth = _psth.psth_across_events(st, reward_events, pre=1.0, post=1.0)
        rows.append({
            "unit": int(unit),
            "n_spikes": int(len(st)),
            "peak_node": peak_node,
            "peak_rate_hz": peak_rate,
            "spatial_info_bits": si,
            "reward_psth_peak_hz": float(np.max(reward_psth["smoothed"]))
            if reward_psth["n_events"] else np.nan,
        })

    neural_summary = pd.DataFrame(rows)
    products = {"neural_unit_summary": neural_summary, "trial_clock": trial_clock}
    if save:
        for name, df in products.items():
            ngo.save_table(df, session, name)
    return products


def _reward_event_times(reward_df: pd.DataFrame, fps: float) -> list:
    if reward_df.empty or "frame_idx_global" not in reward_df.columns:
        return []
    return [_clock.frame_to_abs(f, fps) for f in reward_df["frame_idx_global"].to_numpy()]


__all__ = ["run"]
