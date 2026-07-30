"""Spike <-> video clock alignment.

Consolidates the "stitch spikes onto the video clock" boilerplate that was
duplicated across change_of_mind, turn_analysis and state_hmm (dedup E): build a
per-trial clock from ``frame_idx_global`` and gather spikes relative to an event
time. Aligned spike times are assumed to already share the session clock (that is
what ``aligned_spike_times.mat`` provides); frames map to seconds via
``frame_idx_global / fps``.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from ..columns import resolve


def build_trial_clock(summary_df: pd.DataFrame, fps: float = 40.0) -> pd.DataFrame:
    """Per-trial clock: ``trial_idx, start_frame, end_frame, start_abs, end_abs`` (seconds)."""
    frame_col = resolve(summary_df, "frame_idx_global")
    trial_col = resolve(summary_df, "trial_idx")
    tf = pd.DataFrame({
        "trial_idx": summary_df[trial_col].to_numpy(),
        "frame": pd.to_numeric(summary_df[frame_col], errors="coerce").to_numpy(),
    })
    agg = tf.groupby("trial_idx", sort=True)["frame"].agg(["min", "max"]).reset_index()
    agg = agg.rename(columns={"min": "start_frame", "max": "end_frame"})
    agg["start_abs"] = agg["start_frame"] / fps
    agg["end_abs"] = agg["end_frame"] / fps
    return agg


def frame_to_abs(frame_idx: float, fps: float = 40.0) -> float:
    """Convert a global frame index to absolute seconds on the session clock."""
    return float(frame_idx) / fps


def gather_relative_spikes(
    spike_times_s: Sequence[float],
    event_abs: float,
    pre: float = 1.0,
    post: float = 1.0,
) -> np.ndarray:
    """Spike times relative to *event_abs* (seconds) within ``[-pre, +post]``."""
    st = np.asarray(spike_times_s, dtype=float)
    lo, hi = event_abs - pre, event_abs + post
    sel = st[(st >= lo) & (st <= hi)]
    return sel - event_abs


__all__ = ["build_trial_clock", "frame_to_abs", "gather_relative_spikes"]
