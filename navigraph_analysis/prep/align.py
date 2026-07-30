"""Frame/trial alignment and per-trial windowing.

NaviGraph's raw integrated DataFrame already carries ``frame_idx_global`` and
``trial_idx``; this module normalizes it into the canonical ``summary_df`` used
downstream and derives per-trial frame windows (``trial_windows``) that both the
node-table builders and the neural clock-stitching reuse (previously re-derived
inline in align_frames_trials / change_of_mind / turn_analysis).
"""

from __future__ import annotations

import pandas as pd

from ..columns import resolve, try_resolve


def prepare_summary_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize a raw NaviGraph frame DataFrame into a canonical ``summary_df``.

    Verifies the required columns exist and returns the frame sorted by
    ``frame_idx_global``. Column layout (flat or MultiIndex) is preserved so
    downstream resolvers keep working.
    """
    frame_col = resolve(raw_df, "frame_idx_global")  # raises if missing
    # trial_idx / graph_node are required for the full pipeline; surface early.
    if try_resolve(raw_df, "trial_idx") is None:
        raise KeyError("summary_df is missing 'trial_idx'")
    if try_resolve(raw_df, "graph_node") is None:
        raise KeyError("summary_df has no '*_graph_node' column")
    return raw_df.sort_values(frame_col).reset_index(drop=True)


def trial_windows(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Per-trial global frame window.

    Returns columns ``trial_idx, start_frame, end_frame, n_frames`` sorted by
    trial. Used for trial-context figures and as the basis for the spike/video
    clock (``start_frame / fps``).
    """
    frame_col = resolve(summary_df, "frame_idx_global")
    trial_col = resolve(summary_df, "trial_idx")
    # Build a flat 2-column frame first so MultiIndex tuple keys don't trip up
    # groupby column selection.
    tf = pd.DataFrame({
        "trial_idx": summary_df[trial_col].to_numpy(),
        "frame": pd.to_numeric(summary_df[frame_col], errors="coerce").to_numpy(),
    })
    agg = tf.groupby("trial_idx", sort=True)["frame"].agg(["min", "max"]).reset_index()
    agg = agg.rename(columns={"min": "start_frame", "max": "end_frame"})
    agg["n_frames"] = agg["end_frame"] - agg["start_frame"] + 1
    return agg


__all__ = ["prepare_summary_df", "trial_windows"]
