"""Reward-event annotation.

Derives a tidy per-reward-event table from the frame-level ``summary_df`` reward
columns, labelling each reward with the port node and its Target label. This is
the behavioral-core reward table consumed by the raster / trial-context views;
the richer BPOD-derived ``reward_df_annotated`` (depletion step, transition-type
chains) is layered on top when BPOD data is available.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from ..columns import resolve, try_resolve


def build_reward_table(
    summary_df: pd.DataFrame,
    port_map: Optional[Dict[str, str]] = None,
    bodypart: str = "headstage",
) -> pd.DataFrame:
    """One row per reward delivery.

    Columns: ``trial_idx, frame_idx_global, reward_size, port_node, port_target``,
    plus ``reward_port`` when *summary_df* carries the BPOD port number.
    ``port_node`` is the graph node occupied at the reward frame; ``port_target``
    is its Target label from *port_map* (if provided). ``reward_port`` is the port
    BPOD says paid, which is ground truth rather than an observation.
    """
    frame_col = resolve(summary_df, "frame_idx_global")
    reward_col = resolve(summary_df, "reward")
    size_col = try_resolve(summary_df, "reward_size")
    trial_col = try_resolve(summary_df, "trial_idx")
    node_col = try_resolve(summary_df, bodypart, "graph_node")
    reward_node_col = try_resolve(summary_df, "reward_node")
    # The 1-indexed BPOD port, when the ingest stage recorded it. This is the
    # ground truth the reward-based location correction needs: port_node below
    # is the node tracking *says* the animal was on, which is the thing that may
    # be wrong.
    port_num_col = try_resolve(summary_df, "reward_port")

    rew = pd.to_numeric(summary_df[reward_col], errors="coerce").fillna(0)
    mask = rew > 0
    ev = summary_df[mask]

    if len(ev) == 0:
        return pd.DataFrame(
            columns=["trial_idx", "frame_idx_global", "reward_size", "port_node", "port_target"]
        )

    port_node = (
        ev[reward_node_col] if reward_node_col is not None
        else (ev[node_col] if node_col is not None else pd.Series(np.nan, index=ev.index))
    )
    out = pd.DataFrame({
        "trial_idx": ev[trial_col].values if trial_col is not None else np.nan,
        "frame_idx_global": ev[frame_col].values,
        "reward_size": ev[size_col].values if size_col is not None else np.nan,
        "port_node": np.asarray(port_node).astype(object),
    })
    if port_map is not None:
        out["port_target"] = out["port_node"].map(port_map)
    else:
        out["port_target"] = np.nan
    if port_num_col is not None:
        out["reward_port"] = pd.to_numeric(
            ev[port_num_col], errors="coerce").to_numpy()
    return out.reset_index(drop=True)


__all__ = ["build_reward_table"]
