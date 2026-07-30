"""Per-trial 4-panel context figure.

Reusable version of ``trial_context_viewer.ipynb``: for one trial, a 4-panel
diagnostic figure combining the XY trajectory, the graph node-visit sequence,
a reward raster of nearby trials, and rolling task performance. Composes the
:mod:`~navigraph_analysis.viz.maze` and :mod:`~navigraph_analysis.viz.raster`
primitives instead of re-deriving them.
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from ..columns import resolve, try_resolve
from . import raster as _raster
from .maze import plot_graph, plot_trajectory


def trial_context_figure(
    trial_idx: int,
    summary_df: pd.DataFrame,
    node_df: pd.DataFrame,
    per_trial_reward: pd.DataFrame,
    G: nx.Graph,
    bodypart: str = "headstage",
    context_trials: int = 50,
    rolling_window: int = 10,
    background=None,
) -> plt.Figure:
    """Build the 4-panel context figure for *trial_idx*.

    Panels: (TL) XY trajectory over the optional background frame, (TR) graph
    with this trial's node-visit path highlighted, (mid) reward raster of the
    surrounding trials, (bottom) rolling percent-reward.
    """
    x_col, y_col, frame_col = (
        resolve(summary_df, bodypart, "x"),
        resolve(summary_df, bodypart, "y"),
        resolve(summary_df, "frame_idx_global"),
    )
    trial_col = resolve(summary_df, "trial_idx")

    trial_mask = summary_df[trial_col] == trial_idx
    tx = pd.to_numeric(summary_df.loc[trial_mask, x_col], errors="coerce").to_numpy()
    ty = pd.to_numeric(summary_df.loc[trial_mask, y_col], errors="coerce").to_numpy()

    trial_nodes = node_df[node_df["trial_idx"] == trial_idx]["node_name"].tolist()

    fig = plt.figure(figsize=(12, 9))
    gs = fig.add_gridspec(3, 2, height_ratios=[3, 1, 1])

    ax_traj = fig.add_subplot(gs[0, 0])
    plot_trajectory(tx, ty, ax=ax_traj, background=background, title=f"Trial {trial_idx} trajectory")

    ax_graph = fig.add_subplot(gs[0, 1])
    plot_graph(G, ax=ax_graph, highlight_path=trial_nodes, title="Node-visit path")

    ax_rast = fig.add_subplot(gs[1, :])
    window = per_trial_reward[
        (per_trial_reward["trial_idx"] >= trial_idx - context_trials)
        & (per_trial_reward["trial_idx"] <= trial_idx + context_trials)
    ]
    _raster.plot_reward_raster(window, ax=ax_rast, highlight_trial=trial_idx)

    ax_perf = fig.add_subplot(gs[2, :])
    _raster.plot_rolling_performance(
        per_trial_reward, window=rolling_window, ax=ax_perf, highlight_trial=trial_idx
    )

    fig.tight_layout()
    return fig


__all__ = ["trial_context_figure"]
