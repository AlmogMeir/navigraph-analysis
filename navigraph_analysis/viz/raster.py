"""Reward rasters and rolling task-performance.

Reusable, BPOD-independent core of ``raster_plot.ipynb`` (its
``extract_raster_data`` / ``did_receive_reward`` helpers were also copy-pasted
into ``trial_context_viewer.ipynb``). Works from the reward table produced by
:func:`navigraph_analysis.prep.reward.build_reward_table`.
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def per_trial_reward(reward_df: pd.DataFrame, n_trials: Optional[int] = None) -> pd.DataFrame:
    """Collapse reward events to one row per trial: ``trial_idx, rewarded, port_target``."""
    if reward_df.empty:
        return pd.DataFrame(columns=["trial_idx", "rewarded", "port_target"])
    g = reward_df.groupby("trial_idx")
    out = pd.DataFrame({
        "trial_idx": [k for k, _ in g],
        "rewarded": True,
        "port_target": g["port_target"].first().values,
    })
    if n_trials is not None:
        full = pd.DataFrame({"trial_idx": np.arange(n_trials)})
        out = full.merge(out, on="trial_idx", how="left")
        out["rewarded"] = out["rewarded"].fillna(False)
    return out.sort_values("trial_idx").reset_index(drop=True)


def rolling_reward_rate(per_trial: pd.DataFrame, window: int = 10) -> pd.Series:
    """Rolling fraction-rewarded over trials."""
    return per_trial["rewarded"].astype(float).rolling(window, min_periods=1).mean()


def plot_reward_raster(
    per_trial: pd.DataFrame,
    ax: Optional[plt.Axes] = None,
    highlight_trial: Optional[int] = None,
) -> plt.Axes:
    """Raster of reward port (y) across trials (x); unrewarded trials are blank."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 2.5))
    rew = per_trial[per_trial["rewarded"].astype(bool)]
    ports = rew["port_target"].astype("category")
    ax.scatter(rew["trial_idx"], ports.cat.codes, marker="|", s=60, color="black")
    ax.set_yticks(range(len(ports.cat.categories)))
    ax.set_yticklabels(list(ports.cat.categories), fontsize=7)
    ax.set_xlabel("Trial")
    ax.set_ylabel("Reward port")
    if highlight_trial is not None:
        ax.axvline(highlight_trial, color="red", ls="--", lw=1.0)
    return ax


def plot_rolling_performance(
    per_trial: pd.DataFrame,
    window: int = 10,
    ax: Optional[plt.Axes] = None,
    highlight_trial: Optional[int] = None,
) -> plt.Axes:
    """Line plot of the rolling reward rate."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 2.0))
    rate = rolling_reward_rate(per_trial, window)
    ax.plot(per_trial["trial_idx"], rate, color="#2ca02c", lw=1.5)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Trial")
    ax.set_ylabel(f"% reward\n({window}-trial roll)", fontsize=8)
    if highlight_trial is not None:
        ax.axvline(highlight_trial, color="red", ls="--", lw=1.0)
    return ax


__all__ = [
    "per_trial_reward",
    "rolling_reward_rate",
    "plot_reward_raster",
    "plot_rolling_performance",
]
