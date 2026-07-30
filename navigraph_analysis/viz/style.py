"""Shared plotting palettes and styling.

Consolidates the color dicts and PSTH styling that recurred across notebooks
(dedup J): the LEFT/RIGHT/STRAIGHT/UTURN turn palette, the important vs
non-important decision-node colors, a session color ramp, and a single helper
that draws the canonical PSTH (light bars + smoothed line + t=0 marker).
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

# Turn-type palette (turn_analysis.ipynb)
TURN_COLORS = {
    "LEFT": "#1f77b4",
    "RIGHT": "#d62728",
    "STRAIGHT": "#2ca02c",
    "UTURN": "#9467bd",
    "NONE": "#999999",
}

# Decision-node importance palette (decision_point_velocity_analysis.ipynb)
IMPORTANCE_COLORS = {True: "#2980b9", False: "#e67e22"}

# Behavioral-state palette (state_hmm.ipynb deep-dives)
STATE_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]

PSTH_BAR_COLOR = "lightsteelblue"
PSTH_LINE_COLOR = "steelblue"
EVENT_LINE_COLOR = "red"


def session_colors(n: int, cmap: str = "viridis") -> List:
    """A chronological color ramp with *n* entries (earliest -> latest session)."""
    cm = plt.get_cmap(cmap)
    return [cm(i / max(n - 1, 1)) for i in range(n)]


def plot_psth(
    times: Sequence[float],
    rate: Sequence[float],
    smoothed: Optional[Sequence[float]] = None,
    ax: Optional[plt.Axes] = None,
    event_time: float = 0.0,
    ylabel: str = "Firing rate (Hz)",
    title: Optional[str] = None,
) -> plt.Axes:
    """Draw the canonical PSTH: light bars + smoothed line + dashed event marker."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 3))
    times = np.asarray(times, dtype=float)
    rate = np.asarray(rate, dtype=float)
    width = float(np.median(np.diff(times))) if len(times) > 1 else 0.05
    ax.bar(times, rate, width=width, color=PSTH_BAR_COLOR, alpha=0.6, align="center")
    if smoothed is not None:
        ax.plot(times, np.asarray(smoothed, dtype=float), color=PSTH_LINE_COLOR, lw=1.6)
    ax.axvline(event_time, color=EVENT_LINE_COLOR, ls="--", lw=1.0)
    ax.set_xlabel("Time from event (s)")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=9)
    return ax


__all__ = [
    "TURN_COLORS",
    "IMPORTANCE_COLORS",
    "STATE_COLORS",
    "PSTH_BAR_COLOR",
    "PSTH_LINE_COLOR",
    "EVENT_LINE_COLOR",
    "session_colors",
    "plot_psth",
]
