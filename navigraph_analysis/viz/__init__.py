"""Plotting: shared style + maze overlays + rasters + cross-session annotation."""

from .style import (
    TURN_COLORS,
    IMPORTANCE_COLORS,
    STATE_COLORS,
    session_colors,
    plot_psth,
)
from .maze import plot_graph, plot_trajectory
from .raster import (
    per_trial_reward,
    rolling_reward_rate,
    plot_reward_raster,
    plot_rolling_performance,
)
from .trial_context import trial_context_figure
from .sessions import annotate_date_ranges, mark_sunday_ticks, annotation_blocks

__all__ = [
    "TURN_COLORS",
    "IMPORTANCE_COLORS",
    "STATE_COLORS",
    "session_colors",
    "plot_psth",
    "plot_graph",
    "plot_trajectory",
    "per_trial_reward",
    "rolling_reward_rate",
    "plot_reward_raster",
    "plot_rolling_performance",
    "trial_context_figure",
    "annotate_date_ranges",
    "mark_sunday_ticks",
    "annotation_blocks",
]
