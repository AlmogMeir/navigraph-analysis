"""Speed analysis utilities.

Ported verbatim (de-coupled from ``navigraph``) from
``navigraph/analysis/speed_analysis.py`` — no ``navigraph.core`` dependencies.

Compute and plot mouse movement speed from coordinate tracking, with support
for noise smoothing, a secondary node axis, a PSTH twin-axis overlay, and
mean +/- SEM across time-normalized path instances.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

VIDEO_FPS: float = 40.0
SMOOTH_WINDOW_FRAMES: int = 5
N_GRID: int = 100


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_col(
    df: pd.DataFrame,
    candidates: List[Union[str, tuple]],
) -> Optional[Union[str, tuple]]:
    """Return the first matching column name from *candidates* (flat or MultiIndex)."""
    cols = df.columns
    if isinstance(cols, pd.MultiIndex):
        for cand in candidates:
            if cand in cols:
                return cand
        low = [str(c).lower() for c in candidates]
        for col in cols:
            parts = [str(v).strip().lower() for v in col]
            for cand in low:
                if cand in parts:
                    return col
        return None
    for cand in candidates:
        if cand in cols:
            return cand
    lower_map = {str(c).strip().lower(): c for c in cols}
    for cand in candidates:
        if str(cand).lower() in lower_map:
            return lower_map[str(cand).lower()]
    return None


def _require_col(
    df: pd.DataFrame,
    name: str,
    candidates: List[Union[str, tuple]],
) -> Union[str, tuple]:
    col = _resolve_col(df, candidates)
    if col is None:
        raise KeyError(
            f"Column '{name}' not found. Tried: {candidates}. "
            f"Available: {list(df.columns[:10])}..."
        )
    return col


def _frame_slice(
    summary_df: pd.DataFrame,
    start_frame: int,
    end_frame: int,
    frame_col: Union[str, tuple],
) -> pd.DataFrame:
    """Return rows where frame_col is in [start_frame, end_frame], sorted."""
    fc = _require_col(summary_df, 'frame_col', [frame_col])
    mask = (summary_df[fc] >= start_frame) & (summary_df[fc] <= end_frame)
    return summary_df[mask].sort_values(fc).copy()


def _nan_aware_uniform_smooth(arr: np.ndarray, window: int) -> np.ndarray:
    """Box-filter mean that skips NaN samples (tracking drop-outs)."""
    nan_mask = np.isnan(arr)
    fill = np.where(nan_mask, 0.0, arr)
    wt = np.where(nan_mask, 0.0, 1.0)
    s_fill = uniform_filter1d(fill, size=window, mode='nearest')
    s_wt = uniform_filter1d(wt, size=window, mode='nearest')
    s_wt_safe = np.where(s_wt > 1e-9, s_wt, 1.0)
    return np.where(s_wt > 1e-9, s_fill / s_wt_safe, np.nan)


def _resample_to_grid(values: np.ndarray, n_grid: int) -> np.ndarray:
    """Linearly interpolate *values* onto *n_grid* uniform points in [0, 1]."""
    values = np.asarray(values, dtype=float)
    L = len(values)
    if L == 0:
        return np.full(n_grid, np.nan)
    if L == 1:
        return np.full(n_grid, values[0])
    return np.interp(
        np.linspace(0, 1, n_grid),
        np.linspace(0, 1, L),
        values,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_path_speed(
    summary_df: pd.DataFrame,
    start_frame: int,
    end_frame: int,
    fps: float = VIDEO_FPS,
    x_col: Union[str, tuple] = 'headstage_x',
    y_col: Union[str, tuple] = 'headstage_y',
    frame_col: Union[str, tuple] = 'frame_idx_global',
    smooth_window_frames: int = SMOOTH_WINDOW_FRAMES,
    pixel_to_unit: Optional[float] = None,
    min_likelihood: Optional[float] = None,
    likelihood_col: Union[str, tuple] = 'headstage_likelihood',
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-frame speed (px/s) along a path segment.

    Returns ``(times_s, speed, raw_speed)``, each of length N-1, where
    ``times_s`` is seconds relative to *start_frame* at consecutive-frame
    midpoints, ``speed`` is the smoothed speed and ``raw_speed`` the unsmoothed.
    Frames with NaN coords or below *min_likelihood* yield NaN speed.
    """
    sub = _frame_slice(summary_df, start_frame, end_frame, frame_col)
    fc = _require_col(sub, 'frame_col', [frame_col])
    xc = _require_col(sub, 'x_col', [x_col])
    yc = _require_col(sub, 'y_col', [y_col])

    frames = sub[fc].values.astype(float)
    x = pd.to_numeric(sub[xc], errors='coerce').values.astype(float)
    y = pd.to_numeric(sub[yc], errors='coerce').values.astype(float)

    if min_likelihood is not None:
        lc = _resolve_col(sub, [likelihood_col])
        if lc is not None:
            lk = pd.to_numeric(sub[lc], errors='coerce').values.astype(float)
            bad = lk < min_likelihood
            x[bad] = np.nan
            y[bad] = np.nan

    if len(frames) < 2:
        return np.array([]), np.array([]), np.array([])

    dx = np.diff(x)
    dy = np.diff(y)
    mid_frames = (frames[:-1] + frames[1:]) / 2.0

    raw_speed = np.sqrt(dx**2 + dy**2) * fps
    if pixel_to_unit is not None:
        raw_speed = raw_speed * pixel_to_unit

    speed = _nan_aware_uniform_smooth(raw_speed, smooth_window_frames)
    times_s = (mid_frames - float(start_frame)) / fps
    return times_s, speed, raw_speed


def compute_mean_path_speed(
    path_instances: List[Dict],
    summary_df: pd.DataFrame,
    n_grid: int = N_GRID,
    fps: float = VIDEO_FPS,
    x_col: Union[str, tuple] = 'headstage_x',
    y_col: Union[str, tuple] = 'headstage_y',
    frame_col: Union[str, tuple] = 'frame_idx_global',
    smooth_window_frames: int = SMOOTH_WINDOW_FRAMES,
    pixel_to_unit: Optional[float] = None,
    min_likelihood: Optional[float] = None,
    likelihood_col: Union[str, tuple] = 'headstage_likelihood',
) -> Dict:
    """Mean +/- SEM speed across path instances on a 0..1 normalized grid.

    Each instance supplies its frame range via ``start_frame``+``end_frame`` or
    ``path_start_abs``+``path_end_abs`` (absolute seconds). Returns a dict with
    ``frac_grid, mean, sem, n, mean_duration_s, all_resampled``.
    """
    resampled: List[np.ndarray] = []
    durations: List[float] = []

    for inst in path_instances:
        if 'start_frame' in inst and 'end_frame' in inst:
            sf = int(inst['start_frame'])
            ef = int(inst['end_frame'])
        elif 'path_start_abs' in inst and 'path_end_abs' in inst:
            sf = round(inst['path_start_abs'] * fps)
            ef = round(inst['path_end_abs'] * fps) - 1
        else:
            continue

        _, speed, _ = compute_path_speed(
            summary_df, sf, ef, fps=fps,
            x_col=x_col, y_col=y_col, frame_col=frame_col,
            smooth_window_frames=smooth_window_frames,
            pixel_to_unit=pixel_to_unit,
            min_likelihood=min_likelihood,
            likelihood_col=likelihood_col,
        )
        if len(speed) == 0:
            continue
        resampled.append(_resample_to_grid(speed, n_grid))
        durations.append((ef - sf + 1) / fps)

    frac = np.linspace(0, 1, n_grid)
    if not resampled:
        nan = np.full(n_grid, np.nan)
        return {
            'frac_grid': frac, 'mean': nan, 'sem': nan,
            'n': 0, 'mean_duration_s': np.nan,
            'all_resampled': np.empty((0, n_grid)),
        }

    arr = np.array(resampled)
    n_valid = np.sum(~np.isnan(arr), axis=0).clip(min=1)
    mean = np.nanmean(arr, axis=0)
    sem = np.nanstd(arr, axis=0, ddof=1) / np.sqrt(n_valid)
    return {
        'frac_grid': frac,
        'mean': mean,
        'sem': sem,
        'n': len(resampled),
        'mean_duration_s': float(np.mean(durations)),
        'all_resampled': arr,
    }


def plot_speed(
    summary_df: pd.DataFrame,
    start_frame: int,
    end_frame: int,
    fps: float = VIDEO_FPS,
    x_col: Union[str, tuple] = 'headstage_x',
    y_col: Union[str, tuple] = 'headstage_y',
    frame_col: Union[str, tuple] = 'frame_idx_global',
    smooth_window_frames: int = SMOOTH_WINDOW_FRAMES,
    pixel_to_unit: Optional[float] = None,
    unit_label: str = 'px/s',
    min_likelihood: Optional[float] = None,
    likelihood_col: Union[str, tuple] = 'headstage_likelihood',
    node_rows: Optional[pd.DataFrame] = None,
    psth_times: Optional[np.ndarray] = None,
    psth_rate: Optional[np.ndarray] = None,
    psth_sem: Optional[np.ndarray] = None,
    psth_label: str = 'Firing rate (Hz)',
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    speed_color: str = 'steelblue',
    psth_color: str = 'darkorange',
    show_raw: bool = False,
) -> plt.Figure:
    """Plot speed for a single path with optional node axis and PSTH overlay."""
    times_s, speed, raw_speed = compute_path_speed(
        summary_df, start_frame, end_frame, fps=fps,
        x_col=x_col, y_col=y_col, frame_col=frame_col,
        smooth_window_frames=smooth_window_frames,
        pixel_to_unit=pixel_to_unit,
        min_likelihood=min_likelihood, likelihood_col=likelihood_col,
    )
    path_duration_s = (end_frame - start_frame + 1) / fps

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 3.5))
    else:
        fig = ax.figure

    ax.plot(times_s, speed, color=speed_color, linewidth=1.5, label='Speed (smoothed)')
    if show_raw:
        ax.plot(times_s, raw_speed, color=speed_color, linewidth=0.6,
                alpha=0.3, linestyle='--', label='Speed (raw)')
    ax.set_xlabel('Time from path start (s)')
    ax.set_ylabel(f'Speed ({unit_label})', color=speed_color)
    ax.tick_params(axis='y', labelcolor=speed_color)
    ax.set_xlim(0, path_duration_s)

    node_tick_pos: List[float] = []
    node_tick_lbl: List[str] = []
    if node_rows is not None and len(node_rows) > 0:
        for _, nr in node_rows.iterrows():
            t_node = (float(nr['start_frame']) - float(start_frame)) / fps
            if 0 <= t_node <= path_duration_s:
                ax.axvline(t_node, color='gray', linewidth=0.7,
                           linestyle='--', alpha=0.55, zorder=0)
                node_tick_pos.append(t_node)
                node_tick_lbl.append(str(nr['node_name']))

    if node_tick_pos:
        ax_top = ax.twiny()
        ax_top.set_xlim(ax.get_xlim())
        ax_top.set_xticks(node_tick_pos)
        ax_top.set_xticklabels(node_tick_lbl, fontsize=7, rotation=45, ha='left')
        ax_top.set_xlabel('Node', fontsize=8)

    if psth_times is not None and psth_rate is not None:
        ax_r = ax.twinx()
        ax_r.plot(psth_times, psth_rate, color=psth_color,
                  linewidth=1.5, label=psth_label)
        if psth_sem is not None:
            ax_r.fill_between(psth_times, psth_rate - psth_sem,
                               psth_rate + psth_sem, color=psth_color, alpha=0.25)
        ax_r.set_ylabel(psth_label, color=psth_color)
        ax_r.tick_params(axis='y', labelcolor=psth_color)

    ax.legend(fontsize=8, loc='upper left')
    if title:
        ax.set_title(title, fontsize=9)
    fig.tight_layout()
    return fig


def plot_mean_speed(
    speed_stats: Dict,
    psth_mean: Optional[np.ndarray] = None,
    psth_sem: Optional[np.ndarray] = None,
    psth_label: str = 'Firing rate (Hz)',
    node_labels: Optional[List[str]] = None,
    node_fracs: Optional[List[float]] = None,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    speed_color: str = 'steelblue',
    psth_color: str = 'darkorange',
    unit_label: str = 'px/s',
    show_individual: bool = False,
) -> plt.Figure:
    """Plot mean +/- SEM speed on normalized path time with optional PSTH overlay."""
    frac = speed_stats['frac_grid']
    mean = speed_stats['mean']
    sem = speed_stats['sem']
    n = speed_stats['n']
    dur = speed_stats['mean_duration_s']

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 3.5))
    else:
        fig = ax.figure

    if show_individual:
        for row in speed_stats['all_resampled']:
            ax.plot(frac, row, color=speed_color, linewidth=0.5, alpha=0.2)

    ax.plot(frac, mean, color=speed_color, linewidth=1.8, label=f'Mean speed (n={n})')
    ax.fill_between(frac, mean - sem, mean + sem,
                    color=speed_color, alpha=0.25, label='+/-SEM')
    ax.set_xlabel(
        f'Normalized path time  (0 = start, 1 = end)  [mean duration {dur:.2f} s]'
    )
    ax.set_ylabel(f'Speed ({unit_label})', color=speed_color)
    ax.tick_params(axis='y', labelcolor=speed_color)
    ax.set_xlim(0, 1)

    if node_labels and node_fracs:
        for nf in node_fracs:
            ax.axvline(nf, color='gray', linewidth=0.7, linestyle='--', alpha=0.5, zorder=0)
        ax_top = ax.twiny()
        ax_top.set_xlim(0, 1)
        ax_top.set_xticks(node_fracs)
        ax_top.set_xticklabels(node_labels, fontsize=7, rotation=45, ha='left')
        ax_top.set_xlabel('Node (canonical)', fontsize=8)

    if psth_mean is not None:
        ax_r = ax.twinx()
        ax_r.plot(frac, psth_mean, color=psth_color, linewidth=1.8, label=psth_label)
        if psth_sem is not None:
            ax_r.fill_between(frac, psth_mean - psth_sem, psth_mean + psth_sem,
                               color=psth_color, alpha=0.25)
        ax_r.set_ylabel(psth_label, color=psth_color)
        ax_r.tick_params(axis='y', labelcolor=psth_color)

    ax.legend(fontsize=8, loc='upper left')
    if title:
        ax.set_title(title, fontsize=9)
    fig.tight_layout()
    return fig


__all__ = [
    'compute_path_speed',
    'compute_mean_path_speed',
    'plot_speed',
    'plot_mean_speed',
]
