"""PSTH construction.

The bin-and-smooth PSTH machinery (0.05 s bins, Gaussian smoothing) was
duplicated across the neural notebooks (dedup J). ``binned_and_smoothed_rate``
computes a single event-aligned PSTH; ``psth_across_events`` averages it over
many events (mean +/- SEM), ready for :func:`navigraph_analysis.viz.style.plot_psth`.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .clock import gather_relative_spikes


def bin_edges(pre: float, post: float, bin_size: float) -> np.ndarray:
    return np.arange(-pre, post + bin_size, bin_size)


def binned_and_smoothed_rate(
    rel_spike_times: Sequence[float],
    pre: float,
    post: float,
    bin_size: float = 0.05,
    sigma_s: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Histogram *rel_spike_times* into a rate (Hz) and Gaussian-smooth it.

    Returns ``(bin_centers, rate, smoothed)`` in Hz.
    """
    edges = bin_edges(pre, post, bin_size)
    counts, _ = np.histogram(rel_spike_times, bins=edges)
    rate = counts / bin_size
    sigma_bins = max(sigma_s / bin_size, 1e-6)
    smoothed = gaussian_filter1d(rate.astype(float), sigma_bins)
    centers = (edges[:-1] + edges[1:]) / 2.0
    return centers, rate, smoothed


def psth_across_events(
    spike_times_s: Sequence[float],
    event_abs_times: Sequence[float],
    pre: float = 1.0,
    post: float = 1.0,
    bin_size: float = 0.05,
    sigma_s: float = 0.15,
) -> dict:
    """Mean +/- SEM PSTH for one unit across many events.

    Returns ``{centers, mean, sem, smoothed, n_events}`` (rates in Hz).
    """
    edges = bin_edges(pre, post, bin_size)
    centers = (edges[:-1] + edges[1:]) / 2.0
    per_event: List[np.ndarray] = []
    for ev in event_abs_times:
        rel = gather_relative_spikes(spike_times_s, ev, pre, post)
        counts, _ = np.histogram(rel, bins=edges)
        per_event.append(counts / bin_size)
    if not per_event:
        z = np.zeros_like(centers)
        return {"centers": centers, "mean": z, "sem": z, "smoothed": z, "n_events": 0}
    arr = np.vstack(per_event)
    mean = arr.mean(axis=0)
    sem = arr.std(axis=0, ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else np.zeros_like(mean)
    sigma_bins = max(sigma_s / bin_size, 1e-6)
    smoothed = gaussian_filter1d(mean, sigma_bins)
    return {
        "centers": centers,
        "mean": mean,
        "sem": sem,
        "smoothed": smoothed,
        "n_events": len(arr),
    }


__all__ = ["bin_edges", "binned_and_smoothed_rate", "psth_across_events"]
