"""Graph-node place fields.

A lightweight version of the place-cell analysis: per-node firing-rate maps
(spikes divided by occupancy time at each graph node), plus a spatial-information
score. Occupancy comes from the node-visit table's frame windows.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import pandas as pd

from .clock import frame_to_abs


def node_occupancy_seconds(node_df: pd.DataFrame, fps: float = 40.0) -> Dict[str, float]:
    """Total dwell time (s) per node across the session."""
    dur = node_df.groupby("node_name")["duration_frames"].sum() / fps
    return dur.to_dict()


def node_rate_map(
    spike_times_s: Sequence[float],
    node_df: pd.DataFrame,
    fps: float = 40.0,
) -> pd.DataFrame:
    """Per-node firing rate (Hz) for one unit.

    Returns ``node_name, occupancy_s, n_spikes, rate_hz`` sorted by rate.
    """
    st = np.sort(np.asarray(spike_times_s, dtype=float))
    counts: Dict[str, int] = {}
    for r in node_df.itertuples(index=False):
        a = frame_to_abs(r.start_frame, fps)
        b = frame_to_abs(r.end_frame, fps)
        n = int(np.searchsorted(st, b, "right") - np.searchsorted(st, a, "left"))
        counts[r.node_name] = counts.get(r.node_name, 0) + n

    occ = node_occupancy_seconds(node_df, fps)
    rows = []
    for node, o in occ.items():
        ns = counts.get(node, 0)
        rows.append({
            "node_name": node,
            "occupancy_s": o,
            "n_spikes": ns,
            "rate_hz": (ns / o) if o > 0 else np.nan,
        })
    return pd.DataFrame(rows).sort_values("rate_hz", ascending=False).reset_index(drop=True)


def spatial_information(rate_map: pd.DataFrame) -> float:
    """Skaggs spatial information (bits/spike) from a per-node rate map."""
    df = rate_map.dropna(subset=["rate_hz"])
    if df.empty:
        return np.nan
    occ = df["occupancy_s"].to_numpy(dtype=float)
    rate = df["rate_hz"].to_numpy(dtype=float)
    p = occ / occ.sum()
    mean_rate = np.sum(p * rate)
    if mean_rate <= 0:
        return np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = p * (rate / mean_rate) * np.log2(rate / mean_rate)
    return float(np.nansum(terms))


__all__ = ["node_occupancy_seconds", "node_rate_map", "spatial_information"]
