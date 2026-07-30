"""Neuropixels spike-data loading (optional neural layer).

Consolidates the spike-loading + reshaping boilerplate that was copy-pasted
across change_of_mind, direct_path_analysis, turn_analysis, speed_unit_correlation
and state_hmm (dedup D): ``scipy.io.loadmat`` of ``aligned_spike_clusters.mat`` /
``aligned_spike_times.mat``, ``cluster_info.tsv`` parsing, ``good``/``mua`` unit
selection, the ``_unwrap_mat`` squeeze helper and the ``FS = 30000`` constant.

    from navigraph_analysis.io.spikes import load_spikes
    sp = load_spikes(session.spikes_dir, fs=cfg.fs)
    sp.spike_times_s          # spike times in seconds
    sp.good_units             # list of cluster ids labelled 'good'
    sp.times_for(cluster_id)  # seconds for one unit
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def _unwrap_mat(value):
    """Squeeze the nested object arrays scipy.io.loadmat returns to a flat ndarray."""
    arr = np.asarray(value)
    while arr.dtype == object and arr.size == 1:
        arr = np.asarray(arr.item())
    return np.squeeze(arr)


@dataclass
class SpikeData:
    """Loaded, clock-agnostic spike data for one session."""

    spike_times_s: np.ndarray          # (n_spikes,) seconds
    spike_clusters: np.ndarray         # (n_spikes,) cluster id per spike
    cluster_info: pd.DataFrame         # phy cluster_info.tsv
    fs: float
    ct_offsets: Optional[np.ndarray] = None

    @property
    def good_units(self) -> List[int]:
        return self._units_with_group("good")

    @property
    def mua_units(self) -> List[int]:
        return self._units_with_group("mua")

    def _units_with_group(self, group: str) -> List[int]:
        ci = self.cluster_info
        if "group" not in ci.columns:
            return []
        id_col = "cluster_id" if "cluster_id" in ci.columns else "id"
        return ci.loc[ci["group"] == group, id_col].astype(int).tolist()

    def times_for(self, cluster_id: int) -> np.ndarray:
        """Spike times (seconds) for a single unit."""
        return self.spike_times_s[self.spike_clusters == cluster_id]


def load_spikes(
    spikes_dir: str | Path,
    fs: float = 30000.0,
    clusters_file: str = "aligned_spike_clusters.mat",
    times_file: str = "aligned_spike_times.mat",
    cluster_info_file: str = "cluster_info.tsv",
) -> SpikeData:
    """Load aligned Neuropixels spikes + cluster metadata from *spikes_dir*.

    Spike times are converted to **seconds** (``/ fs``) if they look like sample
    counts (large integers); values already in seconds pass through.
    """
    from scipy.io import loadmat  # local import: only needed for the neural layer

    d = Path(spikes_dir).expanduser()
    times = _load_mat_array(d / times_file, loadmat)
    clusters = _load_mat_array(d / clusters_file, loadmat)

    times = np.asarray(times, dtype=float).ravel()
    clusters = np.asarray(clusters).ravel()

    # heuristic: sample counts -> seconds
    if times.size and np.nanmax(times) > 1e5:
        times = times / fs

    ci_path = d / cluster_info_file
    cluster_info = pd.read_csv(ci_path, sep="\t") if ci_path.exists() else pd.DataFrame()

    ct_path = _first(d.glob("*_ct_offsets.txt"))
    ct_offsets = np.loadtxt(ct_path) if ct_path is not None else None

    return SpikeData(
        spike_times_s=times,
        spike_clusters=clusters,
        cluster_info=cluster_info,
        fs=fs,
        ct_offsets=ct_offsets,
    )


def _load_mat_array(path: Path, loadmat) -> np.ndarray:
    """Load a single-array .mat, returning the first non-metadata entry unwrapped."""
    mat = loadmat(path, squeeze_me=True)
    for key, val in mat.items():
        if not key.startswith("__"):
            return _unwrap_mat(val)
    raise KeyError(f"No data array found in {path}")


def _first(iterable):
    for item in iterable:
        return item
    return None


__all__ = ["SpikeData", "load_spikes"]
