"""Population firing-rate matrices and PCA.

Consolidates ``build_population_rate_matrix`` + ``run_pca`` (SVD on a z-scored
rate matrix) reused by change_of_mind and speed_unit_correlation.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d


def build_population_rate_matrix(
    spike_times_by_unit: Dict[int, Sequence[float]],
    t_start: float,
    t_end: float,
    bin_size: float = 0.05,
    sigma_s: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Binned firing-rate matrix over a continuous window.

    Returns ``(rate_matrix, bin_centers, unit_ids)`` where ``rate_matrix`` is
    ``(n_units, n_bins)`` in Hz, optionally Gaussian-smoothed along time.
    """
    edges = np.arange(t_start, t_end + bin_size, bin_size)
    centers = (edges[:-1] + edges[1:]) / 2.0
    unit_ids = sorted(spike_times_by_unit.keys())
    rows = []
    sigma_bins = (sigma_s / bin_size) if sigma_s > 0 else 0.0
    for u in unit_ids:
        counts, _ = np.histogram(np.asarray(spike_times_by_unit[u], dtype=float), bins=edges)
        rate = counts / bin_size
        if sigma_bins > 0:
            rate = gaussian_filter1d(rate.astype(float), sigma_bins)
        rows.append(rate)
    mat = np.vstack(rows) if rows else np.empty((0, len(centers)))
    return mat, centers, unit_ids


def zscore_rows(mat: np.ndarray) -> np.ndarray:
    """Z-score each unit (row) across time; constant rows -> zeros."""
    mu = mat.mean(axis=1, keepdims=True)
    sd = mat.std(axis=1, keepdims=True)
    sd = np.where(sd > 1e-9, sd, 1.0)
    return (mat - mu) / sd


def run_pca(rate_matrix: np.ndarray, n_components: int = 3) -> Dict[str, np.ndarray]:
    """PCA via SVD on a z-scored (units x time) matrix.

    Returns ``{components, scores, explained_variance_ratio}`` where ``scores``
    is ``(n_components, n_time)`` — the population trajectory in PC space.
    """
    if rate_matrix.size == 0:
        empty = np.empty((0, 0))
        return {"components": empty, "scores": empty, "explained_variance_ratio": np.array([])}
    z = zscore_rows(rate_matrix)
    # time along columns: decompose z^T = U S Vt so scores live in time
    X = z.T  # (time, units)
    X = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    k = min(n_components, len(S))
    scores = (U[:, :k] * S[:k]).T          # (k, time)
    var = (S ** 2)
    evr = (var / var.sum())[:k] if var.sum() > 0 else np.zeros(k)
    return {
        "components": Vt[:k],              # (k, units)
        "scores": scores,
        "explained_variance_ratio": evr,
    }


__all__ = ["build_population_rate_matrix", "zscore_rows", "run_pca"]
