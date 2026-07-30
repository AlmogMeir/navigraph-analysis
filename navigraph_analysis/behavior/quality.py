"""Traversal-quality metrics.

Reusable core of ``traversal_quality_analysis.ipynb`` /
``traversal_correction_audit.ipynb``: summarize path directness and transition
legality, grouped by port-to-port pair — the inputs to the quality heatmaps.
"""

from __future__ import annotations

from typing import Dict

import networkx as nx
import numpy as np
import pandas as pd


def quality_by_port_pair(node_df: pd.DataFrame) -> pd.DataFrame:
    """Per port-to-port pair: path count, fraction direct, mean excess nodes.

    Uses one row per port-to-port segment (the first node-visit of each labelled
    segment), so repeated segment annotations don't inflate counts.
    """
    df = node_df.dropna(subset=["path_pair_label"]).copy()
    if df.empty:
        return pd.DataFrame(columns=["pair_label", "n_paths", "frac_direct", "mean_excess"])

    # collapse each labelled segment to its first row
    df["_seg"] = (df["path_pair_label"].astype(str) != df["path_pair_label"].astype(str).shift()).cumsum()
    seg = df.groupby("_seg").first()

    seg["excess"] = seg["path_seq_length_unique"] - seg["path_shortest_length"]
    seg["is_direct"] = seg["port_to_port_path_type"] == "direct"

    out = (
        seg.groupby(seg["path_pair_label"].astype(str))
        .agg(
            n_paths=("is_direct", "size"),
            frac_direct=("is_direct", "mean"),
            mean_excess=("excess", "mean"),
        )
        .reset_index()
        .rename(columns={"path_pair_label": "pair_label"})
        .sort_values("pair_label")
        .reset_index(drop=True)
    )
    return out


def count_transition_violations(traversal_df: pd.DataFrame, G: nx.Graph) -> Dict[str, int]:
    """Count node->node transitions in the traversal that are not real graph edges.

    A low count indicates a clean, graph-legal traversal. Only *tracked* node
    rows are considered (inferred rows are legal by construction).
    """
    nodes = traversal_df[traversal_df["location_type"] == "node"]
    if "is_inferred" in nodes.columns:
        nodes = nodes[~nodes["is_inferred"].fillna(False).astype(bool)]
    seq = nodes["headstage_graph_node"].tolist()

    total = 0
    violations = 0
    for a, b in zip(seq, seq[1:]):
        if a == b:
            continue
        total += 1
        if not G.has_edge(a, b):
            violations += 1
    return {
        "n_transitions": total,
        "n_violations": violations,
        "violation_rate": (violations / total) if total else 0.0,
    }


def session_quality_summary(node_df: pd.DataFrame, traversal_df: pd.DataFrame, G: nx.Graph) -> Dict:
    """Compact per-session quality summary combining directness + legality."""
    by_pair = quality_by_port_pair(node_df)
    viol = count_transition_violations(traversal_df, G)
    return {
        "n_port_pairs": int(len(by_pair)),
        "overall_frac_direct": float(by_pair["frac_direct"].mean()) if len(by_pair) else np.nan,
        "mean_excess": float(by_pair["mean_excess"].mean()) if len(by_pair) else np.nan,
        **viol,
    }


__all__ = [
    "quality_by_port_pair",
    "count_transition_violations",
    "session_quality_summary",
]
