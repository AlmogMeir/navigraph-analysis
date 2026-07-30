"""Decision-point classification and dwell/speed analysis.

Reusable core of ``decision_point_velocity_analysis.ipynb``: classify each
internal (non-leaf) node as an IMPORTANT decision point — a genuine fork whose
distinct child subtrees each lead to a currently-active reward port — or
NON-IMPORTANT, then compare dwell time and movement speed between the two
classes at each node-visit.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Set

import networkx as nx
import numpy as np
import pandas as pd

from ..columns import resolve
from ..io.graph import node_depth, is_leaf
from .speed import compute_path_speed


def _children(G: nx.Graph, node: str) -> List[str]:
    """Neighbors one level deeper in the tree (away from the root)."""
    d = node_depth(node)
    if d is None:
        return []
    return [nb for nb in G.neighbors(node) if (node_depth(nb) or -1) == d + 1]


def _subtree_nodes(G: nx.Graph, root_child: str, parent: str) -> Set[str]:
    """All nodes in the subtree hanging off ``root_child`` (excluding *parent*)."""
    seen: Set[str] = set()
    stack = [root_child]
    while stack:
        n = stack.pop()
        if n in seen or n == parent:
            continue
        seen.add(n)
        for nb in G.neighbors(n):
            dn, dnb = node_depth(n), node_depth(nb)
            if dn is not None and dnb is not None and dnb == dn + 1:
                stack.append(nb)
    return seen


def classify_important_nodes(
    G: nx.Graph,
    active_ports: Iterable[str],
) -> Dict[str, bool]:
    """Map each internal node -> True if it is an IMPORTANT decision node.

    A node is important when at least two of its child subtrees each contain an
    active reward port (i.e. the animal's choice there commits it toward a
    distinct set of goals).
    """
    active = set(active_ports)
    out: Dict[str, bool] = {}
    for node in G.nodes():
        if is_leaf(node) or node_depth(node) is None:
            continue
        children = _children(G, node)
        if len(children) < 2:
            out[node] = False
            continue
        n_with_reward = 0
        for ch in children:
            sub = _subtree_nodes(G, ch, node)
            if sub & active:
                n_with_reward += 1
        out[node] = n_with_reward >= 2
    return out


def decision_dwell_speed(
    node_df: pd.DataFrame,
    summary_df: Optional[pd.DataFrame],
    important: Dict[str, bool],
    fps: float = 40.0,
    x_col="headstage_x",
    y_col="headstage_y",
    frame_col="frame_idx_global",
    exclude_inferred: bool = True,
) -> pd.DataFrame:
    """Per node-visit dwell + enter/exit speed at decision nodes.

    Returns one row per node-visit that lands on a classified decision node, with
    ``node_name, node_visit_idx, is_important, dwell_s, enter_speed, exit_speed``.
    """
    df = node_df.copy()
    if exclude_inferred and "is_inferred" in df.columns:
        df = df[~df["is_inferred"].fillna(False).astype(bool)]
    df = df[df["node_name"].isin(important.keys())]

    # Resolve real column keys once (handles flat or MultiIndex summary_df).
    have_speed = summary_df is not None
    if have_speed:
        x_col = resolve(summary_df, "headstage", "x")
        y_col = resolve(summary_df, "headstage", "y")
        frame_col = resolve(summary_df, "frame_idx_global")

    rows = []
    for r in df.itertuples(index=False):
        start, end = int(r.start_frame), int(r.end_frame)
        dwell_s = (end - start + 1) / fps
        enter = exit_ = np.nan
        if have_speed:
            _, sp_in, _ = compute_path_speed(
                summary_df, max(start - int(fps // 4), 0), start, fps=fps,
                x_col=x_col, y_col=y_col, frame_col=frame_col)
            _, sp_out, _ = compute_path_speed(
                summary_df, end, end + int(fps // 4), fps=fps,
                x_col=x_col, y_col=y_col, frame_col=frame_col)
            enter = float(np.nanmean(sp_in)) if len(sp_in) else np.nan
            exit_ = float(np.nanmean(sp_out)) if len(sp_out) else np.nan
        rows.append({
            "node_name": r.node_name,
            "node_visit_idx": r.node_visit_idx,
            "is_important": bool(important.get(r.node_name, False)),
            "dwell_s": dwell_s,
            "enter_speed": enter,
            "exit_speed": exit_,
        })
    return pd.DataFrame(rows)


def compare_important_vs_not(metrics_df: pd.DataFrame, value: str = "dwell_s") -> Dict[str, float]:
    """Mann-Whitney U comparison of *value* between important and non-important visits."""
    from scipy.stats import mannwhitneyu

    imp = metrics_df.loc[metrics_df["is_important"], value].dropna()
    non = metrics_df.loc[~metrics_df["is_important"], value].dropna()
    if len(imp) < 2 or len(non) < 2:
        return {"n_important": len(imp), "n_non": len(non), "U": np.nan, "p": np.nan}
    U, p = mannwhitneyu(imp, non, alternative="two-sided")
    return {
        "n_important": int(len(imp)),
        "n_non": int(len(non)),
        "median_important": float(imp.median()),
        "median_non": float(non.median()),
        "U": float(U),
        "p": float(p),
    }


__all__ = [
    "classify_important_nodes",
    "decision_dwell_speed",
    "compare_important_vs_not",
]
