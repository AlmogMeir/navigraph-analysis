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


# ---------------------------------------------------------------------------
# Directional / scoped speed around a node visit
# ---------------------------------------------------------------------------

#: The four directional x scope speed columns :func:`visit_speed_metrics` adds,
#: plus the symmetric window, in display order.
SPEED_METRIC_COLS = [
    "mean_speed",
    "speed_enter_excl",
    "speed_enter_incl",
    "speed_exit_excl",
    "speed_exit_incl",
]

#: Human-readable labels for the four directional metrics.
SPEED_METRIC_LABELS = {
    "speed_enter_excl": "Entering - excl. node\n(pure approach edge)",
    "speed_enter_incl": "Entering - incl. node\n(to turn point)",
    "speed_exit_excl": "Exiting - excl. node\n(pure departure edge)",
    "speed_exit_incl": "Exiting - incl. node\n(from turn point)",
}


def _mean_speed_in(mid_frames: np.ndarray, speed: np.ndarray, lo: float, hi: float) -> float:
    """Mean of speed samples whose frame falls in ``[lo, hi]``.

    NaN when the range is empty or degenerate, so a window that falls off the end
    of the session reads as missing rather than as zero speed.
    """
    if lo >= hi or len(mid_frames) == 0:
        return np.nan
    seg = speed[(mid_frames >= lo) & (mid_frames <= hi)]
    return float(np.nanmean(seg)) if np.any(~np.isnan(seg)) else np.nan


def visit_speed_metrics(
    node_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    window_frames: int = 20,
    fps: float = 40.0,
    bodypart: str = "headstage",
    smooth_window_frames: int = 5,
    exclude_inferred: bool = True,
) -> pd.DataFrame:
    """Per node-visit speed, split by direction (enter/exit) and scope (incl/excl node).

    For each visit, speed is computed once over ``[start - window_frames,
    end + window_frames]`` and then averaged over five sub-windows: the symmetric
    one centred on the turn point, and the four in the table above.

    Parameters
    ----------
    window_frames:
        How far either side of the node's dwell window to extend the speed trace.
    smooth_window_frames:
        Passed to :func:`~navigraph_analysis.behavior.speed.compute_path_speed`.

    Returns
    -------
    One row per node-visit: ``node_visit_idx, node_name, start_frame, end_frame,
    turn_frame`` plus :data:`SPEED_METRIC_COLS`.
    """
    x_col = resolve(summary_df, bodypart, "x")
    y_col = resolve(summary_df, bodypart, "y")
    frame_col = resolve(summary_df, "frame_idx_global")
    max_frame = int(pd.to_numeric(summary_df[frame_col], errors="coerce").max())

    df = node_df.copy()
    if exclude_inferred and "is_inferred" in df.columns:
        df = df[~df["is_inferred"].fillna(False).astype(bool)]

    rows: List[dict] = []
    for r in df.itertuples(index=False):
        s_fr, e_fr = int(r.start_frame), int(r.end_frame)
        turn_fr = (s_fr + e_fr) / 2.0
        sf = max(0, s_fr - window_frames)
        ef = min(max_frame, e_fr + window_frames)

        out = {c: np.nan for c in SPEED_METRIC_COLS}
        try:
            times_s, speed, _ = compute_path_speed(
                summary_df, sf, ef, fps=fps,
                x_col=x_col, y_col=y_col, frame_col=frame_col,
                smooth_window_frames=smooth_window_frames,
            )
            if len(speed):
                mid_frames = np.asarray(times_s) * fps + sf
                out["mean_speed"] = _mean_speed_in(
                    mid_frames, speed, turn_fr - window_frames, turn_fr + window_frames)
                out["speed_enter_excl"] = _mean_speed_in(mid_frames, speed, sf, s_fr)
                out["speed_enter_incl"] = _mean_speed_in(mid_frames, speed, sf, turn_fr)
                out["speed_exit_excl"] = _mean_speed_in(mid_frames, speed, e_fr, ef)
                out["speed_exit_incl"] = _mean_speed_in(mid_frames, speed, turn_fr, ef)
        except (KeyError, ValueError, IndexError):
            pass

        rows.append({
            "node_visit_idx": getattr(r, "node_visit_idx", np.nan),
            "node_name": r.node_name,
            "start_frame": s_fr,
            "end_frame": e_fr,
            "turn_frame": turn_fr,
            **out,
        })
    return pd.DataFrame(rows)


def paired_wilcoxon(
    df: pd.DataFrame,
    col_a: str,
    col_b: str,
    label_a: Optional[str] = None,
    label_b: Optional[str] = None,
) -> Dict[str, float]:
    """Wilcoxon signed-rank between two metrics of the *same* visits.

    Paired rather than independent: ``col_a`` and ``col_b`` are two measurements
    of one row, so the visit itself is controlled for.
    """
    from scipy.stats import wilcoxon

    sub = df[[col_a, col_b]].dropna()
    result = {
        "label_a": label_a or col_a,
        "label_b": label_b or col_b,
        "n": int(len(sub)),
        "median_a": np.nan,
        "median_b": np.nan,
        "W": np.nan,
        "p": np.nan,
    }
    if len(sub) < 2:
        return result
    result["median_a"] = float(sub[col_a].median())
    result["median_b"] = float(sub[col_b].median())
    try:
        W, p = wilcoxon(sub[col_a], sub[col_b])
        result["W"], result["p"] = float(W), float(p)
    except ValueError:
        pass  # all differences zero -> no test to run
    return result


def speed_breakdown(
    df: pd.DataFrame,
    group_col: str,
    groups: Optional[Iterable] = None,
    metrics: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Per group and metric: n, mean, median and SEM, plus the between-group test.

    *group_col* is any categorical split of the visits (important vs not, altering
    vs leaving, toward vs away from reward). Returns one row per
    ``(metric, group)`` with a ``p_vs_first`` column carrying the Mann-Whitney p
    against the first group.
    """
    from scipy.stats import mannwhitneyu

    metrics = list(metrics) if metrics is not None else [
        c for c in SPEED_METRIC_COLS if c in df.columns
    ]
    groups = list(groups) if groups is not None else list(pd.unique(df[group_col].dropna()))

    rows: List[dict] = []
    for metric in metrics:
        ref = (df.loc[df[group_col] == groups[0], metric].dropna()
               if groups else pd.Series(dtype=float))
        for g in groups:
            vals = df.loc[df[group_col] == g, metric].dropna()
            p = np.nan
            if g != groups[0] and len(vals) >= 2 and len(ref) >= 2:
                p = float(mannwhitneyu(ref, vals, alternative="two-sided")[1])
            rows.append({
                "metric": metric,
                "label": SPEED_METRIC_LABELS.get(metric, metric).replace("\n", " "),
                "group": g,
                "n": int(len(vals)),
                "mean": float(vals.mean()) if len(vals) else np.nan,
                "median": float(vals.median()) if len(vals) else np.nan,
                "sem": float(vals.sem()) if len(vals) > 1 else np.nan,
                "p_vs_first": p,
            })
    return pd.DataFrame(rows)


__all__ = [
    "classify_important_nodes",
    "decision_dwell_speed",
    "compare_important_vs_not",
    "SPEED_METRIC_COLS",
    "SPEED_METRIC_LABELS",
    "visit_speed_metrics",
    "paired_wilcoxon",
    "speed_breakdown",
]
