"""Traversal reconstruction: per-frame graph locations -> a legal node/edge sequence.

Produces the canonical ``traversal_df`` that ``node_tables.build_node_df`` /
``build_node_edge_df`` consume. Adapted and de-coupled from
``navigraph.analysis.metrics.reconstruct_full_path`` and the
``traversal_pipeline.ipynb`` / ``improved_node_df.ipynb`` notebooks:

    Pass 1  collapse the per-frame ``{bp}_graph_node`` / ``{bp}_graph_edge`` columns
            into contiguous *tracked* runs (edge takes priority over node).
    Pass 2  walk the runs and insert graph-legal *inferred* node/edge rows
            (shortest path) wherever consecutive tracked positions are not
            directly connected, so the output is a continuous, alternating
            node -> edge -> node sequence.

Output columns (the traversal_df contract):
    location_type ('node'/'edge'), headstage_graph_node, headstage_graph_edge,
    start_frame_global, end_frame_global, duration, trial_idx, is_inferred.
"""

from __future__ import annotations

import ast
from typing import List, Optional

import networkx as nx
import numpy as np
import pandas as pd

from ..columns import flatten_columns, resolve, try_resolve

TRAVERSAL_COLUMNS = [
    "location_type",
    "headstage_graph_node",
    "headstage_graph_edge",
    "start_frame_global",
    "end_frame_global",
    "duration",
    "trial_idx",
    "is_inferred",
]


def _edge_nodes(edge_value: str) -> List[str]:
    """Parse an edge identifier into its two endpoint node names."""
    s = str(edge_value)
    try:
        if s.startswith(("('", '("', "(")):
            t = ast.literal_eval(s)
            if isinstance(t, (tuple, list)) and len(t) >= 2:
                return [str(t[0]), str(t[1])]
    except (ValueError, SyntaxError):
        pass
    if "_" in s:
        return s.split("_", 1)
    return [s]


def _trim_by_likelihood(
    df: pd.DataFrame,
    like_col,
    threshold: float,
    window: int,
) -> pd.DataFrame:
    """Trim leading/trailing frames until *window* consecutive samples clear *threshold*."""
    lk = pd.to_numeric(df[like_col], errors="coerce").values.astype(float)
    n = len(lk)
    if n <= window:
        return df
    start = 0
    for i in range(n - window):
        if np.all(lk[i : i + window] > threshold):
            start = i
            break
    end = n - 1
    for i in range(n - 1, window - 2, -1):
        if np.all(lk[i - window + 1 : i + 1] > threshold):
            end = i
            break
    return df.iloc[start : end + 1]


def reconstruct_traversal(
    summary_df: pd.DataFrame,
    G: nx.Graph,
    bodypart: str = "headstage",
    min_likelihood: Optional[float] = 0.6,
    consistency_window: int = 20,
) -> pd.DataFrame:
    """Reconstruct a legal node/edge traversal from a frame-level ``summary_df``.

    Parameters
    ----------
    summary_df:
        Frame-level table with per-frame ``{bodypart}_graph_node`` /
        ``{bodypart}_graph_edge`` columns, ``frame_idx_global`` and ``trial_idx``.
    G:
        Maze graph (from :func:`navigraph_analysis.io.graph.load_maze_graph`).
    bodypart:
        Which tracked point drives location.
    min_likelihood:
        If set and a likelihood column exists, trim low-confidence start/end.
    consistency_window:
        Frames that must consecutively clear ``min_likelihood`` to anchor the
        trimmed range.
    """
    # Flatten to plain string columns so array extraction is unambiguous.
    df = flatten_columns(summary_df)
    node_col = resolve(df, bodypart, "graph_node")
    edge_col = try_resolve(df, bodypart, "graph_edge")
    frame_col = resolve(df, "frame_idx_global")
    trial_col = try_resolve(df, "trial_idx")

    df = df.sort_values(frame_col).reset_index(drop=True)

    if min_likelihood is not None:
        like_col = try_resolve(df, bodypart, "likelihood")
        if like_col is not None:
            df = _trim_by_likelihood(df, like_col, min_likelihood, consistency_window)

    frames = df[frame_col].to_numpy()
    nodes_arr = df[node_col].to_numpy(dtype=object)
    edges_arr = df[edge_col].to_numpy(dtype=object) if edge_col is not None else np.full(len(df), None)
    trials_arr = df[trial_col].to_numpy() if trial_col is not None else np.full(len(df), np.nan)

    # -- Pass 1: contiguous tracked runs --------------------------------------
    runs: List[dict] = []
    cur = None
    start_frame = None
    start_trial = None
    prev_frame = None

    for i in range(len(df)):
        frame = frames[i]
        node = nodes_arr[i]
        edge = edges_arr[i]
        trial = trials_arr[i]

        if _isna(edge) and _isna(node):
            continue
        pos = ("edge", str(edge)) if not _isna(edge) else ("node", str(node))

        if pos != cur:
            if cur is not None:
                runs.append(_make_run(cur, start_frame, prev_frame, start_trial))
            cur = pos
            start_frame = frame
            start_trial = trial
        prev_frame = frame

    if cur is not None:
        runs.append(_make_run(cur, start_frame, prev_frame, start_trial))

    # -- Pass 2: fill inferred transitions ------------------------------------
    seq: List[dict] = []
    current_location: Optional[str] = None

    for run in runs:
        pos_type, pos_value = run["location_type"], run["_value"]
        nodes = [pos_value] if pos_type == "node" else _edge_nodes(pos_value)

        if current_location is None:
            current_location = pos_value if pos_type == "node" else (nodes[0] if nodes else None)

        # target node we must be at to legally start this run
        if pos_type == "node":
            target = pos_value
        elif len(nodes) == 2:
            if current_location in nodes:
                target = current_location
            else:
                target = _closest_endpoint(G, current_location, nodes)
        else:
            target = current_location

        if current_location and target and current_location != target \
                and not G.has_edge(current_location, target):
            _append_inferred_path(seq, G, current_location, target, run["start_frame_global"])
            current_location = target

        seq.append({
            "location_type": pos_type,
            "headstage_graph_node": pos_value if pos_type == "node" else np.nan,
            "headstage_graph_edge": pos_value if pos_type == "edge" else np.nan,
            "start_frame_global": run["start_frame_global"],
            "end_frame_global": run["end_frame_global"],
            "duration": run["duration"],
            "trial_idx": run["trial_idx"],
            "is_inferred": False,
        })

        if pos_type == "node":
            current_location = pos_value
        elif len(nodes) == 2:
            current_location = nodes[1] if current_location == nodes[0] else nodes[0]

    out = pd.DataFrame(seq, columns=TRAVERSAL_COLUMNS)
    return out


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_run(pos, start_frame, end_frame, trial) -> dict:
    return {
        "location_type": pos[0],
        "_value": pos[1],
        "start_frame_global": start_frame,
        "end_frame_global": end_frame,
        "duration": int(end_frame - start_frame + 1),
        "trial_idx": trial,
    }


def _append_inferred_path(seq, G, src, dst, frame) -> None:
    """Append inferred node/edge rows following the shortest path src..dst."""
    try:
        path = nx.shortest_path(G, src, dst)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return
    for j in range(len(path) - 1):
        a, b = path[j], path[j + 1]
        if j == 0 and not (seq and seq[-1]["headstage_graph_node"] == a):
            seq.append(_inferred_row("node", a, frame))
        seq.append(_inferred_row("edge", f"('{a}', '{b}')", frame))
        seq.append(_inferred_row("node", b, frame))


def _inferred_row(kind, value, frame) -> dict:
    return {
        "location_type": kind,
        "headstage_graph_node": value if kind == "node" else np.nan,
        "headstage_graph_edge": value if kind == "edge" else np.nan,
        "start_frame_global": frame,
        "end_frame_global": frame,
        "duration": 0,
        "trial_idx": np.nan,
        "is_inferred": True,
    }


def _closest_endpoint(G, current, endpoints):
    best, best_d = endpoints[0], np.inf
    for e in endpoints:
        try:
            d = nx.shortest_path_length(G, current, e)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            d = np.inf
        if d < best_d:
            best, best_d = e, d
    return best


def _isna(v) -> bool:
    return v is None or (isinstance(v, float) and np.isnan(v)) or (isinstance(v, str) and v in ("", "nan", "None"))


__all__ = ["reconstruct_traversal", "TRAVERSAL_COLUMNS"]
