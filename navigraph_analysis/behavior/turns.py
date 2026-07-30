"""Turn classification (LEFT / RIGHT / STRAIGHT / UTURN) from graph geometry.

``classify_turn`` / ``turns_from_graph`` are lifted verbatim from
``navigraph/analysis/state_features.py`` (which itself mirrored
``turn_analysis.ipynb``'s ``classify_turn``). ``classify_session_turns`` is the
reusable convenience that turns a node-visit table into a ``turn_df`` — the
structural core the turn_analysis notebook then optionally refines with
trajectory angular-velocity timing.

Geometry note: image coordinates (y increases downward), so ``cross_z > 0`` is a
RIGHT turn (clockwise from above), ``cross_z < 0`` LEFT. ``|angle| < 20`` ->
STRAIGHT, ``|angle| > 160`` -> UTURN.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd

#: Ordered turn categories; NONE covers node-visits with no defined turn.
TURN_CATEGORIES: List[str] = ["LEFT", "RIGHT", "STRAIGHT", "UTURN", "NONE"]


def classify_turn(
    prev_id: object,
    curr_id: object,
    next_id: object,
    node_pos: Dict[object, Tuple[float, float]],
) -> Tuple[str, float]:
    """Signed turn type + angle (degrees) at ``curr_id`` from fixed node positions."""
    px, py = node_pos[prev_id]
    cx, cy = node_pos[curr_id]
    nx_, ny = node_pos[next_id]

    v1 = np.array([cx - px, cy - py], dtype=float)
    v2 = np.array([nx_ - cx, ny - cy], dtype=float)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return "STRAIGHT", 0.0

    v1n, v2n = v1 / n1, v2 / n2
    cross_z = v1n[0] * v2n[1] - v1n[1] * v2n[0]
    dot = np.clip(np.dot(v1n, v2n), -1.0, 1.0)
    angle = float(np.degrees(np.arctan2(cross_z, dot)))

    if abs(angle) < 20:
        return "STRAIGHT", angle
    if abs(angle) > 160:
        return "UTURN", angle
    return ("RIGHT" if cross_z > 0 else "LEFT"), angle


def turns_from_graph(
    node_names: Sequence[object],
    G: "nx.Graph",
) -> Tuple[List[str], List[float]]:
    """Turn type + graph angle for each visit in a session-continuous node sequence.

    NONE only for the first/last visit or a triple where either hop is not a real
    graph edge (traversal violation / bounce artifact).
    """
    pos = nx.get_node_attributes(G, "pos")
    n = len(node_names)
    turn_type = ["NONE"] * n
    angle = [np.nan] * n
    for i in range(1, n - 1):
        prev_n, curr_n, next_n = node_names[i - 1], node_names[i], node_names[i + 1]
        if not all(nd in pos for nd in (prev_n, curr_n, next_n)):
            continue
        if not (G.has_edge(prev_n, curr_n) and G.has_edge(curr_n, next_n)):
            continue
        tt, ang = classify_turn(prev_n, curr_n, next_n, pos)
        turn_type[i] = tt
        angle[i] = ang
    return turn_type, angle


def classify_session_turns(node_df: pd.DataFrame, G: "nx.Graph") -> pd.DataFrame:
    """Return a ``turn_df``: one row per node-visit with turn type + graph angle.

    Walks the whole-session node sequence continuously (not restarted per trial —
    the physical path is continuous across reward boundaries).
    """
    df = node_df.sort_values("node_visit_idx").reset_index(drop=True).copy()
    turn_type, angle = turns_from_graph(df["node_name"].tolist(), G)
    df["turn_type"] = pd.Categorical(turn_type, categories=TURN_CATEGORIES)
    df["graph_angle_deg"] = angle
    df["abs_graph_angle"] = df["graph_angle_deg"].abs()
    keep = [c for c in ("node_visit_idx", "node_name", "trial_idx",
                        "start_frame", "end_frame") if c in df.columns]
    return df[keep + ["turn_type", "graph_angle_deg", "abs_graph_angle"]]


__all__ = ["TURN_CATEGORIES", "classify_turn", "turns_from_graph", "classify_session_turns"]
