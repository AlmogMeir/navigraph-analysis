"""Change-of-mind / path-commitment detection.

Reusable geometric core of ``change_of_mind.ipynb`` (the neural PSTH/PCA parts
live in :mod:`navigraph_analysis.neural`):

- :func:`find_indirect_paths` — long *indirect* port-to-port segments (>= excess
  nodes beyond the graph shortest path).
- :func:`find_commitment_apex` — per indirect segment, the earliest node-visit
  from which the remaining trajectory is the direct shortest path to the
  destination port (the "commitment apex"). The resulting
  ``direct_opportunities`` frame (with ``direct_start_node_visit_idx``) is exactly
  what :func:`navigraph_analysis.behavior.states.add_weak_labels` consumes.
"""

from __future__ import annotations

from typing import List, Optional

import networkx as nx
import numpy as np
import pandas as pd

from .states import _segment_id  # shared contiguous-path-label segmentation


def find_indirect_paths(
    node_df: pd.DataFrame,
    excess_threshold: int = 4,
) -> pd.DataFrame:
    """One row per long-indirect port-to-port segment.

    Columns: ``seg_id, pair_label, src_node, dest_node, start_visit_idx,
    end_visit_idx, n_nodes, shortest_length, path_excess``.
    """
    df = node_df.sort_values("node_visit_idx").reset_index(drop=True).copy()
    df["_seg"] = _segment_id(df["path_pair_label"])

    rows: List[dict] = []
    for seg, sub in df.groupby("_seg"):
        if seg < 0 or len(sub) == 0:
            continue
        if sub.iloc[0].get("port_to_port_path_type") != "indirect":
            continue
        excess = float(sub.iloc[0].get("path_seq_length_unique", np.nan)) - float(
            sub.iloc[0].get("path_shortest_length", np.nan)
        )
        if not (excess >= excess_threshold):
            continue
        rows.append({
            "seg_id": int(seg),
            "pair_label": sub.iloc[0].get("path_pair_label"),
            "src_node": sub.iloc[0]["node_name"],
            "dest_node": sub.iloc[-1]["node_name"],
            "start_visit_idx": int(sub.iloc[0]["node_visit_idx"]),
            "end_visit_idx": int(sub.iloc[-1]["node_visit_idx"]),
            "n_nodes": int(len(sub)),
            "shortest_length": float(sub.iloc[0].get("path_shortest_length", np.nan)),
            "path_excess": excess,
        })
    return pd.DataFrame(rows)


def find_commitment_apex(
    node_df: pd.DataFrame,
    G: nx.Graph,
    excess_threshold: int = 4,
) -> pd.DataFrame:
    """Locate the commitment apex for each long-indirect segment.

    For each indirect segment the apex is the earliest node-visit whose remaining
    (unique-collapsed) node sequence to the destination port is exactly the graph
    shortest path — i.e. from there on the animal takes the direct route.

    Returns a ``direct_opportunities`` frame with one row per segment where an
    apex was found: ``seg_id, dest_node, direct_start_node_visit_idx,
    direct_start_node, nodes_before_commit``.
    """
    df = node_df.sort_values("node_visit_idx").reset_index(drop=True).copy()
    indirect = find_indirect_paths(df, excess_threshold)

    seg_ids = set(indirect["seg_id"].tolist())
    df["_seg"] = _segment_id(df["path_pair_label"])

    out: List[dict] = []
    for seg, sub in df.groupby("_seg"):
        if int(seg) not in seg_ids:
            continue
        sub = sub.reset_index(drop=True)
        dest = sub.iloc[-1]["node_name"]
        names = sub["node_name"].tolist()
        visit_idx = sub["node_visit_idx"].tolist()

        apex_pos: Optional[int] = None
        for i in range(len(sub)):
            remaining = _unique_consecutive(names[i:])
            try:
                sp = nx.shortest_path_length(G, names[i], dest)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
            # remaining trajectory is direct: its unique-node hop count == shortest
            if (len(remaining) - 1) == sp and sp > 0:
                apex_pos = i
                break
        if apex_pos is not None:
            out.append({
                "seg_id": int(seg),
                "dest_node": dest,
                "direct_start_node_visit_idx": int(visit_idx[apex_pos]),
                "direct_start_node": names[apex_pos],
                "nodes_before_commit": int(apex_pos),
            })
    return pd.DataFrame(out)


def _unique_consecutive(seq: List) -> List:
    """Collapse consecutive duplicates in a sequence."""
    out: List = []
    for v in seq:
        if not out or out[-1] != v:
            out.append(v)
    return out


__all__ = ["find_indirect_paths", "find_commitment_apex"]
