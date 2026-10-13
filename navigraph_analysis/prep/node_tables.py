"""Builders for the canonical per-session node/edge traversal DataFrames.

Ported verbatim (de-coupled from ``navigraph``) from
``navigraph/analysis/traversal_builder.py`` — this module had no
``navigraph.core`` dependencies, so it moves across unchanged.

build_node_df       — node-only view with consecutive-visit merging,
                      reward/lick lookup, and port-to-port path-type annotations.
build_node_edge_df  — full node+edge view with short edge-bounce collapsing
                      and port-to-port path-type annotations.
fix_edge_bounces    — standalone utility used by both builders.
"""
from __future__ import annotations

from itertools import combinations
from typing import Dict, Optional

import networkx as nx
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Internal helper: MultiIndex-safe column resolution
# ---------------------------------------------------------------------------

def _resolve_summary_column(df: pd.DataFrame, base_name: str):
    """Return the column key in *df* whose first element matches *base_name*.

    Handles both flat and MultiIndex column structures.
    Raises KeyError when nothing matches.
    """
    if isinstance(df.columns, pd.MultiIndex):
        exact = (base_name, "")
        if exact in df.columns:
            return exact
        for col in df.columns:
            if isinstance(col, tuple) and str(col[0]) == base_name:
                return col
            if base_name in " | ".join(str(p) for p in col):
                return col
    else:
        if base_name in df.columns:
            return base_name
        for col in df.columns:
            if base_name in str(col):
                return col
    raise KeyError(f"Could not resolve summary_df column for '{base_name}'")


# ---------------------------------------------------------------------------
# Public: fix_edge_bounces
# ---------------------------------------------------------------------------

def fix_edge_bounces(
    traversal_df: pd.DataFrame,
    bounce_threshold: int = 5,
) -> pd.DataFrame:
    """Collapse short node->edge->same_node triples into a single node row.

    A triple (rows i, i+1, i+2) is eligible when:
    - rows[i]   is a node visit  (location_type == 'node'),
    - rows[i+1] is an edge visit (location_type == 'edge')
                with duration <= bounce_threshold,
    - rows[i+2] is a node visit to the *same* node as rows[i].

    The merged row keeps rows[i]'s node identity, spans from
    rows[i].start_frame_global to rows[i+2].end_frame_global, and has
    duration = sum of all three rows.  is_inferred is set to True.

    Applied iteratively until no more eligible triples remain (handles
    chained bounces such as node -> edge -> same_node -> edge -> same_node).

    Returns a new DataFrame with bounces collapsed. Input is never mutated.
    """
    df = traversal_df.copy().reset_index(drop=True)

    changed = True
    while changed:
        changed = False
        rows = df.to_dict("records")
        merged: list[dict] = []
        i = 0
        while i < len(rows):
            if (
                i + 2 < len(rows)
                and rows[i]["location_type"] == "node"
                and rows[i + 1]["location_type"] == "edge"
                and rows[i + 2]["location_type"] == "node"
                and rows[i]["headstage_graph_node"] == rows[i + 2]["headstage_graph_node"]
                and rows[i + 1]["duration"] <= bounce_threshold
            ):
                combined = dict(rows[i])
                combined["end_frame_global"] = rows[i + 2]["end_frame_global"]
                combined["duration"] = (
                    rows[i]["duration"]
                    + rows[i + 1]["duration"]
                    + rows[i + 2]["duration"]
                )
                combined["is_inferred"] = True
                merged.append(combined)
                i += 3
                changed = True
            else:
                merged.append(rows[i])
                i += 1
        df = pd.DataFrame(merged).reset_index(drop=True)

    return df


# ---------------------------------------------------------------------------
# Internal: path-type annotation (shared by both builders)
# ---------------------------------------------------------------------------

def _annotate_path_types(
    df: pd.DataFrame,
    port_map: Dict[str, str],
    G: nx.Graph,
    fps: float,
    *,
    node_name_col: str,
    start_frame_col: str,
    visit_idx_col: Optional[str],
) -> pd.DataFrame:
    """Add port-to-port path-type columns to *df* (on a copy).

    Works for both node_df (visit_idx_col is a column name) and node_edge_df
    (visit_idx_col is None; node count is computed from location_type).

    Added columns: path_start_time, port_to_port_path_type, path_pair_label,
    path_seq_length_unique, path_shortest_length.
    """
    df = df.copy()

    df["path_start_time"] = pd.NA
    df["port_to_port_path_type"] = None
    df["path_pair_label"] = None
    df["path_seq_length_unique"] = pd.NA
    df["path_shortest_length"] = pd.NA

    # Build valid-pair set (both directions, cross-target only)
    valid_pairs: set[tuple[str, str]] = set()
    for a, b in combinations(port_map.keys(), 2):
        if port_map[a] != port_map[b]:
            valid_pairs.add((a, b))
            valid_pairs.add((b, a))

    if not valid_pairs:
        return df

    if "location_type" in df.columns:
        is_node_row = df["location_type"] == "node"
    else:
        is_node_row = pd.Series(True, index=df.index)

    is_port_row = is_node_row & df[node_name_col].isin(port_map)
    port_indices = df.index[is_port_row].tolist()

    if len(port_indices) < 2:
        return df

    for i in range(1, len(port_indices)):
        pi_prev = port_indices[i - 1]
        pi_curr = port_indices[i]

        src = df.loc[pi_prev, node_name_col]
        dest = df.loc[pi_curr, node_name_col]

        if (src, dest) not in valid_pairs:
            continue

        if visit_idx_col is not None:
            seq_length_unique = int(
                df.loc[pi_curr, visit_idx_col] - df.loc[pi_prev, visit_idx_col]
            )
        else:
            segment_mask = (df.index >= pi_prev) & (df.index <= pi_curr)
            seq_length_unique = int((is_node_row & segment_mask).sum()) - 1

        try:
            shortest_dist = nx.shortest_path_length(G, source=src, target=dest)
        except nx.NetworkXNoPath:
            shortest_dist = 8

        path_type = "direct" if seq_length_unique <= shortest_dist else "indirect"
        pair_label = (port_map[src], port_map[dest])
        seg_start_time = df.loc[pi_prev, start_frame_col] / fps

        seg_idx = df.loc[pi_prev:pi_curr].index
        df.loc[seg_idx, "path_start_time"] = seg_start_time
        df.loc[seg_idx, "port_to_port_path_type"] = path_type
        df.loc[seg_idx, "path_pair_label"] = pd.Series(
            [pair_label] * len(seg_idx), index=seg_idx, dtype=object
        )
        df.loc[seg_idx, "path_seq_length_unique"] = seq_length_unique
        df.loc[seg_idx, "path_shortest_length"] = shortest_dist

    return df


# ---------------------------------------------------------------------------
# Public: build_node_edge_df
# ---------------------------------------------------------------------------

def build_node_edge_df(
    traversal_df: pd.DataFrame,
    port_map: Dict[str, str],
    G: nx.Graph,
    fps: float = 40.0,
    bounce_threshold: int = 5,
) -> pd.DataFrame:
    """Build the canonical node+edge traversal DataFrame.

    Steps: (1) collapse short edge bounces, (2) mark port node rows, (3) annotate
    port-to-port path segments with direct/indirect classification.
    """
    df = fix_edge_bounces(traversal_df, bounce_threshold)

    df["is_port_node"] = (
        (df["location_type"] == "node")
        & df["headstage_graph_node"].isin(port_map)
    )

    df = _annotate_path_types(
        df,
        port_map,
        G,
        fps,
        node_name_col="headstage_graph_node",
        start_frame_col="start_frame_global",
        visit_idx_col=None,
    )

    return df


# ---------------------------------------------------------------------------
# Public: build_node_df
# ---------------------------------------------------------------------------

def build_node_df(
    traversal_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    port_map: Dict[str, str],
    G: nx.Graph,
    fps: float = 40.0,
    bounce_threshold: int = 5,
) -> pd.DataFrame:
    """Build the canonical node-only traversal DataFrame.

    Output columns (in order): node_visit_idx, node_name, start_frame, end_frame,
    duration_frames, trial_idx, is_inferred, reward_during_node,
    reward_size_during_node, lick_during_node, lick_count_during_node,
    is_port_node, port_to_port_path_type, path_pair_label,
    path_seq_length_unique, path_shortest_length, path_start_time.
    """
    df_fixed = fix_edge_bounces(traversal_df, bounce_threshold)
    nodes = df_fixed[df_fixed["location_type"] == "node"].copy().reset_index(drop=True)

    # Merge consecutive visits to the same node, matching
    # navigraph.analysis.traversal_builder.build_node_df. Note this also merges a
    # node -> edge -> same-node excursion, however long the excursion was; see
    # KNOWN_DIFFERENCES.md for the measured frequency and why it is left as-is.
    nodes["_stay_id"] = (
        nodes["headstage_graph_node"] != nodes["headstage_graph_node"].shift(1)
    ).cumsum()

    agg = (
        nodes.groupby("_stay_id", sort=False)
        .agg(
            node_name=("headstage_graph_node", "first"),
            start_frame=("start_frame_global", "min"),
            end_frame=("end_frame_global", "max"),
            duration_frames=("duration", "sum"),
            trial_idx=("trial_idx", "first"),
            is_inferred=("is_inferred", "any"),
        )
        .reset_index(drop=True)
    )

    try:
        frame_col = _resolve_summary_column(summary_df, "frame_idx_global")
        reward_col = _resolve_summary_column(summary_df, "reward")
        reward_size_col = _resolve_summary_column(summary_df, "reward_size")
        lick_col = _resolve_summary_column(summary_df, "lick")
    except KeyError as exc:
        raise KeyError(
            f"summary_df is missing an expected column: {exc}. "
            "Ensure it contains frame_idx_global, reward, reward_size, lick."
        ) from exc

    reward_during: list = []
    reward_size_during: list = []
    lick_during: list = []
    lick_count_list: list = []

    for row in agg.itertuples(index=False):
        mask = (
            (summary_df[frame_col] >= row.start_frame)
            & (summary_df[frame_col] <= row.end_frame)
        )
        overlap = summary_df[mask]

        if len(overlap) == 0:
            reward_during.append(np.nan)
            reward_size_during.append(np.nan)
            lick_during.append(False)
            lick_count_list.append(0)
            continue

        r_vals = overlap[reward_col].dropna()
        reward_during.append(float(r_vals.iloc[0]) if len(r_vals) > 0 else np.nan)

        s_vals = overlap[reward_size_col].dropna()
        reward_size_during.append(float(s_vals.iloc[0]) if len(s_vals) > 0 else np.nan)

        l_vals = overlap[lick_col]
        has_lick = bool(l_vals.any())
        lick_during.append(has_lick)
        lick_count_list.append(int(l_vals.sum()) if has_lick else 0)

    agg["reward_during_node"] = reward_during
    agg["reward_size_during_node"] = reward_size_during
    agg["lick_during_node"] = lick_during
    agg["lick_count_during_node"] = lick_count_list

    agg["is_port_node"] = agg["node_name"].isin(port_map)
    agg["node_visit_idx"] = np.arange(len(agg))

    agg = _annotate_path_types(
        agg,
        port_map,
        G,
        fps,
        node_name_col="node_name",
        start_frame_col="start_frame",
        visit_idx_col="node_visit_idx",
    )

    col_order = [
        "node_visit_idx", "node_name", "start_frame", "end_frame",
        "duration_frames", "trial_idx", "is_inferred",
        "reward_during_node", "reward_size_during_node",
        "lick_during_node", "lick_count_during_node", "is_port_node",
        "port_to_port_path_type", "path_pair_label", "path_seq_length_unique",
        "path_shortest_length", "path_start_time",
    ]
    return agg[col_order]


__all__ = ["fix_edge_bounces", "build_node_edge_df", "build_node_df"]
