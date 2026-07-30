"""Maze graph loading and node-name classification.

The "build nx graph from FullMazeGrid.pkl" block was copy-pasted near-verbatim
in change_of_mind, decision_point_velocity, turn_analysis, trial_context_viewer
and traversal_quality (dedup B). Node depth/level/leaf parsing from the
``[L|R][level][pos]`` naming was independently reimplemented too (dedup I). Both
live here now.

    from navigraph_analysis.io.graph import load_maze_graph, node_depth, is_leaf

    G, node_pos = load_maze_graph(cfg.graph_pkl)
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import networkx as nx
import numpy as np

_DEPTH_RE = re.compile(r"^[LR](\d)")
_LEAF_RE = re.compile(r"^[RL]5\d+$")


def load_maze_graph(
    graph_pkl: str | Path,
    directed: bool = False,
) -> Tuple[nx.Graph, Dict[str, Tuple[float, float]]]:
    """Build a maze graph from a mapping pickle (``FullMazeGrid.pkl`` format).

    The pickle stores ``mappings['nodes']`` (node -> list of polygon rings) and
    ``mappings['edges']`` (keys ``'A_B'``). Every node is added with its polygon
    centroid as the ``'pos'`` attribute, and a (bi)directional edge is added for
    each ``A_B`` key.

    Parameters
    ----------
    graph_pkl:
        Path to the mapping pickle.
    directed:
        If True return an ``nx.DiGraph`` with edges in both directions (matches
        the change_of_mind notebook); otherwise an undirected ``nx.Graph``
        (sufficient for tree distances / turn geometry).

    Returns
    -------
    (G, node_pos) where ``node_pos`` maps node -> (x, y) centroid.
    """
    with open(Path(graph_pkl).expanduser(), "rb") as fh:
        data = pickle.load(fh)
    mappings = data["mappings"] if "mappings" in data else data

    G: nx.Graph = nx.DiGraph() if directed else nx.Graph()
    node_pos: Dict[str, Tuple[float, float]] = {}
    for node, rings in mappings["nodes"].items():
        poly = np.asarray(rings[0], dtype=float)
        pos = tuple(poly.mean(axis=0))
        node_pos[node] = pos
        G.add_node(node, pos=pos)

    for edge_key in mappings["edges"]:
        key = str(edge_key)
        if "_" not in key:
            continue
        a, b = key.split("_", 1)
        if a in G and b in G:
            G.add_edge(a, b)
            if directed:
                G.add_edge(b, a)
    return G, node_pos


def node_depth(node_id: object) -> Optional[int]:
    """Tree depth encoded in the node name (``'L31'`` -> 3). None if unparseable."""
    m = _DEPTH_RE.match(str(node_id))
    return int(m.group(1)) if m else None


def is_leaf(node_id: object) -> bool:
    """True for reward-port / leaf nodes named like ``R5xx`` / ``L5xx``."""
    return bool(_LEAF_RE.match(str(node_id)))


def node_side(node_id: object) -> Optional[str]:
    """'left' / 'right' from the L/R prefix, or None."""
    s = str(node_id)
    if s[:1] == "L":
        return "left"
    if s[:1] == "R":
        return "right"
    return None


def maze_level(node_id: object, max_inner_depth: int) -> Optional[int]:
    """Maze level as used in turn_analysis: ``max_inner_depth - depth + 1``."""
    d = node_depth(node_id)
    return None if d is None else max_inner_depth - d + 1


__all__ = [
    "load_maze_graph",
    "node_depth",
    "is_leaf",
    "node_side",
    "maze_level",
]
