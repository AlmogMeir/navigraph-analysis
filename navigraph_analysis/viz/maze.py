"""Maze / graph overlay plotting.

Consolidates the recurring "draw the maze graph with node positions, optionally
colored by a per-node value, with the animal's trajectory or a highlighted path"
figures used across trial_context, turn_analysis and change_of_mind notebooks.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def plot_graph(
    G: nx.Graph,
    node_values: Optional[Dict[str, float]] = None,
    ax: Optional[plt.Axes] = None,
    highlight_path: Optional[Sequence[str]] = None,
    node_size: float = 40,
    cmap: str = "viridis",
    title: Optional[str] = None,
    invert_y: bool = True,
) -> plt.Axes:
    """Draw the maze graph using each node's ``pos`` attribute.

    Parameters
    ----------
    node_values:
        Optional node -> scalar map used to color nodes (e.g. visit counts,
        state id). Nodes absent from the map are drawn grey.
    highlight_path:
        Optional node sequence drawn as a thick line over the edges.
    invert_y:
        Invert the y-axis to match image coordinates (default True).
    """
    pos = nx.get_node_attributes(G, "pos")
    if not pos:
        raise ValueError("Graph has no 'pos' node attribute; use io.graph.load_maze_graph")
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    for a, b in G.edges():
        if a in pos and b in pos:
            (xa, ya), (xb, yb) = pos[a], pos[b]
            ax.plot([xa, xb], [ya, yb], color="0.8", lw=1.0, zorder=1)

    xs = [pos[n][0] for n in G.nodes() if n in pos]
    ys = [pos[n][1] for n in G.nodes() if n in pos]
    if node_values is not None:
        vals = [node_values.get(n, np.nan) for n in G.nodes() if n in pos]
        sc = ax.scatter(xs, ys, c=vals, s=node_size, cmap=cmap, zorder=2)
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    else:
        ax.scatter(xs, ys, s=node_size, color="#4C72B0", zorder=2)

    if highlight_path:
        px = [pos[n][0] for n in highlight_path if n in pos]
        py = [pos[n][1] for n in highlight_path if n in pos]
        ax.plot(px, py, color="#d62728", lw=2.5, alpha=0.8, zorder=3)

    if invert_y:
        ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=10)
    return ax


def plot_trajectory(
    x: Sequence[float],
    y: Sequence[float],
    ax: Optional[plt.Axes] = None,
    background=None,
    cmap: str = "plasma",
    title: Optional[str] = None,
) -> plt.Axes:
    """Plot an XY trajectory colored by time (dark=start -> bright=end).

    *background* may be an image array (e.g. the session frame) drawn beneath.
    """
    from matplotlib.collections import LineCollection

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    if background is not None:
        ax.imshow(background)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    pts = np.array([x, y]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap=cmap, array=np.linspace(0, 1, len(segs)), lw=1.5)
    ax.add_collection(lc)
    ax.set_xlim(np.nanmin(x), np.nanmax(x))
    ax.set_ylim(np.nanmax(y), np.nanmin(y))
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=10)
    return ax


__all__ = ["plot_graph", "plot_trajectory"]
