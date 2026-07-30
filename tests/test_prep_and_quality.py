"""Tests for traversal bounce-fixing, node-table building and quality metrics."""

import networkx as nx
import pandas as pd

from navigraph_analysis.prep.node_tables import fix_edge_bounces, build_node_edge_df
from navigraph_analysis.behavior.quality import count_transition_violations


def _toy_traversal():
    # node A -> short edge -> node A (a bounce) -> node B
    return pd.DataFrame([
        {"location_type": "node", "headstage_graph_node": "A", "headstage_graph_edge": None,
         "start_frame_global": 0, "end_frame_global": 5, "duration": 6, "trial_idx": 0, "is_inferred": False},
        {"location_type": "edge", "headstage_graph_node": None, "headstage_graph_edge": "A_B",
         "start_frame_global": 6, "end_frame_global": 8, "duration": 3, "trial_idx": 0, "is_inferred": False},
        {"location_type": "node", "headstage_graph_node": "A", "headstage_graph_edge": None,
         "start_frame_global": 9, "end_frame_global": 12, "duration": 4, "trial_idx": 0, "is_inferred": False},
        {"location_type": "node", "headstage_graph_node": "B", "headstage_graph_edge": None,
         "start_frame_global": 13, "end_frame_global": 20, "duration": 8, "trial_idx": 0, "is_inferred": False},
    ])


def test_fix_edge_bounces_collapses_triple():
    trav = _toy_traversal()
    fixed = fix_edge_bounces(trav, bounce_threshold=5)
    # the A-edge-A triple collapses into a single inferred A row
    a_rows = fixed[fixed["headstage_graph_node"] == "A"]
    assert len(a_rows) == 1
    assert bool(a_rows.iloc[0]["is_inferred"]) is True
    assert int(a_rows.iloc[0]["duration"]) == 6 + 3 + 4


def test_build_node_edge_df_marks_ports():
    trav = _toy_traversal()
    G = nx.Graph([("A", "B")])
    port_map = {"A": "Target1", "B": "Target2"}
    ne = build_node_edge_df(trav, port_map, G, fps=40, bounce_threshold=5)
    assert "is_port_node" in ne.columns
    assert ne.loc[ne["headstage_graph_node"] == "B", "is_port_node"].all()


def test_count_transition_violations():
    G = nx.Graph([("A", "B"), ("B", "C")])
    trav = pd.DataFrame([
        {"location_type": "node", "headstage_graph_node": n, "is_inferred": False}
        for n in ["A", "B", "C", "A"]   # C->A is not an edge -> 1 violation
    ])
    res = count_transition_violations(trav, G)
    assert res["n_transitions"] == 3
    assert res["n_violations"] == 1
