"""Tests for graph node classification and turn geometry."""

import networkx as nx
import numpy as np

from navigraph_analysis.io.graph import node_depth, is_leaf, node_side
from navigraph_analysis.behavior.turns import classify_turn, turns_from_graph


def test_node_classification():
    assert node_depth("L31") == 3
    assert node_depth("R0") == 0
    assert node_depth("weird") is None
    assert is_leaf("R522") and is_leaf("L510")
    assert not is_leaf("R11")
    assert node_side("L21") == "left" and node_side("R21") == "right"


def test_classify_turn_left_right_straight():
    # coordinates chosen in image space (y down): moving +x then turning +y is RIGHT
    pos = {"a": (0.0, 0.0), "b": (1.0, 0.0), "c": (2.0, 1.0)}
    kind, ang = classify_turn("a", "b", "c", pos)
    assert kind == "RIGHT" and ang > 0

    pos_left = {"a": (0.0, 0.0), "b": (1.0, 0.0), "c": (2.0, -1.0)}
    kind_l, ang_l = classify_turn("a", "b", "c", pos_left)
    assert kind_l == "LEFT" and ang_l < 0

    pos_straight = {"a": (0.0, 0.0), "b": (1.0, 0.0), "c": (2.0, 0.0)}
    kind_s, _ = classify_turn("a", "b", "c", pos_straight)
    assert kind_s == "STRAIGHT"


def test_turns_from_graph_marks_endpoints_and_nonedges_none():
    G = nx.Graph()
    coords = {"a": (0, 0), "b": (1, 0), "c": (2, 1), "d": (3, 1)}
    for n, p in coords.items():
        G.add_node(n, pos=p)
    G.add_edges_from([("a", "b"), ("b", "c"), ("c", "d")])
    seq = ["a", "b", "c", "d"]
    types, angles = turns_from_graph(seq, G)
    assert types[0] == "NONE" and types[-1] == "NONE"       # endpoints
    assert types[1] in {"LEFT", "RIGHT", "STRAIGHT", "UTURN"}
    # a triple crossing a non-edge is NONE
    types2, _ = turns_from_graph(["a", "c", "d"], G)
    assert types2[1] == "NONE"
