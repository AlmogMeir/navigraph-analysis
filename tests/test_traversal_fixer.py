"""Tests for the improved traversal fixer and the location-correction stages."""

import networkx as nx
import pandas as pd

from navigraph_analysis.prep.corrections import (
    apply_reward_based_correction,
    detect_zigzags,
    smooth_leaf_parent_zigzags,
)
from navigraph_analysis.prep.traversal_fixer import (
    ImprovedMazeTraversalFixer,
    build_traversal_df,
)

LAYOUT = {1: ("R522", ["R411"]), 2: ("R526", ["R413"])}


def _maze():
    """A small dual-root-tree slice: two ports under one level-3 node."""
    G = nx.Graph()
    G.add_edges_from([
        ("R35", "R411"), ("R411", "R522"),      # R522 is a leaf (virtual link)
        ("R35", "R22"), ("R22", "R11"),
        ("R11", "R23"), ("R23", "R36"),
        ("R36", "R413"), ("R413", "R526"),      # R526 is a leaf
    ])
    return G


def _frames_from_path(G, fixer, path, node_dwell=12, edge_dwell=8, trial=0):
    """Frame-by-frame rows walking *path*, with edge frames where the link is real."""
    rows, frame = [], 0
    for i, n in enumerate(path):
        for _ in range(node_dwell):
            rows.append({"frame_idx_global": frame, "trial_idx": trial,
                         "headstage_graph_node": n, "headstage_graph_edge": None})
            frame += 1
        if i + 1 < len(path):
            nxt = path[i + 1]
            if not fixer._is_virtual_edge(n, nxt):
                for _ in range(edge_dwell):
                    rows.append({"frame_idx_global": frame, "trial_idx": trial,
                                 "headstage_graph_node": None,
                                 "headstage_graph_edge": str(tuple(sorted((n, nxt))))})
                    frame += 1
    return pd.DataFrame(rows)


def test_augmented_graph_keeps_leaf_links_direct():
    G = _maze()
    fixer = ImprovedMazeTraversalFixer(G)
    # a real corridor becomes an intermediate edge-state ...
    assert tuple(sorted(("R35", "R22"))) in fixer.G_aug
    assert not fixer.G_aug.has_edge("R35", "R22")
    # ... while the leaf<->parent link stays a direct node-node connection
    assert fixer.G_aug.has_edge("R411", "R522")
    assert tuple(sorted(("R411", "R522"))) not in fixer.G_aug


def test_clean_session_needs_no_repair():
    G = _maze()
    fixer = ImprovedMazeTraversalFixer(G)
    path = nx.shortest_path(G, "R522", "R526")
    df = _frames_from_path(G, fixer, path)

    trav, _improved, diag = build_traversal_df(df, G)
    assert diag["valid"] == 1
    assert diag["nodes_edges_inferred"] == 0
    assert diag["jumps_interpolated"] == 0
    assert diag["invalid_unresolved"] == 0


def test_short_teleport_is_smoothed_away():
    G = _maze()
    fixer = ImprovedMazeTraversalFixer(G)
    path = nx.shortest_path(G, "R522", "R526")
    df = _frames_from_path(G, fixer, path)

    # two frames claiming a node several hops away, mid-dwell
    df.loc[5:6, "headstage_graph_node"] = "R526"
    trav, _i, diag = build_traversal_df(df, G)
    assert diag["valid"] == 1
    assert (diag["noise_smoothed_long_jump"] + diag["noise_smoothed_forward"]
            + diag["noise_smoothed_backward"]) >= 1
    assert diag["jumps_interpolated"] == 0, "a 2-frame blip is noise, not travel"


def test_tracking_dropout_is_interpolated_and_conserves_frames():
    G = _maze()
    fixer = ImprovedMazeTraversalFixer(G)
    path = nx.shortest_path(G, "R522", "R526")
    clean = _frames_from_path(G, fixer, path)
    total = len(clean)

    noisy = clean.copy()
    noisy.loc[30:70, ["headstage_graph_node", "headstage_graph_edge"]] = None

    trav, _i, diag = build_traversal_df(noisy, G)
    assert diag["valid"] == 1
    # inferred states take real frames from their neighbours rather than
    # appearing as zero-duration rows
    assert trav["duration"].sum() == total
    assert (trav["duration"] > 0).all()


def test_illegal_edge_is_dropped_not_trusted():
    G = _maze()
    fixer = ImprovedMazeTraversalFixer(G)
    path = nx.shortest_path(G, "R522", "R526")
    df = _frames_from_path(G, fixer, path)
    df.loc[5:6, "headstage_graph_node"] = None
    df.loc[5:6, "headstage_graph_edge"] = "('R522', 'R526')"   # not a real corridor

    trav, _i, diag = build_traversal_df(df, G)
    assert diag["valid"] == 1
    edges = {str(e) for e in trav["headstage_graph_edge"].dropna()}
    assert "('R522', 'R526')" not in edges


def test_output_matches_traversal_df_contract():
    G = _maze()
    fixer = ImprovedMazeTraversalFixer(G)
    df = _frames_from_path(G, fixer, nx.shortest_path(G, "R522", "R526"))
    trav, improved, _d = build_traversal_df(df, G)

    for col in ("location_type", "headstage_graph_node", "headstage_graph_edge",
                "start_frame_global", "end_frame_global", "duration",
                "trial_idx", "is_inferred"):
        assert col in trav.columns
    assert set(trav["location_type"]) <= {"node", "edge"}
    assert "location" in improved.columns and "duration_time" in improved.columns


# -- corrections ------------------------------------------------------------

def _port_frames(n=30, flicker_at=(10, 20), long_excursion=(25, 29)):
    seq = ["R522"] * n
    for i in flicker_at:
        seq[i] = "R411"
    lo, hi = long_excursion
    seq[lo:hi] = ["R411"] * (hi - lo)
    return pd.DataFrame({"frame_idx_global": range(n),
                         "headstage_graph_node": seq})


def test_zigzag_smoothing_fixes_flicker_but_spares_real_excursions():
    df = _port_frames()
    out, corr = smooth_leaf_parent_zigzags(df, LAYOUT, max_gap_frames=1)
    nodes = list(out["headstage_graph_node"])

    assert nodes[10] == "R522" and nodes[20] == "R522"
    assert nodes[25:29] == ["R411"] * 4, "a 4-frame excursion is movement, not noise"
    assert corr.frames_corrected == 2
    # the input frame must not be mutated in place
    assert df["headstage_graph_node"].iloc[10] == "R411"


def test_detect_zigzags_counts_without_changing_anything():
    df = _port_frames()
    zz = detect_zigzags(df, "R522", ["R411"], max_gap_frames=1)
    assert len(zz) == 2
    assert set(zz["frame_parent"]) == {10, 20}


def test_reward_correction_indexes_by_frame_not_position():
    """The frame table is trimmed and shifted upstream, so row 0 is not frame 0."""
    df = pd.DataFrame({"frame_idx_global": range(100, 130),
                       "headstage_graph_node": ["R411"] * 30})
    rewards = pd.DataFrame({"frame_idx_global": [105], "reward_port": [1]})

    out, rc = apply_reward_based_correction(df, rewards, LAYOUT,
                                            reward_duration_frames=10)
    nodes = list(out["headstage_graph_node"])
    assert nodes[5:15] == ["R522"] * 10      # frames 105..114
    assert nodes[:5] == ["R411"] * 5
    assert nodes[15:] == ["R411"] * 15
    assert rc.frames_corrected == 10


def test_reward_outside_the_trimmed_range_is_skipped():
    df = pd.DataFrame({"frame_idx_global": range(100, 130),
                       "headstage_graph_node": ["R411"] * 30})
    rewards = pd.DataFrame({"frame_idx_global": [105, 9999],
                            "reward_port": [1, 2]})
    out, rc = apply_reward_based_correction(df, rewards, LAYOUT)
    assert rc.frames_corrected == 10
    assert rc.n_events == 2
