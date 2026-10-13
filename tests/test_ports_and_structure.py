"""Tests for date-ranged port layouts, maze structure and session annotations."""

import datetime as dt

import networkx as nx
import pytest

from navigraph_analysis.config import ProjectConfig, REPO_ROOT
from navigraph_analysis.io.graph import classify_patch, level_of, patch_of
from navigraph_analysis.ports import PortConfig
from navigraph_analysis.viz.sessions import annotation_blocks, _as_date


@pytest.fixture(scope="module")
def ports():
    return PortConfig.load(REPO_ROOT / "config" / "ports.yaml")


def _leaves(ports, subject, iso_date):
    layout = ports.resolve_layout(subject, dt.date.fromisoformat(iso_date))
    return {p: layout[p][0] for p in sorted(layout)}


@pytest.mark.parametrize("iso,port,leaf", [
    # NPC1's change-of-maze window moved port 6 to L518 for one week, and ports
    # 7/8 swapped partway through it. Before and after, port 6 is L525.
    ("2025-08-21", 6, "L525"),
    ("2025-09-04", 6, "L518"),
    ("2025-09-08", 6, "L518"),
    ("2025-09-15", 6, "L525"),
])
def test_npc1_change_of_maze_period(ports, iso, port, leaf):
    assert _leaves(ports, "NPC1", iso)[port] == leaf


@pytest.mark.parametrize("iso,expect_swapped", [
    ("2025-09-04", False),   # ports 7/8 not yet swapped
    ("2025-09-08", True),
    ("2025-09-15", True),
])
def test_npc1_port_78_swap(ports, iso, expect_swapped):
    leaves = _leaves(ports, "NPC1", iso)
    assert (leaves[7] == "R59") is expect_swapped
    assert (leaves[8] == "R55") is expect_swapped


def test_npc4_starts_in_march_and_has_a_change_of_maze_window(ports):
    # The layout opens on 2026-03-01; an earlier date has no period.
    assert _leaves(ports, "NPC4", "2026-03-05")[3] == "L510"
    with pytest.raises(KeyError, match="No port-layout period"):
        ports.resolve_layout("NPC4", dt.date(2026, 2, 1))

    before = _leaves(ports, "NPC4", "2026-05-10")
    during = _leaves(ports, "NPC4", "2026-06-17")
    after = _leaves(ports, "NPC4", "2026-06-22")
    assert (during[1], during[5], during[6]) == ("R521", "L530", "L517")
    assert before == after, "the maze returns to its previous layout"
    assert during != before


def test_every_port_parent_has_grandparents(ports):
    """The port-parent-grandparent analysis needs an entry for every parent used."""
    for subject, periods in ports.port_layouts.items():
        for _lo, _hi, layout in periods:
            for port, (_leaf, parents) in layout.items():
                parent = parents[0]
                assert ports.grandparents.get(parent), (
                    f"{subject} port {port}: no grandparents entry for {parent}")


# -- maze structure ---------------------------------------------------------

def _balanced_dual_root_tree(depth=5):
    G = nx.Graph()
    for side in ("L", "R"):
        level = [f"{side}0"]
        for d in range(1, depth + 1):
            nxt = []
            for i, parent in enumerate(level):
                for c in range(2):
                    child = f"{side}{d}{2 * i + c}"
                    G.add_edge(parent, child)
                    nxt.append(child)
            level = nxt
    G.add_edge("L0", "R0")
    return G


def test_level_and_patch_split_the_maze():
    G = _balanced_dual_root_tree()
    lv = level_of(G)
    assert lv["L0"] == 0 and lv["R0"] == 0
    assert max(lv.values()) == 5
    assert sum(1 for v in lv.values() if v == 5) == 64

    pm = patch_of(G)
    assert pm["L0"] == pm["R0"] == "corridor"
    patches = {v for v in pm.values() if v != "corridor"}
    assert patches == {"L10", "L11", "R10", "R11"}


def test_classify_patch_distinguishes_within_and_across_patch_trips():
    G = _balanced_dual_root_tree()
    pm = patch_of(G)
    leaves = [n for n, d in level_of(G).items() if d == 5]

    same = [n for n in leaves if pm[n] == "R11"][:2]
    assert classify_patch(nx.shortest_path(G, *same), pm) == "in-patch"

    other = next(n for n in leaves if pm[n] == "L10")
    assert classify_patch(nx.shortest_path(G, same[0], other), pm) == "out-of-patch"


# -- session annotations ----------------------------------------------------

def test_as_date_accepts_both_folder_conventions():
    assert _as_date("08_09_2025") == dt.date(2025, 9, 8)     # DD_MM_YYYY
    assert _as_date("2025_09_08") == dt.date(2025, 9, 8)     # YYYY_MM_DD
    assert _as_date("08/09/2025") == dt.date(2025, 9, 8)


def test_annotation_blocks_split_on_gaps_in_the_session_list():
    days = ["01_05_2026", "02_05_2026", "10_05_2026"]
    anns = [{"label": "Post surgery", "start": "01/05/2026"}]
    blocks = annotation_blocks(days, anns)
    assert len(blocks) == 1 and blocks[0][:2] == (0, 2)

    # a session outside the span breaks the run into two blocks
    days = ["01_05_2026", "02_05_2026", "05_05_2026"]
    anns = [{"label": "X", "start": "01/05/2026", "end": "02/05/2026"},
            {"label": "Y", "start": "05/05/2026"}]
    blocks = annotation_blocks(days, anns)
    assert [b[2] for b in blocks] == ["X", "Y"]


def test_project_config_carries_date_annotations():
    cfg = ProjectConfig.load()
    assert set(cfg.date_annotations) >= {"NPC1", "NPC4"}
    labels = [a["label"] for a in cfg.date_annotations["NPC4"]]
    assert "Change of maze" in labels
