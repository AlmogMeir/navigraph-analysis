"""Tests for date parsing and the date-dependent reward-port map."""

import datetime as dt
from pathlib import Path

import pytest

from navigraph_analysis.session import parse_date
from navigraph_analysis.ports import PortConfig

PORTS_YAML = Path(__file__).resolve().parent.parent / "config" / "ports.yaml"


@pytest.mark.parametrize("token,expected", [
    ("21_08_2025", dt.date(2025, 8, 21)),   # DD_MM_YYYY (folder style)
    ("2025_08_21", dt.date(2025, 8, 21)),   # YYYY_MM_DD (file style)
    ("2025-09-01", dt.date(2025, 9, 1)),    # ISO
    ("01/09/2025", dt.date(2025, 9, 1)),    # DD/MM/YYYY
])
def test_parse_date_all_orderings(token, expected):
    assert parse_date(token) == expected


def test_parse_date_rejects_garbage():
    with pytest.raises(ValueError):
        parse_date("not-a-date")


def test_port_map_switch_r55_r59_npc1():
    pc = PortConfig.load(PORTS_YAML)
    before = pc.target_to_node("NPC1", dt.date(2025, 8, 21))
    after = pc.target_to_node("NPC1", dt.date(2025, 9, 10))
    # Target7/Target8 swap between R55 and R59 on 2025-09-07
    assert before["Target7"] == "R55" and before["Target8"] == "R59"
    assert after["Target7"] == "R59" and after["Target8"] == "R55"
    # switch date itself uses the 'after' map
    assert pc.target_to_node("NPC1", dt.date(2025, 9, 7))["Target7"] == "R59"


def test_per_subject_port_map_npc4_swaps_ports_3_4():
    pc = PortConfig.load(PORTS_YAML)
    npc1 = pc.port_map("NPC1", dt.date(2026, 5, 10))
    npc4 = pc.port_map("NPC4", dt.date(2026, 5, 10))
    # NPC4 has ports 3/4 swapped relative to NPC1 (L56 <-> L510)
    assert npc1["L56"] == "Target3" and npc1["L510"] == "Target4"
    assert npc4["L510"] == "Target3" and npc4["L56"] == "Target4"


def test_layout_out_of_range_raises():
    pc = PortConfig.load(PORTS_YAML)
    # NPC4 has no layout before 2026-04-01
    with pytest.raises(KeyError):
        pc.port_map("NPC4", dt.date(2025, 1, 1))


def test_port_pairs_and_patch_lookup():
    pc = PortConfig.load(PORTS_YAML)
    pairs = pc.port_pair_nodes("NPC1", dt.date(2025, 8, 21))
    assert len(pairs) == 4
    assert pc.target_to_patch(7) == (7, 8)
    assert pc.port_map_numeric("NPC1", dt.date(2025, 8, 21))["R522"] == 1


def test_port_parent_grandparents():
    pc = PortConfig.load(PORTS_YAML)
    ppg = pc.port_parent_grandparents("NPC1", dt.date(2025, 8, 21))
    assert ppg[1]["parent"] == "R411"
    assert ppg[1]["grandparents"] == ["R523", "R35"]
