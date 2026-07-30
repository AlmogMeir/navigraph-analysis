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


def test_port_map_switch_r55_r59():
    pc = PortConfig.load(PORTS_YAML)
    before = pc.target_to_node(dt.date(2025, 8, 21))
    after = pc.target_to_node(dt.date(2025, 9, 10))
    # Target7/Target8 swap between R55 and R59 on 2025-09-07
    assert before["Target7"] == "R55" and before["Target8"] == "R59"
    assert after["Target7"] == "R59" and after["Target8"] == "R55"
    # switch date itself uses the 'after' map
    assert pc.target_to_node(dt.date(2025, 9, 7))["Target7"] == "R59"


def test_port_pairs_and_patch_lookup():
    pc = PortConfig.load(PORTS_YAML)
    pairs = pc.port_pairs(dt.date(2025, 8, 21))
    assert len(pairs) == 4
    assert pc.target_to_patch(7) == (7, 8)
    assert pc.port_map_numeric(dt.date(2025, 8, 21))["R522"] == 1
