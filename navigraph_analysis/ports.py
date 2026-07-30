"""Reward-port configuration with date-dependent mapping.

Loads ``config/ports.yaml`` and resolves the correct node->Target map for a
session, applying the 2025-09-07 R55/R59 swap. Replaces the ``get_port_map`` /
``PORT_MAP_BEFORE`` / ``PORT_MAP_AFTER`` / ``pairs_to_check`` blocks that were
copy-pasted across change_of_mind, decision_point_velocity, direct_path and
trial_context notebooks (dedup C).

    from navigraph_analysis.ports import PortConfig
    pc = PortConfig.load(cfg.ports_config)
    port_map = pc.port_map(session.date)        # {'R522': 'Target1', ...}
    pairs    = pc.port_pairs(session.date)       # within-patch node pairs
"""

from __future__ import annotations

import datetime as _dt
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from .session import parse_date


@dataclass
class PortConfig:
    switch_date: _dt.date
    _before: Dict[str, str]
    _after: Dict[str, str]
    patch_pairs: List[Tuple[int, int]]
    within_patch_nodes: Dict[str, dict]

    # -- selection ------------------------------------------------------------
    def port_map(self, date: _dt.date) -> Dict[str, str]:
        """node_name -> Target label for a session on *date*."""
        return dict(self._after if date >= self.switch_date else self._before)

    def target_to_node(self, date: _dt.date) -> Dict[str, str]:
        """Inverse: Target label -> node_name."""
        return {v: k for k, v in self.port_map(date).items()}

    def port_map_numeric(self, date: _dt.date) -> Dict[str, int]:
        """node_name -> 1-indexed port number (e.g. 'Target3' -> 3)."""
        return {n: _target_num(t) for n, t in self.port_map(date).items()}

    def port_pairs(self, date: _dt.date) -> List[Tuple[str, str]]:
        """Within-patch node pairs (e.g. the two goals of each arm) for *date*."""
        t2n = self.target_to_node(date)
        pairs: List[Tuple[str, str]] = []
        for a, b in self.patch_pairs:
            na, nb = t2n.get(f"Target{a}"), t2n.get(f"Target{b}")
            if na and nb:
                pairs.append((na, nb))
        return pairs

    def target_to_patch(self, target_num: int) -> Optional[Tuple[int, int]]:
        for pair in self.patch_pairs:
            if target_num in pair:
                return tuple(pair)
        return None

    # -- loading --------------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path) -> "PortConfig":
        with open(Path(path).expanduser()) as fh:
            raw = yaml.safe_load(fh) or {}
        pm = raw.get("port_map", {})
        return cls(
            switch_date=parse_date(raw["switch_date"]),
            _before=dict(pm.get("before", {})),
            _after=dict(pm.get("after", {})),
            patch_pairs=[tuple(p) for p in raw.get("patch_pairs", [])],
            within_patch_nodes=dict(raw.get("within_patch_nodes", {})),
        )


def _target_num(label: str) -> int:
    m = re.search(r"\d+", str(label))
    return int(m.group()) if m else -1


__all__ = ["PortConfig"]
