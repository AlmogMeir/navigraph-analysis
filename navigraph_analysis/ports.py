"""Reward-port configuration: per-subject, date-ranged layout periods.

Loads ``config/ports.yaml`` and resolves the reward-port map for a given
``(subject, date)`` — a direct port of ``resolve_port_layout`` /
``PORT_LAYOUTS`` from ``align_frames_trials.ipynb``. Replaces the old
before/after single-switch model: the port map is now a **parameter** derived
from the subject and session date, so different mice (and re-wirings within a
mouse) get the right leaf-node -> Target mapping.

    from navigraph_analysis.ports import PortConfig
    pc = PortConfig.load(cfg.ports_config)
    port_map = pc.port_map("NPC1", session.date)        # {'R522': 'Target1', ...}
    pairs    = pc.port_pair_nodes("NPC1", session.date)  # within-alternation leaf pairs
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from .session import parse_date

# One layout period: (valid_from, valid_until, {port_number: (leaf, [parents])}).
Period = Tuple[Optional[_dt.date], Optional[_dt.date], Dict[int, Tuple[str, List[str]]]]


@dataclass
class PortConfig:
    port_layouts: Dict[str, List[Period]]
    port_pairs: List[Tuple[int, int]]
    port_pairs_by_subject: Dict[str, List[Tuple[int, int]]] = field(default_factory=dict)
    grandparents: Dict[str, List[str]] = field(default_factory=dict)
    within_patch_nodes: Dict[str, dict] = field(default_factory=dict)

    # -- core resolution ------------------------------------------------------
    def resolve_layout(self, subject: str, date: _dt.date) -> Dict[int, Tuple[str, List[str]]]:
        """``{port_number: (leaf_node, [parent_nodes])}`` for this (subject, date)."""
        if subject not in self.port_layouts:
            raise KeyError(
                f"No port layout for subject {subject!r}; defined: {sorted(self.port_layouts)}"
            )
        for lo, hi, layout in self.port_layouts[subject]:
            if (lo is None or lo <= date) and (hi is None or date <= hi):
                return layout
        raise KeyError(
            f"No port-layout period covers {subject} on {date.isoformat()}. "
            f"Widen a date window in ports.yaml['port_layouts']['{subject}']."
        )

    def port_map(self, subject: str, date: _dt.date) -> Dict[str, str]:
        """node_name -> Target label, e.g. ``{'R522': 'Target1', ...}``."""
        layout = self.resolve_layout(subject, date)
        return {leaf: f"Target{p}" for p, (leaf, _parents) in sorted(layout.items())}

    def target_to_node(self, subject: str, date: _dt.date) -> Dict[str, str]:
        return {v: k for k, v in self.port_map(subject, date).items()}

    def port_map_numeric(self, subject: str, date: _dt.date) -> Dict[str, int]:
        """node_name -> 1-indexed port number."""
        layout = self.resolve_layout(subject, date)
        return {leaf: p for p, (leaf, _parents) in layout.items()}

    def port_pairs_for(self, subject: str) -> List[Tuple[int, int]]:
        return self.port_pairs_by_subject.get(subject, self.port_pairs)

    def port_pair_nodes(self, subject: str, date: _dt.date) -> List[Tuple[str, str]]:
        """Within-alternation leaf-node pairs for this (subject, date).

        (The raw port-number pairs live in the :attr:`port_pairs` field.)
        """
        layout = self.resolve_layout(subject, date)
        out: List[Tuple[str, str]] = []
        for a, b in self.port_pairs_for(subject):
            if a in layout and b in layout:
                out.append((layout[a][0], layout[b][0]))
        return out

    def port_parent_grandparents(self, subject: str, date: _dt.date) -> Dict[int, dict]:
        """Per port number: its parent node + that parent's grandparent neighbours."""
        layout = self.resolve_layout(subject, date)
        out: Dict[int, dict] = {}
        for p, (_leaf, parents) in layout.items():
            parent = parents[0] if parents else None
            out[p] = {"parent": parent, "grandparents": self.grandparents.get(parent, [])}
        return out

    def target_to_patch(self, target_num: int) -> Optional[Tuple[int, int]]:
        for pair in self.port_pairs:
            if target_num in pair:
                return pair
        return None

    # -- loading --------------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path) -> "PortConfig":
        with open(Path(path).expanduser()) as fh:
            raw = yaml.safe_load(fh) or {}

        layouts: Dict[str, List[Period]] = {}
        for subject, periods in (raw.get("port_layouts") or {}).items():
            parsed: List[Period] = []
            for period in periods:
                lo = _parse_opt_date(period.get("from"))
                hi = _parse_opt_date(period.get("until"))
                ports = {
                    int(num): (spec["leaf"], list(spec.get("parents", [])))
                    for num, spec in (period.get("ports") or {}).items()
                }
                parsed.append((lo, hi, ports))
            layouts[subject] = parsed

        return cls(
            port_layouts=layouts,
            port_pairs=[tuple(p) for p in raw.get("port_pairs", [])],
            port_pairs_by_subject={
                s: [tuple(p) for p in pairs]
                for s, pairs in (raw.get("port_pairs_by_subject") or {}).items()
            },
            grandparents=dict(raw.get("grandparents", {})),
            within_patch_nodes=dict(raw.get("within_patch_nodes", {})),
        )


def _parse_opt_date(value) -> Optional[_dt.date]:
    return None if value in (None, "", "null") else parse_date(str(value))


__all__ = ["PortConfig"]
