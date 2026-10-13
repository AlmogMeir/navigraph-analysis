"""Per-frame graph-location corrections applied before traversal reconstruction.

Tracking puts the animal on the wrong maze node in two systematic ways at the
reward ports, and both distort path structure if left in:

1. **Leaf/parent zigzag.** A port leaf and its parent are adjacent tiles, so a
   head that barely moves flickers ``leaf -> parent -> leaf`` within one or two
   frames. Each flicker reads downstream as two extra node visits and a
   round-trip the animal never made. :func:`smooth_leaf_parent_zigzags` fills the
   flicker back in with the leaf; :func:`detect_zigzags` only counts them, which
   is how the smoothing threshold was chosen (the counts collapse past one
   frame, so ``max_gap_frames=1`` is the default).

2. **Reward-period mislocation.** BPOD knows exactly which port paid and when.
   For the ~0.25 s the animal spends drinking it is at that port by definition,
   whatever the tracking says, so :func:`apply_reward_based_correction` overwrites
   the reward frame and the following few with the paying port's leaf node.

Both operate on the frame-level table *before*
:mod:`navigraph_analysis.prep.traversal_fixer` runs, since they change the node
sequence the traversal is built from. Ported from the zigzag and reward-correction
cells of ``session_pipeline.ipynb``.

    from navigraph_analysis.prep.corrections import (
        smooth_leaf_parent_zigzags, apply_reward_based_correction)

    summary_df, zz = smooth_leaf_parent_zigzags(summary_df, layout)
    summary_df, rc = apply_reward_based_correction(summary_df, reward_df, layout)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from ..columns import resolve

#: Frames the animal is taken to be at the port after a reward (0.25 s at 40 fps).
REWARD_DURATION_FRAMES = 10


@dataclass
class ZigzagCorrections:
    """What :func:`smooth_leaf_parent_zigzags` changed."""

    frames_corrected: int = 0
    events: pd.DataFrame = field(default_factory=pd.DataFrame)

    def by_leaf(self) -> pd.DataFrame:
        """Per leaf node: number of zigzag events and frames rewritten."""
        if self.events.empty:
            return pd.DataFrame(columns=["leaf_node", "events", "frames_corrected"])
        return (
            self.events.groupby("leaf_node", as_index=False)
            .agg(events=("leaf_node", "size"), frames_corrected=("frames_corrected", "sum"))
        )

    def summary(self) -> str:
        return (f"{len(self.events)} leaf/parent zigzag event(s), "
                f"{self.frames_corrected} frame(s) rewritten")


@dataclass
class RewardCorrections:
    """What :func:`apply_reward_based_correction` changed."""

    frames_corrected: int = 0
    per_port: Dict[int, dict] = field(default_factory=dict)
    n_events: int = 0

    def summary(self) -> str:
        return (f"{self.frames_corrected} frame(s) rewritten across "
                f"{self.n_events} reward event(s)")

    def table(self) -> pd.DataFrame:
        """Per port: leaf node, frames corrected, and what they were changed from."""
        rows = []
        for port in sorted(self.per_port):
            info = self.per_port[port]
            rows.append({
                "port": port,
                "leaf_node": info["leaf_node"],
                "frames_corrected": info["frames_corrected"],
                "changed_from": dict(sorted(info["changed_from"].items(),
                                            key=lambda kv: -kv[1])),
            })
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Leaf/parent zigzags
# ---------------------------------------------------------------------------

def detect_zigzags(
    summary_df: pd.DataFrame,
    leaf_node: str,
    parent_nodes: Sequence[str],
    max_gap_frames: int = 5,
    bodypart: str = "headstage",
) -> pd.DataFrame:
    """Count ``leaf -> parent -> leaf`` round trips no longer than *max_gap_frames*.

    Diagnostic only — nothing is changed. Run it across thresholds to see where
    the counts stop growing; that is the point past which the transitions are
    real movement rather than tracking flicker.

    Columns: ``frame_start, frame_parent, frame_return, leaf_node, parent_node,
    frames_before_zigzag, frames_in_parent, frames_after_zigzag,
    total_zigzag_duration``.
    """
    locations = summary_df[resolve(summary_df, bodypart, "graph_node")].to_numpy(dtype=object)
    parent_nodes = list(parent_nodes)
    events: List[dict] = []

    i = 0
    while i < len(locations) - max_gap_frames - 1:
        if locations[i] == leaf_node:
            for gap in range(1, max_gap_frames + 1):
                if i + gap >= len(locations):
                    break
                middle = locations[i + gap]
                if middle not in parent_nodes:
                    continue
                for return_gap in range(1, gap + 1):
                    if i + gap + return_gap >= len(locations):
                        break
                    if locations[i + gap + return_gap] != leaf_node:
                        continue

                    frames_before = 1
                    for j in range(i - 1, -1, -1):
                        if locations[j] != leaf_node:
                            break
                        frames_before += 1

                    frames_after = 1
                    tail = min(i + gap + return_gap + 50, len(locations))
                    for j in range(i + gap + return_gap + 1, tail):
                        if locations[j] != leaf_node:
                            break
                        frames_after += 1

                    events.append({
                        "frame_start": i,
                        "frame_parent": i + gap,
                        "frame_return": i + gap + return_gap,
                        "leaf_node": leaf_node,
                        "parent_node": middle,
                        "frames_before_zigzag": frames_before,
                        "frames_in_parent": return_gap,
                        "frames_after_zigzag": frames_after,
                        "total_zigzag_duration": gap + return_gap,
                    })
                    i = i + gap + return_gap   # don't double-count
                    break
                else:
                    continue
                break
        i += 1

    return pd.DataFrame(events)


def zigzag_threshold_table(
    summary_df: pd.DataFrame,
    layout: Dict[int, Tuple[str, List[str]]],
    thresholds: Sequence[int] = (1, 2, 3, 5),
    bodypart: str = "headstage",
) -> pd.DataFrame:
    """Zigzag counts per port across *thresholds*, for choosing the smoothing gap.

    Columns: ``port, leaf_node, threshold, n_events, mean_frames_in_parent``.
    """
    rows = []
    for port, (leaf, parents) in sorted(layout.items()):
        for threshold in thresholds:
            zz = detect_zigzags(summary_df, leaf, parents,
                                max_gap_frames=threshold, bodypart=bodypart)
            rows.append({
                "port": port,
                "leaf_node": leaf,
                "threshold": threshold,
                "n_events": len(zz),
                "mean_frames_in_parent": (zz["frames_in_parent"].mean()
                                          if len(zz) else np.nan),
            })
    return pd.DataFrame(rows)


def smooth_leaf_parent_zigzags(
    summary_df: pd.DataFrame,
    layout: Dict[int, Tuple[str, List[str]]],
    max_gap_frames: int = 1,
    bodypart: str = "headstage",
) -> Tuple[pd.DataFrame, ZigzagCorrections]:
    """Rewrite ``leaf -> parent(<= max_gap_frames) -> leaf`` flicker back to the leaf.

    The default gap of 1 frame is deliberate: the zigzag counts collapse past a
    single frame, so anything longer is treated as the animal genuinely stepping
    off the port rather than as tracking noise.

    Returns ``(corrected_summary_df, ZigzagCorrections)``; the input is not modified.
    """
    node_col = resolve(summary_df, bodypart, "graph_node")
    out = summary_df.copy()
    locations = out[node_col].to_numpy(dtype=object).copy()

    leaf_to_parents = {leaf: list(parents) for leaf, parents in layout.values()}
    corrected = 0
    events: List[dict] = []

    i = 0
    while i < len(locations) - max_gap_frames - 1:
        leaf_node = locations[i]
        if leaf_node in leaf_to_parents:
            parent_nodes = leaf_to_parents[leaf_node]
            for gap in range(1, max_gap_frames + 1):
                if i + gap >= len(locations):
                    break
                middle = locations[i + gap]
                if middle not in parent_nodes:
                    continue
                # the whole excursion must be on the parent, not just its first frame
                all_parent = all(
                    locations[k] in parent_nodes
                    for k in range(i + gap, min(i + gap + gap, len(locations)))
                )
                if not all_parent:
                    continue
                for return_gap in range(1, gap + 1):
                    if i + gap + return_gap >= len(locations):
                        break
                    if locations[i + gap + return_gap] != leaf_node:
                        continue
                    for fix in range(i + gap, i + gap + return_gap):
                        locations[fix] = leaf_node
                        corrected += 1
                    events.append({
                        "frame_start": i + gap,
                        "frame_end": i + gap + return_gap - 1,
                        "leaf_node": leaf_node,
                        "parent_node": middle,
                        "frames_corrected": return_gap,
                    })
                    i = i + gap + return_gap - 1
                    break
                else:
                    continue
                break
        i += 1

    out[node_col] = locations
    return out, ZigzagCorrections(frames_corrected=corrected,
                                  events=pd.DataFrame(events))


# ---------------------------------------------------------------------------
# Reward-period correction
# ---------------------------------------------------------------------------

def apply_reward_based_correction(
    summary_df: pd.DataFrame,
    reward_frames: pd.DataFrame,
    layout: Dict[int, Tuple[str, List[str]]],
    reward_duration_frames: int = REWARD_DURATION_FRAMES,
    bodypart: str = "headstage",
    port_col: str = "reward_port",
    frame_col: str = "frame_idx_global",
) -> Tuple[pd.DataFrame, RewardCorrections]:
    """Force the paying port's leaf node over each reward's drinking period.

    BPOD says which port paid and when, so for *reward_duration_frames* from the
    reward the animal is at that port regardless of what tracking reports. This is
    the one place in the pipeline where an external ground truth overrides the
    tracking, and it is only sound once the video/BPOD clocks are known to agree —
    run :func:`navigraph_analysis.prep.frame_alignment.align_frames_to_video`
    first, or a misaligned session gets phantom port visits scattered through it.

    *reward_frames* needs a global-frame column and a 1-indexed port column;
    rewards whose port is absent from *layout* are skipped.

    Returns ``(corrected_summary_df, RewardCorrections)``; the input is not modified.
    """
    node_col = resolve(summary_df, bodypart, "graph_node")
    out = summary_df.copy()
    locations = out[node_col].to_numpy(dtype=object).copy()

    # Rewards are indexed by global frame; the table may be trimmed or shifted,
    # so map frame -> row rather than assuming positional identity.
    frames = pd.to_numeric(out[resolve(out, frame_col)], errors="coerce").to_numpy()
    row_of_frame = {int(f): i for i, f in enumerate(frames) if f == f}

    per_port = {p: {"leaf_node": leaf, "frames_corrected": 0, "changed_from": {}}
                for p, (leaf, _parents) in layout.items()}
    corrected = 0
    n_events = 0

    for _, row in reward_frames.iterrows():
        port = int(row[port_col])
        if port not in layout:
            continue
        expected_leaf = layout[port][0]
        start = int(row[frame_col])
        n_events += 1
        for f in range(start, start + reward_duration_frames):
            i = row_of_frame.get(f)
            if i is None:
                continue
            current = locations[i]
            if pd.notna(current) and current != expected_leaf:
                per_port[port]["changed_from"][current] = (
                    per_port[port]["changed_from"].get(current, 0) + 1)
                locations[i] = expected_leaf
                corrected += 1
                per_port[port]["frames_corrected"] += 1

    out[node_col] = locations
    return out, RewardCorrections(frames_corrected=corrected,
                                  per_port=per_port,
                                  n_events=n_events)


__all__ = [
    "REWARD_DURATION_FRAMES",
    "ZigzagCorrections",
    "RewardCorrections",
    "detect_zigzags",
    "zigzag_threshold_table",
    "smooth_leaf_parent_zigzags",
    "apply_reward_based_correction",
]
