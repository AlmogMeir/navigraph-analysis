"""Build the BPOD -> video frame table, and verify the two clocks actually agree.

The frame/trial table is derived by cumulatively summing BPOD trial durations,
which assumes video frame 0 is the start of trial 0. That assumption fails
silently: nothing downstream can tell a misaligned session from a good one, and
the reward-based location correction then injects phantom port visits wherever a
reward frame lands off-target, shortening paths below the graph shortest path and
inflating the direct/indirect ratio.

The frame *budget* (frames BPOD claims vs frames the video holds) does not detect
this. Measured across NPC4's 21 sessions the two are very nearly orthogonal:

    26/05  overshoot +7616 (190 s)  yet 100.0% of rewards on-port -> fine,
                                    the camera merely stopped early
    24/05  overshoot -5007          yet  99.3% on-port            -> fine
    14/05  overshoot     0          but  76.0% on-port            -> poor
    01/06  overshoot +5814          and  13.3% on-port            -> broken,
                                    the camera started 145.3 s late

A budget test would have blocked 26/05 (correct data) while waving 14/05 through.
So :func:`align_frames_to_video` measures alignment directly instead: how often a
reward event finds the animal at the port that paid it. That cleanly separates 13%
from >= 76%. Below :data:`ALIGN_MIN_ON_PORT` it scans for a constant camera-start
offset and applies it, preferring the shift that also makes the two clocks end
together when that sits inside the (broad) peak plateau.

Ported from the alignment cell of ``session_pipeline.ipynb``.

    from navigraph_analysis.prep.frame_alignment import build_frames_df, align_frames_to_video

    frames_df = build_frames_df(trial_end_times, fps=cfg.fps)
    frames_df, alignment = align_frames_to_video(
        frames_df, node_by_frame, reward_frame, reward_port, layout, n_video_frames
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

#: Percent of rewards that must land on the paying port before a session is accepted.
ALIGN_MIN_ON_PORT = 60.0
#: Widest camera lag searched, in frames (8000 = 200 s at 40 fps).
ALIGN_MAX_SHIFT = 8000
#: Fewer reward events than this inside the video and alignment cannot be judged.
ALIGN_MIN_EVENTS = 20


class AlignmentError(ValueError):
    """Raised when a session's video and BPOD clocks cannot be reconciled."""


@dataclass
class FrameAlignment:
    """What :func:`align_frames_to_video` concluded about one session."""

    frame_shift: int                      # frames the camera started after BPOD
    on_port_pct: float                    # % of rewards on the paying port, after the shift
    on_port_pct_raw: float                # ... before it
    n_events: int                         # reward events used for the measurement
    n_video_frames: int
    overshoot: int                        # BPOD-claimed frames - video frames
    snapped_to_exact_fit: bool = False    # shift == overshoot was preferred in the plateau
    n_frames_dropped: int = 0             # frames trimmed for falling outside the video
    trials_before_start: List[int] = field(default_factory=list)
    trials_after_stop: List[int] = field(default_factory=list)

    @property
    def shifted(self) -> bool:
        return self.frame_shift != 0

    @property
    def trials_lost(self) -> List[int]:
        return sorted(self.trials_before_start + self.trials_after_stop)

    def summary(self) -> str:
        parts = [
            f"{self.on_port_pct:.1f}% of {self.n_events} rewards on the paying port",
            f"frame budget overshoot {self.overshoot:+d}",
        ]
        if self.shifted:
            parts.insert(
                1,
                f"camera started {self.frame_shift} frames after BPOD "
                f"(was {self.on_port_pct_raw:.1f}% before correction"
                + (", snapped to exact fit" if self.snapped_to_exact_fit else "")
                + ")",
            )
        if self.n_frames_dropped:
            parts.append(f"{self.n_frames_dropped} frames trimmed outside the video")
        if self.trials_lost:
            parts.append(
                f"{len(self.trials_lost)} trial(s) with no video "
                f"({len(self.trials_before_start)} before the camera started, "
                f"{len(self.trials_after_stop)} after it stopped)"
            )
        return "; ".join(parts)


# ---------------------------------------------------------------------------
# Frame table
# ---------------------------------------------------------------------------

def build_frames_df(
    trial_end_times: Sequence[float],
    fps: float = 40.0,
    tolerance: float = 1e-4,
) -> pd.DataFrame:
    """Frame-to-trial table from BPOD trial durations.

    A trial whose duration is not an exact multiple of the frame interval holds
    one more frame than it has full intervals; that last, partially-exposed frame
    is unreliable, so it is counted in the global frame index but left out of the
    table.

    Columns: ``trial_idx, frame_idx_global, frame_idx_in_trial,
    frame_time_in_trial, trial_length, is_exact_multiple, has_extra_frame``.
    The number of frames BPOD claims for the whole session (including the skipped
    ones) is on the frame's ``attrs['bpod_frame_count']``.
    """
    frame_interval = 1.0 / fps
    rows = []
    current = 0

    for trial_idx, trial_length in enumerate(trial_end_times):
        expected = trial_length * fps
        remainder = trial_length % frame_interval
        is_exact = remainder < tolerance or (frame_interval - remainder) < tolerance

        if is_exact:
            n_frames = int(np.round(expected))
            has_extra = False
        else:
            n_frames = int(np.floor(expected))
            has_extra = True

        for frame_in_trial in range(n_frames):
            rows.append({
                "trial_idx": trial_idx,
                "frame_idx_global": current,
                "frame_idx_in_trial": frame_in_trial,
                "frame_time_in_trial": frame_in_trial * frame_interval,
                "trial_length": trial_length,
                "is_exact_multiple": is_exact,
                "has_extra_frame": has_extra,
            })
            current += 1

        if has_extra:
            current += 1   # count but skip the unreliable final frame

    frames_df = pd.DataFrame(rows)
    frames_df.attrs["bpod_frame_count"] = current
    return frames_df


# ---------------------------------------------------------------------------
# Alignment measurement
# ---------------------------------------------------------------------------

def bpod_reward_frames(
    reward_df: pd.DataFrame,
    frames_df: pd.DataFrame,
    layout: Dict[int, Tuple[str, List[str]]],
    fps: float = 40.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Reward events as ``(global frame, port number)`` on the BPOD clock.

    *reward_df* is the per-session ``reward.csv`` BPOD export: a trial column, a
    time-within-trial column and a port column, found by name. Trials outside the
    session and ports absent from *layout* are dropped.
    """
    trial_col = "trial" if "trial" in reward_df.columns else reward_df.columns[0]
    time_col = next(c for c in reward_df.columns if "time" in c.lower())
    port_col = next(c for c in reward_df.columns if "port" in c.lower())

    trial_start = frames_df.groupby("trial_idx")["frame_idx_global"].min()
    n_trials = int(frames_df["trial_idx"].max()) + 1

    rw = reward_df[(reward_df[trial_col].astype(int) - 1).between(0, n_trials - 1)]
    frame = (trial_start.reindex(rw[trial_col].astype(int).values - 1).values
             + np.round(rw[time_col].values * fps))
    port = rw[port_col].astype(int).values

    keep = ~np.isnan(frame) & np.array([p in layout for p in port], dtype=bool)
    return frame[keep].astype(int), port[keep]


def on_port_fraction(
    reward_frame: np.ndarray,
    reward_port: np.ndarray,
    node_by_frame: Sequence,
    layout: Dict[int, Tuple[str, List[str]]],
    n_video_frames: int,
    shift: int = 0,
    min_events: int = ALIGN_MIN_EVENTS,
) -> Tuple[float, int]:
    """Percent of reward events that find the animal on the port that paid them.

    A hit is the port's own leaf node or one of its parents — the animal is at the
    port either way, and single-frame leaf/parent flicker is pervasive. Returns
    ``(percent, n_events_used)``, with percent NaN when fewer than *min_events*
    reward events fall inside the video at this *shift*.
    """
    idx = np.asarray(reward_frame) - shift
    inside = (idx >= 0) & (idx < n_video_frames)
    n = int(inside.sum())
    if n < min_events:
        return np.nan, n

    hits = 0
    for i, p in zip(idx[inside], np.asarray(reward_port)[inside]):
        leaf, parents = layout[p]
        node = node_by_frame[i]
        if node is not None and (node == leaf or node in parents):
            hits += 1
    return hits / n * 100.0, n


def align_frames_to_video(
    frames_df: pd.DataFrame,
    node_by_frame: Sequence,
    reward_frame: np.ndarray,
    reward_port: np.ndarray,
    layout: Dict[int, Tuple[str, List[str]]],
    n_video_frames: int,
    fps: float = 40.0,
    min_on_port: float = ALIGN_MIN_ON_PORT,
    max_shift: int = ALIGN_MAX_SHIFT,
    session_label: str = "this session",
) -> Tuple[pd.DataFrame, FrameAlignment]:
    """Verify (and if needed correct) the video/BPOD clock offset, then trim.

    Measures reward-on-port alignment at zero shift. If it clears *min_on_port*
    the frame table is left as-is; otherwise a constant camera-start offset is
    searched for — coarsely every 50 frames over ``+/- max_shift``, then frame by
    frame around the best hit — and subtracted from ``frame_idx_global``.

    Frames falling outside the video are trimmed for *every* session, not only
    shifted ones, so the trial-loss report is accurate rather than the trimming
    happening silently somewhere downstream.

    Raises
    ------
    AlignmentError
        If too few reward events fall inside the video to judge alignment, or if
        no shift brings the session up to *min_on_port* — meaning this is not a
        simple camera-start offset and the video, tracking and BPOD files should
        be checked against each other.
    """
    bpod_frames = int(frames_df.attrs.get("bpod_frame_count",
                                          frames_df["frame_idx_global"].max() + 1))
    overshoot = bpod_frames - n_video_frames
    trial_start = frames_df.groupby("trial_idx")["frame_idx_global"].min()
    all_trials = set(frames_df["trial_idx"].unique())

    def pct(shift):
        return on_port_fraction(reward_frame, reward_port, node_by_frame,
                                layout, n_video_frames, shift)

    pct0, n0 = pct(0)
    if np.isnan(pct0):
        raise AlignmentError(
            f"Only {n0} reward events fall inside the video for {session_label} — "
            "too few to verify alignment. Check that the video, the tracking file "
            "and the BPOD .mat all belong to this session."
        )

    frame_shift = 0
    best_pct = pct0
    snapped = False

    if pct0 < min_on_port:
        coarse = [(s, pct(s)[0]) for s in range(-max_shift, max_shift + 1, 50)]
        best = max([(s, v) for s, v in coarse if v == v], key=lambda t: t[1])[0]
        fine = [(s, pct(s)[0]) for s in range(best - 60, best + 61)]
        frame_shift, best_pct = max([(s, v) for s, v in fine if v == v],
                                    key=lambda t: t[1])

        # The peak is a broad plateau (23 frames wide on NPC4 01/06/2026). Where
        # the exact-fit shift -- the one that also makes the two clocks end
        # together -- sits inside that plateau, prefer it: it is the physically
        # meaningful point, and the raw argmax within a plateau is decided by one
        # or two events of noise.
        if abs(overshoot - frame_shift) <= 60 and pct(overshoot)[0] >= best_pct - 1.0:
            frame_shift, best_pct = overshoot, pct(overshoot)[0]
            snapped = True

        if best_pct < min_on_port:
            raise AlignmentError(
                f"Cannot align {session_label}: the best shift found ({frame_shift}) "
                f"still puts only {best_pct:.1f}% of rewards on the paying port. "
                "This is not a simple camera-start offset. Check that the video, "
                "the tracking file and the BPOD .mat all belong to this session, "
                "and that tracking quality on this day is good."
            )

        frame_shift = int(frame_shift)
        frames_df = frames_df.copy()
        frames_df["frame_idx_global"] -= frame_shift

    n_before = len(frames_df)
    out = frames_df[frames_df["frame_idx_global"].between(0, n_video_frames - 1)]
    out = out.reset_index(drop=True)
    out.attrs["bpod_frame_count"] = bpod_frames

    shifted_start = trial_start - frame_shift
    gone = sorted(all_trials - set(out["trial_idx"].unique()))
    alignment = FrameAlignment(
        frame_shift=frame_shift,
        on_port_pct=float(best_pct),
        on_port_pct_raw=float(pct0),
        n_events=n0,
        n_video_frames=n_video_frames,
        overshoot=overshoot,
        snapped_to_exact_fit=snapped,
        n_frames_dropped=n_before - len(out),
        trials_before_start=[int(t) for t in gone if shifted_start.get(t, 0) < 0],
        trials_after_stop=[int(t) for t in gone if shifted_start.get(t, 0) >= n_video_frames],
    )
    return out, alignment


__all__ = [
    "ALIGN_MIN_ON_PORT",
    "ALIGN_MAX_SHIFT",
    "AlignmentError",
    "FrameAlignment",
    "build_frames_df",
    "bpod_reward_frames",
    "on_port_fraction",
    "align_frames_to_video",
]
