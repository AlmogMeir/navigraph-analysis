"""Assemble the canonical ``summary_df`` from raw NaviGraph output + BPOD.

NaviGraph's ``raw_session_data_*.pkl`` is a per-video-frame table of tracked
coordinates and graph locations. It carries **no trial structure and no global
frame index** — those come from the behavioural control system, and deriving
them is the first real stage of the pipeline. This module is that stage; it
reproduces cells 2-64 of ``session_pipeline.ipynb``.

    raw NaviGraph frame table  (one row per video frame)
      + BPOD trial durations   --frame_alignment.build_frames_df-->  frames_df
      + reward events          --frame_alignment.align_frames_to_video--> aligned
      -> join on frame_idx_global, attach reward / lick / bodypart provenance
      -> summary_df

The output column layout matches what the notebook wrote, because the saved
tables are the interface other analyses already read:

- pose columns become a two-level ``(bodypart, coord)`` MultiIndex;
- frame/trial metadata and annotations sit at ``(name, '')``;
- the session's chosen bodypart is copied to the wire names
  ``headstage_graph_node`` / ``headstage_graph_edge`` regardless of which
  bodypart it came from, with ``graph_bodypart`` recording the provenance so a
  nose session is never mistaken for a headstage one downstream.

    from navigraph_analysis.prep.ingest import build_summary_df

    summary_df, report = build_summary_df(raw_df, subject="NPC1", date=date,
                                          layout=layout, bpod_dir=cfg.bpod_dir)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..io import bpod as iobpod
from ..io.dlc import CANONICAL_BODYPARTS, resolve_bodypart
from .frame_alignment import FrameAlignment, align_frames_to_video, build_frames_df

#: Written under these names whatever bodypart actually drove graph location.
GRAPH_NODE_COL = "headstage_graph_node"
GRAPH_EDGE_COL = "headstage_graph_edge"

#: Half a frame interval at 40 fps: a lick is on the frame it falls closest to.
LICK_TIME_TOLERANCE = 0.013

#: Frame-metadata columns carried over from the BPOD frame table.
FRAME_META_COLS = [
    "trial_idx", "frame_idx_global", "frame_idx_in_trial", "frame_time_in_trial",
    "trial_length", "is_exact_multiple", "has_extra_frame",
]

_POSE_SUFFIXES = ("x", "y", "likelihood")


@dataclass
class IngestReport:
    """What the ingest stage did, for logging and for quality gating."""

    bodypart: str
    n_video_frames: int
    n_frames_kept: int
    n_trials: int
    alignment: Optional[FrameAlignment] = None
    n_reward_events: int = 0
    n_lick_frames: int = 0
    notes: List[str] = field(default_factory=list)

    def summary(self) -> str:
        parts = [
            f"bodypart={self.bodypart}",
            f"{self.n_frames_kept}/{self.n_video_frames} frames kept",
            f"{self.n_trials} trials",
            f"{self.n_reward_events} rewards",
            f"{self.n_lick_frames} lick frames",
        ]
        if self.alignment is not None:
            parts.append(self.alignment.summary())
        return "; ".join(parts + self.notes)


def graph_bodyparts(raw_df: pd.DataFrame) -> List[str]:
    """Bodyparts that have a ``*_graph_node`` column with any value."""
    out = []
    for col in raw_df.columns:
        name = str(col)
        if name.endswith("_graph_node") and raw_df[col].notna().any():
            out.append(name[: -len("_graph_node")])
    return out


def _pose_multiindex(raw_df: pd.DataFrame, frame_rows: np.ndarray) -> pd.DataFrame:
    """Pose columns for the kept frames, as a ``(bodypart, coord)`` frame."""
    data: Dict[Tuple[str, str], np.ndarray] = {}
    for col in raw_df.columns:
        name = str(col)
        for suffix in _POSE_SUFFIXES:
            tail = f"_{suffix}"
            if name.endswith(tail) and not name.endswith("_graph_node") \
                    and not name.endswith("_graph_edge"):
                bp = name[: -len(tail)]
                # The heatmap tracker lower-cases ear_R/ear_L; downstream code
                # looks bodyparts up by name, so canonicalise here too.
                bp = CANONICAL_BODYPARTS.get(bp, bp)
                data[(bp, suffix)] = raw_df[col].to_numpy()[frame_rows]
                break
    out = pd.DataFrame(data)
    out.columns = pd.MultiIndex.from_tuples(out.columns)
    return out.reset_index(drop=True)


def build_summary_df(
    raw_df: pd.DataFrame,
    subject: str,
    date,
    layout: Dict[int, Tuple[str, List[str]]],
    bpod_dir: str | Path,
    fps: float = 40.0,
    bodypart: Optional[str] = None,
    check_alignment: bool = True,
    bpod_protocol: Optional[str] = None,
    output_protocol: str = iobpod.DEFAULT_OUTPUT_PROTOCOL,
) -> Tuple[pd.DataFrame, IngestReport]:
    """Build ``summary_df`` for one session from raw NaviGraph output + BPOD.

    Parameters
    ----------
    raw_df:
        NaviGraph's per-video-frame table (from
        :func:`navigraph_analysis.io.navigraph_output.load_raw_session_data`).
        One row per video frame, in order.
    layout:
        ``{port number: (leaf node, [parents])}`` for this session, from
        :meth:`navigraph_analysis.ports.PortConfig.resolve_layout`.
    bodypart:
        Which tracked point drives graph location. ``None`` auto-picks the first
        available of headstage, nose among the bodyparts that actually have
        graph locations in this session.
    check_alignment:
        Run the reward-on-port clock check (recommended). Disable only to
        inspect a session that fails it.

    Returns
    -------
    ``(summary_df, IngestReport)``.
    """
    notes: List[str] = []
    n_video_frames = len(raw_df)

    available = graph_bodyparts(raw_df)
    if not available:
        raise KeyError(
            "No *_graph_node column in the raw session data has any value; "
            "NaviGraph produced no graph locations for this session."
        )
    # Restrict the auto-pick to bodyparts that actually carry graph locations.
    bp = resolve_bodypart(
        pd.DataFrame(columns=pd.MultiIndex.from_product([available, ["x"]])),
        preferred=bodypart,
    )
    node_col, edge_col = f"{bp}_graph_node", f"{bp}_graph_edge"

    # -- trial structure from BPOD -------------------------------------------
    mat_path = iobpod.find_bpod_mat(bpod_dir, subject, date, protocol=bpod_protocol)
    session_data = iobpod.load_bpod_session(mat_path)
    durations = iobpod.trial_end_times(session_data)
    frames_df = build_frames_df(durations, fps=fps)
    n_trials = len(durations)

    # -- reward events on the BPOD clock -------------------------------------
    reward_csv = iobpod.load_reward_csv(bpod_dir, subject, date,
                                        output_protocol=output_protocol)
    events = _reward_events(reward_csv, frames_df, layout, fps)

    # -- verify the two clocks agree, and trim to the video ------------------
    alignment = None
    if check_alignment:
        node_by_frame = raw_df[node_col].astype(object).to_numpy()
        frames_df, alignment = align_frames_to_video(
            frames_df, node_by_frame,
            events["frame_idx_global"].to_numpy(), events["reward_port"].to_numpy(),
            layout, n_video_frames, fps=fps,
            session_label=f"{subject} {date}",
        )
        if alignment.frame_shift:
            # Reward frames were computed on the pre-shift clock.
            events["frame_idx_global"] -= alignment.frame_shift
    else:
        before = len(frames_df)
        frames_df = frames_df[
            frames_df["frame_idx_global"].between(0, n_video_frames - 1)
        ].reset_index(drop=True)
        if before != len(frames_df):
            notes.append(f"trimmed {before - len(frames_df)} out-of-video frame(s)")

    frame_rows = frames_df["frame_idx_global"].to_numpy().astype(int)

    # -- assemble -------------------------------------------------------------
    summary_df = _pose_multiindex(raw_df, frame_rows)
    for col in FRAME_META_COLS:
        if col in frames_df.columns:
            summary_df[(col, "")] = frames_df[col].to_numpy()

    summary_df[(GRAPH_NODE_COL, "")] = raw_df[node_col].to_numpy(dtype=object)[frame_rows]
    if edge_col in raw_df.columns:
        summary_df[(GRAPH_EDGE_COL, "")] = raw_df[edge_col].to_numpy(dtype=object)[frame_rows]
    else:
        summary_df[(GRAPH_EDGE_COL, "")] = np.nan
        notes.append(f"no {edge_col} column; graph edges unavailable")
    summary_df[("graph_bodypart", "")] = bp

    # -- reward annotation ----------------------------------------------------
    events = events[events["frame_idx_global"].between(0, n_video_frames - 1)]
    # frame_idx_global -> row position in summary_df. A reward whose frame was
    # trimmed (outside the video, or the skipped extra frame of a trial) has no
    # row and is dropped rather than misplaced.
    row_of_frame = pd.Series(np.arange(len(frame_rows)), index=frame_rows)
    ev_row = events["frame_idx_global"].map(row_of_frame)
    landed = ev_row.notna()

    reward_node = np.zeros(len(summary_df), dtype=bool)
    reward_size = np.full(len(summary_df), np.nan)
    reward_port = np.full(len(summary_df), np.nan)
    if landed.any():
        idx = ev_row[landed].to_numpy().astype(int)
        matched = events[landed]
        reward_node[idx] = True
        reward_size[idx] = matched["reward_size"].to_numpy()
        reward_port[idx] = matched["reward_port"].to_numpy()
    if (~landed).any():
        notes.append(f"{int((~landed).sum())} reward event(s) fell on a trimmed frame")
    summary_df[("reward_node", "")] = reward_node
    summary_df[("reward_size", "")] = reward_size
    summary_df[("reward_port", "")] = reward_port

    # -- lick annotation ------------------------------------------------------
    n_lick_frames = 0
    try:
        lick_csv = iobpod.load_lick_csv(bpod_dir, subject, date,
                                        output_protocol=output_protocol)
        lick = _lick_mask(lick_csv, frames_df, fps)
        summary_df[("lick", "")] = lick
        n_lick_frames = int(lick.sum())
    except iobpod.BpodDiscoveryError:
        summary_df[("lick", "")] = False
        notes.append("no lick_events.csv; lick column is all False")

    summary_df.columns = pd.MultiIndex.from_tuples(summary_df.columns)

    report = IngestReport(
        bodypart=bp,
        n_video_frames=n_video_frames,
        n_frames_kept=len(summary_df),
        n_trials=n_trials,
        alignment=alignment,
        n_reward_events=int(len(events)),
        n_lick_frames=n_lick_frames,
        notes=notes,
    )
    return summary_df, report


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _reward_events(
    reward_csv: pd.DataFrame,
    frames_df: pd.DataFrame,
    layout: Dict[int, Tuple[str, List[str]]],
    fps: float,
) -> pd.DataFrame:
    """Reward events placed on the global frame clock.

    Columns: ``frame_idx_global, reward_port, reward_size, trial_idx``. Trials
    outside the session and ports absent from *layout* are dropped.
    """
    trial_col = "trial" if "trial" in reward_csv.columns else reward_csv.columns[0]
    time_col = next(c for c in reward_csv.columns
                    if "time" in c.lower() and "abs" not in c.lower())
    port_col = next(c for c in reward_csv.columns if "port" in c.lower())
    size_col = next((c for c in reward_csv.columns if "size" in c.lower()), None)

    trial_start = frames_df.groupby("trial_idx")["frame_idx_global"].min()
    n_trials = int(frames_df["trial_idx"].max()) + 1

    rw = reward_csv[(reward_csv[trial_col].astype(int) - 1).between(0, n_trials - 1)]
    trial_idx = rw[trial_col].astype(int).to_numpy() - 1
    frame = trial_start.reindex(trial_idx).to_numpy() + np.round(
        rw[time_col].to_numpy() * fps)
    port = rw[port_col].astype(int).to_numpy()

    out = pd.DataFrame({
        "frame_idx_global": frame,
        "reward_port": port,
        "reward_size": rw[size_col].to_numpy() if size_col else np.nan,
        "trial_idx": trial_idx,
    })
    out = out[out["frame_idx_global"].notna() & out["reward_port"].isin(layout)]
    out["frame_idx_global"] = out["frame_idx_global"].astype(int)
    return out.reset_index(drop=True)


def _lick_mask(lick_csv: pd.DataFrame, frames_df: pd.DataFrame, fps: float) -> np.ndarray:
    """Per-frame boolean: did a lick land on this frame?

    Matched within the trial by ``|lick_time - frame_time_in_trial| <= tolerance``,
    the same rule the notebook used.
    """
    trial_col = "trial" if "trial" in lick_csv.columns else lick_csv.columns[0]
    time_col = next(c for c in lick_csv.columns
                    if "time" in c.lower() and "absolute" not in c.lower()
                    and "duration" not in c.lower())

    trials = frames_df["trial_idx"].to_numpy()
    times = frames_df["frame_time_in_trial"].to_numpy()
    out = np.zeros(len(frames_df), dtype=bool)

    licks = lick_csv[[trial_col, time_col]].dropna()
    by_trial: Dict[int, np.ndarray] = {
        int(t) - 1: g[time_col].to_numpy()
        for t, g in licks.groupby(trial_col)
    }
    for t in np.unique(trials):
        lt = by_trial.get(int(t))
        if lt is None or not len(lt):
            continue
        mask = trials == t
        ft = times[mask]
        hit = (np.abs(ft[:, None] - lt[None, :]) <= LICK_TIME_TOLERANCE).any(axis=1)
        out[np.flatnonzero(mask)] = hit
    return out


__all__ = [
    "GRAPH_NODE_COL",
    "GRAPH_EDGE_COL",
    "FRAME_META_COLS",
    "IngestReport",
    "build_summary_df",
    "graph_bodyparts",
]
