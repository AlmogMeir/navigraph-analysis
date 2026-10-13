"""BPOD behavioral session loading (optional).

Two things live under a subject's BPOD root, and they are organized differently:

- The **raw** ``<protocol>/Session Data/<subject>_<...>_<YYYYMMDD>_<...>.mat``,
  one per recording. The protocol folder is *not* fixed — a subject runs several
  protocols over an experiment (NPC4 moved from ``Adva_maze_pairs_fix`` to
  ``maze_pairs_fix``), and most maze days also carry a non-maze recording such as
  ``NP_spont`` on the same date. :func:`find_bpod_mat` therefore searches every
  protocol, keeps only maze ones, and refuses to guess when more than one still
  matches rather than silently analysing the wrong recording.
- The **derived** per-session CSVs (``reward``, ``lick_events``, ``port``,
  ``block``) under ``<output_protocol>/Session Data/output/<YYYYMMDD>/``. The
  export puts *every* session's CSVs under one protocol folder, including
  sessions whose raw ``.mat`` lives elsewhere, so
  :data:`DEFAULT_OUTPUT_PROTOCOL` is separate from the raw protocol on purpose.

Ported from the session-discovery cell of ``session_pipeline.ipynb``.

    from navigraph_analysis.io.bpod import find_bpod_mat, load_bpod_csv

    mat = find_bpod_mat(cfg.bpod_dir, "NPC4", session.date)
    reward_df = load_bpod_csv(cfg.bpod_dir, "NPC4", session.date, "reward")
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .spikes import _unwrap_mat

#: Where the derived per-session CSVs are exported for every session.
DEFAULT_OUTPUT_PROTOCOL = "maze_pairs_fix"
#: The derived CSVs each session folder holds.
BPOD_CSV_NAMES = ("reward", "lick_events", "port", "block")


class BpodDiscoveryError(FileNotFoundError):
    """Raised when a subject's BPOD session cannot be located unambiguously."""


def _as_date(date) -> _dt.date:
    if isinstance(date, _dt.datetime):
        return date.date()
    if isinstance(date, _dt.date):
        return date
    from ..session import parse_date
    return parse_date(str(date))


def find_bpod_mat(
    bpod_dir: str | Path,
    subject: str,
    date,
    protocol: Optional[str] = None,
) -> Path:
    """Locate the raw BPOD ``.mat`` for one ``(subject, date)``.

    Searches every protocol folder under ``<bpod_dir>/<subject>/``. Non-maze
    protocols (``NP_spont`` and friends) are dropped when any maze protocol
    matches, since they share the date but not the task. Several ``.mat`` files
    within one protocol on one date are usually an aborted run plus the real one;
    the earliest is taken, which is the long-standing behaviour.

    Parameters
    ----------
    protocol:
        Pin the protocol folder, for a date that matches more than one maze
        protocol or to deliberately analyse a non-maze recording.

    Raises
    ------
    BpodDiscoveryError
        If nothing matches, or if several maze protocols still match and
        *protocol* was not given.
    """
    d = _as_date(date)
    root = Path(bpod_dir).expanduser() / subject
    if not root.is_dir():
        raise BpodDiscoveryError(f"No BPOD folder for subject {subject!r}: {root}")

    stamp = f"{d.year:04d}{d.month:02d}{d.day:02d}"
    hits = sorted(root.glob(f"*/Session Data/{subject}_*_{stamp}_*.mat"))
    if protocol is not None:
        hits = [p for p in hits if p.parent.parent.name == protocol]

    by_protocol: Dict[str, List[Path]] = {}
    for p in hits:
        by_protocol.setdefault(p.parent.parent.name, []).append(p)

    if not by_protocol:
        searched = sorted(x.name for x in root.iterdir() if x.is_dir())
        raise BpodDiscoveryError(
            f"No BPOD .mat for {subject} on {stamp}"
            + (f" in protocol {protocol!r}" if protocol else "")
            + f". Searched under {root}: {searched}"
        )

    maze = {k: v for k, v in by_protocol.items() if "maze" in k.lower()}
    choices = maze or by_protocol
    if len(choices) > 1:
        raise BpodDiscoveryError(
            f"{subject} {stamp} matches several maze protocols: {sorted(choices)}. "
            "Pass protocol= to choose one."
        )

    _name, paths = next(iter(choices.items()))
    return paths[0]


def bpod_output_dir(
    bpod_dir: str | Path,
    subject: str,
    date,
    output_protocol: str = DEFAULT_OUTPUT_PROTOCOL,
) -> Path:
    """Directory holding the derived per-session CSVs for one ``(subject, date)``."""
    d = _as_date(date)
    stamp = f"{d.year:04d}{d.month:02d}{d.day:02d}"
    return (Path(bpod_dir).expanduser() / subject / output_protocol
            / "Session Data" / "output" / stamp)


def load_bpod_csv(
    bpod_dir: str | Path,
    subject: str,
    date,
    name: str = "reward",
    output_protocol: str = DEFAULT_OUTPUT_PROTOCOL,
) -> pd.DataFrame:
    """Load one derived BPOD CSV (see :data:`BPOD_CSV_NAMES`)."""
    directory = bpod_output_dir(bpod_dir, subject, date, output_protocol)
    path = directory / f"{name}.csv"
    if not path.exists():
        available = (sorted(p.name for p in directory.glob("*.csv"))
                     if directory.is_dir() else "the folder does not exist")
        raise BpodDiscoveryError(
            f"No {name}.csv for {subject} on {_as_date(date)} in {directory} "
            f"(present: {available})"
        )
    return pd.read_csv(path)


def load_reward_csv(bpod_dir, subject, date, **kwargs) -> pd.DataFrame:
    """The session's ``reward.csv``: one row per reward delivery.

    Columns include ``trial``, ``port_number``, ``reward_size``,
    ``reward_relative_time`` (seconds into the trial) and ``depletion_step``. This
    is the table :func:`navigraph_analysis.prep.frame_alignment.bpod_reward_frames`
    turns into global frame indices.
    """
    return load_bpod_csv(bpod_dir, subject, date, "reward", **kwargs)


def load_lick_csv(bpod_dir, subject, date, **kwargs) -> pd.DataFrame:
    """The session's ``lick_events.csv``: one row per lick."""
    return load_bpod_csv(bpod_dir, subject, date, "lick_events", **kwargs)


def load_bpod_session(mat_path: str | Path) -> dict:
    """Load a single BPOD ``SessionData`` ``.mat`` into a dict of unwrapped arrays."""
    from scipy.io import loadmat

    mat = loadmat(Path(mat_path).expanduser(), squeeze_me=True, struct_as_record=False)
    key = next((k for k in mat if not k.startswith("__")), None)
    if key is None:
        raise KeyError(f"No SessionData struct in {mat_path}")
    return _struct_to_dict(mat[key])


def _struct_to_dict(struct) -> dict:
    """Recursively convert a scipy mat_struct to plain dict/ndarray."""
    if hasattr(struct, "_fieldnames"):
        return {f: _struct_to_dict(getattr(struct, f)) for f in struct._fieldnames}
    if isinstance(struct, np.ndarray) and struct.dtype == object:
        return _unwrap_mat(struct)
    return struct


#: State whose onset marks the end of the usable part of a trial.
TRIAL_END_STATE = "TrialEnd"


def trial_end_times(session_data: dict, state: str = TRIAL_END_STATE) -> np.ndarray:
    """Per-trial duration in seconds, as the frame table is built from.

    The duration is the **onset of the trial's ``TrialEnd`` state**, read from
    ``RawEvents.Trial[i].States``, not ``TrialEndTimestamp - TrialStartTimestamp``.
    The two differ by the length of the ``TrialEnd`` state itself (0.05 s = 2
    frames at 40 fps), and because the frame table is built by *cumulatively
    summing* these durations, that difference accumulates: over a 500-trial
    session it displaces the last trial by ~1000 frames (25 s), which silently
    moves every reward event off the port that paid it.

    Falls back to the timestamp difference only when ``RawEvents`` is absent,
    and says so via a ``RuntimeWarning``, because that path is known to drift.
    """
    raw_events = session_data.get("RawEvents")
    trials = raw_events.get("Trial") if isinstance(raw_events, dict) else None

    if trials is not None:
        out = []
        for trial in np.atleast_1d(trials):
            states = trial["States"] if isinstance(trial, dict) else getattr(trial, "States", None)
            if states is None:
                break
            value = (states.get(state) if isinstance(states, dict)
                     else getattr(states, state, None))
            if value is None:
                break
            arr = np.asarray(value, dtype=float).ravel()
            if arr.size == 0 or not np.isfinite(arr[0]):
                break
            out.append(float(arr[0]))       # onset of the TrialEnd state
        if len(out) == len(np.atleast_1d(trials)):
            return np.asarray(out, dtype=float)

    if "TrialEndTimestamp" in session_data and "TrialStartTimestamp" in session_data:
        import warnings
        warnings.warn(
            f"SessionData has no usable RawEvents.Trial[...].States.{state}; "
            "falling back to TrialEndTimestamp - TrialStartTimestamp, which "
            "over-counts each trial by the length of the TrialEnd state and "
            "accumulates across the session.",
            RuntimeWarning,
            stacklevel=2,
        )
        end = np.asarray(session_data["TrialEndTimestamp"], dtype=float).ravel()
        start = np.asarray(session_data["TrialStartTimestamp"], dtype=float).ravel()
        return end - start

    raise KeyError(
        "SessionData has no trial-duration source (looked for "
        f"RawEvents.Trial[...].States.{state}, then TrialEndTimestamp/"
        f"TrialStartTimestamp; have: {sorted(session_data)[:15]})"
    )


def find_bpod_file(bpod_dir: str | Path, subject: str) -> Optional[Path]:
    """Deprecated: best-effort discovery with no date or protocol filter.

    Kept so existing callers keep working. Prefer :func:`find_bpod_mat`, which
    takes the session date and will not silently return another day's recording.
    """
    base = Path(bpod_dir).expanduser()
    matches = sorted(base.rglob(f"*{subject}*.mat"))
    return matches[-1] if matches else None


__all__ = [
    "DEFAULT_OUTPUT_PROTOCOL",
    "BPOD_CSV_NAMES",
    "BpodDiscoveryError",
    "find_bpod_mat",
    "bpod_output_dir",
    "load_bpod_csv",
    "load_reward_csv",
    "load_lick_csv",
    "load_bpod_session",
    "trial_end_times",
    "TRIAL_END_STATE",
    "find_bpod_file",
]
