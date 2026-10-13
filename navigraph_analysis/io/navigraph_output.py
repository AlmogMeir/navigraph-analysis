"""Loading NaviGraph pipeline output and reading/writing canonical tables.

Two kinds of artifact:

1. **Raw NaviGraph output** — ``raw_session_data_*.pkl`` (the integrated
   frame-level DataFrame) and ``session_metrics_*.pkl``, discovered under
   ``<navigraph_output>/<DD_MM_YYYY>/`` (latest match wins). This is the entry
   point for the from-raw pipeline.
2. **Canonical intermediate tables** — ``summary_df`` / ``traversal_df`` /
   ``node_df`` / ``node_edge_df`` / ``reward_df_annotated``, written to and
   re-read from the session's output dir as paired ``.pkl``/``.csv``.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import pandas as pd

from ..session import Session


def _latest(directory: Path, pattern: str) -> Optional[Path]:
    """Newest file matching *pattern* in *directory* (by mtime), or None."""
    if not directory.exists():
        return None
    matches = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime)
    return matches[-1] if matches else None


def load_raw_session_data(session: Session) -> pd.DataFrame:
    """Load the newest ``raw_session_data_*.pkl`` for *session*."""
    path = _latest(session.navigraph_dir, "raw_session_data_*.pkl")
    if path is None:
        raise FileNotFoundError(
            f"No raw_session_data_*.pkl in {session.navigraph_dir}"
        )
    with open(path, "rb") as fh:
        obj = pickle.load(fh)
    return _coerce_frame(obj)


def load_session_metrics(session: Session) -> object:
    """Load the newest ``session_metrics_*.pkl`` for *session* (raw object)."""
    path = _latest(session.navigraph_dir, "session_metrics_*.pkl")
    if path is None:
        raise FileNotFoundError(
            f"No session_metrics_*.pkl in {session.navigraph_dir}"
        )
    with open(path, "rb") as fh:
        return pickle.load(fh)


#: Keys a NaviGraph raw pickle may store its frame table under, at any nesting depth.
_FRAME_KEYS = ("summary_df", "dataframe", "df", "data")


def _coerce_frame(obj: object, _depth: int = 0) -> pd.DataFrame:
    """Extract the frame table from a NaviGraph raw pickle.

    The layout has varied across NaviGraph versions. Current output nests one
    level per session::

        {"session_<subject>_<DD_MM_YYYY>": {"dataframe": <DataFrame>,
                                            "session_path": str},
         "timestamp": str, "experiment_path": str}

    Older output stored the DataFrame at the top level, or under one of
    :data:`_FRAME_KEYS`. All are accepted; a multi-session pickle is rejected
    rather than guessed at.
    """
    if isinstance(obj, pd.DataFrame):
        return obj
    if isinstance(obj, dict) and _depth < 3:
        for key in _FRAME_KEYS:
            if key in obj and isinstance(obj[key], pd.DataFrame):
                return obj[key]
        frames = [v for v in obj.values() if isinstance(v, pd.DataFrame)]
        if len(frames) == 1:
            return frames[0]
        if len(frames) > 1:
            raise TypeError(
                f"Raw session pickle holds {len(frames)} DataFrames; expected one. "
                "This looks like a multi-session export — split it per session first."
            )
        # Recurse into nested per-session dicts (the current layout).
        nested = [_coerce_frame(v, _depth + 1) for v in obj.values()
                  if isinstance(v, dict)]
        if len(nested) == 1:
            return nested[0]
        if len(nested) > 1:
            raise TypeError(
                f"Raw session pickle holds {len(nested)} sessions; expected one. "
                "Load the session you want directly."
            )
    raise TypeError(f"Cannot coerce raw session object of type {type(obj)} to DataFrame")


# ---------------------------------------------------------------------------
# Canonical intermediate tables (paired .pkl / .csv in the session output dir)
# ---------------------------------------------------------------------------

def save_table(df: pd.DataFrame, session: Session, name: str, csv: bool = True) -> Path:
    """Write *df* to ``<output_dir>/<name>.pkl`` (+ ``.csv``). Returns the pkl path."""
    out = session.output_dir
    pkl = out / f"{name}.pkl"
    df.to_pickle(pkl)
    if csv:
        df.to_csv(out / f"{name}.csv", index=False)
    return pkl


def load_table(session: Session, name: str) -> pd.DataFrame:
    """Load ``<output_dir>/<name>.pkl`` (falls back to ``.csv``)."""
    out = session.output_dir
    pkl = out / f"{name}.pkl"
    if pkl.exists():
        return pd.read_pickle(pkl)
    csv = out / f"{name}.csv"
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(f"No table {name!r} (.pkl/.csv) in {out}")


def has_table(session: Session, name: str) -> bool:
    out = session.output_dir
    return (out / f"{name}.pkl").exists() or (out / f"{name}.csv").exists()


__all__ = [
    "load_raw_session_data",
    "load_session_metrics",
    "save_table",
    "load_table",
    "has_table",
]
