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


def _coerce_frame(obj: object) -> pd.DataFrame:
    """NaviGraph raw pickles are sometimes a DataFrame, sometimes a dict wrapper."""
    if isinstance(obj, pd.DataFrame):
        return obj
    if isinstance(obj, dict):
        for key in ("summary_df", "dataframe", "df", "data"):
            if key in obj and isinstance(obj[key], pd.DataFrame):
                return obj[key]
        # single-session dict keyed by session name
        frames = [v for v in obj.values() if isinstance(v, pd.DataFrame)]
        if len(frames) == 1:
            return frames[0]
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
