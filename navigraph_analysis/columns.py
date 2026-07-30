"""``summary_df`` column resolution helpers.

``summary_df`` may arrive with flat columns (``headstage_x``) or a two-level
MultiIndex (``('headstage', 'x')``, ``('frame_idx_global', '')``). Every notebook
re-derived accessors for this; here they live once.

    from navigraph_analysis.columns import resolve, flatten_columns, xy_frame_cols

    x = summary_df[resolve(summary_df, "headstage", "x")]
    flat = flatten_columns(summary_df)          # returns a flat-column copy
    xc, yc, fc = xy_frame_cols(summary_df, bodypart="headstage")
"""

from __future__ import annotations

from typing import List, Sequence, Tuple, Union

import pandas as pd

ColKey = Union[str, tuple]


def resolve(df: pd.DataFrame, *wanted: str) -> ColKey:
    """Resolve the column whose (any-level) name parts contain all *wanted* tokens.

    Case-insensitive substring match. Works on flat or MultiIndex columns.
    Raises KeyError if nothing matches.

    Examples
    --------
    ``resolve(df, "headstage", "x")`` -> ``('headstage', 'x')`` or ``'headstage_x'``
    ``resolve(df, "frame_idx_global")`` -> ``('frame_idx_global', '')`` or the flat name
    """
    wanted_l = [w.lower() for w in wanted]
    for col in df.columns:
        parts = [str(v).strip().lower() for v in (col if isinstance(col, tuple) else (col,))]
        if all(any(w == p or w in p for p in parts) for w in wanted_l):
            return col
    raise KeyError(
        f"No column matching {wanted}; have e.g. {list(df.columns[:8])}"
    )


def try_resolve(df: pd.DataFrame, *wanted: str) -> ColKey | None:
    """Like :func:`resolve` but returns ``None`` instead of raising."""
    try:
        return resolve(df, *wanted)
    except KeyError:
        return None


def flatten_columns(df: pd.DataFrame, sep: str = "_") -> pd.DataFrame:
    """Return a copy of *df* with MultiIndex columns joined into flat strings.

    ``('headstage', 'x')`` -> ``'headstage_x'``; ``('frame_idx_global', '')`` ->
    ``'frame_idx_global'``. No-op for already-flat frames.
    """
    if not isinstance(df.columns, pd.MultiIndex):
        return df.copy()
    out = df.copy()
    out.columns = [
        sep.join(str(p) for p in col if str(p) != "") for col in df.columns
    ]
    return out


def xy_frame_cols(
    df: pd.DataFrame,
    bodypart: str = "headstage",
) -> Tuple[ColKey, ColKey, ColKey]:
    """Resolve the ``(x, y, frame_idx_global)`` column keys for *bodypart*."""
    return (
        resolve(df, bodypart, "x"),
        resolve(df, bodypart, "y"),
        resolve(df, "frame_idx_global"),
    )


def likelihood_col(df: pd.DataFrame, bodypart: str = "headstage") -> ColKey | None:
    return try_resolve(df, bodypart, "likelihood")


def bodyparts(df: pd.DataFrame) -> List[str]:
    """List distinct bodypart names present as the first level of the columns."""
    seen: list[str] = []
    for col in df.columns:
        first = str(col[0]) if isinstance(col, tuple) else str(col)
        # heuristic: bodypart columns pair with x/y/likelihood
        base = first.split("_")[0]
        if base and base not in seen:
            seen.append(base)
    return seen


__all__ = [
    "resolve",
    "try_resolve",
    "flatten_columns",
    "xy_frame_cols",
    "likelihood_col",
    "bodyparts",
]
