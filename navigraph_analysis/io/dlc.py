"""DeepLabCut pose loading.

DLC ``*_corrected.h5`` files are pandas HDF stores (key ``df_with_missing``)
holding a MultiIndex-column DataFrame. Depending on export they carry a
``(scorer, bodypart, coord)``, ``(bodypart, coord)`` or flat ``{bp}_x`` layout.
:func:`load_dlc_pose` normalizes to a two-level ``(bodypart, coord)`` frame and
optionally NaNs out low-likelihood samples.

Normalization also canonicalizes bodypart *case*: the newer heatmap tracker
exports ``ear_r`` / ``ear_l`` where the older DLC model exported ``ear_R`` /
``ear_L``. Everything downstream indexes ``(bodypart, 'x')`` by name, so the two
exports would otherwise silently look like different bodyparts.

:func:`resolve_bodypart` picks which tracked point drives graph location: the
preferred one if this session tracked it, else the first available fallback.
Sessions differ in what was tracked, so pinning one name in a config means
editing it per session.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def load_dlc_pose(
    h5_path: str | Path,
    likelihood_threshold: Optional[float] = None,
) -> pd.DataFrame:
    """Load a DLC ``.h5`` into a ``(bodypart, coord)`` MultiIndex DataFrame.

    Parameters
    ----------
    h5_path:
        Path to the DLC HDF file.
    likelihood_threshold:
        If given, x/y are set to NaN wherever that bodypart's likelihood is below
        the threshold.
    """
    path = Path(h5_path).expanduser()
    df = pd.read_hdf(path)
    if not isinstance(df, pd.DataFrame):  # pragma: no cover - defensive
        raise TypeError(f"{path} did not contain a DataFrame")

    df = normalize_bodypart_case(_drop_scorer_level(df))

    if likelihood_threshold is not None and isinstance(df.columns, pd.MultiIndex):
        for bp in df.columns.get_level_values(0).unique():
            if (bp, "likelihood") in df.columns:
                bad = df[(bp, "likelihood")] < likelihood_threshold
                df.loc[bad, (bp, "x")] = np.nan
                df.loc[bad, (bp, "y")] = np.nan
    return df


def _drop_scorer_level(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse a 3-level ``(scorer, bodypart, coord)`` column index to 2 levels.

    Also handles the >3-level case some exports produce.
    """
    if isinstance(df.columns, pd.MultiIndex) and df.columns.nlevels > 2:
        df = df.copy()
        while df.columns.nlevels > 2:
            df.columns = df.columns.droplevel(0)
    return df


#: Bodypart spellings that differ between tracker versions, mapped to canonical.
CANONICAL_BODYPARTS = {"ear_r": "ear_R", "ear_l": "ear_L"}

#: Order in which :func:`resolve_bodypart` picks a tracked point when none is asked for.
BODYPART_PREFERENCE = ("headstage", "nose")


def normalize_bodypart_case(df: pd.DataFrame) -> pd.DataFrame:
    """Rename bodyparts whose case differs between tracker versions.

    The heatmap tracker lower-cases ``ear_R`` / ``ear_L``; downstream code looks
    them up by name, so normalize here rather than in each user.
    """
    if not isinstance(df.columns, pd.MultiIndex) or df.columns.nlevels != 2:
        return df
    renamed = [(CANONICAL_BODYPARTS.get(str(b), str(b)), c) for b, c in df.columns]
    if renamed == list(df.columns):
        return df
    df = df.copy()
    df.columns = pd.MultiIndex.from_tuples(renamed)
    return df


def resolve_bodypart(
    df: pd.DataFrame,
    preferred: Optional[str] = None,
    preference: tuple = BODYPART_PREFERENCE,
) -> str:
    """Which tracked point to drive graph location with.

    *preferred* wins when this session tracked it. Otherwise the first name in
    *preference* that is present is used, so a session without a headstage falls
    back to the nose instead of failing.

    Raises
    ------
    KeyError
        If *preferred* was named but not tracked, or if no fallback is present.
    """
    available = bodyparts(df)
    if preferred is not None:
        if preferred in available:
            return preferred
        raise KeyError(
            f"Bodypart {preferred!r} is not tracked in this session; "
            f"available: {available}"
        )
    for name in preference:
        if name in available:
            return name
    raise KeyError(
        f"None of the preferred bodyparts {list(preference)} are tracked; "
        f"available: {available}"
    )


def bodyparts(df: pd.DataFrame) -> list[str]:
    """Distinct bodypart names in a ``(bodypart, coord)`` pose frame."""
    if isinstance(df.columns, pd.MultiIndex):
        return list(dict.fromkeys(df.columns.get_level_values(0)))
    return list(dict.fromkeys(c.rsplit("_", 1)[0] for c in df.columns))


__all__ = [
    "load_dlc_pose",
    "bodyparts",
    "normalize_bodypart_case",
    "resolve_bodypart",
    "CANONICAL_BODYPARTS",
    "BODYPART_PREFERENCE",
]
