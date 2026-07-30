"""DeepLabCut pose loading.

DLC ``*_corrected.h5`` files are pandas HDF stores (key ``df_with_missing``)
holding a MultiIndex-column DataFrame. Depending on export they carry a
``(scorer, bodypart, coord)``, ``(bodypart, coord)`` or flat ``{bp}_x`` layout.
:func:`load_dlc_pose` normalizes to a two-level ``(bodypart, coord)`` frame and
optionally NaNs out low-likelihood samples.
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

    df = _drop_scorer_level(df)

    if likelihood_threshold is not None and isinstance(df.columns, pd.MultiIndex):
        for bp in df.columns.get_level_values(0).unique():
            if (bp, "likelihood") in df.columns:
                bad = df[(bp, "likelihood")] < likelihood_threshold
                df.loc[bad, (bp, "x")] = np.nan
                df.loc[bad, (bp, "y")] = np.nan
    return df


def _drop_scorer_level(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse a 3-level ``(scorer, bodypart, coord)`` column index to 2 levels."""
    if isinstance(df.columns, pd.MultiIndex) and df.columns.nlevels == 3:
        df = df.copy()
        df.columns = df.columns.droplevel(0)
    return df


def bodyparts(df: pd.DataFrame) -> list[str]:
    """Distinct bodypart names in a ``(bodypart, coord)`` pose frame."""
    if isinstance(df.columns, pd.MultiIndex):
        return list(dict.fromkeys(df.columns.get_level_values(0)))
    return list(dict.fromkeys(c.rsplit("_", 1)[0] for c in df.columns))


__all__ = ["load_dlc_pose", "bodyparts"]
