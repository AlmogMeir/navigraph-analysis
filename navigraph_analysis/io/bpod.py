"""BPOD behavioral session loading (optional).

BPOD ``Session Data/*.mat`` files hold the trial-by-trial reward protocol. This
loader returns the raw MATLAB struct (unwrapped) plus, where derivable, a tidy
per-trial reward table used by the raster / reward views.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .spikes import _unwrap_mat


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


def find_bpod_file(bpod_dir: str | Path, subject: str) -> Optional[Path]:
    """Best-effort discovery of a subject's BPOD Session Data ``.mat``."""
    base = Path(bpod_dir).expanduser()
    matches = sorted(base.rglob(f"*{subject}*.mat"))
    return matches[-1] if matches else None


__all__ = ["load_bpod_session", "find_bpod_file"]
