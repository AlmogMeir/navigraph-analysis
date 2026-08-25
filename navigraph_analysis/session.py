"""Session identity, date normalization and per-session path resolution.

Every notebook re-implemented session/date parsing, and the dataset mixes two
date orderings (``DD_MM_YYYY`` in folder names, ``YYYY_MM_DD`` in file stems).
:class:`Session` normalizes both once, and derives every per-session path from
a :class:`~navigraph_analysis.config.ProjectConfig`.

    from navigraph_analysis.config import ProjectConfig
    from navigraph_analysis.session import Session

    cfg = ProjectConfig.load()
    s = Session.parse("NPC1", "21_08_2025", cfg)   # or "2025-08-21", "2025_08_21"
    s.date_iso          # '2025-08-21'
    s.folder_ddmmyyyy   # '21_08_2025'   (navigraph_output / spikes folders)
    s.file_stem         # 'NPC1_2025_08_21'  (DLC .h5, images, videos)
    s.output_dir        # <intermediate>/NPC1/2025-08-21/  (created on access)
"""

from __future__ import annotations

import datetime as _dt
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .config import ProjectConfig

_SEP = re.compile(r"[/_\-.]")


def parse_date(token: str) -> _dt.date:
    """Parse a date from any ordering/separator used across the dataset.

    Accepts ``DD_MM_YYYY``, ``YYYY_MM_DD``, ``DD/MM/YYYY``, ``YYYY-MM-DD`` etc.
    Disambiguates by which field is the 4-digit year.
    """
    parts = [p for p in _SEP.split(str(token).strip()) if p]
    if len(parts) != 3:
        raise ValueError(f"Cannot parse date from {token!r}: need 3 numeric fields")
    nums = [int(p) for p in parts]
    if len(parts[0]) == 4:                 # YYYY_MM_DD
        y, m, d = nums
    elif len(parts[2]) == 4:               # DD_MM_YYYY
        d, m, y = nums
    else:
        raise ValueError(f"Cannot find 4-digit year in {token!r}")
    return _dt.date(y, m, d)


@dataclass(frozen=True)
class Session:
    """One recording session for one subject."""

    subject: str
    date: _dt.date
    cfg: ProjectConfig

    # -- id / date renderings -------------------------------------------------
    @property
    def date_iso(self) -> str:
        return self.date.isoformat()                       # 2025-08-21

    @property
    def folder_ddmmyyyy(self) -> str:
        return self.date.strftime("%d_%m_%Y")              # 21_08_2025

    @property
    def date_yyyy_mm_dd(self) -> str:
        return self.date.strftime("%Y_%m_%d")              # 2025_08_21

    @property
    def file_stem(self) -> str:
        return f"{self.subject}_{self.date_yyyy_mm_dd}"     # NPC1_2025_08_21

    @property
    def session_id(self) -> str:
        return f"{self.subject}_{self.folder_ddmmyyyy}"     # NPC1_21_08_2025

    # -- input directories ----------------------------------------------------
    def _session_subdir(self, base: Optional[Path]) -> Optional[Path]:
        """``<base>/<subject>/<DD_MM_YYYY>`` (subject-nested), falling back to the
        flat ``<base>/<DD_MM_YYYY>`` when the subject subfolder is absent.

        Paths were re-organized into per-subject subfolders; this keeps both
        layouts working during the migration.
        """
        if base is None:
            return None
        nested = base / self.subject / self.folder_ddmmyyyy
        flat = base / self.folder_ddmmyyyy
        if nested.exists() or not flat.exists():
            return nested
        return flat

    @property
    def navigraph_dir(self) -> Path:
        return self._session_subdir(self.cfg.navigraph_output_dir)

    @property
    def spikes_dir(self) -> Optional[Path]:
        return self._session_subdir(self.cfg.spikes_dir)

    def dlc_h5(self, suffix: str = "_corrected") -> Optional[Path]:
        """Resolve the DLC pose file for this session, if a dlc dir is configured."""
        base = self.cfg.dlc_dir
        if base is None:
            return None
        cand = base / f"{self.file_stem}{suffix}.h5"
        return cand

    def maze_image(self) -> Optional[Path]:
        base = self.cfg.maze_images_dir
        return None if base is None else base / f"{self.file_stem}.png"

    def video(self) -> Optional[Path]:
        base = self.cfg.videos_dir
        return None if base is None else base / f"{self.file_stem}.mp4"

    def graph_mapping_pkl(self) -> Optional[Path]:
        """Per-session graph mapping ``maze_mappings/<subject>_<YYYY_MM_DD>.pkl``."""
        base = self.cfg.paths.get("maze_mappings")
        return None if base is None else base / f"{self.file_stem}.pkl"

    def calibration_npy(self) -> Optional[Path]:
        """Per-session calibration ``transform_matrix/<YYYY_MM_DD>.npy``.

        Calibration became per-session (a ``transform_matrix/`` directory of
        dated ``.npy`` files); falls back to a legacy flat ``transform_matrix.npy``
        sitting next to that directory.
        """
        base = self.cfg.paths.get("calibration")
        if base is None:
            return None
        dated = base / f"{self.date_yyyy_mm_dd}.npy"
        if dated.exists():
            return dated
        legacy = base.parent / "transform_matrix.npy"
        return dated if not legacy.exists() else legacy

    # -- output directory -----------------------------------------------------
    @property
    def output_dir(self) -> Path:
        """Per-session output dir, ordered by ISO date. Created on access."""
        d = self.cfg.intermediate_dir / self.subject / self.date_iso
        d.mkdir(parents=True, exist_ok=True)
        return d

    # -- construction ---------------------------------------------------------
    @classmethod
    def parse(cls, subject: str, date_token: str, cfg: ProjectConfig) -> "Session":
        return cls(subject=subject, date=parse_date(date_token), cfg=cfg)

    def __str__(self) -> str:
        return f"{self.subject}@{self.date_iso}"


def discover_sessions(
    cfg: ProjectConfig,
    subject: Optional[str] = None,
) -> List[Session]:
    """Discover sessions for the configured subject(s), chronologically ordered.

    Unions two sources so both un-processed and already-processed sessions are
    found:
    - raw NaviGraph output: ``<navigraph_output>/<subject>/<DD_MM_YYYY>/`` folders
      (subject-nested), or the legacy flat ``<navigraph_output>/<DD_MM_YYYY>/``;
    - processed output: ``<intermediate>/<subject>/<YYYY-MM-DD>/`` folders.
    """
    subjects = [subject] if subject else cfg.subjects
    seen: set[tuple[str, object]] = set()
    sessions: List[Session] = []

    def _add(subj: str, date) -> None:
        if (subj, date) not in seen:
            seen.add((subj, date))
            sessions.append(Session(subject=subj, date=date, cfg=cfg))

    def _scan_date_folders(root: Path, subj: str) -> None:
        for child in root.iterdir():
            if not child.is_dir():
                continue
            try:
                date = parse_date(child.name)
            except ValueError:
                continue
            _add(subj, date)

    # Source 1: raw NaviGraph output — prefer per-subject subfolders, fall back
    # to flat date folders shared across subjects.
    raw_root = cfg.paths.get("navigraph_output")
    if raw_root and raw_root.exists():
        for subj in subjects:
            subj_root = raw_root / subj
            if subj_root.is_dir():
                _scan_date_folders(subj_root, subj)
            else:
                _scan_date_folders(raw_root, subj)

    # Source 2: already-processed sessions under the intermediate dir.
    inter = cfg.paths.get("intermediate")
    if inter and inter.exists():
        for subj in subjects:
            subj_dir = inter / subj
            if not subj_dir.is_dir():
                continue
            for child in subj_dir.iterdir():
                if not child.is_dir() or child.name == "cross_session":
                    continue
                try:
                    date = parse_date(child.name)
                except ValueError:
                    continue
                _add(subj, date)

    sessions.sort(key=lambda s: (s.subject, s.date))
    return sessions


__all__ = ["Session", "parse_date", "discover_sessions"]
