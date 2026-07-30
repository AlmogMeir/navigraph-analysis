"""Project configuration loading and path resolution.

A single :class:`ProjectConfig` (loaded from ``config/project.yaml``) is the one
place that knows where the data lives. Every notebook used to hard-code
``/var/home/almogmeir/Documents/...`` paths; this module replaces all of them.

Usage
-----
    from navigraph_analysis.config import ProjectConfig

    cfg = ProjectConfig.load()                 # finds config/project.yaml (or .local)
    cfg = ProjectConfig.load("path/to.yaml")   # explicit
    cfg.fps                                     # -> 40
    cfg.graph_pkl                               # -> resolved absolute Path
    cfg.navigraph_output_dir                    # -> resolved absolute Path
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# Repo root = parent of the package directory.
REPO_ROOT = Path(__file__).resolve().parent.parent


def _resolve(base: Path, value: Optional[str]) -> Optional[Path]:
    """Resolve a config path string against *base*, expanding ``~``.

    Absolute paths (after ``~`` expansion) are returned as-is; relative paths are
    taken relative to the repo root. ``None`` passes through so optional paths
    stay optional.
    """
    if value is None:
        return None
    p = Path(value).expanduser()
    return p if p.is_absolute() else (base / p).resolve()


@dataclass
class ProjectConfig:
    """Resolved project configuration."""

    subjects: List[str]
    constants: Dict[str, Any]
    paths: Dict[str, Optional[Path]]
    ports_config: Optional[Path]
    source_file: Path = field(repr=False)

    # -- constants convenience accessors -------------------------------------
    @property
    def fps(self) -> float:
        return float(self.constants.get("fps", 40))

    @property
    def fs(self) -> float:
        return float(self.constants.get("fs", 30000))

    @property
    def maze_start(self) -> int:
        return int(self.constants.get("maze_start", 30))

    @property
    def maze_end(self) -> int:
        return int(self.constants.get("maze_end", 767))

    # -- path convenience accessors ------------------------------------------
    @property
    def navigraph_output_dir(self) -> Path:
        return self._require("navigraph_output")

    @property
    def graph_pkl(self) -> Path:
        return self._require("graph_pkl")

    @property
    def intermediate_dir(self) -> Path:
        return self._require("intermediate")

    @property
    def dlc_dir(self) -> Optional[Path]:
        return self.paths.get("dlc")

    @property
    def maze_images_dir(self) -> Optional[Path]:
        return self.paths.get("maze_images")

    @property
    def videos_dir(self) -> Optional[Path]:
        return self.paths.get("videos")

    @property
    def spikes_dir(self) -> Optional[Path]:
        return self.paths.get("spikes")

    @property
    def bpod_dir(self) -> Optional[Path]:
        return self.paths.get("bpod")

    def _require(self, key: str) -> Path:
        p = self.paths.get(key)
        if p is None:
            raise KeyError(
                f"Required path '{key}' is not set in {self.source_file}."
            )
        return p

    # -- loading -------------------------------------------------------------
    @classmethod
    def load(cls, path: Optional[str | Path] = None) -> "ProjectConfig":
        """Load configuration.

        If *path* is None, prefers ``config/project.local.yaml`` (gitignored,
        machine-specific) then falls back to ``config/project.yaml``.
        """
        if path is None:
            local = REPO_ROOT / "config" / "project.local.yaml"
            default = REPO_ROOT / "config" / "project.yaml"
            src = local if local.exists() else default
        else:
            src = Path(path).expanduser().resolve()
        if not src.exists():
            raise FileNotFoundError(f"Project config not found: {src}")

        with open(src) as fh:
            raw = yaml.safe_load(fh) or {}

        base = REPO_ROOT
        raw_paths = raw.get("paths", {}) or {}
        paths = {k: _resolve(base, v) for k, v in raw_paths.items()}

        return cls(
            subjects=list(raw.get("subjects", [])),
            constants=dict(raw.get("constants", {})),
            paths=paths,
            ports_config=_resolve(base, raw.get("ports_config")),
            source_file=src,
        )


__all__ = ["ProjectConfig", "REPO_ROOT"]
