"""navigraph_analysis — standalone analysis pipelines for dual-root-tree maze data.

Fully independent of the ``navigraph`` package: it consumes NaviGraph *output*
(the integrated frame DataFrame / DLC pose / graph-mapping pickle) plus optional
Neuropixels + BPOD data, and produces ordered per-subject/per-session analyses.

Top-level convenience re-exports:

    from navigraph_analysis import ProjectConfig, Session, PortConfig
"""

from .config import ProjectConfig
from .ports import PortConfig
from .session import Session, discover_sessions, parse_date

__version__ = "0.1.0"

__all__ = [
    "ProjectConfig",
    "PortConfig",
    "Session",
    "discover_sessions",
    "parse_date",
    "__version__",
]
