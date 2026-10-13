"""Input loaders for every data modality the pipelines consume.

- :mod:`~navigraph_analysis.io.graph`            maze graph + node classification
- :mod:`~navigraph_analysis.io.navigraph_output` NaviGraph raw output + canonical tables
- :mod:`~navigraph_analysis.io.dlc`              DeepLabCut pose ``.h5``
- :mod:`~navigraph_analysis.io.spikes`           Neuropixels spikes (neural layer)
- :mod:`~navigraph_analysis.io.bpod`             BPOD behavioral ``.mat`` (optional)
"""

from .graph import (
    load_maze_graph,
    node_depth,
    is_leaf,
    node_side,
    maze_level,
    level_of,
    patch_of,
    classify_patch,
)
from .dlc import load_dlc_pose, resolve_bodypart, normalize_bodypart_case
from .navigraph_output import (
    load_raw_session_data,
    load_session_metrics,
    load_table,
    save_table,
    has_table,
)

__all__ = [
    "load_maze_graph",
    "node_depth",
    "is_leaf",
    "node_side",
    "maze_level",
    "level_of",
    "patch_of",
    "classify_patch",
    "load_dlc_pose",
    "resolve_bodypart",
    "normalize_bodypart_case",
    "load_raw_session_data",
    "load_session_metrics",
    "load_table",
    "save_table",
    "has_table",
]
