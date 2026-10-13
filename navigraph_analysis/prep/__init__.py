"""Data-prep: raw NaviGraph output -> canonical intermediate tables.

Pipeline order (see :mod:`navigraph_analysis.pipelines.session_pipeline`):

    raw NaviGraph output + BPOD --ingest.build_summary_df-->         summary_df
       (internally: frame_alignment.build_frames_df -> align_frames_to_video,
        then reward / lick annotation and bodypart selection)
    summary_df           --corrections.smooth_leaf_parent_zigzags--> summary_df
    summary_df + rewards --corrections.apply_reward_based_correction--> summary_df
    summary_df + G       --traversal_fixer.build_traversal_df-->     traversal_df
    traversal_df         --node_tables.build_node_df / build_node_edge_df--> node_df / node_edge_df
    summary_df           --reward.build_reward_table-->              reward_df

The two correction stages and the alignment check run *before* traversal
reconstruction, because they change the node sequence it is built from.
:func:`~navigraph_analysis.prep.traversal.reconstruct_traversal` is the older
two-pass reconstruction, kept for comparison;
:func:`~navigraph_analysis.prep.traversal_fixer.build_traversal_df` is the default.
"""

from .align import prepare_summary_df, trial_windows
from .ingest import IngestReport, build_summary_df
from .frame_alignment import (
    AlignmentError,
    FrameAlignment,
    align_frames_to_video,
    bpod_reward_frames,
    build_frames_df,
    on_port_fraction,
)
from .corrections import (
    RewardCorrections,
    ZigzagCorrections,
    apply_reward_based_correction,
    detect_zigzags,
    smooth_leaf_parent_zigzags,
    zigzag_threshold_table,
)
from .traversal import reconstruct_traversal, TRAVERSAL_COLUMNS
from .traversal_fixer import (
    ImprovedMazeTraversalFixer,
    build_traversal_df,
    make_improved_traversal,
)
from .node_tables import build_node_df, build_node_edge_df, fix_edge_bounces
from .reward import build_reward_table

__all__ = [
    "prepare_summary_df",
    "trial_windows",
    "IngestReport",
    "build_summary_df",
    "AlignmentError",
    "FrameAlignment",
    "align_frames_to_video",
    "bpod_reward_frames",
    "build_frames_df",
    "on_port_fraction",
    "RewardCorrections",
    "ZigzagCorrections",
    "apply_reward_based_correction",
    "detect_zigzags",
    "smooth_leaf_parent_zigzags",
    "zigzag_threshold_table",
    "reconstruct_traversal",
    "TRAVERSAL_COLUMNS",
    "ImprovedMazeTraversalFixer",
    "build_traversal_df",
    "make_improved_traversal",
    "build_node_df",
    "build_node_edge_df",
    "fix_edge_bounces",
    "build_reward_table",
]
