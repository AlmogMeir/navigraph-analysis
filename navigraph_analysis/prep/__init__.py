"""Data-prep: raw NaviGraph output -> canonical intermediate tables.

Pipeline order (see :func:`navigraph_analysis.pipelines.session_pipeline.run_prep`):
    raw_session_data  --align.prepare_summary_df-->  summary_df
    summary_df + G    --traversal.reconstruct_traversal-->  traversal_df
    traversal_df      --node_tables.build_node_df / build_node_edge_df-->  node_df / node_edge_df
    summary_df        --reward.build_reward_table-->  reward_df
"""

from .align import prepare_summary_df, trial_windows
from .traversal import reconstruct_traversal, TRAVERSAL_COLUMNS
from .node_tables import build_node_df, build_node_edge_df, fix_edge_bounces
from .reward import build_reward_table

__all__ = [
    "prepare_summary_df",
    "trial_windows",
    "reconstruct_traversal",
    "TRAVERSAL_COLUMNS",
    "build_node_df",
    "build_node_edge_df",
    "fix_edge_bounces",
    "build_reward_table",
]
