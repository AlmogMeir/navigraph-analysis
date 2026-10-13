"""Behavioral analyses over the canonical tables + maze graph (no neural deps).

- :mod:`~navigraph_analysis.behavior.speed`           per-frame / mean path speed
- :mod:`~navigraph_analysis.behavior.turns`           LEFT/RIGHT/STRAIGHT/UTURN classification
- :mod:`~navigraph_analysis.behavior.decision_points` important-node dwell/speed
- :mod:`~navigraph_analysis.behavior.change_of_mind`  indirect paths + commitment apex
- :mod:`~navigraph_analysis.behavior.states`          per-node-visit state features + HMM validation
- :mod:`~navigraph_analysis.behavior.state_models`    HMM families + cross-family comparison (optional deps)
- :mod:`~navigraph_analysis.behavior.quality`         traversal quality metrics
"""

from .speed import compute_path_speed, compute_mean_path_speed
from .turns import classify_turn, turns_from_graph, classify_session_turns, TURN_CATEGORIES
from .decision_points import (
    classify_important_nodes,
    decision_dwell_speed,
    compare_important_vs_not,
    visit_speed_metrics,
    paired_wilcoxon,
    speed_breakdown,
    SPEED_METRIC_COLS,
)
from .change_of_mind import find_indirect_paths, find_commitment_apex
from .states import (
    build_state_feature_frame,
    add_weak_labels,
    build_design_matrix,
    validate_states,
)
from .quality import quality_by_port_pair, count_transition_violations, session_quality_summary

# state_models pulls in hmmlearn / IOHMM / ssm lazily; import the module (cheap)
# rather than its names, so the behavioral core keeps installing without them.
from . import state_models

__all__ = [
    "compute_path_speed",
    "compute_mean_path_speed",
    "classify_turn",
    "turns_from_graph",
    "classify_session_turns",
    "TURN_CATEGORIES",
    "classify_important_nodes",
    "decision_dwell_speed",
    "compare_important_vs_not",
    "visit_speed_metrics",
    "paired_wilcoxon",
    "speed_breakdown",
    "SPEED_METRIC_COLS",
    "find_indirect_paths",
    "find_commitment_apex",
    "build_state_feature_frame",
    "add_weak_labels",
    "build_design_matrix",
    "validate_states",
    "quality_by_port_pair",
    "count_transition_violations",
    "session_quality_summary",
    "state_models",
]
