"""Pipeline orchestration + CLI.

- :mod:`~navigraph_analysis.pipelines.session_pipeline`  per-session end-to-end
- :mod:`~navigraph_analysis.pipelines.cross_session`     date-ordered aggregation
- :mod:`~navigraph_analysis.pipelines.cli`               the ``nga`` command
"""

from .session_pipeline import run_session, build_tables, run_behavior, SessionProducts

__all__ = ["run_session", "build_tables", "run_behavior", "SessionProducts"]
