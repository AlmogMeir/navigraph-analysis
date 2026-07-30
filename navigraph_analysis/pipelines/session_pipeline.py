"""Per-session pipeline: raw NaviGraph output -> canonical tables -> analyses.

Orchestrates the behavioral core end-to-end and writes every product into the
session's date-ordered output dir (``<intermediate>/<subject>/<YYYY-MM-DD>/``).
The neural layer is attached only when requested and available.

    from navigraph_analysis import ProjectConfig, Session
    from navigraph_analysis.pipelines.session_pipeline import run_session

    cfg = ProjectConfig.load()
    s = Session.parse("NPC1", "21_08_2025", cfg)
    products = run_session(s, cfg)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import pandas as pd

from ..config import ProjectConfig
from ..io import graph as iograph
from ..io import navigraph_output as ngo
from ..ports import PortConfig
from ..session import Session
from .. import prep
from .. import behavior as B


@dataclass
class SessionProducts:
    """Handles + in-memory references for one session's pipeline run."""

    session: Session
    tables: Dict[str, pd.DataFrame] = field(default_factory=dict)

    def __getitem__(self, key: str) -> pd.DataFrame:
        return self.tables[key]


# ---------------------------------------------------------------------------
# Stage 1: canonical tables
# ---------------------------------------------------------------------------

def build_tables(
    session: Session,
    cfg: ProjectConfig,
    port_config: Optional[PortConfig] = None,
    summary_source: str = "auto",
    save: bool = True,
) -> Dict[str, pd.DataFrame]:
    """Produce summary_df, traversal_df, node_df, node_edge_df, reward_df.

    Parameters
    ----------
    summary_source:
        ``'raw'`` — build from ``raw_session_data_*.pkl``;
        ``'table'`` — reuse an existing ``summary_df`` in the output dir;
        ``'auto'`` (default) — raw if available, else an existing table.
    """
    port_config = port_config or PortConfig.load(cfg.ports_config)
    port_map = port_config.port_map(session.date)
    G, _pos = iograph.load_maze_graph(cfg.graph_pkl)

    summary_df = _load_summary(session, cfg, summary_source)
    summary_df = prep.prepare_summary_df(summary_df)

    traversal_df = prep.reconstruct_traversal(summary_df, G, bodypart="headstage")
    node_df = prep.build_node_df(traversal_df, summary_df, port_map, G, fps=cfg.fps)
    node_edge_df = prep.build_node_edge_df(traversal_df, port_map, G, fps=cfg.fps)
    reward_df = prep.build_reward_table(summary_df, port_map)

    tables = {
        "summary_df": summary_df,
        "traversal_df": traversal_df,
        "node_df": node_df,
        "node_edge_df": node_edge_df,
        "reward_df": reward_df,
    }
    if save:
        # summary_df keeps its (possibly MultiIndex) columns -> pickle only
        ngo.save_table(summary_df, session, "summary_df", csv=False)
        for name in ("traversal_df", "node_df", "node_edge_df", "reward_df"):
            ngo.save_table(tables[name], session, name)
    return tables


def _load_summary(session: Session, cfg: ProjectConfig, source: str) -> pd.DataFrame:
    if source == "table":
        return ngo.load_table(session, "summary_df")
    if source == "raw":
        return ngo.load_raw_session_data(session)
    # auto
    try:
        return ngo.load_raw_session_data(session)
    except FileNotFoundError:
        if ngo.has_table(session, "summary_df"):
            return ngo.load_table(session, "summary_df")
        raise


# ---------------------------------------------------------------------------
# Stage 2: behavioral analyses
# ---------------------------------------------------------------------------

def run_behavior(
    session: Session,
    cfg: ProjectConfig,
    tables: Dict[str, pd.DataFrame],
    port_config: Optional[PortConfig] = None,
    save: bool = True,
) -> Dict[str, pd.DataFrame]:
    """Compute turn / decision-point / change-of-mind / state-feature products."""
    port_config = port_config or PortConfig.load(cfg.ports_config)
    port_map = port_config.port_map(session.date)
    G, _ = iograph.load_maze_graph(cfg.graph_pkl)

    node_df = tables["node_df"]
    summary_df = tables["summary_df"]

    turn_df = B.classify_session_turns(node_df, G)

    important = B.classify_important_nodes(G, list(port_map.keys()))
    decision_df = B.decision_dwell_speed(node_df, summary_df, important, fps=cfg.fps)

    indirect_df = B.find_indirect_paths(node_df, excess_threshold=4)
    apex_df = B.find_commitment_apex(node_df, G, excess_threshold=4)

    feat_df = B.build_state_feature_frame(node_df, summary_df, G, fps=cfg.fps)
    feat_df = B.add_weak_labels(feat_df, apex_df if len(apex_df) else None)

    quality = B.session_quality_summary(node_df, tables["traversal_df"], G)

    products = {
        "turn_df": turn_df,
        "decision_metrics": decision_df,
        "indirect_paths": indirect_df,
        "direct_opportunities": apex_df,
        "state_features": feat_df,
    }
    if save:
        for name, df in products.items():
            ngo.save_table(df, session, name)
        pd.Series(quality).to_json(session.output_dir / "quality_summary.json")
    return products


# ---------------------------------------------------------------------------
# Full session
# ---------------------------------------------------------------------------

def run_session(
    session: Session,
    cfg: Optional[ProjectConfig] = None,
    summary_source: str = "auto",
    neural: bool = False,
    save: bool = True,
) -> SessionProducts:
    """Run the full per-session pipeline (tables + behavior, optionally neural)."""
    cfg = cfg or ProjectConfig.load()
    port_config = PortConfig.load(cfg.ports_config)

    tables = build_tables(session, cfg, port_config, summary_source=summary_source, save=save)
    products = run_behavior(session, cfg, tables, port_config, save=save)

    all_tables = {**tables, **products}
    if neural:
        from . import _neural_stage  # lazy: only import when requested
        all_tables.update(_neural_stage.run(session, cfg, tables, save=save))

    return SessionProducts(session=session, tables=all_tables)


__all__ = ["SessionProducts", "build_tables", "run_behavior", "run_session"]
