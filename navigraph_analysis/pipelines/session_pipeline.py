"""Per-session pipeline: raw NaviGraph output -> canonical tables -> analyses.

Orchestrates the behavioral core end-to-end and writes every product into the
session's date-ordered output dir (``<intermediate>/<subject>/<YYYY-MM-DD>/``).
The neural layer is attached only when requested and available.

Stage order:

    summary_df --zigzag smoothing--> --reward correction--> corrected summary_df
    corrected summary_df --improved fixer--> traversal_df --> node_df / node_edge_df

Both corrections rewrite per-frame graph locations, so they precede traversal
reconstruction. ``reward_df`` is derived before them, since the reward events are
the ground truth the correction is applied from.

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

from ..columns import try_resolve
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
    #: What each prep stage did -- see build_tables(diagnostics=...).
    diagnostics: Dict[str, object] = field(default_factory=dict)

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
    bodypart: str = "headstage",
    corrections: bool = True,
    smooth_zigzags: bool = True,
    reward_correction: bool = True,
    traversal_method: str = "fixer",
    diagnostics: Optional[Dict[str, object]] = None,
) -> Dict[str, pd.DataFrame]:
    """Produce summary_df, traversal_df, node_df, node_edge_df, reward_df.

    Stage order matters: both corrections rewrite per-frame graph locations, so
    they run *before* the traversal is reconstructed from those locations.

    Parameters
    ----------
    summary_source:
        ``'raw'`` — build from ``raw_session_data_*.pkl``;
        ``'table'`` — reuse an existing ``summary_df`` in the output dir;
        ``'auto'`` (default) — raw if available, else an existing table.
    corrections:
        Master switch for both location-correction stages; set False to
        reconstruct straight from the raw tracking (useful for comparing against
        the uncorrected pipeline).
    smooth_zigzags, reward_correction:
        Individual switches for
        :func:`~navigraph_analysis.prep.corrections.smooth_leaf_parent_zigzags`
        and
        :func:`~navigraph_analysis.prep.corrections.apply_reward_based_correction`.
        The reward correction needs a ``reward_port`` column in ``reward_df`` and
        is skipped with a note in *diagnostics* when the session has none.
    traversal_method:
        ``'fixer'`` (default) —
        :func:`~navigraph_analysis.prep.traversal_fixer.build_traversal_df`;
        ``'two_pass'`` — the older
        :func:`~navigraph_analysis.prep.traversal.reconstruct_traversal`.
    diagnostics:
        Optional dict; what each stage did is recorded into it under
        ``zigzag`` / ``reward_correction`` / ``traversal``.
    """
    port_config = port_config or PortConfig.load(cfg.ports_config)
    port_map = port_config.port_map(session.subject, session.date)
    layout = port_config.resolve_layout(session.subject, session.date)
    G, _pos = iograph.load_maze_graph(cfg.graph_pkl)
    diagnostics = diagnostics if diagnostics is not None else {}

    summary_df = _load_summary(session, cfg, summary_source, layout, diagnostics)
    summary_df = prep.prepare_summary_df(summary_df)

    # reward_df is built from the *uncorrected* frame table: the reward events
    # are the ground truth the correction is applied from, so deriving them after
    # the correction would make it circular.
    reward_df = prep.build_reward_table(summary_df, port_map, bodypart=bodypart)

    if corrections and smooth_zigzags:
        summary_df, zz = prep.smooth_leaf_parent_zigzags(
            summary_df, layout, bodypart=bodypart)
        diagnostics["zigzag"] = zz

    if corrections and reward_correction:
        rewards = _reward_events_with_port(reward_df, port_config, session)
        if rewards is None:
            diagnostics["reward_correction"] = (
                "skipped: reward_df has no reward_port and port_node could not be "
                "mapped back to a port number"
            )
        else:
            summary_df, rc = prep.apply_reward_based_correction(
                summary_df, rewards, layout, bodypart=bodypart)
            diagnostics["reward_correction"] = rc

    if traversal_method == "fixer":
        traversal_df, _improved, diag = prep.build_traversal_df(
            summary_df, G, fps=cfg.fps, bodypart=bodypart)
        diagnostics["traversal"] = diag
    elif traversal_method == "two_pass":
        traversal_df = prep.reconstruct_traversal(summary_df, G, bodypart=bodypart)
    else:
        raise ValueError(
            f"traversal_method must be 'fixer' or 'two_pass', got {traversal_method!r}")

    node_df = prep.build_node_df(traversal_df, summary_df, port_map, G, fps=cfg.fps)
    node_edge_df = prep.build_node_edge_df(traversal_df, port_map, G, fps=cfg.fps)

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


def _reward_events_with_port(
    reward_df: pd.DataFrame,
    port_config: PortConfig,
    session: Session,
) -> Optional[pd.DataFrame]:
    """``reward_df`` with a 1-indexed ``reward_port``, or None if it cannot be derived.

    ``build_reward_table`` labels each reward with the graph node the animal was
    on, which is exactly what may be wrong; the port *number* is what the
    correction needs. It is recovered from ``port_node`` via the layout when the
    table does not already carry it.
    """
    if "reward_port" in reward_df.columns:
        return reward_df
    if "port_node" not in reward_df.columns:
        return None
    numeric = port_config.port_map_numeric(session.subject, session.date)
    out = reward_df.copy()
    out["reward_port"] = out["port_node"].map(numeric)
    out = out[out["reward_port"].notna()]
    if out.empty:
        return None
    out["reward_port"] = out["reward_port"].astype(int)
    return out


def _load_summary(
    session: Session,
    cfg: ProjectConfig,
    source: str,
    layout: Optional[Dict[int, tuple]] = None,
    diagnostics: Optional[Dict[str, object]] = None,
) -> pd.DataFrame:
    """Get this session's frame-level table, ingesting from raw when needed.

    NaviGraph's raw output has no trial structure and no global frame index, so
    a raw table is passed through :func:`navigraph_analysis.prep.ingest.build_summary_df`,
    which derives them from BPOD and verifies the two clocks agree. An already
    built ``summary_df`` (which carries those columns) is used as-is.
    """
    if source == "table":
        return ngo.load_table(session, "summary_df")

    if source == "raw":
        raw = ngo.load_raw_session_data(session)
    else:  # auto
        try:
            raw = ngo.load_raw_session_data(session)
        except FileNotFoundError:
            if ngo.has_table(session, "summary_df"):
                return ngo.load_table(session, "summary_df")
            raise

    if try_resolve(raw, "frame_idx_global") is not None:
        return raw   # already ingested

    if layout is None:
        raise ValueError(
            "Raw NaviGraph output needs the session's port layout to be ingested "
            "(it is used to verify the video/BPOD clock alignment)."
        )
    if cfg.bpod_dir is None:
        raise KeyError(
            "Raw NaviGraph output carries no trial structure; ingesting it needs "
            "BPOD. Set paths.bpod in the project config."
        )
    summary_df, report = prep.build_summary_df(
        raw, session.subject, session.date, layout, cfg.bpod_dir, fps=cfg.fps,
    )
    if diagnostics is not None:
        diagnostics["ingest"] = report
    return summary_df


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
    port_map = port_config.port_map(session.subject, session.date)
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
    bodypart: str = "headstage",
    corrections: bool = True,
    traversal_method: str = "fixer",
) -> SessionProducts:
    """Run the full per-session pipeline (tables + behavior, optionally neural).

    ``corrections`` and ``traversal_method`` are forwarded to :func:`build_tables`;
    what each prep stage did is left on :attr:`SessionProducts.diagnostics`.
    """
    cfg = cfg or ProjectConfig.load()
    port_config = PortConfig.load(cfg.ports_config)

    diagnostics: Dict[str, object] = {}
    tables = build_tables(session, cfg, port_config, summary_source=summary_source,
                          save=save, bodypart=bodypart, corrections=corrections,
                          traversal_method=traversal_method, diagnostics=diagnostics)
    products = run_behavior(session, cfg, tables, port_config, save=save)

    all_tables = {**tables, **products}
    if neural:
        from . import _neural_stage  # lazy: only import when requested
        all_tables.update(_neural_stage.run(session, cfg, tables, save=save))

    return SessionProducts(session=session, tables=all_tables, diagnostics=diagnostics)


__all__ = ["SessionProducts", "build_tables", "run_behavior", "run_session"]
