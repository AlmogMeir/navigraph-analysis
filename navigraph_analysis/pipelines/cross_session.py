"""Cross-session aggregation, ordered by session date.

Reusable core of ``multisession.ipynb``: walk every discovered session in
chronological order, pull its per-session behavioral products (building them if
absent), and assemble learning curves — per-session directness, indirect-path
count and mean path excess — plus a learning-curve figure. Output lands in
``<intermediate>/<subject>/cross_session/``.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd

from ..config import ProjectConfig
from ..io import navigraph_output as ngo
from ..session import Session, discover_sessions


def _session_row(session: Session, cfg: ProjectConfig) -> Optional[dict]:
    """One learning-curve row for *session*, computing tables on demand."""
    try:
        if ngo.has_table(session, "node_df"):
            node_df = ngo.load_table(session, "node_df")
        else:
            from .session_pipeline import build_tables
            node_df = build_tables(session, cfg)["node_df"]
    except (FileNotFoundError, KeyError):
        return None

    from ..behavior.quality import quality_by_port_pair
    from ..behavior.change_of_mind import find_indirect_paths

    by_pair = quality_by_port_pair(node_df)
    indirect = find_indirect_paths(node_df, excess_threshold=4)
    return {
        "subject": session.subject,
        "date": session.date_iso,
        "n_port_pairs": int(len(by_pair)),
        "frac_direct": float(by_pair["frac_direct"].mean()) if len(by_pair) else float("nan"),
        "mean_excess": float(by_pair["mean_excess"].mean()) if len(by_pair) else float("nan"),
        "n_indirect_paths": int(len(indirect)),
    }


def build_learning_curves(cfg: ProjectConfig, subject: Optional[str] = None) -> pd.DataFrame:
    """Assemble the date-ordered learning-curve table across sessions."""
    sessions = discover_sessions(cfg, subject=subject)
    rows: List[dict] = []
    for s in sessions:
        row = _session_row(s, cfg)
        if row is not None:
            rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["subject", "date"]).reset_index(drop=True)
    return df


def run_cross_session(cfg: ProjectConfig, subject: Optional[str] = None) -> Path:
    """Build learning curves + figure; write to ``<intermediate>/<subject>/cross_session/``."""
    df = build_learning_curves(cfg, subject=subject)
    subj = subject or (cfg.subjects[0] if cfg.subjects else "all")
    out_dir = cfg.intermediate_dir / subj / "cross_session"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv = out_dir / "learning_curves.csv"
    df.to_csv(csv, index=False)

    if not df.empty:
        _plot_learning_curves(df, out_dir / "learning_curves.png")
    return csv


def _plot_learning_curves(df: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    x = range(len(df))
    axes[0].plot(x, df["frac_direct"], "-o", color="#2ca02c")
    axes[0].set_ylabel("Fraction direct")
    axes[1].plot(x, df["mean_excess"], "-o", color="#e67e22")
    axes[1].set_ylabel("Mean excess nodes")
    axes[2].plot(x, df["n_indirect_paths"], "-o", color="#4C72B0")
    axes[2].set_ylabel("# indirect paths")
    axes[2].set_xticks(list(x))
    axes[2].set_xticklabels(df["date"], rotation=45, ha="right", fontsize=7)
    axes[2].set_xlabel("Session")
    axes[0].set_title("Cross-session learning curves")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


__all__ = ["build_learning_curves", "run_cross_session"]
