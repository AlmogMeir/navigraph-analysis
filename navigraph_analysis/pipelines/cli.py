"""``nga`` command-line entry point.

    nga list                              # discover sessions from the config
    nga prep   --session 21_08_2025       # build canonical tables only
    nga run    --session 21_08_2025       # full behavioral pipeline
    nga run    --all                      # every discovered session
    nga run    --all --neural             # include the neural layer
    nga cross                             # cross-session aggregation

All commands take ``--config PATH`` (defaults to config/project.yaml or .local).
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from ..config import ProjectConfig
from ..session import Session, discover_sessions


def _sessions_for(args, cfg: ProjectConfig) -> List[Session]:
    if getattr(args, "all", False):
        return discover_sessions(cfg, subject=args.subject)
    if not args.session:
        raise SystemExit("Specify --session <date> or --all")
    subject = args.subject or (cfg.subjects[0] if cfg.subjects else None)
    if subject is None:
        raise SystemExit("No subject configured; pass --subject")
    return [Session.parse(subject, args.session, cfg)]


def _cmd_list(args, cfg: ProjectConfig) -> int:
    sessions = discover_sessions(cfg, subject=args.subject)
    if not sessions:
        print(f"No sessions found under {cfg.navigraph_output_dir}")
        return 0
    print(f"{len(sessions)} session(s):")
    for s in sessions:
        print(f"  {s.subject}  {s.date_iso}  (folder {s.folder_ddmmyyyy})")
    return 0


def _cmd_prep(args, cfg: ProjectConfig) -> int:
    from .session_pipeline import build_tables

    for s in _sessions_for(args, cfg):
        print(f"[prep] {s} ...", flush=True)
        tables = build_tables(s, cfg, summary_source=args.summary_source)
        print(f"       -> {', '.join(f'{k}({len(v)})' for k, v in tables.items())}")
        print(f"       written to {s.output_dir}")
    return 0


def _cmd_run(args, cfg: ProjectConfig) -> int:
    from .session_pipeline import run_session

    for s in _sessions_for(args, cfg):
        print(f"[run] {s} (neural={args.neural}) ...", flush=True)
        products = run_session(s, cfg, summary_source=args.summary_source, neural=args.neural)
        print(f"      -> {len(products.tables)} products in {s.output_dir}")
    return 0


def _cmd_cross(args, cfg: ProjectConfig) -> int:
    from .cross_session import run_cross_session

    out = run_cross_session(cfg, subject=args.subject)
    print(f"[cross] wrote {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="nga", description="NaviGraph-Analysis pipelines")
    p.add_argument("--config", default=None, help="Path to project.yaml")
    sub = p.add_subparsers(dest="command", required=True)

    def add_common(sp):
        sp.add_argument("--subject", default=None)
        sp.add_argument("--session", default=None, help="Session date (any format)")
        sp.add_argument("--all", action="store_true", help="All discovered sessions")
        sp.add_argument("--summary-source", default="auto",
                        choices=["auto", "raw", "table"], dest="summary_source")

    sp_list = sub.add_parser("list", help="Discover sessions")
    sp_list.add_argument("--subject", default=None)
    sp_list.set_defaults(func=_cmd_list)

    sp_prep = sub.add_parser("prep", help="Build canonical tables")
    add_common(sp_prep)
    sp_prep.set_defaults(func=_cmd_prep)

    sp_run = sub.add_parser("run", help="Full per-session pipeline")
    add_common(sp_run)
    sp_run.add_argument("--neural", action="store_true", help="Include neural layer")
    sp_run.set_defaults(func=_cmd_run)

    sp_cross = sub.add_parser("cross", help="Cross-session aggregation")
    sp_cross.add_argument("--subject", default=None)
    sp_cross.set_defaults(func=_cmd_cross)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = ProjectConfig.load(args.config)
    return args.func(args, cfg)


if __name__ == "__main__":
    sys.exit(main())
