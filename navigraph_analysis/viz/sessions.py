"""Annotating cross-session figures whose x axis is "one session per day".

Day-by-day figures are hard to read without knowing which days were special. Two
markings, applied the same way on every such figure so they mean the same thing
each time:

- :func:`annotate_date_ranges` draws each configured span of sessions (change of
  maze, pre/post surgery, ...) as dotted boundary lines plus a label.
- :func:`mark_sunday_ticks` colours the tick labels of Sunday sessions, which run
  under a different schedule.

Spans come from ``date_annotations`` in ``config/project.yaml`` — one list per
subject, each entry ``{label, start, end, color}`` with inclusive, optional
bounds — so they are configuration rather than something each figure repeats.

Ported from the cross-session cells of ``change_of_mind.ipynb``.

    from navigraph_analysis.viz.sessions import annotate_date_ranges

    annotate_date_ranges(ax, session_dates, subject="NPC4", annotations=cfg.date_annotations)
"""

from __future__ import annotations

import datetime as _dt
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt

#: Fallback colour cycle for spans that do not name their own.
ANNOTATION_COLORS = ["darkorange", "purple", "seagreen", "saddlebrown", "magenta"]
#: Sundays run under a different schedule, so their tick labels are marked.
SUNDAY_TICK_COLOR = "red"


def _as_date(v) -> Optional[_dt.date]:
    """date/datetime/Timestamp, ``'DD/MM/YYYY'`` or a ``'DD_MM_YYYY'`` folder name -> date.

    ``'YYYY_MM_DD'`` / ``'YYYY-MM-DD'`` are accepted too, distinguished by a
    4-digit first field.
    """
    if v is None:
        return None
    if isinstance(v, _dt.datetime):
        return v.date()
    if isinstance(v, _dt.date):
        return v
    if hasattr(v, "to_pydatetime"):
        return v.to_pydatetime().date()
    parts = str(v).strip().replace("/", "_").replace("-", "_").split("_")
    if len(parts) < 3:
        raise ValueError(f"Cannot parse date: {v!r}")
    if len(parts[0]) == 4:                       # YYYY_MM_DD
        y, m, d = (int(p) for p in parts[:3])
    else:                                        # DD_MM_YYYY
        d, m, y = (int(p) for p in parts[:3])
    return _dt.date(y, m, d)


def annotation_blocks(
    session_dates: Sequence,
    annotations: Sequence[dict],
) -> List[Tuple[int, int, str, str, Optional[_dt.date], Optional[_dt.date]]]:
    """Contiguous runs of sessions matching each annotation.

    Returns ``[(i_first, i_last, label, color, start, end)]`` with indices into
    *session_dates*. A span is split into several blocks where the session list
    itself has gaps, so a missing recording day does not draw a boundary through
    the middle of a span.
    """
    dates = [_as_date(d) for d in session_dates]
    blocks = []
    for k, ann in enumerate(annotations):
        start, end = _as_date(ann.get("start")), _as_date(ann.get("end"))
        color = ann.get("color", ANNOTATION_COLORS[k % len(ANNOTATION_COLORS)])
        hits = [i for i, d in enumerate(dates)
                if (start is None or d >= start) and (end is None or d <= end)]
        if not hits:
            continue
        run = [hits[0]]
        for i in hits[1:]:
            if i == run[-1] + 1:
                run.append(i)
            else:
                blocks.append((run[0], run[-1], ann["label"], color, start, end))
                run = [i]
        blocks.append((run[0], run[-1], ann["label"], color, start, end))
    return blocks


def mark_sunday_ticks(
    ax: plt.Axes,
    session_dates: Sequence,
    color: str = SUNDAY_TICK_COLOR,
    bold: bool = True,
) -> List[int]:
    """Colour the x tick labels of sessions that fall on a Sunday.

    Call after ``set_xticklabels``; x positions must be ``0, 1, ... n-1``. Returns
    the indices of the Sunday sessions.
    """
    dates = [_as_date(d) for d in session_dates]
    sundays = [i for i, d in enumerate(dates) if d.weekday() == 6]
    for i, lbl in enumerate(ax.get_xticklabels()):
        if i in sundays:
            lbl.set_color(color)
            if bold:
                lbl.set_fontweight("bold")
    return sundays


def annotate_date_ranges(
    ax: plt.Axes,
    session_dates: Sequence,
    subject: Optional[str] = None,
    annotations: Optional[Sequence[dict] | Dict[str, Sequence[dict]]] = None,
    fontsize: int = 16,
    label_y: float = 0.99,
    headroom: float = 0.16,
    show_label: bool = True,
    linewidth: float = 1.6,
    alpha: float = 0.9,
    sunday_ticks: bool = True,
) -> plt.Axes:
    """Mark a subject's special date ranges on a per-session (x = 0..n-1) axis.

    Parameters
    ----------
    session_dates:
        Ordered session dates matching x positions ``0, 1, ... n-1`` (date objects
        or ``'DD_MM_YYYY'`` folder names).
    subject:
        Used to pick this subject's list when *annotations* is the whole
        per-subject mapping (e.g. ``cfg.date_annotations``).
    annotations:
        Either a list of spans, or the ``{subject: [spans]}`` mapping. Each span is
        ``{'label': str, 'start': date?, 'end': date?, 'color': str?}`` with
        inclusive bounds; an omitted bound is open on that side.
    headroom:
        Fraction of the y-range added on top to fit the labels.
    """
    if sunday_ticks:
        mark_sunday_ticks(ax, session_dates)

    if annotations is None:
        return ax
    if isinstance(annotations, dict):
        annotations = annotations.get(subject, []) if subject else []
    if not annotations:
        return ax

    blocks = annotation_blocks(session_dates, annotations)
    if not blocks:
        return ax

    n = len(session_dates)
    if headroom:
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + (hi - lo) * headroom)

    drawn = set()
    for lvl, (i0, i1, label, color, start, end) in enumerate(blocks):
        left, right = i0 - 0.5, i1 + 0.5
        # A boundary is drawn only where the span really ends inside the plot --
        # not at the edge of the session list, where it just runs off.
        for x, bounded in ((left, i0 > 0 or start is not None),
                           (right, i1 < n - 1 or end is not None)):
            if bounded and (round(x, 3), color) not in drawn:
                ax.axvline(x, color=color, linestyle=":", linewidth=linewidth,
                           alpha=alpha, zorder=1)
                drawn.add((round(x, 3), color))
        if not show_label:
            continue
        rot = 0 if (i1 - i0 + 1) >= 3 else 90        # narrow spans get a vertical label
        y = label_y if rot else label_y - 0.085 * (lvl % 2)
        ax.text((left + right) / 2, y, label, transform=ax.get_xaxis_transform(),
                ha="center", va="top", rotation=rot, fontsize=fontsize, color=color,
                fontweight="bold", zorder=6, clip_on=False,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                          edgecolor="none", alpha=0.7))
    return ax


__all__ = [
    "ANNOTATION_COLORS",
    "SUNDAY_TICK_COLOR",
    "annotation_blocks",
    "mark_sunday_ticks",
    "annotate_date_ranges",
]
