"""Traversal reconstruction, improved fixer: per-frame graph locations -> legal sequence.

Port of ``navigraph.analysis.traversal_fixer.ImprovedMazeTraversalFixer``, which
replaced the two-pass reconstruction in ``traversal_pipeline.ipynb`` (kept here as
:func:`~navigraph_analysis.prep.traversal.reconstruct_traversal`). It produces the
same ``traversal_df`` contract but reconstructs it differently, and is what
:mod:`navigraph_analysis.pipelines.session_pipeline` uses by default.

What it does that the two-pass version does not:

1. **Augmented state graph.** Physical maze edges become first-class
   intermediate *states*, so "am I allowed to go from here to there in one step"
   is a single ``has_edge`` question over nodes *and* edges. Leaf<->parent links
   (level 4 <-> level 5) are virtual — there is no physical corridor — so they
   stay direct node-node connections.
2. **Noise smoothing.** A short run that cannot be reached from its neighbour is
   tracking noise, not travel. Three patterns are absorbed rather than
   interpolated: forward (``[A, x]`` with ``x`` short), backward (``[A, x, A]``
   sandwich), and long-jump (a graph distance above ``max_plausible_hops`` where
   the *shorter* of the two runs is itself short). Two long runs separated by a
   big gap are a genuine port-to-port traversal with a tracking dropout, and
   fall through to interpolation instead.
3. **Frame-conserving interpolation.** Inferred intermediate states take real
   frames from the runs on either side, so the frame budget is conserved and
   ``duration`` stays meaningful. The two-pass version emitted zero-duration
   rows all stamped with the same frame.
4. **Invalid states are dropped, not trusted.** An edge whose endpoints are not
   actually adjacent in the maze is treated as an unknown frame and absorbed
   into a neighbouring run instead of leaving a ghost edge in the traversal.
5. **Validation + diagnostics.** :meth:`ImprovedMazeTraversalFixer.validate`
   proves every consecutive pair is graph-legal, and ``diagnostics`` counts what
   each rule did, so a session that needed heavy repair is visible rather than
   silently different.

The fixer encodes dual-root-tree maze assumptions: node names are ``L``/``R``
prefixed with the level in the second character, level 4 to level 5 links are
virtual edges, and degree-1 nodes are leaves reached through such an edge. It is
not a general-purpose graph utility.

    from navigraph_analysis.prep.traversal_fixer import build_traversal_df

    traversal_df, diagnostics = build_traversal_df(summary_df, G)
"""

from __future__ import annotations

import ast
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from ..columns import resolve, try_resolve
from .traversal import TRAVERSAL_COLUMNS

#: ``traversal_df`` columns the fixer adds on top of :data:`TRAVERSAL_COLUMNS`.
IN_TRIAL_COLUMNS = ["start_frame_in_trial", "end_frame_in_trial"]


class ImprovedMazeTraversalFixer:
    """Reconstruct a validated node/edge traversal from a frame-level ``summary_df``.

    Parameters
    ----------
    G:
        Maze graph (from :func:`navigraph_analysis.io.graph.load_maze_graph`).
    noise_threshold:
        A run this many frames or shorter may be absorbed into a neighbour when
        the two are not graph-adjacent.
    max_plausible_hops:
        A graph distance above this between consecutive runs is not travel the
        animal could have made unobserved; if the shorter run is also short, the
        pair is treated as a tracking artifact.
    bodypart:
        Which tracked point drives location. The *output* column names stay
        ``headstage_graph_node`` / ``headstage_graph_edge`` either way — that is
        the ``traversal_df`` contract, not a claim about the bodypart.
    """

    def __init__(
        self,
        G: nx.Graph,
        noise_threshold: int = 5,
        max_plausible_hops: int = 8,
        bodypart: str = "headstage",
    ):
        self.noise_threshold = noise_threshold
        self.max_plausible_hops = max_plausible_hops
        self.bodypart = bodypart
        self.G_aug = self._build_augmented_graph(G)
        self.diagnostics: Dict[str, int] = {
            "trials_processed": 0,
            "virtual_edges_ignored": 0,
            "noise_smoothed_forward": 0,     # runs[i+1] smoothed to runs[i]
            "noise_smoothed_backward": 0,    # runs[i] smoothed (sandwiched by same state)
            "noise_smoothed_long_jump": 0,   # path > max_plausible_hops, short run -> artifact
            "jumps_interpolated": 0,
            "nodes_edges_inferred": 0,
            "invalid_unresolved": 0,
        }

    # -- graph helpers -------------------------------------------------------

    @staticmethod
    def _is_virtual_edge(u_str: str, v_str: str) -> bool:
        """True for L4<->L5 / R4<->R5 pairs (no physical maze corridor)."""
        def level(s):
            return s[1] if len(s) >= 2 and s[0] in ("L", "R") else None
        l_u, l_v = level(u_str), level(v_str)
        return bool(l_u and l_v and {l_u, l_v} == {"4", "5"})

    def _build_augmented_graph(self, G: nx.Graph) -> nx.Graph:
        """Augmented state graph: physical edges become intermediate tuple-nodes.

        Virtual edges (leaf<->parent) remain direct node-node connections.
        """
        G_aug = nx.Graph()
        for u in G.nodes():
            G_aug.add_node(str(u), state_type="node")
        for u, v in G.edges():
            u_s, v_s = str(u), str(v)
            if G.degree(u) == 1 or G.degree(v) == 1 or self._is_virtual_edge(u_s, v_s):
                G_aug.add_edge(u_s, v_s)
            else:
                edge_state = tuple(sorted((u_s, v_s)))
                G_aug.add_node(edge_state, state_type="edge")
                G_aug.add_edge(u_s, edge_state)
                G_aug.add_edge(v_s, edge_state)
        return G_aug

    # -- row accessors (handle MultiIndex summary_df columns) ----------------

    @staticmethod
    def _is_missing(val) -> bool:
        if val is None:
            return True
        if isinstance(val, float) and np.isnan(val):
            return True
        if pd.api.types.is_scalar(val) and pd.isna(val):
            return True
        return False

    @staticmethod
    def _scalar(row, col):
        v = row.get(col)
        if isinstance(v, pd.Series):
            v = v.dropna()
            return v.iloc[0] if not v.empty else np.nan
        return v

    def _get_state(self, row):
        """The augmented-graph state for one frame, or NaN if unknown/illegal."""
        node = self._scalar(row, self._node_col)
        if not self._is_missing(node):
            if isinstance(node, float) and float(node).is_integer():
                return str(int(node))
            return str(node)

        if self._edge_col is None:
            return np.nan
        edge = self._scalar(row, self._edge_col)
        if self._is_missing(edge):
            return np.nan

        if isinstance(edge, str):
            try:
                edge = ast.literal_eval(edge)
            except (ValueError, SyntaxError):
                clean = edge.strip("() ")
                if "_" in clean and "," not in clean:
                    clean = clean.replace("_", ",", 1)
                if "," in clean:
                    parts = [p.strip(" '\"") for p in clean.split(",")]
                    if len(parts) == 2:
                        edge = parts
        if isinstance(edge, (list, tuple)) and len(edge) == 2:
            u_s, v_s = str(edge[0]), str(edge[1])
            if self._is_virtual_edge(u_s, v_s):
                self.diagnostics["virtual_edges_ignored"] += 1
                return np.nan
            edge_state = tuple(sorted((u_s, v_s)))
            # An edge whose endpoints are not adjacent in the maze is not a
            # place the animal can be; treat the frame as unknown.
            if edge_state not in self.G_aug:
                return np.nan
            return edge_state
        return np.nan

    # -- run compression ----------------------------------------------------

    def _compress_to_runs(self, df_sorted: pd.DataFrame) -> List[dict]:
        """Collapse frame-by-frame rows into ``(state, frame-range)`` runs.

        NaN frames are absorbed into the adjacent run when there is no gap in
        the global frame index.
        """
        runs: List[dict] = []
        cur: Optional[dict] = None

        for _, row in df_sorted.iterrows():
            state = self._get_state(row)
            f_glob = self._scalar(row, self._frame_col)
            f_trial = self._scalar(row, self._in_trial_col) if self._in_trial_col else np.nan
            t_idx = self._scalar(row, self._trial_col) if self._trial_col else np.nan

            if self._is_missing(state):
                if cur is not None and (f_glob - cur["end_frame_global"]) == 1:
                    cur["end_frame_global"] = f_glob
                    cur["end_frame_in_trial"] = f_trial
                    cur["duration"] += 1
                continue

            is_gap = cur is not None and (f_glob - cur["end_frame_global"]) > 1

            if cur is None or cur["state"] != state or cur["trial_idx"] != t_idx or is_gap:
                if cur is not None:
                    runs.append(cur)
                cur = {
                    "state": state,
                    "start_frame_global": f_glob,
                    "end_frame_global": f_glob,
                    "start_frame_in_trial": f_trial,
                    "end_frame_in_trial": f_trial,
                    "trial_idx": t_idx,
                    "duration": 1,
                    "is_inferred": False,
                }
            else:
                cur["end_frame_global"] = f_glob
                cur["end_frame_in_trial"] = f_trial
                cur["duration"] += 1

        if cur is not None:
            runs.append(cur)
        return runs

    # -- run fixing ---------------------------------------------------------

    def _fix_runs(self, runs: List[dict]) -> List[dict]:
        i = 0
        while i < len(runs) - 1:
            s1 = runs[i]["state"]
            s2 = runs[i + 1]["state"]

            if s1 not in self.G_aug or s2 not in self.G_aug:
                self.diagnostics["invalid_unresolved"] += 1
                i += 1
                continue

            if s1 == s2 or self.G_aug.has_edge(s1, s2):
                i += 1
                continue

            try:
                path = nx.shortest_path(self.G_aug, s1, s2)
            except nx.NetworkXNoPath:
                self.diagnostics["invalid_unresolved"] += 1
                i += 1
                continue

            # Long-jump check: only a tracking artifact when the SHORTER of the
            # two runs is also short. A genuine port-to-port traversal whose
            # tracking dropped out mid-path has two long runs separated by a
            # large gap, and must be interpolated rather than smoothed away.
            if len(path) - 1 > self.max_plausible_hops:
                shorter_dur = min(runs[i]["duration"], runs[i + 1]["duration"])
                if shorter_dur <= self.noise_threshold:
                    if runs[i + 1]["duration"] <= runs[i]["duration"]:
                        runs[i + 1]["state"] = s1
                        runs[i + 1]["is_inferred"] = True
                    else:
                        runs[i]["state"] = s2
                        runs[i]["is_inferred"] = True
                    self.diagnostics["noise_smoothed_long_jump"] += 1
                    continue  # re-check the pair; now they match
                # Both runs long -> genuine traversal with a mid-path gap.

            # Backward noise: runs[i] is sandwiched between the same state.
            # Pattern [... s2, s1(short), s2 ...] -> s1 is noise.
            if (i > 0
                    and runs[i]["duration"] <= self.noise_threshold
                    and runs[i - 1]["state"] == s2):
                runs[i]["state"] = s2
                runs[i]["is_inferred"] = True
                self.diagnostics["noise_smoothed_backward"] += 1
                continue

            # Forward noise: runs[i+1] is short -> absorb into runs[i].
            if runs[i + 1]["duration"] <= self.noise_threshold:
                runs[i + 1]["state"] = s1
                runs[i + 1]["is_inferred"] = True
                self.diagnostics["noise_smoothed_forward"] += 1
                continue

            # Jump interpolation: take frames from both sides and inject the
            # intermediate states, so the frame budget is conserved.
            missing = path[1:-1]
            n_miss = len(missing)

            avail_left = max(0, runs[i]["duration"] - 1)
            avail_right = max(0, runs[i + 1]["duration"] - 1)

            if avail_left + avail_right < n_miss:
                self.diagnostics["invalid_unresolved"] += 1
                i += 1
                continue

            take_left = min(n_miss, avail_left)
            take_right = n_miss - take_left

            orig_end_G = runs[i]["end_frame_global"]
            orig_end_T = runs[i]["end_frame_in_trial"]
            orig_start_G = runs[i + 1]["start_frame_global"]
            orig_start_T = runs[i + 1]["start_frame_in_trial"]

            if take_left > 0:
                runs[i]["end_frame_global"] -= take_left
                runs[i]["end_frame_in_trial"] -= take_left
                runs[i]["duration"] -= take_left
            if take_right > 0:
                runs[i + 1]["start_frame_global"] += take_right
                runs[i + 1]["start_frame_in_trial"] += take_right
                runs[i + 1]["duration"] -= take_right

            insert_idx = i + 1
            left_done = 0
            right_done = 0

            for state in missing:
                if left_done < take_left:
                    offset = take_left - left_done - 1
                    fg, ft = orig_end_G - offset, orig_end_T - offset
                    t = runs[i]["trial_idx"]
                    left_done += 1
                else:
                    fg, ft = orig_start_G + right_done, orig_start_T + right_done
                    t = runs[i + 1]["trial_idx"]
                    right_done += 1

                runs.insert(insert_idx, {
                    "state": state,
                    "start_frame_global": fg,
                    "end_frame_global": fg,
                    "start_frame_in_trial": ft,
                    "end_frame_in_trial": ft,
                    "trial_idx": t,
                    "duration": 1,
                    "is_inferred": True,
                })
                insert_idx += 1

            self.diagnostics["jumps_interpolated"] += 1
            self.diagnostics["nodes_edges_inferred"] += n_miss
            i = insert_idx

        return runs

    # -- validation (same-state consecutive runs allowed) --------------------

    def validate(self, runs: List[dict]) -> Tuple[bool, str]:
        """True when every consecutive pair of runs is a legal one-step move."""
        for i in range(len(runs) - 1):
            s1, s2 = runs[i]["state"], runs[i + 1]["state"]
            if s1 == s2:
                continue  # same state either side of a frame gap is fine
            if not self.G_aug.has_edge(s1, s2):
                return False, f"Invalid transition @ index {i}: {s1} -> {s2}"
        return True, "Traversal valid."

    # -- main entry point ---------------------------------------------------

    def process_session(self, summary_df: pd.DataFrame) -> pd.DataFrame:
        """Reconstruct the traversal for one session's ``summary_df``."""
        self._node_col = resolve(summary_df, self.bodypart, "graph_node")
        self._edge_col = try_resolve(summary_df, self.bodypart, "graph_edge")
        self._frame_col = resolve(summary_df, "frame_idx_global")
        self._trial_col = try_resolve(summary_df, "trial_idx")
        self._in_trial_col = try_resolve(summary_df, "frame_idx_in_trial")

        if self._trial_col is not None:
            t_col = summary_df[self._trial_col]
            self.diagnostics["trials_processed"] = (
                len(t_col.iloc[:, 0].unique()) if isinstance(t_col, pd.DataFrame)
                else len(t_col.unique())
            )

        df = summary_df.loc[:, ~summary_df.columns.duplicated()].copy()
        df = df.sort_values(self._frame_col)

        runs = self._compress_to_runs(df)
        runs = self._fix_runs(runs)

        # Final merge: collapse consecutive same-state runs with no frame gap.
        merged: List[dict] = []
        for r in runs:
            if (merged
                    and merged[-1]["state"] == r["state"]
                    and merged[-1]["trial_idx"] == r["trial_idx"]
                    and r["start_frame_global"] - merged[-1]["end_frame_global"] <= 1):
                merged[-1]["end_frame_global"] = r["end_frame_global"]
                merged[-1]["end_frame_in_trial"] = r["end_frame_in_trial"]
                merged[-1]["duration"] += r["duration"]
                merged[-1]["is_inferred"] = merged[-1]["is_inferred"] or r["is_inferred"]
            else:
                merged.append(r)

        ok, msg = self.validate(merged)
        self.diagnostics["valid"] = int(ok)
        self.validation_message = msg

        out = pd.DataFrame(merged)
        if out.empty:
            return pd.DataFrame(columns=TRAVERSAL_COLUMNS + IN_TRIAL_COLUMNS)

        out["location_type"] = out["state"].apply(
            lambda x: "edge" if isinstance(x, tuple) else "node"
        )
        out["headstage_graph_node"] = np.where(
            out["location_type"] == "node", out["state"], np.nan
        )
        # np.where would coerce the tuples, so build the edge column directly.
        out["headstage_graph_edge"] = [
            s if isinstance(s, tuple) else np.nan for s in out["state"]
        ]

        out = out[TRAVERSAL_COLUMNS + IN_TRIAL_COLUMNS]
        for c in ["start_frame_global", "end_frame_global", "duration",
                  "start_frame_in_trial", "end_frame_in_trial"]:
            out[c] = pd.to_numeric(out[c], errors="coerce")
        return out.reset_index(drop=True)

    def format_diagnostics(self) -> str:
        """Human-readable summary of what each repair rule did."""
        d = self.diagnostics
        lines = [
            f"  Trials:                {d['trials_processed']}",
            f"  Virtual edges ignored: {d['virtual_edges_ignored']:5d} frames",
            f"  Noise (forward):       {d['noise_smoothed_forward']:5d}   (i+1 smoothed to i)",
            f"  Noise (backward):      {d['noise_smoothed_backward']:5d}   "
            "(i smoothed; sandwiched)",
        ]
        if d["noise_smoothed_long_jump"]:
            lines.append(
                f"  Noise (long-jump):     {d['noise_smoothed_long_jump']:5d}   "
                f"(path > {self.max_plausible_hops} hops, short run -> artifact)"
            )
        lines += [
            f"  Jumps interpolated:    {d['jumps_interpolated']:5d}",
            f"  States inferred:       {d['nodes_edges_inferred']:5d}",
        ]
        if d["invalid_unresolved"]:
            lines.append(f"  ! Unresolved:          {d['invalid_unresolved']:5d}")
        if not d.get("valid", 1):
            lines.append(f"  ! {getattr(self, 'validation_message', 'invalid traversal')}")
        return "\n".join(lines)


def build_traversal_df(
    summary_df: pd.DataFrame,
    G: nx.Graph,
    *,
    fps: float = 40.0,
    noise_threshold: int = 5,
    max_plausible_hops: int = 8,
    bodypart: str = "headstage",
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    """Build both traversal frames for one session.

    Returns ``(traversal_df, improved_traversal_df, diagnostics)``. ``traversal_df``
    is drop-in compatible with
    :func:`navigraph_analysis.prep.traversal.reconstruct_traversal` output and
    additionally carries :data:`IN_TRIAL_COLUMNS`; ``improved_traversal_df`` is the
    ``improved_traversal_df.csv``-compatible view the upstream notebooks read.
    """
    fixer = ImprovedMazeTraversalFixer(
        G,
        noise_threshold=noise_threshold,
        max_plausible_hops=max_plausible_hops,
        bodypart=bodypart,
    )
    traversal_df = fixer.process_session(summary_df)
    improved_df = make_improved_traversal(traversal_df, 1.0 / fps)
    return traversal_df, improved_df, dict(fixer.diagnostics)


def make_improved_traversal(trav: pd.DataFrame, frame_dt: float = 0.025) -> pd.DataFrame:
    """The ``improved_traversal_df.csv``-compatible view of fixer output.

    Mirrors ``navigraph.analysis.traversal_fixer.make_improved_traversal`` so the
    CSVs this repo writes stay readable by the upstream notebooks.
    """
    out = trav.copy()
    out["location"] = [
        str(e) if lt == "edge" else n
        for lt, n, e in zip(out["location_type"],
                            out["headstage_graph_node"],
                            out["headstage_graph_edge"])
    ]
    out = out.rename(columns={
        "start_frame_global": "start_frame",
        "end_frame_global": "end_frame",
        "duration": "duration_frames",
    })
    out["start_time"] = out["start_frame"] * frame_dt
    out["end_time"] = out["end_frame"] * frame_dt
    out["duration_time"] = out["duration_frames"] * frame_dt
    out["inference_reason"] = out["is_inferred"].map(
        {True: "noise_smoothed_or_interpolated", False: None}
    )
    keep = ["location", "location_type", "start_frame", "end_frame",
            "duration_frames", "start_time", "end_time", "duration_time",
            "is_inferred", "inference_reason"]
    return out[[c for c in keep if c in out.columns]]


__all__ = [
    "ImprovedMazeTraversalFixer",
    "build_traversal_df",
    "make_improved_traversal",
    "IN_TRIAL_COLUMNS",
]
