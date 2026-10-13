"""Tests for the raw-NaviGraph -> summary_df ingest stage.

The trial-duration test guards a bug that is invisible per trial and severe in
aggregate, so it is the most important test in this file.
"""

import numpy as np
import pandas as pd
import pytest

from navigraph_analysis.io.bpod import trial_end_times
from navigraph_analysis.io.navigraph_output import _coerce_frame
from navigraph_analysis.prep.frame_alignment import build_frames_df
from navigraph_analysis.prep.ingest import _lick_mask, _reward_events, graph_bodyparts


class _Struct:
    """Stand-in for scipy's mat_struct (attribute access, no dict interface)."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


def _session_data(n_trials=5, trial_end_state_len=0.05):
    """A SessionData dict with both duration sources, which disagree by design."""
    onsets = np.array([9.8465, 4.3207, 5.4309, 4.1745, 4.2995])[:n_trials]
    trials = np.array([
        _Struct(States=_Struct(TrialEnd=np.array([o, o + trial_end_state_len])))
        for o in onsets
    ], dtype=object)
    starts = np.cumsum(np.r_[0.0, onsets[:-1] + trial_end_state_len])
    return {
        "RawEvents": {"Trial": trials},
        "TrialStartTimestamp": starts,
        "TrialEndTimestamp": starts + onsets + trial_end_state_len,
    }, onsets


def test_trial_duration_comes_from_the_trial_end_state_onset():
    """Not TrialEndTimestamp - TrialStartTimestamp, which over-counts every trial.

    The two differ by the length of the TrialEnd state (0.05 s = 2 frames at
    40 fps). build_frames_df cumulatively sums these, so the error accumulates
    and displaces later reward events off the port that paid them.
    """
    sd, onsets = _session_data()
    got = trial_end_times(sd)
    assert np.allclose(got, onsets)

    timestamp_based = np.asarray(sd["TrialEndTimestamp"]) - np.asarray(sd["TrialStartTimestamp"])
    assert not np.allclose(got, timestamp_based), "the two sources must not be conflated"
    # the drift is 2 frames per trial and compounds
    drift = (timestamp_based - got).sum() * 40
    assert drift == pytest.approx(2 * len(onsets))


def test_trial_duration_falls_back_with_a_warning():
    sd, _ = _session_data()
    del sd["RawEvents"]
    with pytest.warns(RuntimeWarning, match="accumulates"):
        out = trial_end_times(sd)
    assert len(out) == 5


def test_trial_duration_raises_when_no_source():
    with pytest.raises(KeyError, match="no trial-duration source"):
        trial_end_times({"nTrials": 5})


def test_accumulated_drift_moves_reward_events_off_port():
    """The end-to-end consequence the fix prevents."""
    sd, onsets = _session_data()
    good = build_frames_df(trial_end_times(sd), fps=40.0)
    bad = build_frames_df(
        np.asarray(sd["TrialEndTimestamp"]) - np.asarray(sd["TrialStartTimestamp"]),
        fps=40.0,
    )
    good_starts = good.groupby("trial_idx")["frame_idx_global"].min()
    bad_starts = bad.groupby("trial_idx")["frame_idx_global"].min()
    # the last trial is displaced by 2 frames per preceding trial
    assert bad_starts.iloc[-1] - good_starts.iloc[-1] == 2 * (len(onsets) - 1)


# -- raw pickle layouts ------------------------------------------------------

def test_coerce_frame_reads_the_nested_per_session_layout():
    """Current NaviGraph output nests the table one level under the session name."""
    df = pd.DataFrame({"headstage_x": [1.0, 2.0]})
    obj = {
        "session_NPC1_08_09_2025": {"dataframe": df, "session_path": "/tmp"},
        "timestamp": "20260924_142558",
        "experiment_path": "/tmp",
    }
    assert _coerce_frame(obj).equals(df)


@pytest.mark.parametrize("obj_factory", [
    lambda df: df,                        # bare DataFrame
    lambda df: {"dataframe": df},         # flat, keyed
    lambda df: {"anything": df},          # flat, single frame under any key
])
def test_coerce_frame_accepts_older_layouts(obj_factory):
    df = pd.DataFrame({"headstage_x": [1.0]})
    assert _coerce_frame(obj_factory(df)).equals(df)


def test_coerce_frame_refuses_to_guess_between_sessions():
    df = pd.DataFrame({"headstage_x": [1.0]})
    obj = {"session_a": {"dataframe": df}, "session_b": {"dataframe": df}}
    with pytest.raises(TypeError, match="2 sessions"):
        _coerce_frame(obj)


def test_graph_bodyparts_ignores_empty_columns():
    raw = pd.DataFrame({
        "headstage_graph_node": ["R0", "R11"],
        "nose_graph_node": [None, None],       # tracked but never localised
        "headstage_x": [1.0, 2.0],
    })
    assert graph_bodyparts(raw) == ["headstage"]


# -- annotation --------------------------------------------------------------

def test_reward_events_land_on_the_right_global_frame():
    frames = build_frames_df([10.0, 10.0], fps=40.0)   # 400 frames per trial
    reward = pd.DataFrame({
        "trial": [1, 2],
        "port_number": [1, 2],
        "reward_size": [0.03, 0.015],
        "reward_relative_time": [1.0, 2.0],
    })
    layout = {1: ("R522", ["R411"]), 2: ("R526", ["R413"])}
    ev = _reward_events(reward, frames, layout, fps=40.0)
    assert list(ev["frame_idx_global"]) == [40, 400 + 80]
    assert list(ev["reward_size"]) == [0.03, 0.015]


def test_reward_events_drop_ports_not_in_the_layout():
    frames = build_frames_df([10.0], fps=40.0)
    reward = pd.DataFrame({"trial": [1, 1], "port_number": [1, 9],
                           "reward_size": [0.03, 0.0],
                           "reward_relative_time": [1.0, 2.0]})
    ev = _reward_events(reward, frames, {1: ("R522", ["R411"])}, fps=40.0)
    assert list(ev["reward_port"]) == [1]      # port 9 is not a physical port


def test_lick_mask_marks_the_nearest_frame_within_tolerance():
    frames = build_frames_df([1.0], fps=40.0)     # 40 frames, 0.025 s apart
    licks = pd.DataFrame({"trial": [1, 1], "lick_time": [0.100, 0.500]})
    mask = _lick_mask(licks, frames, fps=40.0)
    assert mask.sum() == 2
    assert mask[4] and mask[20]                    # 0.100/0.025 and 0.500/0.025


def test_lick_mask_marks_every_frame_within_tolerance():
    """A lick falling exactly between two frames marks both.

    The tolerance (0.013 s) is half a frame interval at 40 fps, so a lick at
    0.0125 s is within tolerance of both frame 0 and frame 1. This is the
    notebook's rule kept verbatim, and it is why a session can report slightly
    more lick frames than it has lick events.
    """
    frames = build_frames_df([1.0], fps=40.0)
    licks = pd.DataFrame({"trial": [1], "lick_time": [0.0125]})
    mask = _lick_mask(licks, frames, fps=40.0)
    assert mask.sum() == 2
    assert mask[0] and mask[1]


def test_lick_mask_ignores_licks_from_other_trials():
    frames = build_frames_df([1.0, 1.0], fps=40.0)
    licks = pd.DataFrame({"trial": [2], "lick_time": [0.100]})
    mask = _lick_mask(licks, frames, fps=40.0)
    assert mask[:40].sum() == 0          # trial 1 untouched
    assert mask[40:].sum() == 1          # marked in trial 2 only
