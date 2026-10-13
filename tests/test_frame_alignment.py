"""Tests for the BPOD->video frame table and the clock-offset detection."""

import numpy as np
import pytest

from navigraph_analysis.prep.frame_alignment import (
    AlignmentError,
    align_frames_to_video,
    build_frames_df,
    on_port_fraction,
)

FPS = 40.0
LAYOUT = {1: ("R522", ["R411"]), 2: ("R526", ["R413"]),
          7: ("R59", ["R44"]), 8: ("R55", ["R42"])}


def _session(n_trials=250, seed=0):
    """A synthetic session: trial durations, its frame table, and reward events."""
    rng = np.random.default_rng(seed)
    durations = np.round(rng.uniform(8, 13, n_trials), 4)
    frames = build_frames_df(durations, fps=FPS)
    trial_start = frames.groupby("trial_idx")["frame_idx_global"].min()
    ports = rng.choice(list(LAYOUT), n_trials)
    times = np.array([rng.uniform(2.5, d - 2.5) for d in durations])
    rframe = (trial_start.values + np.round(times * FPS)).astype(int)
    return durations, frames, rframe, ports


def _node_track(frames, rframe, ports, dwell=40):
    """Per-frame node, with the animal on the paying port around each reward."""
    node = np.full(frames.attrs["bpod_frame_count"], None, dtype=object)
    for f, p in zip(rframe, ports):
        node[max(0, f - dwell):f + dwell] = LAYOUT[p][0]
    return node


def test_build_frames_df_skips_one_frame_per_inexact_trial():
    # 10.0 s is an exact multiple of 1/40; 10.01 s is not.
    frames = build_frames_df([10.0, 10.01], fps=FPS)
    per_trial = frames.groupby("trial_idx").size()
    assert per_trial[0] == 400
    assert per_trial[1] == 400              # floor(400.4), the 401st is dropped
    # the skipped frame is still counted in the global index
    assert frames.attrs["bpod_frame_count"] == 801
    assert frames.loc[frames["trial_idx"] == 1, "frame_idx_global"].min() == 400


def test_aligned_session_is_not_shifted():
    _d, frames, rframe, ports = _session()
    node = _node_track(frames, rframe, ports)
    out, al = align_frames_to_video(frames, node, rframe, ports, LAYOUT,
                                    frames.attrs["bpod_frame_count"], fps=FPS)
    assert al.frame_shift == 0
    assert al.on_port_pct == pytest.approx(100.0)
    assert al.n_frames_dropped == 0
    assert len(out) == len(frames)


@pytest.mark.parametrize("shift", [137, 5800, 5814])
def test_late_camera_start_is_detected_and_corrected(shift):
    _d, frames, rframe, ports = _session()
    node = _node_track(frames, rframe, ports)
    n_video = frames.attrs["bpod_frame_count"] - shift

    raw, _n = on_port_fraction(rframe, ports, node[shift:], LAYOUT, n_video, 0)
    assert raw < 60.0, "the uncorrected session must look broken"

    out, al = align_frames_to_video(frames, node[shift:], rframe, ports, LAYOUT,
                                    n_video, fps=FPS)
    assert al.frame_shift == shift
    assert al.on_port_pct > 95.0
    assert out["frame_idx_global"].between(0, n_video - 1).all()
    # A late start can only cost trials from the front, never the back. A shift
    # shorter than the first trial costs none at all, so only the direction is
    # asserted here.
    assert not al.trials_after_stop
    if shift > 5000:
        assert al.trials_before_start


def test_early_camera_stop_is_trimmed_not_shifted():
    """A large frame-budget overshoot is not on its own a misalignment."""
    _d, frames, rframe, ports = _session()
    node = _node_track(frames, rframe, ports)
    n_video = frames.attrs["bpod_frame_count"] - 7616

    out, al = align_frames_to_video(frames, node[:n_video], rframe, ports, LAYOUT,
                                    n_video, fps=FPS)
    assert al.frame_shift == 0, "an early stop must not be 'corrected' by a shift"
    assert al.overshoot == 7616
    assert al.n_frames_dropped > 0
    assert al.trials_after_stop and not al.trials_before_start
    assert out["frame_idx_global"].max() < n_video


def test_mismatched_files_raise_rather_than_inventing_a_shift():
    _d, frames, rframe, ports = _session()
    rng = np.random.default_rng(3)
    junk = rng.choice(["R522", "L56", "R0", "L0"],
                      frames.attrs["bpod_frame_count"]).astype(object)
    with pytest.raises(AlignmentError, match="not a simple camera-start offset"):
        align_frames_to_video(frames, junk, rframe, ports, LAYOUT,
                              frames.attrs["bpod_frame_count"], fps=FPS)


def test_too_few_reward_events_raises():
    _d, frames, rframe, ports = _session(n_trials=10)
    node = _node_track(frames, rframe, ports)
    with pytest.raises(AlignmentError, match="too few to verify alignment"):
        align_frames_to_video(frames, node, rframe, ports, LAYOUT,
                              frames.attrs["bpod_frame_count"], fps=FPS)


def test_on_port_fraction_counts_parent_as_on_port():
    """Leaf/parent flicker is pervasive; the parent still means 'at the port'."""
    node = np.array(["R411"] * 100, dtype=object)       # the parent of port 1
    rframe = np.arange(0, 100, 4)
    ports = np.ones(len(rframe), dtype=int)
    pct, n = on_port_fraction(rframe, ports, node, LAYOUT, 100)
    assert pct == pytest.approx(100.0)
    assert n == len(rframe)
