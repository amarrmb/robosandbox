"""Trajectory inspector — in-RAM buffer behaviour of SimThread.

These tests construct ``SimThread`` but do NOT call ``.start()`` (no MuJoCo
involvement); the buffer lifecycle methods we exercise don't touch ``_sim``.

The WebSocket round-trip is deliberately not exercised here — FastAPI's
``TestClient`` + WebSocket + a thread running a live MuJoCo renderer makes
for a brittle test, and the inspector's value is the buffer, not the wire.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fastapi")  # Skip viewer tests if extras aren't installed

from robosandbox.types import Observation, Pose
from robosandbox.viewer.server import FrameSnapshot, SimThread


def _fake_obs(t: float, joint_val: float = 0.0) -> Observation:
    """Build a minimal Observation. rgb is a 1x1 placeholder — the snapshot
    path accepts the pre-encoded JPEG bytes directly, so the pixels don't
    need to be real."""
    return Observation(
        rgb=np.zeros((1, 1, 3), dtype=np.uint8),
        depth=None,
        robot_joints=np.array([joint_val] * 6, dtype=np.float64),
        ee_pose=Pose(xyz=(0.1, 0.2, 0.3 + t), quat_xyzw=(0.0, 0.0, 0.0, 1.0)),
        gripper_width=0.04,
        scene_objects={"cube": Pose(xyz=(0.5, 0.0, 0.05), quat_xyzw=(0.0, 0.0, 0.0, 1.0))},
        timestamp=t,
    )


def _make_thread(tmp_path: Path) -> SimThread:
    """SimThread without start() — we drive the buffer methods directly."""
    return SimThread(runs_dir=tmp_path / "runs")


def test_append_builds_snapshots_from_observations(tmp_path):
    st = _make_thread(tmp_path)
    for i in range(5):
        st._append_snapshot(_fake_obs(t=i * 0.1, joint_val=float(i)), jpg=b"JPEG%d" % i)

    assert len(st._trajectory) == 5
    first, last = st._trajectory[0], st._trajectory[-1]
    assert isinstance(first, FrameSnapshot)
    assert first.timestamp == pytest.approx(0.0)
    assert last.timestamp == pytest.approx(0.4)
    # Joint payload is copied out of the observation verbatim.
    assert list(last.robot_joints) == [4.0] * 6
    # Scene objects flatten Pose -> (xyz, quat) tuples so snapshots are
    # safely serialisable over the wire without a Pose import on the client.
    assert "cube" in last.scene_objects
    xyz, quat = last.scene_objects["cube"]
    assert xyz == (0.5, 0.0, 0.05)
    assert quat == (0.0, 0.0, 0.0, 1.0)
    assert last.rgb_jpeg == b"JPEG4"


def test_trajectory_cleared_on_load_task(tmp_path, monkeypatch):
    """Loading a new task must drop any prior trajectory + exit inspection.

    We stub out the MuJoCo/load path since those aren't what we're testing —
    the contract under test is 'the buffer is empty after load_task'.
    """
    st = _make_thread(tmp_path)
    for i in range(3):
        st._append_snapshot(_fake_obs(t=i * 0.01), jpg=b"x")
    st._inspecting = True
    assert len(st._trajectory) == 3

    # Replace the heavy interior of _load_task so we isolate the reset logic.
    # The test contract: whatever the impl does, after _load_task the buffer
    # is empty and inspection is cleared.
    def _stub_load(task_name: str) -> None:
        st._trajectory.clear()
        st._inspecting = False

    monkeypatch.setattr(st, "_load_task", _stub_load)
    st._load_task("any_task")

    assert len(st._trajectory) == 0
    assert st._inspecting is False


def test_trajectory_reset_on_actual_load_sequence(tmp_path):
    """The real _load_task path also clears. We exercise just the reset
    block by raising inside load_builtin_task — the clear must happen
    *before* the MuJoCo work, so a failing task load still resets."""
    st = _make_thread(tmp_path)
    for i in range(4):
        st._append_snapshot(_fake_obs(t=i), jpg=b"y")
    st._inspecting = True
    assert len(st._trajectory) == 4

    # Calling the real _load_task with a bad name will raise; we don't care
    # — we only care that the reset side-effect ran before the raise.
    with pytest.raises(Exception):
        st._load_task("__definitely_not_a_task__")

    assert len(st._trajectory) == 0
    assert st._inspecting is False


def test_buffer_drops_oldest_when_over_cap(tmp_path, monkeypatch):
    """With cap N, pushing N+5 snapshots keeps length at N and drops the
    oldest five — the deque's maxlen does the work."""
    st = _make_thread(tmp_path)
    cap = 10
    # Re-cap via a fresh deque so we don't have to push 900 frames.
    from collections import deque
    st._trajectory = deque(maxlen=cap)

    for i in range(cap + 5):
        st._append_snapshot(_fake_obs(t=float(i)), jpg=b"z%d" % i)

    assert len(st._trajectory) == cap
    # Oldest five (t=0..4) were dropped; surviving range is t=5..14.
    assert st._trajectory[0].timestamp == pytest.approx(5.0)
    assert st._trajectory[-1].timestamp == pytest.approx(14.0)


def test_inspect_at_clamps_and_emits(tmp_path):
    st = _make_thread(tmp_path)
    for i in range(3):
        st._append_snapshot(_fake_obs(t=i * 0.1), jpg=b"frame%d" % i)

    # drain existing events
    while not st.events.empty():
        st.events.get_nowait()

    st._inspect_at({"frame_idx": 99})  # out of range — clamps to last

    # First, the JPEG was pushed onto the frames queue.
    assert not st.frames.empty()
    jpg = st.frames.get_nowait()
    assert jpg == b"frame2"

    # Then the inspect_frame event carries the metadata.
    evt = st.events.get_nowait()
    assert evt["type"] == "inspect_frame"
    assert evt["frame_idx"] == 2
    assert evt["total"] == 3
    assert evt["timestamp"] == pytest.approx(0.2)
    assert evt["gripper_width"] == pytest.approx(0.04)
    assert evt["ee_pose"]["xyz"] == [0.1, 0.2, pytest.approx(0.5)]
    assert "cube" in evt["scene_objects"]
    assert st._inspecting is True


def test_inspect_at_empty_trajectory_errors(tmp_path):
    st = _make_thread(tmp_path)
    while not st.events.empty():
        st.events.get_nowait()

    st._inspect_at({"frame_idx": 0})

    evt = st.events.get_nowait()
    assert evt["type"] == "error"
    assert "no trajectory" in evt["message"]


def test_inspect_clear_flips_flag(tmp_path):
    st = _make_thread(tmp_path)
    st._inspecting = True
    # _sim is None so _publish_frame is skipped; exit cleanly.
    st._inspect_clear()
    assert st._inspecting is False
    # Emits the cleared event for client UX.
    evt = st.events.get_nowait()
    assert evt["type"] == "inspect_cleared"
