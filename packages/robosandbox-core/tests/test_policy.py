"""Tests for the policy-in-the-loop replay module.

Covers the Policy protocol, the ReplayTrajectoryPolicy concrete impl,
the run_policy loop, and the stub load_policy loader.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from robosandbox.policy import (
    Policy,
    ReplayTrajectoryPolicy,
    load_policy,
    run_policy,
)
from robosandbox.sim.mujoco_backend import MuJoCoBackend
from robosandbox.tasks.loader import SuccessCriterion, load_builtin_task
from robosandbox.types import Observation


# --------- fixtures -----------------------------------------------------


def _dummy_obs(n_dof: int = 6) -> Observation:
    from robosandbox.types import Pose

    return Observation(
        rgb=np.zeros((4, 4, 3), dtype=np.uint8),
        depth=None,
        robot_joints=np.zeros(n_dof),
        ee_pose=Pose(xyz=(0.0, 0.0, 0.3)),
        gripper_width=0.07,
    )


# --------- ReplayTrajectoryPolicy --------------------------------------


def test_replay_trajectory_policy_reads_jsonl_and_replays_in_order(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    rows = []
    for i in range(10):
        joints = rng.uniform(-1.0, 1.0, size=6).tolist()
        rows.append({"joints": joints, "gripper": float(i) / 10.0})
    jsonl = tmp_path / "traj.jsonl"
    jsonl.write_text("\n".join(json.dumps(r) for r in rows))

    policy = ReplayTrajectoryPolicy.from_jsonl(jsonl)
    assert isinstance(policy, Policy)  # runtime_checkable Protocol

    obs = _dummy_obs()
    for i, row in enumerate(rows):
        action = policy.act(obs)
        assert action.shape == (7,)
        np.testing.assert_allclose(action[:6], row["joints"])
        assert float(action[6]) == pytest.approx(row["gripper"])


def test_replay_trajectory_policy_supports_events_jsonl_shape(tmp_path: Path) -> None:
    """LocalRecorder writes `robot_joints` + `gripper_width` keys; we accept both."""
    rows = []
    for i in range(3):
        rows.append(
            {
                "t": 0.01 * i,
                "frame_idx": i,
                "robot_joints": [0.1 * i] * 6,
                "gripper_width": 0.035,  # ~ half-open
            }
        )
    events = tmp_path / "events.jsonl"
    events.write_text("\n".join(json.dumps(r) for r in rows))

    policy = ReplayTrajectoryPolicy.from_jsonl(events)
    obs = _dummy_obs()
    a0 = policy.act(obs)
    np.testing.assert_allclose(a0[:6], [0.0] * 6)
    a1 = policy.act(obs)
    np.testing.assert_allclose(a1[:6], [0.1] * 6)


def test_replay_trajectory_policy_lookahead(tmp_path: Path) -> None:
    rows = [{"joints": [float(i)] * 6, "gripper": 0.0} for i in range(6)]
    jsonl = tmp_path / "traj.jsonl"
    jsonl.write_text("\n".join(json.dumps(r) for r in rows))

    policy = ReplayTrajectoryPolicy.from_jsonl(jsonl, action_lookahead=3)
    obs = _dummy_obs()
    a0 = policy.act(obs)
    # First call advances by 3 → next call returns row 3.
    np.testing.assert_allclose(a0[:6], [0.0] * 6)
    a1 = policy.act(obs)
    np.testing.assert_allclose(a1[:6], [3.0] * 6)


def test_replay_trajectory_policy_clamps_past_end(tmp_path: Path) -> None:
    rows = [{"joints": [1.0] * 6, "gripper": 0.5}]
    jsonl = tmp_path / "traj.jsonl"
    jsonl.write_text(json.dumps(rows[0]))

    policy = ReplayTrajectoryPolicy.from_jsonl(jsonl)
    obs = _dummy_obs()
    a0 = policy.act(obs)
    a1 = policy.act(obs)  # past end → hold last action
    np.testing.assert_allclose(a0, a1)


# --------- run_policy --------------------------------------------------


class _HomePolicy:
    """Always commands the known-safe home pose."""

    _HOME = np.array([0.0, -0.4, 1.2, -0.8, 0.0, 0.0])

    def act(self, obs: Observation) -> np.ndarray:
        return np.concatenate([self._HOME, [0.0]])


def test_run_policy_runs_100_steps_on_home_task() -> None:
    task = load_builtin_task("home")
    sim = MuJoCoBackend(render_size=(120, 160))
    sim.load(task.scene)
    try:
        result = run_policy(sim, _HomePolicy(), max_steps=100)
    finally:
        sim.close()
    assert result["steps"] == 100
    # success is None when no criterion supplied
    assert result["success"] is None
    assert result["final_obs"] is not None


def test_run_policy_evaluates_success_criterion() -> None:
    task = load_builtin_task("home")
    sim = MuJoCoBackend(render_size=(120, 160))
    sim.load(task.scene)
    try:
        # Criterion that is trivially true (empty "all")
        crit = SuccessCriterion(data={"kind": "all", "checks": []})
        result = run_policy(sim, _HomePolicy(), max_steps=40, success=crit)
    finally:
        sim.close()
    assert result["success"] is True


def test_run_policy_invokes_on_step_callback() -> None:
    task = load_builtin_task("home")
    sim = MuJoCoBackend(render_size=(120, 160))
    sim.load(task.scene)
    counter = {"n": 0}

    def _cb(obs: Observation, action: np.ndarray) -> None:
        counter["n"] += 1

    try:
        run_policy(sim, _HomePolicy(), max_steps=7, on_step=_cb)
    finally:
        sim.close()
    assert counter["n"] == 7


# --------- load_policy stub --------------------------------------------


def test_load_policy_missing_path_raises_with_helpful_message(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist"
    with pytest.raises(ImportError) as excinfo:
        load_policy(missing)
    msg = str(excinfo.value)
    assert "checkpoint" in msg.lower()
    # Point the user at the extension seam.
    assert "load_policy" in msg or "LeRobot" in msg or "lerobot" in msg


def test_load_policy_directory_with_policy_json_returns_replay(tmp_path: Path) -> None:
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    # policy.json references a trajectory file in the same dir.
    (ckpt / "traj.jsonl").write_text(json.dumps({"joints": [0.1] * 6, "gripper": 0.0}))
    (ckpt / "policy.json").write_text(
        json.dumps({"kind": "replay_trajectory", "trajectory": "traj.jsonl"})
    )
    pol = load_policy(ckpt)
    assert isinstance(pol, Policy)
    a = pol.act(_dummy_obs())
    np.testing.assert_allclose(a[:6], [0.1] * 6)
