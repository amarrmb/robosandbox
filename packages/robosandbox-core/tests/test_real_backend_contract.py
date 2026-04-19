"""Tests for the sim-to-real handoff contract.

Guard the minimum surface a real ``RealRobotBackend`` subclass must
honour for observation+step skills to transfer unchanged from sim:

- ``n_dof`` / ``joint_names`` / ``home_qpos`` all stay length-consistent.
- ``step(target_joints=…)`` mutates the observable state in the
  expected direction.
- ``Home`` skill drives an arbitrary start pose to the backend's
  reported home with zero residual.

If these break, the SO-101 handoff tutorial stops being honest.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from robosandbox.agent.context import AgentContext
from robosandbox.skills.home import Home
from robosandbox.types import Scene

# Make the so101_handoff example importable as a sibling module.
_HANDOFF_DIR = Path(__file__).resolve().parents[3] / "examples" / "so101_handoff"
sys.path.insert(0, str(_HANDOFF_DIR))

pytest.importorskip("numpy")
if not (_HANDOFF_DIR / "so101_backend.py").exists():
    pytest.skip("examples/so101_handoff not present", allow_module_level=True)

from so101_backend import SO101Backend  # noqa: E402


def test_config_shapes_are_length_consistent() -> None:
    b = SO101Backend()
    assert b.n_dof == len(b.joint_names) == len(b.home_qpos)
    # Matches the Menagerie trs_so_arm100 exposed DoF so sim-trained
    # policies for that asset carry over without remapping.
    assert b.n_dof == 5


def test_step_mutates_observed_joint_state() -> None:
    b = SO101Backend()
    b.load(Scene())
    target = np.array([0.1, -0.2, 0.3, 0.4, -0.5], dtype=np.float64)
    b.step(target_joints=target)
    observed = b.observe().robot_joints
    assert np.allclose(observed, target), (
        f"step did not update observed state: got {observed}, expected {target}"
    )


def test_gripper_open_width_exceeds_closed() -> None:
    b = SO101Backend()
    b.load(Scene())
    b.step(gripper=1.0)  # closed
    closed = b.observe().gripper_width
    b.step(gripper=0.0)  # open
    open_ = b.observe().gripper_width
    assert open_ > closed, (
        f"open gripper_width ({open_}) must exceed closed ({closed}); "
        "gripper_open / gripper_closed likely swapped in the backend config"
    )


def test_home_skill_runs_against_real_backend() -> None:
    """The headline handoff claim: observation+step skills transfer unchanged.

    Regression for the bug that had ``Home`` hard-coded to the built-in
    arm's 6-DoF home vector. With ``home_qpos`` now exposed on every
    backend, the skill broadcasts against whatever shape the backend
    reports.
    """
    b = SO101Backend()
    b.load(Scene())
    # Start somewhere non-home so the skill has real work to do.
    b._joints = np.array([0.5, -0.5, 0.5, 0.3, -0.3], dtype=np.float64)
    ctx = AgentContext(sim=b, perception=None, grasp=None, motion=None)
    result = Home()(ctx)
    assert result.success, f"Home failed: {result.reason_detail}"
    final = b.observe().robot_joints
    expected_home = np.asarray(b.home_qpos, dtype=np.float64)
    assert np.allclose(final, expected_home, atol=1e-3), (
        f"after Home, joints={final} != home_qpos={expected_home}"
    )


def test_home_skill_reports_dim_mismatch_cleanly() -> None:
    """When a backend's home_qpos length contradicts its n_dof
    observation, Home must surface a structured failure rather than
    raising a ValueError mid-skill.
    """
    b = SO101Backend()
    b.load(Scene())
    # Corrupt the observed-state dimension to simulate a bad sidecar.
    b._joints = np.zeros(6, dtype=np.float64)  # home_qpos is len 5
    ctx = AgentContext(sim=b, perception=None, grasp=None, motion=None)
    result = Home()(ctx)
    assert not result.success
    assert result.reason == "home_dim_mismatch"
