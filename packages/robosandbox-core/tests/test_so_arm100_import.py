"""Regression tests for the non-bundled SO-ARM100 URDF + sidecar.

Codex flagged that `examples/so_arm100/` shipped a hand-authored sidecar
(TCP placement, joint order, gripper open/closed semantics) with no
automated coverage. These tests guard the assumptions that the
`examples/so_arm100/smoke_test.py` verifies manually, so a regression
in scene loading or the sidecar schema can't silently break the
documented BYO-robot flow.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from robosandbox.sim.mujoco_backend import MuJoCoBackend
from robosandbox.types import Pose, Scene, SceneObject


os.environ.setdefault("MUJOCO_GL", "egl")

# Resolve paths relative to the repo root so pytest can run from anywhere.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_SO100_DIR = _REPO_ROOT / "examples" / "so_arm100"
_ROBOT_XML = _SO100_DIR / "so_arm100.xml"
_ROBOT_YAML = _SO100_DIR / "so_arm100.robosandbox.yaml"


def _scene_with_cube() -> Scene:
    return Scene(
        robot_urdf=_ROBOT_XML,
        robot_config=_ROBOT_YAML,
        objects=(
            SceneObject(
                id="red_cube", kind="box",
                size=(0.012, 0.012, 0.012),
                pose=Pose(xyz=(0.0, -0.25, 0.06)),
                rgba=(0.85, 0.2, 0.2, 1.0),
                mass=0.05,
            ),
        ),
    )


@pytest.mark.skipif(not _ROBOT_XML.exists(), reason="SO-ARM100 assets not present")
def test_so_arm100_loads_and_reports_expected_dof() -> None:
    """Sidecar drives correctly into MuJoCoBackend's runtime state.

    Catches regressions where the sidecar schema changes or a mismatch
    between `arm.joints` and the MJCF body tree silently shifts ordering.
    """
    sim = MuJoCoBackend(render_size=(240, 320), camera="scene")
    sim.load(_scene_with_cube())
    try:
        assert sim.n_dof == 5, f"expected 5 arm DoF, got {sim.n_dof}"
        assert sim.joint_names == [
            "Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll",
        ], f"joint order regressed: {sim.joint_names}"
    finally:
        sim.close()


@pytest.mark.skipif(not _ROBOT_XML.exists(), reason="SO-ARM100 assets not present")
def test_so_arm100_gripper_open_closed_ordering() -> None:
    """Driving gripper=1.0 (sim-side 'closed') yields a *smaller* observed
    width than gripper=0.0 ('open'). Guards against the open_qpos /
    closed_qpos swap Codex flagged by catching a backwards sidecar.
    """
    sim = MuJoCoBackend(render_size=(240, 320), camera="scene")
    sim.load(_scene_with_cube())
    try:
        # Settle before the command so the position actuator starts from
        # home and the comparison reflects steady-state tracking.
        for _ in range(100):
            sim.step()

        # Command closed, observe.
        for _ in range(200):
            sim.step(gripper=1.0)
        closed_w = sim.observe().gripper_width

        # Command open, observe.
        for _ in range(200):
            sim.step(gripper=0.0)
        open_w = sim.observe().gripper_width

        assert open_w > closed_w, (
            f"open gripper_width ({open_w:.4f}) should be larger than "
            f"closed ({closed_w:.4f}); open_qpos/closed_qpos likely swapped"
        )
    finally:
        sim.close()


@pytest.mark.skipif(not _ROBOT_XML.exists(), reason="SO-ARM100 assets not present")
def test_so_arm100_reachability_passes_for_workspace_cube() -> None:
    """A cube inside the SO-ARM100's reach envelope must be flagged
    reachable. Regression for sidecar home_qpos that doesn't put the
    gripper within IK range of its own workspace.
    """
    from robosandbox.scene.reachability import check_scene_reachability

    warnings = check_scene_reachability(_scene_with_cube())
    assert warnings == [], (
        f"cube at (0, -0.25, 0.06) should be reachable; warnings: "
        f"{[(w.id, w.first_failed_phase, w.detail[:60]) for w in warnings]}"
    )
