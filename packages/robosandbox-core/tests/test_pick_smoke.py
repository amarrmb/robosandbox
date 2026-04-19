"""Smoke test: the hello-pick vertical slice lifts the cube."""

from __future__ import annotations

import pytest
from robosandbox.agent.context import AgentContext
from robosandbox.grasp.analytic import AnalyticTopDown
from robosandbox.motion.ik import DLSMotionPlanner
from robosandbox.perception.ground_truth import GroundTruthPerception
from robosandbox.sim.mujoco_backend import MuJoCoBackend
from robosandbox.skills.pick import Pick
from robosandbox.types import Pose, Scene, SceneObject


@pytest.fixture
def cube_scene() -> Scene:
    return Scene(
        objects=(
            SceneObject(
                id="red_cube",
                kind="box",
                size=(0.012, 0.012, 0.012),
                pose=Pose(xyz=(0.05, 0.0, 0.07)),
                mass=0.05,
                rgba=(0.85, 0.2, 0.2, 1.0),
            ),
        ),
    )


def test_pick_lifts_cube(cube_scene: Scene) -> None:
    sim = MuJoCoBackend(render_size=(240, 320))
    sim.load(cube_scene)
    try:
        # settle gravity
        for _ in range(80):
            sim.step()
        ctx = AgentContext(
            sim=sim,
            perception=GroundTruthPerception(),
            grasp=AnalyticTopDown(),
            motion=DLSMotionPlanner(n_waypoints=160, dt=0.005),
        )
        result = Pick()(ctx, object="red cube")
    finally:
        sim.close()

    assert result.success, f"Pick failed: {result.reason} ({result.reason_detail})"
    assert result.artifacts["lifted_m"] > 0.05, result.artifacts


def test_perception_ground_truth_matches_id(cube_scene: Scene) -> None:
    sim = MuJoCoBackend(render_size=(120, 160))
    sim.load(cube_scene)
    try:
        for _ in range(30):
            sim.step()
        obs = sim.observe()
        matches = GroundTruthPerception().locate("red cube", obs)
    finally:
        sim.close()
    assert len(matches) == 1
    assert matches[0].label == "red_cube"
    assert matches[0].pose_3d is not None


def test_observation_shapes(cube_scene: Scene) -> None:
    sim = MuJoCoBackend(render_size=(240, 320))
    sim.load(cube_scene)
    try:
        sim.step()
        obs = sim.observe()
    finally:
        sim.close()
    assert obs.rgb.shape == (240, 320, 3)
    assert obs.depth is not None and obs.depth.shape == (240, 320)
    assert obs.robot_joints.shape == (6,)
    assert obs.camera_intrinsics is not None
