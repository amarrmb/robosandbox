"""Integration test: Franka Panda loads + homes + picks a cube end-to-end.

This is the acceptance test for the URDF-import slice. If this passes,
Scene(robot_urdf=Path("panda.xml")) does what the README promises — any
arm, any object, any command — at least for the bundled Franka.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from robosandbox.agent.context import AgentContext
from robosandbox.grasp.analytic import AnalyticTopDown
from robosandbox.motion.ik import DLSMotionPlanner
from robosandbox.perception.ground_truth import GroundTruthPerception
from robosandbox.sim.mujoco_backend import MuJoCoBackend
from robosandbox.skills.pick import Pick
from robosandbox.types import Pose, Scene, SceneObject


@pytest.fixture
def franka_scene() -> Scene:
    from importlib.resources import files

    urdf = Path(str(files("robosandbox").joinpath("assets/robots/franka_panda/panda.xml")))
    config = Path(
        str(files("robosandbox").joinpath("assets/robots/franka_panda/panda.robosandbox.yaml"))
    )
    return Scene(
        robot_urdf=urdf,
        robot_config=config,
        objects=(
            SceneObject(
                id="red_cube",
                kind="box",
                size=(0.012, 0.012, 0.012),
                pose=Pose(xyz=(0.4, 0.0, 0.06)),
                mass=0.05,
                rgba=(0.85, 0.2, 0.2, 1.0),
            ),
        ),
    )


def test_franka_loads_with_7_dof(franka_scene: Scene) -> None:
    sim = MuJoCoBackend(render_size=(240, 320))
    sim.load(franka_scene)
    try:
        assert sim.n_dof == 7
        assert sim.joint_names == ["joint1", "joint2", "joint3", "joint4",
                                    "joint5", "joint6", "joint7"]
    finally:
        sim.close()


def test_franka_home_pose_matches_sidecar(franka_scene: Scene) -> None:
    """After reset, arm qpos should equal the sidecar's home_qpos."""
    sim = MuJoCoBackend(render_size=(240, 320))
    sim.load(franka_scene)
    try:
        obs = sim.observe()
        expected = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])
        assert np.allclose(obs.robot_joints, expected, atol=1e-4), (obs.robot_joints, expected)
    finally:
        sim.close()


def test_franka_pick_lifts_cube(franka_scene: Scene) -> None:
    """Acceptance test: Pick skill on bundled Franka picks a cube >5 cm."""
    sim = MuJoCoBackend(render_size=(240, 320))
    sim.load(franka_scene)
    try:
        for _ in range(100):
            sim.step()
        ctx = AgentContext(
            sim=sim,
            perception=GroundTruthPerception(),
            grasp=AnalyticTopDown(),
            motion=DLSMotionPlanner(n_waypoints=160, dt=0.005),
        )
        result = Pick()(ctx, object="red cube")
        obs_final = sim.observe()
        final_z = obs_final.scene_objects["red_cube"].xyz[2]
    finally:
        sim.close()

    assert result.success, (result.reason, result.reason_detail)
    # Initial cube z ~0.06 (settled to ~0.05); success requires >50mm rise = z > 0.10
    assert final_z > 0.10, f"cube final z={final_z} < 0.10"
