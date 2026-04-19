"""Integration tests for the Pour and Tap skills (Sprint B 2.1).

Both skills run end-to-end on the bundled Franka. Neither needs a new
scene primitive — Pour tilts the held object over a target; Tap
descends onto a target and retreats.
"""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

from robosandbox.agent.agent import Agent
from robosandbox.agent.context import AgentContext
from robosandbox.agent.planner import StubPlanner
from robosandbox.grasp.analytic import AnalyticTopDown
from robosandbox.motion.ik import DLSMotionPlanner
from robosandbox.perception.ground_truth import GroundTruthPerception
from robosandbox.sim.mujoco_backend import MuJoCoBackend
from robosandbox.skills.home import Home
from robosandbox.skills.pick import Pick
from robosandbox.skills.place import PlaceOn
from robosandbox.skills.pour import Pour
from robosandbox.skills.push import Push
from robosandbox.skills.tap import Tap
from robosandbox.types import Pose, Scene, SceneObject


def _franka_scene_with_two_ycb() -> Scene:
    urdf = Path(str(files("robosandbox").joinpath("assets/robots/franka_panda/panda.xml")))
    cfg = Path(
        str(files("robosandbox").joinpath("assets/robots/franka_panda/panda.robosandbox.yaml"))
    )
    can_sidecar = Path(
        str(
            files("robosandbox").joinpath(
                "assets/objects/ycb/005_tomato_soup_can/tomato_soup_can.robosandbox.yaml"
            )
        )
    )
    bowl_sidecar = Path(
        str(files("robosandbox").joinpath("assets/objects/ycb/024_bowl/bowl.robosandbox.yaml"))
    )
    return Scene(
        robot_urdf=urdf,
        robot_config=cfg,
        objects=(
            SceneObject(
                id="tomato_soup_can",
                kind="mesh",
                size=(0.0,),
                pose=Pose(xyz=(0.35, -0.12, 0.065)),
                mass=0.0,
                mesh_sidecar=can_sidecar,
            ),
            SceneObject(
                id="bowl",
                kind="mesh",
                size=(0.0,),
                pose=Pose(xyz=(0.45, 0.12, 0.06)),
                mass=0.0,
                mesh_sidecar=bowl_sidecar,
            ),
        ),
    )


def _franka_scene_with_apple() -> Scene:
    urdf = Path(str(files("robosandbox").joinpath("assets/robots/franka_panda/panda.xml")))
    cfg = Path(
        str(files("robosandbox").joinpath("assets/robots/franka_panda/panda.robosandbox.yaml"))
    )
    apple = Path(
        str(files("robosandbox").joinpath("assets/objects/ycb/013_apple/apple.robosandbox.yaml"))
    )
    return Scene(
        robot_urdf=urdf,
        robot_config=cfg,
        objects=(
            SceneObject(
                id="apple",
                kind="mesh",
                size=(0.0,),
                pose=Pose(xyz=(0.4, 0.0, 0.05)),
                mass=0.0,
                mesh_sidecar=apple,
            ),
        ),
    )


# ---- Tap --------------------------------------------------------------


def test_tap_lowers_end_effector_near_target() -> None:
    scene = _franka_scene_with_apple()
    sim = MuJoCoBackend(render_size=(240, 320))
    sim.load(scene)
    try:
        for _ in range(140):
            sim.step()
        ctx = AgentContext(
            sim=sim,
            perception=GroundTruthPerception(),
            grasp=AnalyticTopDown(),
            motion=DLSMotionPlanner(n_waypoints=160, dt=0.005),
        )
        result = Tap()(ctx, object="apple")
        assert result.success, f"Tap failed: {result.reason}/{result.reason_detail}"

        # After the tap, ee should be above the apple (retreated), not below.
        obs = sim.observe()
        apple_z = obs.scene_objects["apple"].xyz[2]
        ee_z = obs.ee_pose.xyz[2]
        assert ee_z > apple_z, f"ee_z={ee_z} should be above apple_z={apple_z}"
    finally:
        sim.close()


def test_tap_unknown_object() -> None:
    scene = _franka_scene_with_apple()
    sim = MuJoCoBackend(render_size=(240, 320))
    sim.load(scene)
    try:
        ctx = AgentContext(
            sim=sim,
            perception=GroundTruthPerception(),
            grasp=AnalyticTopDown(),
            motion=DLSMotionPlanner(n_waypoints=160, dt=0.005),
        )
        result = Tap()(ctx, object="unicorn")
        assert not result.success
        assert result.reason == "object_not_found"
    finally:
        sim.close()


def test_tap_through_agent_loop() -> None:
    """StubPlanner's tap regex matches, agent dispatches Tap."""
    scene = _franka_scene_with_apple()
    sim = MuJoCoBackend(render_size=(240, 320))
    sim.load(scene)
    try:
        for _ in range(140):
            sim.step()
        ctx = AgentContext(
            sim=sim,
            perception=GroundTruthPerception(),
            grasp=AnalyticTopDown(),
            motion=DLSMotionPlanner(n_waypoints=160, dt=0.005),
        )
        skills = [Pick(), PlaceOn(), Push(), Home(), Pour(), Tap()]
        agent = Agent(ctx=ctx, skills=skills, planner=StubPlanner(skills))
        ep = agent.run("tap the apple")
        assert ep.success, f"episode failed: {ep.final_reason}"
        assert [s.name for s in ep.plan] == ["tap"], f"unexpected plan: {ep.plan}"
    finally:
        sim.close()


# ---- Pour -------------------------------------------------------------


def test_pour_moves_ee_over_target_and_tilts() -> None:
    scene = _franka_scene_with_two_ycb()
    sim = MuJoCoBackend(render_size=(240, 320))
    sim.load(scene)
    try:
        for _ in range(140):
            sim.step()
        ctx = AgentContext(
            sim=sim,
            perception=GroundTruthPerception(),
            grasp=AnalyticTopDown(),
            motion=DLSMotionPlanner(n_waypoints=160, dt=0.005),
        )
        # Pour requires the pouring object to be held; pick the can first.
        pick_result = Pick()(ctx, object="tomato soup can")
        assert pick_result.success, (
            f"pre-pour Pick failed: {pick_result.reason}/{pick_result.reason_detail}"
        )
        result = Pour()(ctx, target="bowl")
        assert result.success, f"Pour failed: {result.reason}/{result.reason_detail}"

        obs = sim.observe()
        bowl_xy = obs.scene_objects["bowl"].xyz[:2]
        ee_xy = obs.ee_pose.xyz[:2]
        # EE should end up within ~8 cm xy of the bowl (IK tolerance + motion).
        dist = ((ee_xy[0] - bowl_xy[0]) ** 2 + (ee_xy[1] - bowl_xy[1]) ** 2) ** 0.5
        assert dist < 0.08, f"ee xy {ee_xy} too far from bowl {bowl_xy} (dist={dist})"
    finally:
        sim.close()


def test_pour_unknown_target() -> None:
    scene = _franka_scene_with_apple()
    sim = MuJoCoBackend(render_size=(240, 320))
    sim.load(scene)
    try:
        ctx = AgentContext(
            sim=sim,
            perception=GroundTruthPerception(),
            grasp=AnalyticTopDown(),
            motion=DLSMotionPlanner(n_waypoints=160, dt=0.005),
        )
        result = Pour()(ctx, target="gravity_well")
        assert not result.success
        assert result.reason == "object_not_found"
    finally:
        sim.close()


def test_pour_through_agent_loop_plans_pick_then_pour() -> None:
    """StubPlanner decomposes 'pour X into Y' into [pick(X), pour(Y)]."""
    scene = _franka_scene_with_two_ycb()
    sim = MuJoCoBackend(render_size=(240, 320))
    sim.load(scene)
    try:
        for _ in range(140):
            sim.step()
        ctx = AgentContext(
            sim=sim,
            perception=GroundTruthPerception(),
            grasp=AnalyticTopDown(),
            motion=DLSMotionPlanner(n_waypoints=160, dt=0.005),
        )
        skills = [Pick(), PlaceOn(), Push(), Home(), Pour(), Tap()]
        agent = Agent(ctx=ctx, skills=skills, planner=StubPlanner(skills))
        ep = agent.run("pour the tomato soup can into the bowl")
        assert ep.success, f"episode failed: {ep.final_reason}"
        assert [s.name for s in ep.plan] == ["pick", "pour"], f"plan: {ep.plan}"
    finally:
        sim.close()
