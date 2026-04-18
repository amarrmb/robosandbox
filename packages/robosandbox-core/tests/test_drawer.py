"""Tests for the drawer scene primitive + OpenDrawer / CloseDrawer skills."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

import pytest

from robosandbox.agent.agent import Agent
from robosandbox.agent.context import AgentContext
from robosandbox.agent.planner import StubPlanner
from robosandbox.grasp.analytic import AnalyticTopDown
from robosandbox.motion.ik import DLSMotionPlanner
from robosandbox.perception.ground_truth import GroundTruthPerception
from robosandbox.scene.mjcf_builder import build_model
from robosandbox.sim.mujoco_backend import MuJoCoBackend
from robosandbox.skills.drawer import CloseDrawer, OpenDrawer
from robosandbox.skills.home import Home
from robosandbox.skills.pick import Pick
from robosandbox.skills.place import PlaceOn
from robosandbox.skills.pour import Pour
from robosandbox.skills.push import Push
from robosandbox.skills.tap import Tap
from robosandbox.types import Pose, Scene, SceneObject


def _drawer_scene() -> Scene:
    urdf = Path(str(files("robosandbox").joinpath("assets/robots/franka_panda/panda.xml")))
    cfg = Path(
        str(files("robosandbox").joinpath("assets/robots/franka_panda/panda.robosandbox.yaml"))
    )
    return Scene(
        robot_urdf=urdf,
        robot_config=cfg,
        objects=(
            SceneObject(
                id="drawer_a",
                kind="drawer",
                size=(0.15, 0.12, 0.05),
                pose=Pose(xyz=(0.42, 0.0, 0.08)),
                rgba=(0.55, 0.35, 0.2, 1.0),
                drawer_max_open=0.12,
            ),
        ),
    )


def test_drawer_scene_compiles_with_cabinet_drawer_and_handle() -> None:
    scene = _drawer_scene()
    model, _ = build_model(scene)
    # All three body names should exist in the compiled model.
    assert model.body("drawer_a__cabinet").id > 0
    assert model.body("drawer_a").id > 0          # sliding inner
    assert model.body("drawer_a_handle").id > 0
    # Slide joint present.
    assert model.joint("drawer_a__slide").id >= 0


def test_drawer_handle_observable_in_scene_objects() -> None:
    scene = _drawer_scene()
    sim = MuJoCoBackend(render_size=(240, 320))
    sim.load(scene)
    try:
        obs = sim.observe()
        assert "drawer_a" in obs.scene_objects
        assert "drawer_a_handle" in obs.scene_objects
        # Handle should sit in front of the cabinet (smaller x).
        assert obs.scene_objects["drawer_a_handle"].xyz[0] < obs.scene_objects["drawer_a"].xyz[0]
    finally:
        sim.close()


def test_open_drawer_displaces_sliding_body_backward() -> None:
    scene = _drawer_scene()
    sim = MuJoCoBackend(render_size=(240, 320))
    sim.load(scene)
    try:
        for _ in range(140):
            sim.step()
        obs0 = sim.observe()
        x0 = obs0.scene_objects["drawer_a"].xyz[0]

        ctx = AgentContext(
            sim=sim,
            perception=GroundTruthPerception(),
            grasp=AnalyticTopDown(),
            motion=DLSMotionPlanner(n_waypoints=160, dt=0.005),
        )
        r = OpenDrawer()(ctx, drawer="drawer_a")
        assert r.success, f"OpenDrawer failed: {r.reason}/{r.reason_detail}"

        obs1 = sim.observe()
        dx = obs1.scene_objects["drawer_a"].xyz[0] - x0
        # Opening pulls the drawer in -x by at least 50 mm.
        assert dx < -0.05, f"expected drawer dx < -0.05 m, got {dx}"
    finally:
        sim.close()


def test_open_then_close_returns_drawer_near_start() -> None:
    scene = _drawer_scene()
    sim = MuJoCoBackend(render_size=(240, 320))
    sim.load(scene)
    try:
        for _ in range(140):
            sim.step()
        x0 = sim.observe().scene_objects["drawer_a"].xyz[0]
        ctx = AgentContext(
            sim=sim,
            perception=GroundTruthPerception(),
            grasp=AnalyticTopDown(),
            motion=DLSMotionPlanner(n_waypoints=160, dt=0.005),
        )
        assert OpenDrawer()(ctx, drawer="drawer_a").success
        x_open = sim.observe().scene_objects["drawer_a"].xyz[0]
        push_dist = float(x0 - x_open + 0.02)   # push back where we came from, +20 mm overshoot budget
        assert CloseDrawer()(ctx, drawer="drawer_a", distance=push_dist).success
        x_close = sim.observe().scene_objects["drawer_a"].xyz[0]
        # Drawer should be within ~2 cm of the start (slide joint range clamps it).
        assert abs(x_close - x0) < 0.02, f"drawer {x_close} not near start {x0}"
    finally:
        sim.close()


def test_open_drawer_unknown_name() -> None:
    scene = _drawer_scene()
    sim = MuJoCoBackend(render_size=(240, 320))
    sim.load(scene)
    try:
        ctx = AgentContext(
            sim=sim,
            perception=GroundTruthPerception(),
            grasp=AnalyticTopDown(),
            motion=DLSMotionPlanner(n_waypoints=160, dt=0.005),
        )
        r = OpenDrawer()(ctx, drawer="cabinet_b")
        assert not r.success
        assert r.reason == "object_not_found"
    finally:
        sim.close()


def test_open_drawer_through_agent_loop() -> None:
    scene = _drawer_scene()
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
        skills = [Pick(), PlaceOn(), Push(), Home(), Pour(), Tap(),
                  OpenDrawer(), CloseDrawer()]
        agent = Agent(ctx=ctx, skills=skills, planner=StubPlanner(skills))
        ep = agent.run("open the drawer_a")
        assert ep.success, f"episode failed: {ep.final_reason}"
        assert [s.name for s in ep.plan] == ["open_drawer"], f"plan: {ep.plan}"
    finally:
        sim.close()
