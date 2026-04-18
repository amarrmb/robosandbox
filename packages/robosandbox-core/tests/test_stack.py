"""Tests for the Stack skill — repeated (pick, place_on) orchestration."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

import pytest

from robosandbox.agent.context import AgentContext
from robosandbox.grasp.analytic import AnalyticTopDown
from robosandbox.motion.ik import DLSMotionPlanner
from robosandbox.perception.ground_truth import GroundTruthPerception
from robosandbox.sim.mujoco_backend import MuJoCoBackend
from robosandbox.skills.stack import Stack
from robosandbox.types import Pose, Scene, SceneObject


def _three_cube_scene() -> Scene:
    urdf = Path(str(files("robosandbox").joinpath("assets/robots/franka_panda/panda.xml")))
    cfg = Path(
        str(files("robosandbox").joinpath("assets/robots/franka_panda/panda.robosandbox.yaml"))
    )
    return Scene(
        robot_urdf=urdf,
        robot_config=cfg,
        objects=(
            SceneObject(
                id="base_cube",
                kind="box",
                size=(0.018, 0.018, 0.018),
                pose=Pose(xyz=(0.45, -0.10, 0.06)),
                rgba=(0.2, 0.2, 0.8, 1.0),
                mass=0.05,
            ),
            SceneObject(
                id="red_cube",
                kind="box",
                size=(0.014, 0.014, 0.014),
                pose=Pose(xyz=(0.35, 0.05, 0.06)),
                rgba=(0.8, 0.2, 0.2, 1.0),
                mass=0.05,
            ),
        ),
    )


def test_stack_requires_sources() -> None:
    scene = _three_cube_scene()
    sim = MuJoCoBackend(render_size=(240, 320))
    sim.load(scene)
    try:
        ctx = AgentContext(
            sim=sim,
            perception=GroundTruthPerception(),
            grasp=AnalyticTopDown(),
            motion=DLSMotionPlanner(n_waypoints=160, dt=0.005),
        )
        r = Stack()(ctx, sources=[], base="base_cube")
        assert not r.success
        assert r.reason == "empty_sources"
    finally:
        sim.close()


def test_stack_one_source_on_base() -> None:
    """Stack([red], base_cube) should behave like pick(red) + place_on(base)."""
    scene = _three_cube_scene()
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
        r = Stack()(ctx, sources=["red_cube"], base="base_cube")
        assert r.success, f"Stack failed: {r.reason}/{r.reason_detail}"

        # red_cube should end up above base_cube.
        obs = sim.observe()
        red = obs.scene_objects["red_cube"].xyz
        base = obs.scene_objects["base_cube"].xyz
        # xy within 3 cm and red z > base z.
        assert abs(red[0] - base[0]) < 0.03
        assert abs(red[1] - base[1]) < 0.03
        assert red[2] > base[2], f"red_cube z={red[2]} not above base z={base[2]}"
        assert r.artifacts["placed"] == ["red_cube"]
    finally:
        sim.close()


def test_stack_pick_failure_aborts_with_artifacts() -> None:
    """If the first pick can't find the object, stack fails immediately
    with no ``placed`` entries."""
    scene = _three_cube_scene()
    sim = MuJoCoBackend(render_size=(240, 320))
    sim.load(scene)
    try:
        ctx = AgentContext(
            sim=sim,
            perception=GroundTruthPerception(),
            grasp=AnalyticTopDown(),
            motion=DLSMotionPlanner(n_waypoints=160, dt=0.005),
        )
        r = Stack()(ctx, sources=["unicorn"], base="base_cube")
        assert not r.success
        assert r.reason == "pick_failed"
        assert r.artifacts["placed"] == []
        assert r.artifacts["failed_at"] == "unicorn"
    finally:
        sim.close()
