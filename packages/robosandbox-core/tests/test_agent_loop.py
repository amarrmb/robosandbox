"""Agent-loop tests — planner-agnostic.

Uses a `MockPlanner` that returns canned plans. Validates plan dispatch,
replan on failure, replan exhaustion, unknown-skill handling.
"""

from __future__ import annotations

from collections import deque
from typing import Any

from robosandbox.agent.agent import Agent
from robosandbox.agent.context import AgentContext
from robosandbox.agent.planner import SkillCall, StubPlanner
from robosandbox.grasp.analytic import AnalyticTopDown
from robosandbox.motion.ik import DLSMotionPlanner
from robosandbox.perception.ground_truth import GroundTruthPerception
from robosandbox.sim.mujoco_backend import MuJoCoBackend
from robosandbox.skills.home import Home
from robosandbox.skills.pick import Pick
from robosandbox.skills.place import PlaceOn
from robosandbox.types import Pose, Scene, SceneObject


class MockPlanner:
    """Returns pre-canned plans; one per call."""

    def __init__(self, plans: list[list[SkillCall]]) -> None:
        self._q: deque[list[SkillCall]] = deque(plans)
        self.calls = 0

    def plan(
        self,
        task: str,
        obs: Any,
        prior_attempts: list[dict[str, Any]],
    ) -> tuple[list[SkillCall], int]:
        self.calls += 1
        if not self._q:
            raise RuntimeError("MockPlanner ran out of canned plans")
        return self._q.popleft(), 1


def _scene() -> Scene:
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


def _ctx() -> tuple[AgentContext, MuJoCoBackend]:
    """Settle long enough to give pick a reliable starting state.
    140 sim steps @ 5ms = 700ms — past the initial physics wobble window.
    """
    sim = MuJoCoBackend(render_size=(240, 320))
    sim.load(_scene())
    for _ in range(140):
        sim.step()
    ctx = AgentContext(
        sim=sim,
        perception=GroundTruthPerception(),
        grasp=AnalyticTopDown(),
        motion=DLSMotionPlanner(n_waypoints=160, dt=0.005),
    )
    return ctx, sim


def test_agent_executes_pick_from_plan() -> None:
    ctx, sim = _ctx()
    try:
        # Pick is probabilistic (~64% on the built-in arm at this settle
        # depth); give the agent up to 4 plan emissions so the replan
        # loop can recover if the first attempt is unlucky.
        planner = MockPlanner([[SkillCall("pick", {"object": "red cube"})]] * 4)
        agent = Agent(ctx, [Pick(), PlaceOn(), Home()], planner, max_replans=3)
        ep = agent.run("pick up the red cube")
    finally:
        sim.close()

    assert ep.success, (ep.final_reason, ep.final_detail)
    assert ep.plan[0] == SkillCall("pick", {"object": "red cube"})


def test_agent_replans_on_skill_failure() -> None:
    ctx, sim = _ctx()
    try:
        planner = MockPlanner(
            [
                [SkillCall("pick", {"object": "nonexistent gadget"})],
                *([[SkillCall("pick", {"object": "red cube"})]] * 4),
            ]
        )
        agent = Agent(ctx, [Pick(), PlaceOn(), Home()], planner, max_replans=4)
        ep = agent.run("pick whatever")
    finally:
        sim.close()

    assert ep.success, (ep.final_reason, ep.final_detail)
    assert ep.replans >= 1
    assert ep.steps[0].result.success is False
    assert ep.steps[0].result.reason == "object_not_found"
    assert any(s.result.success for s in ep.steps[1:])


def test_agent_gives_up_after_max_replans() -> None:
    ctx, sim = _ctx()
    try:
        planner = MockPlanner([[SkillCall("pick", {"object": "ghost"})]] * 5)
        agent = Agent(ctx, [Pick()], planner, max_replans=2)
        ep = agent.run("pick ghost")
    finally:
        sim.close()

    assert ep.success is False
    assert ep.final_reason == "replan_exhausted"
    assert ep.replans == 2


def test_agent_returns_failure_on_empty_plan() -> None:
    ctx, sim = _ctx()
    try:
        planner = MockPlanner([[]])
        agent = Agent(ctx, [Pick()], planner)
        ep = agent.run("do something unrecognized")
    finally:
        sim.close()

    assert ep.success is False
    assert ep.final_reason == "unrecognized_prompt"


def test_agent_rejects_unknown_skill() -> None:
    ctx, sim = _ctx()
    try:
        planner = MockPlanner([[SkillCall("fly", {})]] * 5)
        agent = Agent(ctx, [Pick()], planner, max_replans=2)
        ep = agent.run("fly to the moon")
    finally:
        sim.close()

    assert ep.success is False
    assert ep.steps[0].result.reason == "unknown_skill"


def test_stub_planner_parses_simple_pick() -> None:
    ctx, sim = _ctx()
    try:
        sp = StubPlanner(skills=[Pick(), PlaceOn(), Home()])
        obs = ctx.sim.observe()
        plan, calls = sp.plan("pick up the red cube", obs, [])
    finally:
        sim.close()
    assert calls == 0  # zero model calls — it's rule-based
    assert plan == [SkillCall("pick", {"object": "red_cube"})]


def test_stub_planner_parses_pick_and_place() -> None:
    # Scene with two cubes so 'on the green cube' resolves.
    scene = Scene(
        objects=(
            SceneObject(
                id="red_cube",
                kind="box",
                size=(0.012, 0.012, 0.012),
                pose=Pose(xyz=(0.03, 0.0, 0.07)),
                mass=0.05,
                rgba=(0.85, 0.2, 0.2, 1.0),
            ),
            SceneObject(
                id="green_cube",
                kind="box",
                size=(0.012, 0.012, 0.012),
                pose=Pose(xyz=(0.07, 0.0, 0.07)),
                mass=0.05,
                rgba=(0.2, 0.8, 0.2, 1.0),
            ),
        ),
    )
    sim = MuJoCoBackend(render_size=(240, 320))
    sim.load(scene)
    try:
        for _ in range(50):
            sim.step()
        obs = sim.observe()
        sp = StubPlanner(skills=[Pick(), PlaceOn(), Home()])
        for phrasing in [
            "pick up the red cube and put it on the green cube",
            "pick the red cube, then place it on the green cube",
            "stack red cube on green cube",
        ]:
            plan, _ = sp.plan(phrasing, obs, [])
            assert plan == [
                SkillCall("pick", {"object": "red_cube"}),
                SkillCall("place_on", {"target": "green_cube"}),
            ], f"failed on: {phrasing!r}"
    finally:
        sim.close()


def test_stub_planner_home() -> None:
    ctx, sim = _ctx()
    try:
        obs = ctx.sim.observe()
        sp = StubPlanner(skills=[Pick(), PlaceOn(), Home()])
        plan, _ = sp.plan("go home", obs, [])
        assert plan == [SkillCall("home", {})]
    finally:
        sim.close()


def test_stub_planner_returns_empty_for_unknown_task() -> None:
    ctx, sim = _ctx()
    try:
        obs = ctx.sim.observe()
        sp = StubPlanner(skills=[Pick(), PlaceOn(), Home()])
        plan, _ = sp.plan("dance the macarena", obs, [])
        assert plan == []
    finally:
        sim.close()
