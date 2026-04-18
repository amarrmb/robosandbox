"""Agent-loop tests with a stub VLM — no network.

Validates: plan parsing, skill dispatch, failure → replan, replan exhaustion.
"""

from __future__ import annotations

import json
from collections import deque
from typing import Any

import pytest

from robosandbox.agent.agent import Agent
from robosandbox.agent.context import AgentContext
from robosandbox.grasp.analytic import AnalyticTopDown
from robosandbox.motion.ik import DLSMotionPlanner
from robosandbox.perception.ground_truth import GroundTruthPerception
from robosandbox.sim.mujoco_backend import MuJoCoBackend
from robosandbox.skills.home import Home
from robosandbox.skills.pick import Pick
from robosandbox.skills.place import PlaceOn
from robosandbox.types import Pose, Scene, SceneObject, SkillResult


class StubVLM:
    """Returns a queue of canned responses, one per chat() call."""

    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self._q: deque[dict[str, Any]] = deque(responses)
        self.calls: list[dict[str, Any]] = []

    def chat(self, messages, tools=None, tool_choice=None, **kw) -> dict[str, Any]:
        self.calls.append({"messages": messages, "tools": tools})
        if not self._q:
            raise RuntimeError("StubVLM ran out of canned responses")
        return self._q.popleft()


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
    sim = MuJoCoBackend(render_size=(240, 320))
    sim.load(_scene())
    for _ in range(100):
        sim.step()
    ctx = AgentContext(
        sim=sim,
        perception=GroundTruthPerception(),
        grasp=AnalyticTopDown(),
        motion=DLSMotionPlanner(n_waypoints=160, dt=0.005),
    )
    return ctx, sim


def _tool_call(name: str, args: dict[str, Any], id_: str = "tc_1") -> dict[str, Any]:
    return {"id": id_, "name": name, "arguments": json.dumps(args)}


def test_agent_executes_pick_from_vlm_plan() -> None:
    ctx, sim = _ctx()
    try:
        vlm = StubVLM(
            [
                {
                    "content": None,
                    "tool_calls": [_tool_call("pick", {"object": "red cube"})],
                    "finish_reason": "tool_calls",
                    "raw": None,
                }
            ]
        )
        agent = Agent(ctx, [Pick(), PlaceOn(), Home()], vlm)
        ep = agent.run("pick up the red cube")
    finally:
        sim.close()

    assert ep.success, (ep.final_reason, ep.final_detail)
    assert len(ep.plan) == 1
    assert ep.plan[0].name == "pick"
    assert ep.plan[0].arguments == {"object": "red cube"}
    assert ep.replans == 0
    assert ep.vlm_calls == 1


def test_agent_replans_on_skill_failure() -> None:
    ctx, sim = _ctx()
    try:
        # First plan tries a non-existent object, fails; second plan uses the right name.
        vlm = StubVLM(
            [
                {
                    "content": None,
                    "tool_calls": [_tool_call("pick", {"object": "nonexistent gadget"})],
                    "finish_reason": "tool_calls",
                    "raw": None,
                },
                {
                    "content": None,
                    "tool_calls": [_tool_call("pick", {"object": "red cube"}, "tc_2")],
                    "finish_reason": "tool_calls",
                    "raw": None,
                },
            ]
        )
        agent = Agent(ctx, [Pick(), PlaceOn(), Home()], vlm, max_replans=2)
        ep = agent.run("pick whatever")
    finally:
        sim.close()

    assert ep.success
    assert ep.replans == 1
    assert ep.vlm_calls == 2
    # First attempt failed with object_not_found; second succeeded.
    assert ep.steps[0].result.success is False
    assert ep.steps[0].result.reason == "object_not_found"
    assert ep.steps[1].result.success is True


def test_agent_gives_up_after_max_replans() -> None:
    ctx, sim = _ctx()
    try:
        vlm = StubVLM(
            [
                {
                    "content": None,
                    "tool_calls": [_tool_call("pick", {"object": "ghost"})],
                    "finish_reason": "tool_calls",
                    "raw": None,
                }
            ]
            * 5
        )
        agent = Agent(ctx, [Pick()], vlm, max_replans=2)
        ep = agent.run("pick ghost")
    finally:
        sim.close()

    assert ep.success is False
    assert ep.final_reason == "replan_exhausted"
    assert ep.replans == 2


def test_agent_respects_done_tool() -> None:
    ctx, sim = _ctx()
    try:
        vlm = StubVLM(
            [
                {
                    "content": None,
                    "tool_calls": [{"id": "x", "name": "done", "arguments": "{}"}],
                    "finish_reason": "tool_calls",
                    "raw": None,
                }
            ]
        )
        agent = Agent(ctx, [Pick()], vlm)
        ep = agent.run("already done")
    finally:
        sim.close()

    assert ep.success is True
    assert ep.final_reason == "already_done"


def test_agent_rejects_unknown_skill() -> None:
    ctx, sim = _ctx()
    try:
        vlm = StubVLM(
            [
                {
                    "content": None,
                    "tool_calls": [_tool_call("fly", {})],
                    "finish_reason": "tool_calls",
                    "raw": None,
                }
            ]
            * 5
        )
        agent = Agent(ctx, [Pick()], vlm, max_replans=2)
        ep = agent.run("fly to the moon")
    finally:
        sim.close()

    assert ep.success is False
    assert ep.steps[0].result.reason == "unknown_skill"
