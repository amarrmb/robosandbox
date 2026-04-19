"""Trace the replan loop on a deliberately unreachable task.

Prints PLAN/EXECUTE log lines per iteration so you can watch the
ReAct feedback loop fail, feed prior_attempts back, and fail again
until max_replans is hit.

Usage:
    uv run python examples/replan_demo.py
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

from robosandbox.agent.agent import Agent
from robosandbox.agent.context import AgentContext
from robosandbox.agent.planner import StubPlanner
from robosandbox.grasp.analytic import AnalyticTopDown
from robosandbox.motion.ik import DLSMotionPlanner
from robosandbox.perception.ground_truth import GroundTruthPerception
from robosandbox.recorder.local import LocalRecorder
from robosandbox.sim.mujoco_backend import MuJoCoBackend
from robosandbox.skills.home import Home
from robosandbox.skills.pick import Pick
from robosandbox.skills.place import PlaceOn
from robosandbox.tasks.loader import load_task


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-20s  %(message)s",
        datefmt="%H:%M:%S",
    )

    task = load_task(Path("examples/tasks/pick_unreachable_cube.yaml"))
    sim = MuJoCoBackend(render_size=(360, 480), camera="scene")
    sim.load(task.scene)
    for _ in range(50):
        sim.step()

    skills = [Pick(), PlaceOn(), Home()]
    ctx = AgentContext(
        sim=sim,
        perception=GroundTruthPerception(),
        grasp=AnalyticTopDown(),
        motion=DLSMotionPlanner(),
        recorder=LocalRecorder(Path("runs")),
    )
    # max_replans=2 gives PLAN → EXECUTE (fail) → PLAN → EXECUTE (fail) → PLAN → EXECUTE (fail)
    agent = Agent(ctx, skills=skills, planner=StubPlanner(skills), max_replans=2)

    t0 = time.time()
    result = agent.run(task.prompt, max_steps=10)
    wall = time.time() - t0

    print()
    print("─" * 66)
    print(f"FINAL VERDICT")
    print("─" * 66)
    print(f"  success:     {result.success}")
    print(f"  replans:     {result.replans}")
    print(f"  steps:       {len(result.steps)}")
    print(f"  final reason: {result.final_reason}")
    print(f"  detail:      {result.final_detail}")
    print(f"  wall:        {wall:.1f}s")
    sim.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
