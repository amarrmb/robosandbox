"""Run the built-in benchmark tasks programmatically and print JSON results.

Wraps ``robo-sandbox-bench`` at the library level so you can embed it
into CI or a notebook. Uses the StubPlanner (no VLM calls, no API key).

Run:
    uv run python examples/headless_eval.py
    uv run python examples/headless_eval.py --tasks pick_cube,pick_ycb_mug
"""

from __future__ import annotations

import argparse
import json

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
from robosandbox.skills.push import Push
from robosandbox.tasks.loader import list_builtin_tasks, load_builtin_task


def run_one(name: str, settle_steps: int = 140) -> dict:
    task = load_builtin_task(name)
    sim = MuJoCoBackend(render_size=(240, 320), camera="scene")
    sim.load(task.scene)
    try:
        for _ in range(settle_steps):
            sim.step()
        ctx = AgentContext(
            sim=sim,
            perception=GroundTruthPerception(),
            grasp=AnalyticTopDown(),
            motion=DLSMotionPlanner(n_waypoints=160, dt=0.005),
        )
        skills = [Pick(), PlaceOn(), Push(), Home()]
        agent = Agent(ctx=ctx, skills=skills, planner=StubPlanner(skills))
        ep = agent.run(task.prompt)
        return {
            "task": name,
            "success": ep.success,
            "plan": [s.name for s in ep.plan],
            "steps": len(ep.steps),
            "replans": ep.replans,
            "final_reason": ep.final_reason,
        }
    finally:
        sim.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks", help="comma-separated task names (default: all builtin)")
    args = ap.parse_args()

    names = args.tasks.split(",") if args.tasks else list_builtin_tasks()
    results = [run_one(n) for n in names]
    print(json.dumps({
        "n_tasks": len(results),
        "n_success": sum(1 for r in results if r["success"]),
        "results": results,
    }, indent=2))


if __name__ == "__main__":
    main()
