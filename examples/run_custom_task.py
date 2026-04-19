"""Run a user-authored task YAML through the same agent loop the benchmark uses.

Demonstrates loading a task from an arbitrary path (so your YAML doesn't
have to live inside the package).

Usage:
    uv run python examples/run_custom_task.py examples/tasks/pick_yellow_cube.yaml
    uv run python examples/run_custom_task.py path/to/your_task.yaml --seeds 5
"""
from __future__ import annotations

import argparse
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
from robosandbox.skills.push import Push
from robosandbox.scene.reachability import check_scene_reachability, format_warnings
from robosandbox.tasks.loader import load_task
from robosandbox.tasks.randomize import jitter_scene


def run_once(task, seed: int) -> tuple[bool, float, str]:
    scene = jitter_scene(task.scene, task.randomize, seed)
    sim = MuJoCoBackend(render_size=(360, 480), camera="scene")
    sim.load(scene)
    for _ in range(100):
        sim.step()

    skills = [Pick(), PlaceOn(), Push(), Home()]
    ctx = AgentContext(
        sim=sim,
        perception=GroundTruthPerception(),
        grasp=AnalyticTopDown(),
        motion=DLSMotionPlanner(),
        recorder=LocalRecorder(Path("runs")),
    )
    agent = Agent(ctx, skills=skills, planner=StubPlanner(skills), max_replans=2)
    t0 = time.time()
    result = agent.run(task.prompt, max_steps=10)
    wall = time.time() - t0
    sim.close()
    return bool(result.success), wall, result.final_reason


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("task_yaml", type=Path)
    p.add_argument("--seeds", type=int, default=1)
    args = p.parse_args()

    task = load_task(args.task_yaml)
    print(f"Task: {task.name}")
    print(f"Prompt: {task.prompt}")
    print(f"Success: {task.success.data.get('kind')} (object={task.success.data.get('object')})")
    print(f"Seeds: {args.seeds}")

    warnings = check_scene_reachability(task.scene)
    if warnings:
        print(format_warnings(warnings))
    print()

    wins = 0
    for seed in range(args.seeds):
        ok, wall, reason = run_once(task, seed)
        flag = "OK " if ok else "FAIL"
        print(f"  seed={seed}  {flag}  wall={wall:.1f}s  reason={reason}")
        wins += int(ok)

    print(f"\nSUMMARY: {wins}/{args.seeds} successful")
    return 0 if wins == args.seeds else 1


if __name__ == "__main__":
    raise SystemExit(main())
