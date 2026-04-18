"""Benchmark runner: execute the 5-task starter suite and report results.

Runs each task N seeds (default 1) with the StubPlanner + ground-truth
perception by default — isolates sandbox reliability from VLM variance.
Use ``--vlm-provider`` to switch to a real model.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from robosandbox.agent.agent import Agent
from robosandbox.agent.context import AgentContext
from robosandbox.agent.planner import StubPlanner, VLMPlanner
from robosandbox.grasp.analytic import AnalyticTopDown
from robosandbox.motion.ik import DLSMotionPlanner
from robosandbox.perception.ground_truth import GroundTruthPerception
from robosandbox.sim.mujoco_backend import MuJoCoBackend
from robosandbox.skills.home import Home
from robosandbox.skills.pick import Pick
from robosandbox.skills.place import PlaceOn
from robosandbox.skills.push import Push
from robosandbox.tasks.loader import (
    SuccessCriterion,
    Task,
    list_builtin_tasks,
    load_builtin_task,
)
from robosandbox.types import Observation
from robosandbox.vlm.client import OpenAIVLMClient, VLMConfig, VLMTransportError

log = logging.getLogger("robosandbox.bench")


@dataclass
class TaskResult:
    name: str
    prompt: str
    success: bool
    wall_seconds: float
    replans: int
    vlm_calls: int
    plan: list[dict[str, Any]]
    final_reason: str
    success_detail: dict[str, Any]


# ---------- success evaluation -----------------------------------------

def _eval_criterion(c: SuccessCriterion, initial: Observation, final: Observation) -> tuple[bool, dict]:
    return _eval_check(c.data, initial, final)


def _eval_check(check: dict, initial: Observation, final: Observation) -> tuple[bool, dict]:
    kind = check.get("kind")
    if kind == "lifted":
        oid = check["object"]
        min_mm = float(check.get("min_mm", 50.0))
        z0 = initial.scene_objects.get(oid, None)
        zf = final.scene_objects.get(oid, None)
        if z0 is None or zf is None:
            return False, {"reason": "object missing"}
        dz_mm = (zf.xyz[2] - z0.xyz[2]) * 1000
        return dz_mm >= min_mm, {"dz_mm": dz_mm, "min_mm": min_mm}
    if kind == "moved_above":
        oid = check["object"]
        tid = check["target"]
        xy_tol = float(check.get("xy_tol", 0.03))
        min_dz = float(check.get("min_dz", 0.01))
        o = final.scene_objects.get(oid)
        t = final.scene_objects.get(tid)
        if o is None or t is None:
            return False, {"reason": "object or target missing"}
        xy = float(np.linalg.norm(np.array(o.xyz[:2]) - np.array(t.xyz[:2])))
        dz = o.xyz[2] - t.xyz[2]
        return (xy <= xy_tol and dz >= min_dz), {"xy": xy, "dz": dz, "xy_tol": xy_tol, "min_dz": min_dz}
    if kind == "displaced":
        oid = check["object"]
        direction = str(check["direction"]).lower()
        min_mm = float(check.get("min_mm", 30.0))
        vec_map = {
            "forward": (1.0, 0.0), "back": (-1.0, 0.0), "backward": (-1.0, 0.0),
            "left": (0.0, -1.0), "right": (0.0, 1.0),
        }
        dx, dy = vec_map.get(direction, (0.0, 0.0))
        o0 = initial.scene_objects.get(oid)
        of = final.scene_objects.get(oid)
        if o0 is None or of is None:
            return False, {"reason": "object missing"}
        displacement_mm = float(np.dot(
            [(of.xyz[0] - o0.xyz[0]) * 1000, (of.xyz[1] - o0.xyz[1]) * 1000],
            [dx, dy],
        ))
        return displacement_mm >= min_mm, {"displacement_mm": displacement_mm, "min_mm": min_mm}
    if kind == "all":
        results = [_eval_check(sub, initial, final) for sub in check.get("checks", [])]
        ok = all(r[0] for r in results) if results else True  # empty "all" is trivially true
        return ok, {"sub": [r[1] for r in results]}
    if kind == "any":
        results = [_eval_check(sub, initial, final) for sub in check.get("checks", [])]
        ok = any(r[0] for r in results) if results else False
        return ok, {"sub": [r[1] for r in results]}
    return False, {"reason": f"unknown criterion kind {kind!r}"}


# ---------- task runner ------------------------------------------------

def _run_one(
    task: Task,
    *,
    vlm_provider: str,
    vlm_client: OpenAIVLMClient | None,
    max_replans: int,
    settle_steps: int,
    seed: int = 0,
) -> TaskResult:
    from robosandbox.tasks.randomize import jitter_scene

    scene = jitter_scene(task.scene, task.randomize, seed)
    sim = MuJoCoBackend(render_size=(240, 320), camera="scene")
    sim.load(scene)
    try:
        for _ in range(settle_steps):
            sim.step()
        initial = sim.observe()

        perception = GroundTruthPerception()
        skills = [Pick(), PlaceOn(), Push(), Home()]
        if vlm_provider == "stub":
            planner = StubPlanner(skills=skills)
        else:
            assert vlm_client is not None
            planner = VLMPlanner(vlm=vlm_client, skills=skills)

        ctx = AgentContext(
            sim=sim,
            perception=perception,
            grasp=AnalyticTopDown(),
            motion=DLSMotionPlanner(n_waypoints=160, dt=0.005),
        )
        agent = Agent(ctx=ctx, skills=skills, planner=planner, max_replans=max_replans)

        t0 = time.time()
        ep = agent.run(task.prompt)
        elapsed = time.time() - t0

        final = sim.observe()
        success_ok, success_detail = _eval_criterion(task.success, initial, final)

        return TaskResult(
            name=task.name,
            prompt=task.prompt,
            success=success_ok,
            wall_seconds=elapsed,
            replans=ep.replans,
            vlm_calls=ep.vlm_calls,
            plan=[{"skill": c.name, "args": c.arguments} for c in ep.plan],
            final_reason=ep.final_reason,
            success_detail=success_detail,
        )
    finally:
        sim.close()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Run the built-in RoboSandbox task benchmark"
    )
    p.add_argument("--tasks", nargs="*", default=None, help="Task names (default: all)")
    p.add_argument("--seeds", type=int, default=1, help="Repetitions per task")
    p.add_argument(
        "--vlm-provider",
        choices=["stub", "openai", "ollama", "custom"],
        default="stub",
    )
    p.add_argument("--model", default=None)
    p.add_argument("--base-url", default=None)
    p.add_argument("--api-key-env", default=None)
    p.add_argument("--max-replans", type=int, default=3)
    p.add_argument("--settle-steps", type=int, default=140)
    p.add_argument(
        "--out",
        default=None,
        help="Write results as JSON to this path (default: benchmark_results.json)",
    )
    p.add_argument("--log-level", default="WARNING")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    task_names = args.tasks or list_builtin_tasks()

    # VLM setup.
    vlm_client: OpenAIVLMClient | None = None
    if args.vlm_provider != "stub":
        from robosandbox.agentic_demo import PROVIDER_DEFAULTS

        defaults = PROVIDER_DEFAULTS.get(args.vlm_provider, {})
        model = args.model or defaults.get("model") or "gpt-4o-mini"
        base_url = args.base_url or defaults.get("base_url")
        api_key_env = args.api_key_env or defaults.get("api_key_env")
        import os
        if api_key_env is None or not os.environ.get(api_key_env or ""):
            os.environ.setdefault("_ROBOSANDBOX_PLACEHOLDER_KEY", "ollama-local")
            api_key_env = "_ROBOSANDBOX_PLACEHOLDER_KEY"
        try:
            vlm_client = OpenAIVLMClient(
                VLMConfig(model=model, base_url=base_url, api_key_env=api_key_env)
            )
        except VLMTransportError as e:
            print(f"[bench] VLM unavailable: {e}", file=sys.stderr)
            return 2

    all_results: list[dict[str, Any]] = []
    print(f"{'TASK':<18} {'SEED':<5} {'RESULT':<6} {'SECS':>6} {'REPLANS':>8} {'DETAIL'}")
    print("-" * 90)
    for name in task_names:
        try:
            task = load_builtin_task(name)
        except FileNotFoundError as e:
            print(f"[bench] skipping unknown task {name!r}: {e}", file=sys.stderr)
            continue
        for seed in range(args.seeds):
            r = _run_one(
                task,
                vlm_provider=args.vlm_provider,
                vlm_client=vlm_client,
                max_replans=args.max_replans,
                settle_steps=args.settle_steps,
                seed=seed,
            )
            detail_str = ", ".join(
                f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in r.success_detail.items()
                if k not in ("sub",)
            )
            print(
                f"{r.name:<18} {seed:<5} {'OK' if r.success else 'FAIL':<6} "
                f"{r.wall_seconds:>6.1f} {r.replans:>8} {detail_str}"
            )
            all_results.append(
                {
                    "task": r.name,
                    "seed": seed,
                    "prompt": r.prompt,
                    "success": r.success,
                    "wall_seconds": r.wall_seconds,
                    "replans": r.replans,
                    "vlm_calls": r.vlm_calls,
                    "plan": r.plan,
                    "final_reason": r.final_reason,
                    "success_detail": r.success_detail,
                }
            )

    print()
    # Aggregate.
    by_task: dict[str, list[bool]] = {}
    for r in all_results:
        by_task.setdefault(r["task"], []).append(r["success"])
    overall_ok = sum(1 for r in all_results if r["success"])
    print(f"SUMMARY: {overall_ok}/{len(all_results)} successful")
    for task_name, runs in by_task.items():
        ok = sum(1 for r in runs if r)
        n = len(runs)
        mean = ok / n
        # Bernoulli stderr: sqrt(p*(1-p)/n). Shown only when n>1.
        if n > 1:
            stderr = (mean * (1.0 - mean) / n) ** 0.5
            print(f"  {task_name:<20} {ok}/{n}  mean={mean:.2f} ± {stderr:.2f}")
        else:
            print(f"  {task_name:<20} {ok}/{n}")

    out_path = Path(args.out or "benchmark_results.json")
    out_path.write_text(json.dumps({"runs": all_results}, indent=2))
    print(f"\nResults written to {out_path}")

    return 0 if overall_ok == len(all_results) else 1


if __name__ == "__main__":
    sys.exit(main())
