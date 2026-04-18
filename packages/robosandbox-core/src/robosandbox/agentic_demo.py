"""`python -m robosandbox.agentic_demo "pick up the red cube"`.

Fires up a 3-cube scene and runs the VLM-driven Agent. Requires an
OpenAI-compatible endpoint; set ``OPENAI_API_KEY`` (or ``--base-url``
+ ``--api-key-env`` for a local vLLM / Ollama).

Falls back gracefully if no key is set — prints a helpful message.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

from robosandbox.agent.agent import Agent
from robosandbox.agent.context import AgentContext
from robosandbox.grasp.analytic import AnalyticTopDown
from robosandbox.motion.ik import DLSMotionPlanner
from robosandbox.perception.ground_truth import GroundTruthPerception
from robosandbox.perception.vlm_pointer import VLMPointer
from robosandbox.recorder.local import LocalRecorder
from robosandbox.sim.mujoco_backend import MuJoCoBackend
from robosandbox.skills.home import Home
from robosandbox.skills.pick import Pick
from robosandbox.skills.place import PlaceOn
from robosandbox.types import Pose, Scene, SceneObject
from robosandbox.vlm.client import OpenAIVLMClient, VLMConfig, VLMTransportError


def build_three_cube_scene() -> Scene:
    """A red / green / blue cube on the table, plus a 'plate' region for placement."""
    return Scene(
        objects=(
            SceneObject(
                id="red_cube",
                kind="box",
                size=(0.012, 0.012, 0.012),
                pose=Pose(xyz=(0.03, -0.05, 0.07)),
                mass=0.05,
                rgba=(0.85, 0.2, 0.2, 1.0),
            ),
            SceneObject(
                id="green_cube",
                kind="box",
                size=(0.012, 0.012, 0.012),
                pose=Pose(xyz=(0.07, 0.0, 0.07)),
                mass=0.05,
                rgba=(0.2, 0.75, 0.3, 1.0),
            ),
            SceneObject(
                id="blue_cube",
                kind="box",
                size=(0.012, 0.012, 0.012),
                pose=Pose(xyz=(0.05, 0.06, 0.07)),
                mass=0.05,
                rgba=(0.2, 0.45, 0.9, 1.0),
            ),
        ),
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("task", nargs="?", default="pick up the red cube", help="Natural-language task")
    p.add_argument("--model", default="gpt-4o-mini", help="VLM model id")
    p.add_argument("--base-url", default=None, help="OpenAI-compatible endpoint URL")
    p.add_argument("--api-key-env", default="OPENAI_API_KEY", help="Env var holding the API key")
    p.add_argument(
        "--perception",
        choices=["vlm", "ground_truth"],
        default="vlm",
        help="Perception backend. 'vlm' needs an API key; 'ground_truth' cheats.",
    )
    p.add_argument(
        "--max-replans",
        type=int,
        default=3,
        help="How many times the agent re-plans after a skill failure",
    )
    p.add_argument("--log-level", default="INFO", help="Python logging level")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    log = logging.getLogger("robosandbox.agentic_demo")

    # ---- wire the VLM (may fail cleanly with no key) -------------------
    vlm: OpenAIVLMClient | None = None
    try:
        vlm = OpenAIVLMClient(
            VLMConfig(
                model=args.model,
                base_url=args.base_url,
                api_key_env=args.api_key_env,
            )
        )
    except VLMTransportError as e:
        print(f"[agentic_demo] VLM unavailable: {e}", file=sys.stderr)
        if args.perception == "vlm":
            print(
                "[agentic_demo] Hint: export OPENAI_API_KEY=... "
                "or use --perception ground_truth for an offline run",
                file=sys.stderr,
            )
            return 2

    # ---- build the sim + skills + recorder -----------------------------
    scene = build_three_cube_scene()
    sim = MuJoCoBackend(render_size=(480, 640), camera="scene")
    sim.load(scene)

    recorder = LocalRecorder(root=Path("runs"), video_fps=30)

    if args.perception == "vlm":
        perception = VLMPointer(vlm=vlm)
    else:
        perception = GroundTruthPerception()

    ctx = AgentContext(
        sim=sim,
        perception=perception,
        grasp=AnalyticTopDown(),
        motion=DLSMotionPlanner(n_waypoints=160, dt=0.005),
        recorder=recorder,
    )

    def _on_step() -> None:
        recorder.write_frame(sim.observe())

    ctx.on_step = _on_step

    episode_id = recorder.start_episode(
        task=args.task,
        metadata={
            "source": "robosandbox.agentic_demo",
            "source_version": "0.1.0",
            "sim_dt": 0.005,
            "vlm_model": args.model,
            "perception": args.perception,
            "scene": {
                "objects": [{"id": o.id, "xyz": o.pose.xyz} for o in scene.objects]
            },
        },
    )
    log.info("episode_id=%s", episode_id)

    # Settle the scene so the VLM sees cubes on the table, not falling.
    for _ in range(100):
        sim.step()
        recorder.write_frame(sim.observe())

    # If we have no VLM, fall back to a direct skill invocation so the
    # demo still ends with a video and a verdict.
    if vlm is None:
        log.warning("no VLM available — running a scripted Pick(red_cube) fallback")
        result = Pick()(ctx, object="red_cube")
        recorder.end_episode(
            success=result.success,
            result={"reason": result.reason, "reason_detail": result.reason_detail},
        )
        sim.close()
        return 0 if result.success else 1

    agent = Agent(
        ctx=ctx,
        skills=[Pick(), PlaceOn(), Home()],
        vlm=vlm,
        max_replans=args.max_replans,
    )

    t0 = time.time()
    ep = agent.run(args.task)
    elapsed = time.time() - t0

    print()
    print(f"[agentic_demo] task:     {ep.task}")
    print(f"[agentic_demo] success:  {ep.success}")
    print(f"[agentic_demo] plan:     {[(c.name, c.arguments) for c in ep.plan]}")
    print(f"[agentic_demo] steps:    {len(ep.steps)}  replans: {ep.replans}  vlm_calls: {ep.vlm_calls}")
    print(f"[agentic_demo] wall:     {elapsed:.1f}s")
    print(f"[agentic_demo] final:    {ep.final_reason} — {ep.final_detail}")

    recorder.end_episode(
        success=ep.success,
        result={
            "plan": [{"skill": c.name, "args": c.arguments} for c in ep.plan],
            "steps": [
                {"skill": s.skill, "args": s.args, "success": s.result.success, "reason": s.result.reason}
                for s in ep.steps
            ],
            "replans": ep.replans,
            "vlm_calls": ep.vlm_calls,
            "final_reason": ep.final_reason,
            "final_detail": ep.final_detail,
            "wall_seconds": elapsed,
        },
    )
    sim.close()
    return 0 if ep.success else 1


if __name__ == "__main__":
    sys.exit(main())
