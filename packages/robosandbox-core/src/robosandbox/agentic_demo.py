"""`python -m robosandbox.agentic_demo "pick up the red cube"`.

Fires up a 3-cube scene and runs the Agent. Planner backend is
configurable:

- ``--vlm-provider stub``   — rule-based NLU, zero deps, zero setup
- ``--vlm-provider openai`` — OpenAI cloud (needs OPENAI_API_KEY)
- ``--vlm-provider ollama`` — local Ollama (``ollama serve``)
- ``--vlm-provider custom`` — any OpenAI-compatible endpoint via --base-url
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from robosandbox.agent.agent import Agent
from robosandbox.agent.context import AgentContext
from robosandbox.agent.planner import StubPlanner, VLMPlanner
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


PROVIDER_DEFAULTS = {
    "stub": {"base_url": None, "model": None, "api_key_env": None},
    "openai": {
        "base_url": None,
        "model": "gpt-4o-mini",
        "api_key_env": "OPENAI_API_KEY",
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "model": "llama3.2-vision",
        "api_key_env": None,  # ollama ignores auth; we pass a placeholder
    },
    "custom": {
        "base_url": None,  # user supplies
        "model": None,
        "api_key_env": "OPENAI_API_KEY",
    },
}


def build_three_cube_scene() -> Scene:
    """Three cubes inside the arm's comfortable reach envelope.

    Arm base is at (-0.32, 0, 0.04). `red_cube` sits at the center
    (y=0) — the reach sweet spot validated by the pick_cube benchmark
    — since that's the target of the README's headline
    ``robo-sandbox run "pick up the red cube"`` example. Green/blue
    flank it on ±y so perception still has three candidates to
    disambiguate by color.
    """
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
            SceneObject(
                id="green_cube",
                kind="box",
                size=(0.012, 0.012, 0.012),
                pose=Pose(xyz=(0.05, -0.07, 0.07)),
                mass=0.05,
                rgba=(0.2, 0.75, 0.3, 1.0),
            ),
            SceneObject(
                id="blue_cube",
                kind="box",
                size=(0.012, 0.012, 0.012),
                pose=Pose(xyz=(0.05, 0.07, 0.07)),
                mass=0.05,
                rgba=(0.2, 0.45, 0.9, 1.0),
            ),
        ),
    )


def _build_vlm_client(args: argparse.Namespace) -> OpenAIVLMClient:
    """Construct an OpenAIVLMClient honoring provider defaults + overrides."""
    defaults = PROVIDER_DEFAULTS.get(args.vlm_provider, {})
    model = args.model or defaults.get("model") or "gpt-4o-mini"
    base_url = args.base_url or defaults.get("base_url")
    api_key_env = args.api_key_env or defaults.get("api_key_env")

    import os

    # For local / no-auth endpoints, the server doesn't need a key. Inject a
    # placeholder so the OpenAI SDK stops complaining.
    if api_key_env is None or not os.environ.get(api_key_env or ""):
        os.environ.setdefault("_ROBOSANDBOX_PLACEHOLDER_KEY", "ollama-local")
        api_key_env = "_ROBOSANDBOX_PLACEHOLDER_KEY"

    return OpenAIVLMClient(
        VLMConfig(model=model, base_url=base_url, api_key_env=api_key_env)
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("task", nargs="?", default="pick up the red cube")
    p.add_argument(
        "--vlm-provider",
        choices=list(PROVIDER_DEFAULTS),
        default="stub",
        help="Planner backend: stub (zero-dep), openai, ollama, or custom.",
    )
    p.add_argument("--model", default=None, help="Model id (overrides provider default)")
    p.add_argument("--base-url", default=None, help="OpenAI-compatible endpoint URL")
    p.add_argument(
        "--api-key-env",
        default=None,
        help="Env var holding the API key (for providers that need one)",
    )
    p.add_argument(
        "--perception",
        choices=["vlm", "ground_truth"],
        default=None,
        help="Perception backend. Auto-picks ground_truth for stub, vlm otherwise.",
    )
    p.add_argument("--max-replans", type=int, default=3)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    log = logging.getLogger("robosandbox.agentic_demo")

    # ---- assemble planner + perception ---------------------------------
    scene = build_three_cube_scene()
    sim = MuJoCoBackend(render_size=(480, 640), camera="scene")
    sim.load(scene)
    recorder = LocalRecorder(root=Path("runs"), video_fps=30)

    default_perception = "ground_truth" if args.vlm_provider == "stub" else "vlm"
    perception_choice = args.perception or default_perception

    vlm_client: OpenAIVLMClient | None = None
    if args.vlm_provider != "stub":
        try:
            vlm_client = _build_vlm_client(args)
        except VLMTransportError as e:
            log.error("VLM unavailable (%s). Falling back to stub planner.", e)
            args.vlm_provider = "stub"
            perception_choice = "ground_truth"

    skills = [Pick(), PlaceOn(), Home()]
    if args.vlm_provider == "stub":
        planner = StubPlanner(skills=skills)
    else:
        assert vlm_client is not None
        planner = VLMPlanner(vlm=vlm_client, skills=skills)

    if perception_choice == "vlm":
        if vlm_client is None:
            log.warning("VLM perception requested but no VLM client — using ground_truth")
            perception = GroundTruthPerception()
        else:
            perception = VLMPointer(vlm=vlm_client)
    else:
        perception = GroundTruthPerception()

    ctx = AgentContext(
        sim=sim,
        perception=perception,
        grasp=AnalyticTopDown(),
        # n_waypoints default (200) spaces the trajectory so the gripper
        # arrives gently; lower values (e.g. 160) cause the descend to
        # bump the cube out of the grasp. See bench runner which uses the
        # default too.
        motion=DLSMotionPlanner(),
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
            "vlm_provider": args.vlm_provider,
            "planner": type(planner).__name__,
            "perception": type(perception).__name__,
            "scene": {"objects": [{"id": o.id, "xyz": o.pose.xyz} for o in scene.objects]},
        },
    )
    log.info(
        "episode_id=%s planner=%s perception=%s",
        episode_id,
        type(planner).__name__,
        type(perception).__name__,
    )

    # Let the scene settle so observations show cubes at rest.
    for _ in range(100):
        sim.step()
        recorder.write_frame(sim.observe())

    agent = Agent(ctx=ctx, skills=skills, planner=planner, max_replans=args.max_replans)

    t0 = time.time()
    ep = agent.run(args.task)
    elapsed = time.time() - t0

    print()
    print(f"[agentic_demo] task:      {ep.task}")
    print(f"[agentic_demo] planner:   {type(planner).__name__}")
    print(f"[agentic_demo] success:   {ep.success}")
    print(f"[agentic_demo] plan:      {[(c.name, c.arguments) for c in ep.plan]}")
    print(f"[agentic_demo] steps:     {len(ep.steps)}  replans: {ep.replans}  vlm_calls: {ep.vlm_calls}")
    print(f"[agentic_demo] wall:      {elapsed:.1f}s")
    print(f"[agentic_demo] final:     {ep.final_reason} — {ep.final_detail}")

    recorder.end_episode(
        success=ep.success,
        result={
            "planner": type(planner).__name__,
            "plan": [{"skill": c.name, "args": c.arguments} for c in ep.plan],
            "steps": [
                {
                    "skill": s.skill,
                    "args": s.args,
                    "success": s.result.success,
                    "reason": s.result.reason,
                }
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
