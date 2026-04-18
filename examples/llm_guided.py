"""Run a task with the VLMPlanner (OpenAI-compatible endpoint).

Set ``OPENAI_API_KEY`` to use the real path, or export
``ROBOSANDBOX_VLM_BASE_URL=http://localhost:11434/v1`` to point at Ollama
(``ollama serve && ollama pull llama3.2-vision``). Without a key, this
script exits early with a clear message rather than failing.

Run:
    OPENAI_API_KEY=sk-... uv run python examples/llm_guided.py "pick the apple"
"""

from __future__ import annotations

import argparse
import os
import sys
from importlib.resources import files
from pathlib import Path

from robosandbox.agent.agent import Agent
from robosandbox.agent.context import AgentContext
from robosandbox.agent.planner import VLMPlanner
from robosandbox.grasp.analytic import AnalyticTopDown
from robosandbox.motion.ik import DLSMotionPlanner
from robosandbox.perception.ground_truth import GroundTruthPerception
from robosandbox.sim.mujoco_backend import MuJoCoBackend
from robosandbox.skills.home import Home
from robosandbox.skills.pick import Pick
from robosandbox.skills.place import PlaceOn
from robosandbox.skills.push import Push
from robosandbox.types import Pose, Scene, SceneObject
from robosandbox.vlm.client import OpenAIVLMClient, VLMConfig


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("prompt", nargs="?", default="pick up the apple")
    ap.add_argument("--model", default=os.environ.get("ROBOSANDBOX_VLM_MODEL", "gpt-4o-mini"))
    ap.add_argument(
        "--base-url",
        default=os.environ.get("ROBOSANDBOX_VLM_BASE_URL", "https://api.openai.com/v1"),
    )
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY") and "api.openai.com" in args.base_url:
        print(
            "OPENAI_API_KEY not set and --base-url points at OpenAI. Set the key or "
            "override --base-url to a local/Ollama endpoint.",
            file=sys.stderr,
        )
        sys.exit(0)

    urdf = Path(str(files("robosandbox").joinpath("assets/robots/franka_panda/panda.xml")))
    cfg = Path(
        str(files("robosandbox").joinpath("assets/robots/franka_panda/panda.robosandbox.yaml"))
    )
    sidecar = Path(
        str(files("robosandbox").joinpath("assets/objects/ycb/013_apple/apple.robosandbox.yaml"))
    )
    scene = Scene(
        robot_urdf=urdf,
        robot_config=cfg,
        objects=(
            SceneObject(
                id="apple",
                kind="mesh",
                size=(0.0,),
                pose=Pose(xyz=(0.42, 0.0, 0.05)),
                mass=0.0,
                mesh_sidecar=sidecar,
            ),
        ),
    )

    vlm = OpenAIVLMClient(
        VLMConfig(
            model=args.model,
            base_url=args.base_url,
            api_key=os.environ.get("OPENAI_API_KEY", "ollama"),
        )
    )
    planner = VLMPlanner(vlm, skills=[Pick(), PlaceOn(), Push(), Home()])

    sim = MuJoCoBackend(render_size=(240, 320))
    sim.load(scene)
    try:
        ctx = AgentContext(
            sim=sim,
            perception=GroundTruthPerception(),
            grasp=AnalyticTopDown(),
            motion=DLSMotionPlanner(n_waypoints=160, dt=0.005),
        )
        agent = Agent(
            ctx=ctx,
            skills=[Pick(), PlaceOn(), Push(), Home()],
            planner=planner,
        )
        result = agent.run(args.prompt)
        print(f"success={result.success}  plan={[s.name for s in result.plan]}  "
              f"vlm_calls={result.vlm_calls}  final_reason={result.final_reason!r}")
    finally:
        sim.close()


if __name__ == "__main__":
    main()
