"""Author a task in a YAML and run it.

Task YAMLs bundle a Scene + prompt + declarative success criterion. No
executable check code — diffable and versionable. This example writes a
small task, loads it, runs it with the stub planner, and prints the
success-criterion evaluation.

Run:
    uv run python examples/custom_task.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

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
from robosandbox.tasks.loader import load_task


TASK_YAML = """\
name: my_pick_apple
prompt: "pick up the apple"
seed_note: "Demo task authored in examples/custom_task.py."
scene:
  robot_urdf: "@builtin:robots/franka_panda/panda.xml"
  robot_config: "@builtin:robots/franka_panda/panda.robosandbox.yaml"
  objects:
    - id: apple
      kind: mesh
      mesh: "@ycb:013_apple"
      pose:
        xyz: [0.42, 0.0, 0.05]
success:
  kind: lifted
  object: apple
  min_mm: 50
"""


def main() -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as fh:
        fh.write(TASK_YAML)
        yaml_path = Path(fh.name)

    task = load_task(yaml_path)
    print(f"loaded task: {task.name!r} — {task.prompt!r}")
    print(f"scene has {len(task.scene.objects)} objects")

    sim = MuJoCoBackend(render_size=(240, 320))
    sim.load(task.scene)
    try:
        ctx = AgentContext(
            sim=sim,
            perception=GroundTruthPerception(),
            grasp=AnalyticTopDown(),
            motion=DLSMotionPlanner(n_waypoints=160, dt=0.005),
        )
        skills = [Pick(), PlaceOn(), Push(), Home()]
        agent = Agent(ctx=ctx, skills=skills, planner=StubPlanner(skills))
        result = agent.run(task.prompt)
        print(f"\nepisode success: {result.success}")
        print(f"  plan: {[s.name for s in result.plan]}")
        print(f"  steps: {len(result.steps)}")
        for s in result.steps:
            print(f"    {s.skill}({s.args}) -> {s.result}")
    finally:
        sim.close()


if __name__ == "__main__":
    main()
