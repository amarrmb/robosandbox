# Tutorial: custom arm

Drop any URDF (or MJCF) into a `Scene`, write a short sidecar YAML,
pick a cube. No core changes.

## What you need

- A URDF / MJCF file for your robot, with its own meshes / actuators.
- Knowledge of: one joint list, the gripper primary joint, where the
  end-effector TCP sits, the home pose, and the base pose in the
  world.

## 1. Drop the URDF in

```python
from pathlib import Path
from robosandbox.types import Scene

scene = Scene(
    robot_urdf=Path("/abs/path/to/my_arm.urdf"),   # or .xml
    robot_config=Path("/abs/path/to/my_arm.robosandbox.yaml"),
    objects=(...),
)
```

If `robot_config` is omitted, RoboSandbox auto-discovers a sibling
file `<stem>.robosandbox.yaml` next to the URDF.

## 2. Write the sidecar

The sidecar maps RoboSandbox roles (arm, gripper, ee_site, base) onto
your robot's element names.

Reference schema — the bundled Franka Panda's
`panda.robosandbox.yaml`:

```yaml
# RoboSandbox role-to-element map for the bundled Franka Panda.
# Consumed by scene/robot_loader.py alongside panda.xml.

arm:
  joints: [joint1, joint2, joint3, joint4, joint5, joint6, joint7]
  actuators: [actuator1, actuator2, actuator3, actuator4, actuator5, actuator6, actuator7]
  # Home pose — arm's preferred neutral. Must be reachable and not in
  # self-collision. Menagerie's keyframe for Franka, minus gripper.
  home_qpos: [0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853]

gripper:
  joints: [finger_joint1, finger_joint2]
  primary_joint: finger_joint1     # the one skills set open/closed qpos on
  actuator: actuator8
  open_qpos: 0.04                  # ~80 mm between fingertips
  closed_qpos: 0.0

ee_site:
  inject:
    attach_body: hand              # name of the body the TCP rides on
    # Offset of the tool center point in the attached body's frame.
    # For Franka: 0.1034 m down the z-axis of the hand flange.
    xyz: [0.0, 0.0, 0.1034]

base_pose:
  xyz: [-0.3, 0.0, 0.0]            # robot base in world frame
```

### Field-by-field

- **`arm.joints`** — ordered list of arm DoF joints.
- **`arm.actuators`** — one actuator per joint, same order. Skills
  write to these.
- **`arm.home_qpos`** — `len(arm.joints)` floats. Must be reachable
  and well-conditioned.
- **`gripper.primary_joint`** — the joint the motion planner reads to
  expose `gripper_width`. For parallel grippers with mimic joints,
  the primary is the one with the actuator.
- **`gripper.open_qpos` / `closed_qpos`** — joint values, not widths.
  If your gripper reports width in mm, pick the qpos equivalents.
- **`ee_site.inject.attach_body`** — name of the MJCF body the
  end-effector site should be welded to. Usually the last link
  before the gripper.
- **`ee_site.inject.xyz`** — offset from that body's frame to the TCP
  (point between closed fingertips).
- **`base_pose.xyz`** — where the base sits in world frame. Tune so
  the workspace overlaps the default table area `(-0.5..0.5) x
  (-0.5..0.5)` at `z ≈ 0.04 m`.

## 3. Pick a cube with it

Here's a minimal end-to-end run — loads the robot, spawns a cube in
reach, runs `Pick` with the stub planner:

```python
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
from robosandbox.types import Pose, Scene, SceneObject


def main():
    scene = Scene(
        robot_urdf=Path("/abs/path/to/my_arm.urdf"),
        robot_config=Path("/abs/path/to/my_arm.robosandbox.yaml"),
        objects=(
            SceneObject(
                id="red_cube", kind="box",
                size=(0.012, 0.012, 0.012),
                pose=Pose(xyz=(0.4, 0.0, 0.06)),
                rgba=(0.85, 0.2, 0.2, 1.0), mass=0.05,
            ),
        ),
    )

    sim = MuJoCoBackend(render_size=(240, 320))
    sim.load(scene)
    try:
        ctx = AgentContext(
            sim=sim,
            perception=GroundTruthPerception(),
            grasp=AnalyticTopDown(),
            motion=DLSMotionPlanner(n_waypoints=160, dt=0.005),
        )
        skills = [Pick(), Home()]
        agent = Agent(ctx=ctx, skills=skills, planner=StubPlanner(skills))
        ep = agent.run("pick up the red cube")
        print(f"success={ep.success}  steps={len(ep.steps)}  reason={ep.final_reason}")
    finally:
        sim.close()


if __name__ == "__main__":
    main()
```

Expected output on success:

```
success=True  steps=1  reason=plan_complete
```

## 4. Run from a YAML task

Wrap it in a task YAML to plug into the benchmark runner:

```yaml
name: pick_cube_myarm
prompt: "pick up the red cube"
scene:
  robot_urdf: "/abs/path/to/my_arm.urdf"
  robot_config: "/abs/path/to/my_arm.robosandbox.yaml"
  objects:
    - id: red_cube
      kind: box
      size: [0.012, 0.012, 0.012]
      pose:
        xyz: [0.4, 0.0, 0.06]
      rgba: [0.85, 0.2, 0.2, 1.0]
      mass: 0.05
success:
  kind: lifted
  object: red_cube
  min_mm: 50
```

Load + run:

```python
from robosandbox.tasks.loader import load_task
task = load_task(Path("pick_cube_myarm.yaml"))
# ... same Agent setup as above, agent.run(task.prompt)
```

Drop the file under
`packages/robosandbox-core/src/robosandbox/tasks/definitions/` to
include it in `robo-sandbox-bench`.

## Debugging tips

- **Arm won't reach the cube.** Tune `base_pose.xyz` so the
  workspace overlaps the table (default `z ≈ 0.04 m`). Or move the
  cube closer.
- **Gripper can't close.** Check that `gripper.primary_joint` is the
  one with the actuator, and that `open_qpos`/`closed_qpos` are the
  right sign. Some grippers move in `-qpos` to close.
- **`UnreachableError` on every pick.** TCP probably sits off — the
  motion planner is computing IK for the wrong point. Re-check
  `ee_site.inject.xyz`.
- **Self-collision on home.** Tweak `arm.home_qpos` — the viewer
  (`robo-sandbox viewer --task pick_cube_franka`) is the fastest way
  to eyeball it.

## See also

- `examples/custom_robot.py` — runnable reference.
- [Scenes & objects](../concepts/scenes.md) for the full `Scene` API.
- [Perception & grasping](../concepts/perception-and-grasping.md) for
  why TCP accuracy matters for `AnalyticTopDown`.
