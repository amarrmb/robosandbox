# Real-robot bridge

Swapping from sim to real is a constructor change. The
`SimBackend` Protocol is the seam.

## `SimBackend` Protocol

```python
class SimBackend(Protocol):
    def load(self, scene: Scene) -> None: ...
    def reset(self) -> None: ...
    def step(self, target_joints=None, gripper=None) -> None: ...
    def observe(self) -> Observation: ...
    def get_object_pose(self, object_id: str) -> Pose | None: ...
    def set_object_pose(self, object_id: str, pose: Pose) -> None: ...
    @property
    def n_dof(self) -> int: ...
    @property
    def joint_names(self) -> list[str]: ...
    def close(self) -> None: ...
```

Every skill, the motion planner, the grasp planner, and the Agent
loop go through this interface. They don't care whether it's MuJoCo
or a real robot on the other end.

## The swap pattern

```python
# sim
from robosandbox.sim.mujoco_backend import MuJoCoBackend
sim = MuJoCoBackend(render_size=(240, 320))
sim.load(scene)

# real
from my_driver import MySO101Backend
sim = MySO101Backend("/dev/ttyACM0")
sim.load(scene)

# Observation+step skills (Home, teleop, LeRobotPolicyAdapter-wrapped
# policies via run_policy) then run against MySO101Backend through the
# same SimBackend Protocol. Motion-planning skills (Pick, PlaceOn, Push)
# are the exception — they read sim.model / sim.data, which a real
# backend doesn't expose. See tutorials/sim-to-real-handoff.md for the
# "plan in sim, execute on real" pattern.
```

## `RealRobotBackend` stub

RoboSandbox ships
[`robosandbox.backends.real.RealRobotBackend`](../reference/api.md#backends)
— a stub that satisfies `SimBackend` at the Protocol level and raises
`NotImplementedError` with actionable messages from every method.

Subclass and fill in the hardware driver:

```python
from robosandbox.backends.real import (
    RealRobotBackend, RealRobotBackendConfig,
)

class MySO101Backend(RealRobotBackend):
    def __init__(self, serial_port: str):
        super().__init__(RealRobotBackendConfig(
            n_dof=6,
            joint_names=("j1", "j2", "j3", "j4", "j5", "j6"),
            control_hz=200.0,
            home_qpos=(0.0, 0.2, -0.8, 0.0, 0.5, 0.0),
            gripper_open=0.04,
            gripper_closed=0.0,
        ))
        self._arm = SO101Arm(serial_port)

    def load(self, scene):
        self._arm.connect()
        self._arm.home()

    def reset(self):
        self._arm.goto(self._config.home_qpos)

    def step(self, target_joints=None, gripper=None):
        if target_joints is not None:
            self._arm.write_joint_positions(target_joints)
        if gripper is not None:
            self._arm.write_gripper(gripper)

    def observe(self):
        return Observation(
            rgb=self._camera.read(),
            depth=None,
            robot_joints=self._arm.read_joints(),
            ee_pose=self._arm.fk(),
            gripper_width=self._arm.read_gripper(),
            scene_objects={},   # or fill from a pose estimator
            timestamp=time.time(),
        )

    # get_object_pose: query your tracker (AprilTag, OptiTrack, keypoint net)
    # set_object_pose: no-op (real hardware can't teleport)
```

## What a production backend typically wires

- **`load`** — connect cameras, zero the arm, home the gripper,
  verify `scene.workspace_aabb`.
- **`step`** — stream `target_joints` to the position controller at
  roughly the sim rate (200 Hz). Clamp against joint limits and
  max-velocity safety.
- **`observe`** — read joints from the robot state interface, capture
  one frame from the overhead camera, optionally fuse with an
  external pose tracker for `scene_objects`.
- **`reset`** — send home_qpos; block until settled.
- **`get_object_pose`** — AprilTag / OptiTrack / learned keypoint
  detector. Return `None` if you don't have one and rely on VLM
  perception instead.

## Worked example

See `examples/real_robot_swap.py`. It defines a `FakeSO101Backend` —
prints what it would do instead of talking to hardware — so you can
verify your skills+motion plumbing before wiring a real driver.

Run:

```bash
uv run python examples/real_robot_swap.py
```

Expected:

```
[fake_so101] load scene with 0 objects
[fake_so101] reset to home
[example] sim-or-real run ok.  n_dof=6  t=0.025
```

## What's not done

- `RealRobotBackend` ships as a stub only. No concrete subclass ships
  in core — the first target is SO-101 (LeRobot's reference driver).
  See [roadmap 4.6](../reference/roadmap.md#pillar-4-loop-closure).

## Related

- [Policy replay](../tutorials/policy-replay.md) — same `SimBackend`
  interface, but driven by a policy rather than the agent.
- [Skills & agents](skills-and-agents.md) — what consumes the
  backend.
