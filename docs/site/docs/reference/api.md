# API reference

Top-level public surface. Every symbol below is importable under the
path shown.

For the full source tree, browse
[`packages/robosandbox-core/src/robosandbox/`](https://github.com/amarrmb/robosandbox/tree/main/packages/robosandbox-core/src/robosandbox).
This page covers the symbols downstream code should rely on.

## Core types — `robosandbox.types`

```python
from robosandbox.types import (
    Pose, Scene, SceneObject,
    Observation, CameraIntrinsics,
    DetectedObject,
    Grasp, JointTrajectory,
    SkillResult,
    XYZ, Quat,
)
```

All dataclasses are frozen.

### `Pose`

```python
@dataclass(frozen=True)
class Pose:
    xyz: tuple[float, float, float]
    quat_xyzw: tuple[float, float, float, float] = (0, 0, 0, 1)

    def as_array(self) -> np.ndarray          # shape (7,) float64
    @classmethod
    def from_array(cls, a: np.ndarray) -> Pose
```

### `SceneObject`

```python
@dataclass(frozen=True)
class SceneObject:
    id: str
    kind: str                  # 'box' | 'sphere' | 'cylinder' | 'mesh' | 'drawer'
    size: tuple[float, ...]    # semantics depend on kind
    pose: Pose
    mass: float = 0.1
    rgba: tuple[float, float, float, float] = (0.7, 0.7, 0.7, 1.0)
    mesh_path: Path | None = None
    mesh_sidecar: Path | None = None
    collision: str = "coacd"   # 'coacd' | 'hull'
    drawer_max_open: float = 0.12
```

See [Scenes & objects](../concepts/scenes.md) for the per-kind
semantics.

### `Scene`

```python
@dataclass(frozen=True)
class Scene:
    robot_urdf: Path | None = None
    robot_config: Path | None = None
    objects: tuple[SceneObject, ...] = ()
    workspace_aabb: tuple[XYZ, XYZ] = ((-0.5, -0.5, 0.0), (0.5, 0.5, 0.8))
    table_height: float = 0.0
    gravity: XYZ = (0, 0, -9.81)
```

### `Observation`

```python
@dataclass(frozen=True)
class Observation:
    rgb: np.ndarray                                # (H, W, 3) uint8
    depth: np.ndarray | None                        # (H, W) float32, meters
    robot_joints: np.ndarray                        # (n_dof,)
    ee_pose: Pose
    gripper_width: float
    scene_objects: dict[str, Pose] = {}
    timestamp: float = 0.0
    camera_intrinsics: CameraIntrinsics | None = None
    camera_extrinsics: Pose | None = None           # camera in world
```

### `SkillResult`

```python
class SkillResult:
    success: bool
    reason: str = ""                # short structured string
    reason_detail: str = ""
    artifacts: dict = {}
```

## Protocols — `robosandbox.protocols`

```python
from robosandbox.protocols import (
    SimBackend, Perception, GraspPlanner,
    MotionPlanner, RecordSink, VLMClient, Skill,
)
```

All are `@runtime_checkable`. Shapes documented in
[Skills & agents](../concepts/skills-and-agents.md) and
[Perception & grasping](../concepts/perception-and-grasping.md).

## Agent loop — `robosandbox.agent`

```python
from robosandbox.agent.agent import Agent, AgentState, EpisodeResult, StepRecord
from robosandbox.agent.context import AgentContext
from robosandbox.agent.planner import Planner, SkillCall, VLMPlanner, StubPlanner
```

### `Agent`

```python
Agent(ctx: AgentContext, skills: list[Skill], planner: Planner, *, max_replans: int = 3)
    .run(task: str, max_steps: int = 20) -> EpisodeResult
```

### `AgentContext`

The bundle every skill receives:

```python
@dataclass
class AgentContext:
    sim: SimBackend
    perception: Perception
    grasp: GraspPlanner
    motion: MotionPlanner
    recorder: RecordSink | None = None
    on_step: Callable[[], None] | None = None
```

### `SkillCall`

```python
@dataclass
class SkillCall:
    name: str
    arguments: dict[str, Any]
    tool_call_id: str | None = None
```

### `EpisodeResult`

```python
@dataclass
class EpisodeResult:
    success: bool
    task: str
    plan: list[SkillCall]
    steps: list[StepRecord]
    replans: int = 0
    vlm_calls: int = 0
    final_reason: str = ""
    final_detail: str = ""
```

## Sim backend — `robosandbox.sim.mujoco_backend`

```python
from robosandbox.sim.mujoco_backend import MuJoCoBackend

sim = MuJoCoBackend(render_size=(240, 320), camera="scene")
sim.load(scene)            # builds MJCF, compiles model
sim.step(target_joints=None, gripper=None)
obs = sim.observe()
sim.close()
```

Implements the full `SimBackend` protocol. See
[Real-robot bridge](../concepts/real-robot.md) for the swap pattern.

## <a id="backends"></a>Real-robot stub — `robosandbox.backends.real`

```python
from robosandbox.backends.real import RealRobotBackend, RealRobotBackendConfig
```

`RealRobotBackend` is a stub: satisfies `SimBackend` at the Protocol
level, raises `NotImplementedError` from every method. Subclass and
fill in a hardware driver.

## Perception — `robosandbox.perception`

```python
from robosandbox.perception.ground_truth import GroundTruthPerception
from robosandbox.perception.vlm_pointer import VLMPointer
```

Both match `Perception`. See
[perception & grasping](../concepts/perception-and-grasping.md).

## Grasping — `robosandbox.grasp.analytic`

```python
from robosandbox.grasp.analytic import AnalyticTopDown

grasp = AnalyticTopDown(default_object_width=0.04, grasp_height_offset=0.013)
```

## Motion — `robosandbox.motion.ik`

```python
from robosandbox.motion.ik import DLSMotionPlanner, UnreachableError

motion = DLSMotionPlanner(n_waypoints=160, dt=0.005)
traj = motion.plan(sim, start_joints, target_pose, constraints={"orientation": "z_down"})
```

Raises `UnreachableError` when IK diverges.

## Skills — `robosandbox.skills`

```python
from robosandbox.skills.pick import Pick
from robosandbox.skills.place import PlaceOn
from robosandbox.skills.push import Push
from robosandbox.skills.home import Home
from robosandbox.skills.pour import Pour
from robosandbox.skills.tap import Tap
from robosandbox.skills.drawer import OpenDrawer, CloseDrawer
from robosandbox.skills.stack import Stack
```

All match `Skill`. See
[skills & agents](../concepts/skills-and-agents.md) for signatures.

## Recorder — `robosandbox.recorder.local`

```python
from robosandbox.recorder.local import LocalRecorder

recorder = LocalRecorder(root="runs", video_fps=30, subsample_to_fps=True)
recorder.start_episode(task, metadata) -> episode_id
recorder.write_frame(obs, action=None)
recorder.end_episode(success, result)
```

## Export — `robosandbox.export.lerobot`

```python
from robosandbox.export.lerobot import export_episode

out_dir = export_episode(src: Path, dst: Path, *, task: str | None = None, fps: int = 30)
```

Needs `pyarrow` — `uv pip install -e 'packages/robosandbox-core[lerobot]'`.

## Policy — `robosandbox.policy`

```python
from robosandbox.policy import Policy, ReplayTrajectoryPolicy, run_policy, load_policy

# Protocol:
#   Policy.act(obs: Observation) -> np.ndarray  shape (n_dof + 1,)

policy = ReplayTrajectoryPolicy.from_jsonl("runs/.../events.jsonl")
result = run_policy(sim, policy, max_steps=1000, success=task.success)
# {"success": bool | None, "steps": int, "initial_obs": ..., "final_obs": ...}
```

`load_policy(path)` auto-wraps a directory containing `events.jsonl`
or a `policy.json` of `{"kind": "replay_trajectory", ...}`. Any other
format raises `ImportError` — extend it to dispatch on your
checkpoint format.

## Tasks — `robosandbox.tasks`

```python
from robosandbox.tasks.loader import (
    Task, SuccessCriterion,
    load_task, load_builtin_task,
    list_builtin_tasks, list_builtin_ycb_objects,
)

tasks = list_builtin_tasks()      # ['home', 'pick_cube', 'pick_cube_franka', ...]
task = load_builtin_task("pick_cube_franka")   # Task(name, scene, prompt, success, ...)
task = load_task(Path("/path/to/my_task.yaml"))
```

Returned `Task` has `name`, `scene: Scene`, `prompt: str`,
`success: SuccessCriterion`, `seed_note: str`, and an optional
`randomize: dict`.

## Scene presets — `robosandbox.scene.presets`

```python
from robosandbox.scene.presets import tabletop_clutter

scene = tabletop_clutter(n_objects=5, seed=0)
# Franka + N non-overlapping YCB objects. seed 0 is deterministic.
```

## VLM client — `robosandbox.vlm.client`

```python
from robosandbox.vlm.client import OpenAIVLMClient, VLMConfig, VLMTransportError

client = OpenAIVLMClient(VLMConfig(
    api_key="sk-...",
    base_url="https://api.openai.com/v1",
    model="gpt-4o-mini",
))
resp = client.chat(messages, tools=..., tool_choice="auto")
# resp: {"content": str | None, "tool_calls": [{"name", "arguments", "id"}, ...]}
```

`VLMTransportError` wraps network / auth failures so the agent loop
can surface them as a `final_reason`.

## Robot loader — `robosandbox.scene.robot_loader`

```python
from robosandbox.scene.robot_loader import load_robot

mjcf_string, robot_spec = load_robot(urdf_path, config_path)
# robot_spec: arm_joint_names, arm_actuator_names, gripper_primary_joint,
#             ee_site_name, home_qpos, base_pose, ...
```

Used internally by the MJCF builder when `Scene.robot_urdf` is set.
Exposed so you can introspect a sidecar without a sim —
`examples/custom_robot.py` uses it.

## Related

- [Concepts](../concepts/scenes.md) for prose explanations.
- [CLI reference](cli.md) for the command-line layer on top of this
  API.
