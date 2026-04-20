# Perception & grasping

Between "pick up the red cube" and an actual joint trajectory, there
are three main pieces:

- **`Perception`** — text + observation → `DetectedObject` list.
- **`GraspPlanner`** — observation + target → candidate `Grasp` list.
- **`MotionPlanner`** — start joints + target pose → `JointTrajectory`.

Each one is a narrow protocol, so swapping implementations is meant to
be straightforward.

## Perception

```python
class Perception(Protocol):
    def locate(self, query: str, obs: Observation) -> list[DetectedObject]: ...
```

Two implementations ship in core:

### `GroundTruthPerception`

Substring match against `obs.scene_objects`. Used in the benchmark
runner and by all Tier-3 skill tests so reliability numbers measure
the sandbox, not the VLM.

```python
from robosandbox.perception.ground_truth import GroundTruthPerception
perception = GroundTruthPerception()
perception.locate("red cube", obs)
# [DetectedObject(label='red_cube', pose_3d=Pose(xyz=(0.4, 0.0, 0.05), ...), confidence=1.0)]
```

Use this when you want to debug the rest of the stack without involving
the model.

### `VLMPointer`

Default perception in the agentic flow. Calls an OpenAI-compatible
chat endpoint, passes the RGB frame + intrinsics + extrinsics, asks
the VLM for pixel coords + bbox, back-projects through depth into
world frame.

Output shape the VLM must produce:

```json
{"objects": [
  {"label": "...", "bbox": [x1, y1, x2, y2], "point": [cx, cy], "confidence": 0.0-1.0}
]}
```

`VLMOutputError` is raised on malformed output; the caller can choose
to retry. `vlm.json_recovery.parse_json_loose` tolerates code-fenced
or prose-wrapped responses.

### When to use which

| Use case | Perception |
|---|---|
| Running the benchmark, regression-testing sim behaviour | `GroundTruthPerception` |
| Running the agent loop with a real VLM | `VLMPointer` |
| Deterministic tests that exercise the skill (not the model) | `GroundTruthPerception` |

A `CassetteVLMClient` (see `tests/cassettes/`) records + replays VLM
responses — lets VLMPointer participate in CI without network access.

## Grasping

```python
class GraspPlanner(Protocol):
    def plan(self, obs: Observation, target: DetectedObject) -> list[Grasp]: ...
```

There is one implementation in v0.1:

### `AnalyticTopDown`

Assumes the gripper approaches along `-Z`, palm down, jaws along `X`.
It works well enough for cubes, short cylinders, cans, and mugs picked
from above. It is not the right tool for flatter or more awkward shapes
that really need side grasps.

Key parameters:

- `default_object_width=0.04` — fallback when bounds unknown.
- `grasp_height_offset=0.013` — added above the object centroid when
  targeting the end-effector site. Tuned for v0.1's 4 cm fingers.

Output `Grasp(pose, gripper_width, approach_offset=0.08, score)`:

- `pose` — gripper pose at closed-on-object.
- `gripper_width` — width at which to close.
- `approach_offset` — distance above `pose` to approach from.

The planner returns one candidate today, and `Pick` executes it.

## Motion planning

```python
class MotionPlanner(Protocol):
    def plan(
        self,
        sim: SimBackend,
        start_joints: np.ndarray,
        target_pose: Pose,
        constraints: dict | None = None,
    ) -> JointTrajectory: ...
```

There is one implementation in v0.1:

### `DLSMotionPlanner`

Damped least-squares Jacobian IK + Cartesian interpolation.

- Interpolates `n_waypoints` Cartesian poses from current EE pose to
  `target_pose`.
- Solves IK at each waypoint; `UnreachableError` is raised if the
  solver diverges.
- Supports a `{"orientation": "z_down"}` constraint that locks the
  palm-down quaternion — used by every pick/place skill.

Construction:

```python
from robosandbox.motion.ik import DLSMotionPlanner
motion = DLSMotionPlanner(n_waypoints=160, dt=0.005)  # default values
```

`dt` is per-step time; total trajectory duration is `n_waypoints *
dt`. Skills call `motion.plan(...)` once per Cartesian segment.

If a GPU motion planner gets added later, this is the seam where it
fits.

## Typical skill shape

Most pick/place skills follow this shape, and `Pick` is the clearest
example:

```python
def __call__(self, ctx, object):
    obs = ctx.sim.observe()
    detections = ctx.perception.locate(object, obs)
    if not detections:
        return SkillResult(False, reason="not_found", reason_detail=object)
    grasps = ctx.grasp.plan(obs, detections[0])
    grasp = grasps[0]

    # approach → descend → close → lift
    for pose in (approach_pose, grasp.pose, grasp.pose, lift_pose):
        traj = ctx.motion.plan(
            ctx.sim,
            start_joints=ctx.sim.observe().robot_joints,
            target_pose=pose,
            constraints={"orientation": "z_down"},
        )
        execute_trajectory(ctx, traj, gripper=...)
    return SkillResult(True, reason="picked", artifacts={"grasp": grasp})
```

The shared `skills._common.execute_trajectory` helper streams waypoints
to the sim at `traj.dt`.

## Related

- [Skills & agents](skills-and-agents.md) — who calls Perception /
  Grasp / Motion.
- [Scenes & objects](scenes.md) — what `DetectedObject.label` refers
  back to.
