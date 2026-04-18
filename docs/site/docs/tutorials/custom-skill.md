# Tutorial: custom skill

Skills are any object matching the `Skill` protocol. No base class, no
registry required — just match the shape, pass an instance to
`Agent(skills=[...])`.

## The protocol

```python
class Skill(Protocol):
    name: str
    description: str
    parameters_schema: dict               # JSON schema for arguments
    def __call__(self, ctx, **kwargs): ...   # returns SkillResult
```

- `name` — lowercase identifier; matches what a planner emits.
- `description` — one-line string the VLM sees in its tool definition.
- `parameters_schema` — standard JSON schema. Converted directly to
  an OpenAI tool definition.
- `__call__` — receives `ctx: AgentContext` plus keyword args parsed
  from the planner's `SkillCall.arguments`. Returns `SkillResult`.

## Tap skill — minimal working example

```python
from robosandbox.agent.context import AgentContext
from robosandbox.motion.ik import UnreachableError
from robosandbox.skills._common import execute_trajectory
from robosandbox.types import Pose, SkillResult


class Tap:
    """Touch the top of an object with the end-effector, no gripper."""

    name = "tap"
    description = "Tap the named object by touching its top with the end-effector."
    parameters_schema = {
        "type": "object",
        "properties": {
            "object": {"type": "string", "description": "Scene object id to tap."},
        },
        "required": ["object"],
    }

    def __call__(self, ctx: AgentContext, object: str) -> SkillResult:
        obs = ctx.sim.observe()
        target_pose = obs.scene_objects.get(object)
        if target_pose is None:
            return SkillResult(False, reason="not_found", reason_detail=object)

        tx, ty, tz = target_pose.xyz
        above   = Pose(xyz=(tx, ty, tz + 0.12), quat_xyzw=(1.0, 0.0, 0.0, 0.0))
        contact = Pose(xyz=(tx, ty, tz + 0.025), quat_xyzw=(1.0, 0.0, 0.0, 0.0))

        try:
            for pose in (above, contact, above):
                now = ctx.sim.observe()
                traj = ctx.motion.plan(
                    ctx.sim,
                    start_joints=now.robot_joints,
                    target_pose=pose,
                    constraints={"orientation": "z_down"},
                )
                execute_trajectory(ctx, traj, gripper=0.0)
        except UnreachableError as e:
            return SkillResult(False, reason="unreachable", reason_detail=str(e))
        return SkillResult(True, reason="tapped", artifacts={"object": object})
```

## Routing through a planner

### Via `StubPlanner`

`StubPlanner` already recognises "tap/press/poke/touch the `<obj>`"
and emits `SkillCall("tap", {"object": <obj>})` when a `tap` skill is
in the skill list.

```python
from robosandbox.agent.agent import Agent
from robosandbox.agent.planner import StubPlanner
from robosandbox.skills.home import Home

skills = [Tap(), Home()]
agent = Agent(ctx=ctx, skills=skills, planner=StubPlanner(skills))
episode = agent.run("tap the apple")
```

### Via a tiny custom planner

For more control — or to test the skill in isolation — any object
with a `plan(task, obs, prior_attempts) -> (list[SkillCall], int)`
method is a valid planner:

```python
from robosandbox.agent.planner import SkillCall

class TapStub:
    def plan(self, task, obs, prior_attempts):
        return ([SkillCall(name="tap", arguments={"object": "apple"})], 0)

agent = Agent(ctx=ctx, skills=[Tap(), Home()], planner=TapStub())
agent.run("anything — ignored")
```

### Via `VLMPlanner`

Drop the `Tap()` instance into the skill list — `VLMPlanner` converts
`parameters_schema` to an OpenAI tool definition automatically.
Description text matters here: it's what the model reads.

## Registering as a plugin

If you want your skill available to downstream users without them
copying code, ship it in a package with a
`robosandbox.skills` entry point in `pyproject.toml`:

```toml
[project.entry-points."robosandbox.skills"]
tap = "my_package.skills:Tap"
```

Core registers its built-ins the same way — see
`packages/robosandbox-core/pyproject.toml`.

## Error handling conventions

`SkillResult.reason` is a structured string the replan loop reads:

| `reason` | When to raise |
|---|---|
| `not_found` | Object isn't in the scene / perception returned nothing. |
| `unreachable` | Motion planner couldn't IK the target. |
| `missed_grasp` | Gripper closed but object is no longer held. |
| `bad_arguments` | (Emitted by Agent on `TypeError`.) |
| `unknown_skill` | (Emitted by Agent when planner calls a name it doesn't have.) |

Pass structured reasons so prior-attempt feedback is informative to
the VLM on replan.

## Worked reference

Full runnable script: `examples/custom_skill.py`. Builds the scene,
runs `Tap()` in isolation, then runs the same skill through the
Agent loop.

Run:

```bash
uv run python examples/custom_skill.py
```

Expected final line:

```
agent episode success=True, steps=1
```

## See also

- [Skills & agents](../concepts/skills-and-agents.md) for the
  protocol, the agent state machine, and built-in skill list.
- [Perception & grasping](../concepts/perception-and-grasping.md)
  for what `ctx.perception` / `ctx.grasp` / `ctx.motion` expose.
