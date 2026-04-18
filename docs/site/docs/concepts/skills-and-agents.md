# Skills & agents

A task runs through three abstractions:

1. **Skills** — the action vocabulary (`pick`, `place_on`, …). Each is
   a callable with a JSON schema.
2. **Planner** — turns a natural-language task + current observation
   into a list of `SkillCall`s.
3. **Agent** — ReAct loop: plan → execute → (on failure) replan.

## Skill protocol

```python
class Skill(Protocol):
    name: str
    description: str
    parameters_schema: dict            # JSON schema
    def __call__(self, ctx, **kwargs) -> SkillResult: ...
```

No base class, no registry. Any object matching the shape works. The
VLMPlanner converts `parameters_schema` directly to an OpenAI tool
definition; the stub planner dispatches by `name`.

`ctx` is an `AgentContext` carrying `sim`, `perception`, `grasp`,
`motion`, and (optionally) `recorder`. Skills do I/O through `ctx`.

## Skills shipped in v0.1

| Skill | Signature | What it does |
|---|---|---|
| `pick` | `pick(object: str)` | Locate object, plan top-down grasp, execute, close gripper. |
| `place_on` | `place_on(target: str)` | Move above target, release. |
| `push` | `push(object: str, direction: str)` | Cartesian push in `forward / back / left / right`. |
| `home` | `home()` | Return to the home pose from the robot sidecar. |
| `pour` | `pour(target: str)` | Tilt end-effector over target. |
| `tap` | `tap(object: str)` | Touch the top of an object with the fingertip. |
| `open_drawer` | `open_drawer(drawer: str)` | Grasp handle, pull toward base. |
| `close_drawer` | `close_drawer(drawer: str)` | Push drawer back in. |
| `stack` | `stack(object: str, target: str)` | Pick + place in one. |

Every skill returns `SkillResult(success, reason, reason_detail,
artifacts)`. Failure reasons are structured strings (e.g.
`unreachable`, `not_found`, `missed_grasp`) so the replan loop can
reason about them.

Skills register at the `robosandbox.skills` entry point — see
`packages/robosandbox-core/pyproject.toml`. A plugin package can add
new skills without core changes.

## Planner protocol

```python
class Planner(Protocol):
    def plan(
        self,
        task: str,
        obs: Observation,
        prior_attempts: list[dict],
    ) -> tuple[list[SkillCall], int]:
        """Returns (plan, n_model_calls). Empty plan == 'already done'."""
```

Two implementations ship:

### `VLMPlanner`

Calls an OpenAI-compatible chat endpoint with tool-calling + image
input. Converts each Skill's `parameters_schema` into a tool
definition. Passes an RGB frame of the current observation plus a
text summary of `scene_objects` keys/xyz.

Used by `robo-sandbox run --vlm-provider {openai,ollama,custom}` and
the `llm_guided.py` example.

Retry nudge: if the model responds with prose instead of tool calls,
VLMPlanner re-asks once with `"respond with tool calls only — no
prose."`.

### `StubPlanner`

Deterministic, zero-dep regex NLU. No "AI" — just enough to prove the
agent loop without a model. Handles:

- `pick (up) the <obj>`
- `pick (up) the <obj> (and|then|,) (put|place) (it) on (the) <obj2>`
- `stack <obj> on (top of) <obj2>`
- `push the <obj> forward|back|left|right`
- `pour <obj> into <obj2>`
- `tap/press/poke/touch the <obj>`
- `open/close the <drawer>`
- `(go) home`

Object names are fuzzy-matched against scene object IDs (exact →
substring → word-overlap).

Used by default in the benchmark runner and in tests.

## Agent loop

```
IDLE → PLAN → EXECUTE (one skill at a time) → EVALUATE →
                   │ success                      │ failure
                   ▼                              ▼
                 next in plan                   REPLAN ─► (max N times)
                   │                              │
                   ▼                              ▼
                 DONE                           FAILED
```

```python
from robosandbox.agent.agent import Agent
from robosandbox.agent.context import AgentContext
from robosandbox.agent.planner import StubPlanner
from robosandbox.skills.pick import Pick
from robosandbox.skills.home import Home

skills = [Pick(), Home()]
agent = Agent(ctx=ctx, skills=skills, planner=StubPlanner(skills))
episode = agent.run("pick up the red cube")
# episode.success, episode.steps, episode.replans, episode.plan, ...
```

ReAct replan semantics:

- On any `SkillResult(success=False)`, the agent collects the failure
  (`step_idx`, `skill`, `args`, `reason`, `reason_detail`) into
  `prior_attempts`.
- The planner is called again with `prior_attempts` so a VLM can
  avoid repeating the same move. The stub planner ignores it (it's
  deterministic).
- The loop terminates on `replans >= max_replans` with
  `final_reason="replan_exhausted"`.

## Composing your own

Custom planner — any object with a `plan(task, obs, prior_attempts)
-> (list[SkillCall], int)` method. See `examples/custom_skill.py` for
a `TapStub` that emits one skill call unconditionally.

Custom skill — [tutorial](../tutorials/custom-skill.md). The Tap
example in the tutorial is under 40 lines.

## Related

- [Perception & grasping](perception-and-grasping.md) — how skills
  find objects and compute gripper poses.
- [Recording & export](recording-and-export.md) — what the agent
  writes to disk during a run.
- [CLI reference](../reference/cli.md#run) — `robo-sandbox run`
  flags.
