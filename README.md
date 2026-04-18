# RoboSandbox

> Sim-first agentic manipulation sandbox.
> **Any arm. Any object. Any command.**

Drop a URDF, drop some objects, type a task. A VLM (or rule-based stub)
decomposes the task into skills; the sandbox executes them in MuJoCo
with analytic grasping + IK; optionally records the result so you can
fine-tune your own policies.

```
user: "pick up the red cube and put it on the green cube"
       │
       ▼
 planner ─► [pick(red_cube), place_on(green_cube)]
       │
       ▼
 perception (VLM or ground truth) locates both in 3D
       │
       ▼
 motion (DLS Jacobian IK + Cartesian interpolation) executes
       │
       ▼
 recorder writes runs/<id>/video.mp4 + events.jsonl
```

## Status

v0.1 in active development. The architecture is the product — the
built-in arm is the cheapest way to run the stack. Plumbing works end
to end; built-in arm physics can be flaky for stacking (force-
controlled grip lands in v0.2). See **Roadmap** at the bottom.

## Install

```bash
git clone <this-repo> robosandbox
cd robosandbox
uv sync                                 # one-time
uv pip install -e packages/robosandbox-core
```

Requires Python 3.10+. MuJoCo 3.2+ comes in as a dep; no GPU needed for
the built-in arm.

## Three quickstart paths

### Zero setup — rule-based stub planner

```bash
uv run robo-sandbox run "pick up the red cube"
# → plan: [pick(object=red_cube)]
# → MuJoCo opens a 3-cube scene, arm picks the cube
# → writes runs/<timestamp>/{video.mp4, events.jsonl, result.json}
```

No API key, no model download, no vendor lock-in. The stub planner
handles:

- `pick (up) the <obj>`
- `pick (up) the <obj> (and|then|,) (put|place) (it) on (the) <obj2>`
- `stack <obj> on (top of) <obj2>`
- `push the <obj> forward|back|left|right`
- `(go) home`

### Local, free — Ollama

```bash
ollama pull llama3.2-vision
ollama serve &
uv run robo-sandbox run --vlm-provider ollama \
  "pick up the blue cube and put it on the green cube"
```

Any OpenAI-compatible Ollama model works. Provider defaults use
`llama3.2-vision` on `localhost:11434/v1`; override with `--model` or
`--base-url`.

### Cloud, best quality — OpenAI

```bash
export OPENAI_API_KEY=sk-...
uv run robo-sandbox run --vlm-provider openai \
  "stack all three cubes by colour — red on green on blue"
```

`gpt-4o-mini` is the default; ~$0.002 per episode. Swap with
`--model gpt-4o` for better planning on harder tasks.

`--vlm-provider custom --base-url https://...` works with together.ai,
vLLM, any OpenAI-compatible endpoint.

## Benchmark

```bash
uv run robo-sandbox-bench                    # run all default tasks
uv run robo-sandbox-bench --seeds 5          # 5 seeds each
uv run robo-sandbox-bench --vlm-provider ollama   # use a real VLM
```

Five built-in tasks (YAML scenes + success criteria under
`packages/robosandbox-core/src/robosandbox/tasks/definitions/`):

| Task | What it exercises |
|---|---|
| `home` | Skill dispatch with no spatial reasoning |
| `pick_cube` | Single-object pick (core reliability) |
| `pick_from_three` | Perception disambiguation by colour name |
| `push_forward` | Non-pick manipulation, verifies directional displacement |
| `_experimental_stack_two` | Multi-step plan. Excluded from default run — needs force-controlled grip (v0.2). |

Each task bundles a `Scene`, a natural-language `prompt`, and a
declarative `SuccessCriterion` evaluated against the final
`Observation`. No executable success check code — safer, diffable.

Results are appended to `benchmark_results.json` for regression
tracking.

## Architecture

Every swappable component is a narrow `Protocol` (PEP 544) with an
entry-point registration. `pip install robosandbox-<plugin>` drops a
new implementation in; no core changes needed.

```
packages/robosandbox-core/
├── src/robosandbox/
│   ├── types.py          Pose, Scene, Observation, Grasp, SkillResult
│   ├── protocols.py      SimBackend, Perception, GraspPlanner,
│   │                     MotionPlanner, RecordSink, VLMClient, Skill
│   ├── sim/              MuJoCo backend (built-in 6-DOF arm, no meshes)
│   ├── scene/            MJCF builder — spawns any Scene into MuJoCo
│   ├── perception/       ground_truth (sim cheat), vlm_pointer (VLM)
│   ├── grasp/            analytic top-down (v0.1)
│   ├── motion/           DLS Jacobian IK + Cartesian interpolation
│   ├── skills/           Pick, PlaceOn, Push, Home
│   ├── agent/            Planner protocol, VLMPlanner, StubPlanner,
│   │                     ReAct-style Agent with replan loop
│   ├── vlm/              OpenAI-compatible client + JSON recovery
│   ├── recorder/         MP4 + JSONL per episode
│   ├── tasks/            Task loader + benchmark runner
│   ├── cli.py            `robo-sandbox` entry point
│   ├── demo.py           Scripted pick (no VLM, no API)
│   └── agentic_demo.py   Full agent loop
└── tests/                21 tests covering types, IK, skills, agent,
                          planner, JSON recovery, VLM pointer projection
```

### Agent loop

```
IDLE → PLAN → EXECUTE (one skill at a time) → EVALUATE →
                   │ success                      │ failure
                   ▼                              ▼
                 next in plan                   REPLAN ─► (max N times)
                   │                              │
                   ▼                              ▼
                 DONE                           FAILED
```

The Planner protocol is the key seam:

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

`VLMPlanner` calls an OpenAI-compatible endpoint with tool-calling +
image input. `StubPlanner` is regex-based.

### Skills expose themselves via JSON schema

Every skill has `name`, `description`, `parameters_schema` (JSON
schema). VLMPlanner converts them to OpenAI tool definitions; the
model's tool calls become skill dispatches. Add a skill by shipping a
package that registers at the `robosandbox.skills` entry point.

## Roadmap

- **v0.1** (current): MuJoCo + built-in arm + 4 core skills + stub/
  OpenAI-compatible planner + 5-task benchmark.
- **v0.2**: force-controlled grip (reliable stacking), `robosandbox-
  curobo` plugin for GPU motion planning, `robosandbox-molmo`
  dedicated pointing model, real-robot bridge (LeRobot adapter), web
  UI for scene authoring.
- **v0.3**: mesh-object import, `robosandbox-anygrasp` opt-in plugin,
  force-based control API, leaderboard.

## License

Core: Apache 2.0.

Optional `contrib/` plugins carry their own licenses — research-
licensed grasp predictors etc. live there; they are opt-in installs
and not pulled in by the base `pip install robosandbox`.
