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

## Documentation

Full docs — concepts, tutorials (custom arm / task / skill / policy
replay), and CLI + API reference — live under
[`docs/site/`](docs/site/) and build with MkDocs-material:

```bash
uv pip install -e 'packages/robosandbox-core[docs]'
mkdocs serve -f docs/site/mkdocs.yml          # live preview
mkdocs build --strict -f docs/site/mkdocs.yml # one-shot build
```

Start at [`docs/site/docs/index.md`](docs/site/docs/index.md) if
you're reading on GitHub.

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
uv run robo-sandbox-bench --seeds 50         # randomize and aggregate
uv run robo-sandbox-bench --vlm-provider ollama   # use a real VLM
```

Tasks that declare a `randomize:` block in their YAML get per-seed
jitter (xy translation + yaw) applied to every object. Seed 0 is always
the deterministic base layout (bit-exact with `--seeds 1`); seeds ≥ 1
sample uniform perturbations keyed on the seed. The summary reports
`mean ± stderr` per task when `--seeds > 1`.

Five built-in tasks (YAML scenes + success criteria under
`packages/robosandbox-core/src/robosandbox/tasks/definitions/`):

| Task | What it exercises |
|---|---|
| `home` | Skill dispatch with no spatial reasoning |
| `pick_cube` | Single-object pick (core reliability) |
| `pick_cube_franka` | URDF-import path — bundled Franka picks a cube |
| `pick_from_three` | Perception disambiguation by colour name |
| `pick_ycb_mug` | Mesh-import path — bundled YCB mug (15 convex hulls) picked by Franka |
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
│   ├── sim/              MuJoCo backend (built-in 6-DOF arm + URDF robots)
│   ├── scene/            MJCF builder + URDF/mesh loaders — spawns any Scene into MuJoCo
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
└── tests/                65 tests covering types, IK, skills, agent,
                          planner, JSON recovery, VLM pointer projection,
                          URDF import (Franka), mesh import (YCB pack)
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
- **v0.3**: `robosandbox-anygrasp` opt-in plugin, force-based control
  API, leaderboard.

## Browser live viewer

Install the optional extra and open a browser — you'll see MuJoCo render
in real time and can kick off tasks from a dropdown.

```bash
pip install 'robosandbox[viewer]'
robo-sandbox viewer
# → open http://localhost:8000
```

Pick a task, click Run. Events log to the sidebar; frames stream at
~15–50 fps depending on how fast the sim is stepping. Pass
`--task pick_cube_franka` (or any other) to preload a specific scene.
`--host 0.0.0.0` exposes it to other machines on your LAN.

## Bundled assets

### Robots

`assets/robots/franka_panda/` ships a trimmed copy of Franka Emika
Panda adapted from [mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie)
under Apache 2.0. Visual meshes removed (collision-only, ~160 KB); the
tendon-driven gripper actuator was replaced with a simple position
actuator on `finger_joint1` so the standard RobotSpec interface
(open_qpos / closed_qpos) applies directly. See `LICENSE` in that
directory for menagerie's attribution.

To bring your own robot:

```python
Scene(
    robot_urdf=Path("/path/to/ur5.urdf"),     # .urdf or .xml
    robot_config=Path("/path/to/ur5.robosandbox.yaml"),  # optional — sibling auto-discovered
    objects=(...),
)
```

The sidecar YAML tells RoboSandbox which joint is the primary finger,
where the end-effector TCP sits, the home pose, and gripper open/closed
qpos. See `packages/robosandbox-core/src/robosandbox/assets/robots/franka_panda/panda.robosandbox.yaml`
for the schema.

### Objects (mesh import)

`assets/objects/ycb/` ships 10 pre-decomposed YCB benchmark objects: a
visual OBJ + N CoACD convex hulls + per-object sidecar YAML each.

| YCB id | Description | Mass (kg) |
|---|---|---|
| `003_cracker_box` | cracker box | 0.411 |
| `005_tomato_soup_can` | tomato soup can | 0.349 |
| `006_mustard_bottle` | mustard bottle | 0.603 |
| `011_banana` | banana | 0.066 |
| `013_apple` | apple | 0.068 |
| `024_bowl` | bowl (hollow; 11 hulls) | 0.147 |
| `025_mug` | mug (handled; 15 hulls) | 0.118 |
| `035_power_drill` | power drill | 0.895 |
| `042_adjustable_wrench` | adjustable wrench | 0.252 |
| `055_baseball` | baseball | 0.148 |

Drop any of them into a task with the `@ycb:` shorthand:

```yaml
objects:
  - id: box_1
    kind: mesh
    mesh: "@ycb:003_cracker_box"
    pose: {xyz: [0.4, 0.0, 0.08]}
  - id: soup
    kind: mesh
    mesh: "@ycb:005_tomato_soup_can"
    pose: {xyz: [0.4, 0.15, 0.06]}
```

Or discover the bundled catalog from Python:

```python
from robosandbox.tasks.loader import list_builtin_ycb_objects
list_builtin_ycb_objects()
# ['003_cracker_box', '005_tomato_soup_can', ..., '055_baseball']
```

See `assets/objects/ycb/LICENSE` for the YCB project's terms.

**Bring-your-own meshes.** The sandbox decomposes user OBJ/STL files
with CoACD and caches the hulls at `~/.cache/robosandbox/mesh_hulls/`:

```bash
pip install 'robosandbox[meshes]'    # pulls in coacd
```

```python
SceneObject(
    id="widget",
    kind="mesh",
    mesh_path=Path("/abs/path/to/widget.obj"),
    collision="coacd",                # or "hull" (skip decomp if mesh is already convex)
    pose=Pose(xyz=(0.4, 0.0, 0.05)),
    mass=0.1,
)
```

`collision="hull"` is a cheap fallback for already-convex meshes — no
CoACD install required, but the sandbox does not compute a hull for
you; it trusts the mesh is convex. For concave objects, always use
`collision="coacd"`.

Pre-decompose a mesh once for a bundled asset with the authoring tool:

```bash
python scripts/decompose_mesh.py \
  --input /path/to/drill.obj \
  --out-dir assets/objects/custom/drill \
  --name drill --mass 0.3 --center-bottom
```

## License

Core: Apache 2.0.

Optional `contrib/` plugins carry their own licenses — research-
licensed grasp predictors etc. live there; they are opt-in installs
and not pulled in by the base `pip install robosandbox`.
