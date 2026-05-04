# RoboSandbox

Two checkpoints, same task. Which one actually got better?

In sim today, you can't tell. Different teams settle physics differently, run different numbers of trials, score success differently. The numbers don't compose. Read a 43% pick rate in one paper and a 51% pick rate in another and you're comparing weather reports from different planets.

`robo-sandbox eval` makes the contract explicit. One command, one JSON: success rate, Wilson 95% CI, per-position breakdown of where it failed, full provenance. Same JSON whether you ran 64 trials in MuJoCo or 1024 parallel trials in Newton.

```bash
robo-sandbox eval \
    --task pick_cube_franka_random \
    --policy outputs/act_franka_pick/checkpoints/050000/pretrained_model \
    --sim-backend mujoco \
    --n-trials 64 --action-repeat 6 --settle-steps 60 \
    --output outputs/eval_50k.json
# 28/64 (43.8%), CI [32.3, 55.9], spatial_breakdown by cube x/y
```

That's the whole product. Everything below is how to use it.

## What's run, what scored

Today's policy support is `LeRobotPolicyAdapter`. The `Policy` protocol is framework-agnostic — wrappers for Diffusion Policy, Octo, π0 are a few hours of glue each, but they aren't shipped.

Numbers from real runs on this branch:

| Run | Result |
|---|---|
| ACT 50k on `pick_cube_franka_random`, n=64, MuJoCo | 28/64 (43.8%), CI [32.3, 55.9] |
| Reach, 1024 parallel Newton worlds | 1024/1024 (100%) |
| USB-A 1.5mm insertion, 64 Newton worlds | 64/64 (100%) |
| USB-A 1.5mm insertion, 1024 Newton worlds | 988/1024 (96.5%), CI [95.2, 97.4] |
| Same insertion policy, replayed in classical MuJoCo, n=32 | 32/32 (100%), CI [89.3, 100.0] |

The cross-sim row matters most. A policy trained in Newton runs in classical MuJoCo at full success without re-training or code changes. If it didn't transfer, the sim numbers wouldn't mean much.

<p align="center">
  <video src="https://github.com/amarrmb/robosandbox/raw/main/docs/site/docs/assets/demos/usb_a_insertion_zoom.mp4" width="640" controls muted></video>
</p>

Four Franka arms, four different port positions, USB-A spec (1.5mm clearance), all four insertions complete. Same policy ran in classical MuJoCo at 32/32 with zero code change.

The `eval` CLI lives on `experimental/newton-eval` today and lands on `main` with the eval-and-recording-hygiene PR. Until that ships, reproducing any of the rows above means checking out the branch.

## Try it

```bash
git clone https://github.com/amarrmb/robosandbox.git
cd robosandbox
uv sync
uv pip install -e 'packages/robosandbox-core[viewer]'

uv run robo-sandbox viewer
# open http://localhost:8000
# pick a task, type "pick up the red cube", click Run
# hit Record before running to save the episode for export
```

No API key, no model download. The built-in planner runs without a VLM so you can verify the install before plugging anything in.

## What this isn't

A training framework. Train your policy elsewhere — `lerobot train`, your own RL loop, whatever — and bring the checkpoint here.

A photorealistic simulator. MuJoCo + Newton, no rendering tricks. If your policy needs realistic textures or lighting, this is the wrong tool.

A drop-in for arbitrary policy frameworks. The `LeRobotPolicyAdapter` is what's shipped. Other frameworks need their own wrapper around the `Policy` protocol.

If you outgrow this and move to Isaac Sim or your team's internal stack, that's success.

## Make It Yours

### Providers

`robo-sandbox run` takes a `--vlm-provider` flag. Pick one:

| Provider | Command | Setup |
|---|---|---|
| `stub` (default) | `uv run robo-sandbox run "pick up the red cube"` | none — regex-based planner |
| `ollama` | `uv run robo-sandbox run --vlm-provider ollama "pick up the blue cube and put it on the green cube"` | `ollama pull llama3.2-vision && ollama serve &` |
| `openai` | `uv run robo-sandbox run --vlm-provider openai "stack all three cubes by colour — red on green on blue"` | `export OPENAI_API_KEY=sk-...` |
| `custom` | `uv run robo-sandbox run --vlm-provider custom --base-url https://... ...` | any OpenAI-compatible endpoint (together.ai, vLLM, ...) |

Override the model with `--model` (defaults: `llama3.2-vision` for ollama, `gpt-4o-mini` for openai). For richer reasoning on open-ended tasks, try `--model gpt-4o`.

### System prerequisites

Requires Python 3.10–3.13. MuJoCo 3.2+ comes in as a dependency; no GPU needed for the IL track.

**macOS (Apple Silicon or Intel):** works out of the box.

**Linux (Ubuntu 22.04 / 24.04):** CI-tested. Headless GL needs one apt-get line:

```bash
sudo apt-get install -y libosmesa6 libosmesa6-dev libgl1-mesa-dri
export MUJOCO_GL=osmesa    # or `egl` if a GPU is available
```

**Windows:** WSL2 + Ubuntu 22.04. Native Windows isn't supported.

### Bring your own…

- [Bring your own robot](docs/site/docs/guides/bring-your-own-robot.md) — URDF + sidecar YAML
- [Bring your own object](docs/site/docs/guides/bring-your-own-object.md) — YCB meshes + BYO OBJ
- [Bring your own task](docs/site/docs/guides/bring-your-own-task.md) — author a task YAML, randomize, score
- [Add a skill](docs/site/docs/guides/add-a-skill.md) — extend the agent's vocabulary

## Extras

Each extra is two lines: install the optional dependency, then run the command.

### Benchmark

```bash
uv run robo-sandbox-bench                           # run all default tasks
uv run robo-sandbox-bench --seeds 50                # randomize and aggregate
uv run robo-sandbox-bench --vlm-provider ollama     # use a real VLM
```

Tasks with a `randomize:` block get per-seed perturbations. Seed 0 is the deterministic baseline; seeds ≥ 1 apply uniform jitter keyed on the seed. With multiple seeds the summary reports `mean ± stderr`. Results append to `benchmark_results.json` locally for regression tracking (the file is gitignored).

Eight default tasks ship under `packages/robosandbox-core/src/robosandbox/tasks/definitions/` (plus one experimental):

| Task | What it exercises |
|---|---|
| `home` | Skill dispatch with no spatial reasoning |
| `pick_cube` | Single-object pick (core reliability) |
| `pick_cube_franka` | URDF-import path — bundled Franka picks a cube |
| `pick_cube_scrambled` | Pick under per-seed pose/size/mass/rgba randomization |
| `pick_from_three` | Perception disambiguation by colour name |
| `pick_ycb_mug` | Mesh-import path — bundled YCB mug picked by Franka |
| `pour_can_into_bowl` | Long-horizon composite (pick → pour) |
| `push_forward` | Non-pick manipulation, verifies directional displacement |
| `open_drawer` | First articulated primitive — drawer + `OpenDrawer` skill |

`_experimental_stack_two` is excluded from default runs because stacking is still open work.

### Browser live viewer

```bash
uv pip install -e 'packages/robosandbox-core[viewer]'
uv run robo-sandbox viewer
# open http://localhost:8000
```

Pick a task, click Run. Events log to the sidebar; frames stream at ~15–50 fps depending on how fast the sim is stepping. Pass `--task pick_cube_franka` to preload a specific scene, `--host 0.0.0.0` to expose it on your LAN.

### Documentation preview

```bash
uv pip install -e 'packages/robosandbox-core[docs]'
uv run mkdocs serve -f docs/site/mkdocs.yml           # live preview
uv run mkdocs build --strict -f docs/site/mkdocs.yml  # one-shot build
```

If you're reading this on GitHub, start at [`docs/site/docs/index.md`](docs/site/docs/index.md).

### Bring-your-own meshes

The sandbox decomposes user OBJ/STL files with CoACD and caches the hulls at `~/.cache/robosandbox/mesh_hulls/`:

```bash
uv pip install -e 'packages/robosandbox-core[meshes]'    # pulls in coacd
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

`collision="hull"` is a cheap fallback for already-convex meshes — no CoACD install required, but the sandbox does not compute a hull for you; it trusts the mesh is convex. For concave objects, always use `collision="coacd"`.

Pre-decompose once for a bundled asset with the authoring tool:

```bash
uv run python scripts/decompose_mesh.py \
  --input /path/to/drill.obj \
  --out-dir assets/objects/custom/drill \
  --name drill --mass 0.3 --center-bottom
```

## Bundled Assets

### Robots

`packages/robosandbox-core/src/robosandbox/assets/robots/franka_panda/` ships a trimmed copy of Franka Emika Panda adapted from [mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie) under Apache 2.0. Visual meshes removed (collision-only, ~160 KB); the tendon-driven gripper actuator was replaced with a simple position actuator on `finger_joint1` so the standard RobotSpec interface (open_qpos / closed_qpos) applies directly. See `LICENSE` in that directory for menagerie's attribution.

To bring your own robot:

```python
Scene(
    robot_urdf=Path("/path/to/ur5.urdf"),     # .urdf or .xml
    robot_config=Path("/path/to/ur5.robosandbox.yaml"),  # optional — sibling auto-discovered
    objects=(...),
)
```

The sidecar YAML tells RoboSandbox which joint is the primary finger, where the end-effector TCP sits, the home pose, and gripper open/closed qpos. See `packages/robosandbox-core/src/robosandbox/assets/robots/franka_panda/panda.robosandbox.yaml` for the schema.

### Objects

`packages/robosandbox-core/src/robosandbox/assets/objects/ycb/` ships 10 pre-decomposed YCB benchmark objects: a visual OBJ + N CoACD convex hulls + per-object sidecar YAML each.

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

See `packages/robosandbox-core/src/robosandbox/assets/objects/ycb/LICENSE` for the YCB project's terms.

## Architecture

The codebase is deliberately small. Most extension points are plain `Protocol`s, so the seams are easy to find and reason about.

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
│   ├── skills/           Pick, PlaceOn, Push, Home, Pour, Tap,
│   │                     OpenDrawer, CloseDrawer, Stack
│   ├── agent/            Planner protocol, VLMPlanner, StubPlanner,
│   │                     ReAct-style Agent with replan loop
│   ├── policy/           Policy protocol + LeRobotPolicyAdapter
│   ├── vlm/              OpenAI-compatible client + JSON recovery
│   ├── recorder/         MP4 + JSONL per episode; `export-lerobot` CLI
│   ├── backends/         RealRobotBackend (sim-to-real Protocol stub)
│   ├── tasks/            Task loader + benchmark runner
│   ├── cli.py            `robo-sandbox` entry point
│   ├── demo.py           Scripted pick (no VLM, no API)
│   └── agentic_demo.py   Full agent loop
└── tests/                Test suite covering types, IK, skills, agent,
                          planner, JSON recovery, VLM pointer projection,
                          URDF import, mesh import, policy adapter,
                          real-backend contract, reachability pre-flight.
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

One important seam is the planner:

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

`VLMPlanner` talks to an OpenAI-compatible endpoint with tool-calling and image input. `StubPlanner` is a regex parser.

### Skills as tools

Each skill exposes `name`, `description`, and a JSON `parameters_schema`. `VLMPlanner` turns that into tool definitions; the model's tool calls become skill dispatches. To add a skill, register it at the `robosandbox.skills` entry point.

## Status

Still early. Most moving parts are narrow `Protocol`s, so swapping in a different robot, object set, planner, recorder, or policy is a small integration job instead of a rewrite. Solid on pick/push/pour/drawer-style tasks today. Stacking is rougher than the rest and remains open work.

The [roadmap](docs/site/docs/reference/roadmap.md) is the best place to see what already ships and what's deferred. Short version: better stacking, collision-aware planning, a cleaner real-policy path, and a concrete SO-101 hardware backend are the main next steps.

## Development

```bash
uv sync --extra dev --extra viewer --extra meshes

uv run ruff check packages/
uv run pytest packages/robosandbox-core/tests/ -q
uv run robo-sandbox-bench --tasks pick_cube pick_cube_franka home pick_ycb_mug
```

These are the exact commands CI runs on every PR (see `.github/workflows/ci.yml`).

## License

Core: Apache 2.0.

Optional `contrib/` plugins carry their own licenses — research-licensed grasp predictors etc. live there; they are opt-in installs and not pulled in by the base source install from `packages/robosandbox-core`.
