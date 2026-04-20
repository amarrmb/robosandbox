# RoboSandbox

> A sim-first sandbox for robot manipulation.
> **Bring your own arm, objects, and tasks.**

!!! note "v0.1 is Linux-first"
    Developed and CI-tested on Ubuntu 22.04/24.04 with Python
    3.11/3.12/3.13. macOS and Windows are not regression-gated —
    platform-specific issues (headless GL, Apple Silicon MuJoCo
    wheels, Windows paths) are not tracked. See the
    [Quickstart](quickstart.md) for the exact apt-get line the
    viewer and rendering tests need.

RoboSandbox is for building and testing small manipulation loops in
simulation without needing a giant stack around them. Load a robot,
drop in a few objects, give the system a task, and it will plan and run
through a skill-based loop in MuJoCo. If you want to record episodes or
export them for policy work later, that is part of the same flow.

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

## Three quick ways in

All three paths below use the same
[agent loop](concepts/skills-and-agents.md). The only thing that changes
is where the planner gets its intelligence from.

### 1. Zero setup: stub planner

No API key, no model download, just the built-in parser.

```bash
uv run robo-sandbox run "pick up the red cube"
```

The [stub planner](concepts/skills-and-agents.md#stubplanner) handles a
small grammar:

- `pick (up) the <obj>`
- `pick (up) the <obj> (and|then|,) (put|place) (it) on (the) <obj2>`
- `stack <obj> on (top of) <obj2>`
- `push the <obj> forward|back|left|right`
- `pour <obj> into <obj2>`
- `tap/press the <obj>`
- `open/close the <drawer>`
- `(go) home`

### 2. Local model: Ollama

```bash
ollama pull llama3.2-vision
ollama serve &
uv run robo-sandbox run --vlm-provider ollama \
  "pick up the blue cube and put it on the green cube"
```

Any OpenAI-compatible local endpoint works here too; use `--base-url`
if you want something other than the default Ollama setup.

### 3. Hosted model: OpenAI

```bash
export OPENAI_API_KEY=sk-...
uv run robo-sandbox run --vlm-provider openai \
  "stack all three cubes by colour — red on green on blue"
```

Default model is `gpt-4o-mini` (~$0.002 per episode). Use `--model
gpt-4o` for harder plans.

## Where to go next

- **[Quickstart](quickstart.md)** — install, run the benchmark, open
  the viewer, record an episode. 5 minutes end-to-end.
- **Concepts** — [Scenes & objects](concepts/scenes.md), [Skills & agents](concepts/skills-and-agents.md), [Perception & grasping](concepts/perception-and-grasping.md), [Recording & export](concepts/recording-and-export.md), [Real-robot bridge](concepts/real-robot.md).
- **Tutorials** — [Custom arm](tutorials/custom-arm.md), [Custom task](tutorials/custom-task.md), [Custom skill](tutorials/custom-skill.md), [Policy replay](tutorials/policy-replay.md).
- **Reference** — [CLI](reference/cli.md), [API](reference/api.md),
  [roadmap](reference/roadmap.md).

## What ships in v0.1

- MuJoCo physics backend + built-in 6-DOF arm + bundled Franka Panda
  (URDF import path).
- 9 skills: `pick`, `place_on`, `push`, `home`, `pour`, `tap`,
  `open_drawer`, `close_drawer`, `stack`.
- 9 default benchmark tasks including a long-horizon
  `pour_can_into_bowl` and an articulated-drawer primitive.
- 10 bundled YCB objects, drop-in via `@ycb:<id>`.
- Browser live viewer with record + keyboard teleop.
- LeRobot v3 parquet export + policy replay loop.
- Real-robot bridge stub — subclass, fill the hardware driver;
  observation+step skills (`Home`, teleop, policy rollouts) carry
  over unchanged. Motion-planning skills (`Pick`, `PlaceOn`, `Push`)
  still depend on MuJoCo kinematics — see the
  [sim-to-real handoff tutorial](tutorials/sim-to-real-handoff.md).

See the [roadmap](reference/roadmap.md) for what is coming next.
