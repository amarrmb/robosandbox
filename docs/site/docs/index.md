# RoboSandbox

> Sim-first agentic manipulation sandbox.
> **Any arm. Any object. Any command.**

!!! note "v0.1 is Linux-first"
    Developed and CI-tested on Ubuntu 22.04/24.04 with Python
    3.11/3.12/3.13. macOS and Windows are not regression-gated —
    platform-specific issues (headless GL, Apple Silicon MuJoCo
    wheels, Windows paths) are not tracked. See the
    [Quickstart](quickstart.md) for the exact apt-get line the
    viewer and rendering tests need.

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

## Three quickstart paths

Pick whichever matches what you have installed. All three drive the
same [agent loop](concepts/skills-and-agents.md).

### 1. Zero setup — rule-based stub planner

No API key, no model download.

```bash
uv run robo-sandbox run "pick up the red cube"
```

The [stub planner](concepts/skills-and-agents.md#stubplanner) parses a
small grammar:

- `pick (up) the <obj>`
- `pick (up) the <obj> (and|then|,) (put|place) (it) on (the) <obj2>`
- `stack <obj> on (top of) <obj2>`
- `push the <obj> forward|back|left|right`
- `pour <obj> into <obj2>`
- `tap/press the <obj>`
- `open/close the <drawer>`
- `(go) home`

### 2. Local, free — Ollama

```bash
ollama pull llama3.2-vision
ollama serve &
uv run robo-sandbox run --vlm-provider ollama \
  "pick up the blue cube and put it on the green cube"
```

Any OpenAI-compatible endpoint works — override with `--base-url`.

### 3. Cloud — OpenAI

```bash
export OPENAI_API_KEY=sk-...
uv run robo-sandbox run --vlm-provider openai \
  "stack all three cubes by colour — red on green on blue"
```

Default model is `gpt-4o-mini` (~$0.002 per episode). Use `--model
gpt-4o` for harder plans.

---

## Where to go next

- **[Quickstart](quickstart.md)** — install, run the benchmark, open
  the viewer, record an episode. 5 minutes end-to-end.
- **Concepts** — the architecture one page at a time:
    - [Scenes & objects](concepts/scenes.md)
    - [Skills & agents](concepts/skills-and-agents.md)
    - [Perception & grasping](concepts/perception-and-grasping.md)
    - [Recording & export](concepts/recording-and-export.md)
    - [Real-robot bridge](concepts/real-robot.md)
- **Tutorials** — task-driven walk-throughs:
    - [Custom arm](tutorials/custom-arm.md) — drop any URDF.
    - [Custom task](tutorials/custom-task.md) — author YAML, run it.
    - [Custom skill](tutorials/custom-skill.md) — extend the action vocabulary.
    - [Policy replay](tutorials/policy-replay.md) — record → parquet → replay.
- **Reference** — [CLI](reference/cli.md), [API](reference/api.md),
  [roadmap](reference/roadmap.md).

## What ships in v0.1

- MuJoCo physics backend + built-in 6-DOF arm + bundled Franka Panda
  (URDF import path).
- 9 skills: `pick`, `place_on`, `push`, `home`, `pour`, `tap`,
  `open_drawer`, `close_drawer`, `stack`.
- 8 default benchmark tasks including a long-horizon
  `pour_can_into_bowl`.
- 10 bundled YCB objects, drop-in via `@ycb:<id>`.
- Browser live viewer with record + keyboard teleop.
- LeRobot v3 parquet export + policy replay loop.
- Real-robot bridge stub — subclass, fill hardware driver, skills
  work unchanged.

See the [roadmap](reference/roadmap.md) for what's next.
