# RoboSandbox

> A sim-first sandbox for robot manipulation.
> **Bring your own arm, objects, and tasks.**

!!! info "Platform support"
    **macOS** (Apple Silicon and Intel) works out of the box — no GL configuration needed.
    **Linux** (Ubuntu 22.04/24.04) is the CI-tested platform; headless rendering needs one `apt-get` line (see [Quickstart](quickstart.md)).
    **Windows** is not directly supported; WSL2 running Ubuntu 22.04 works.

RoboSandbox is a small manipulation sandbox for building and testing
manipulation loops without needing a giant stack around them. Load a
robot, drop in a few objects, define a task, run a planner or policy,
and record the result. If you want to export episodes for policy work
later, that is part of the same flow.

<video controls preload="metadata" playsinline style="width: 100%; border-radius: 12px; margin: 1rem 0;">
  <source src="assets/demos/robosandbox_teaser_phase.mp4" type="video/mp4">
</video>

## Why This Project Exists

RoboSandbox is a small manipulation sandbox for learning, prototyping,
and integration work.

It exists for the gap between toy demos and heavyweight robotics
stacks. You can bring in a robot, define a task, run a planner or
policy, record the result, and inspect the interfaces without a large
simulator setup.

The project is deliberately scoped. It is meant to help you understand
the workflow, try ideas quickly, and make the seams between robot, task,
skills, recorder, and policy visible. It is not trying to be the final
simulator you use forever.

If you outgrow RoboSandbox and move to MuJoCo, Isaac Sim, LeRobot
training pipelines, or real hardware, that is success, not failure.

## Who This Is For

RoboSandbox is a good fit if you are:

- learning how a manipulation stack fits together
- doing robotics work but want a lighter-weight way to prototype in simulation
- already comfortable with simulation and need a small, hackable integration harness

It is especially useful when you want to answer questions like:

- How do I add a new robot?
- How do I describe a task?
- What does a policy need to consume and emit?
- What gets recorded and exported?
- What breaks when I swap embodiments?

## When To Move Beyond It

RoboSandbox is a starting point, not an end state.

You may want to move beyond it when you need:

- lower-level simulator control than the current abstractions expose
- photorealism or richer sensor simulation
- large-scale industrial workflows
- large scenes or more complex multi-robot environments
- production deployment infrastructure

The intended path is simple: start small, understand the workflow,
validate the seams, then move to a heavier stack when your requirements
become sharper.

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

## Evaluating Trained Policies

Beyond the scripted demo, there is an evaluation harness for trained
policies. `robo-sandbox eval` takes a checkpoint and a task and writes
a JSON with the success rate, a Wilson 95% CI on the rate, per-trial
details (initial object pose, peak lift, EE-object min distance, success
step), a spatial breakdown of where the policy failed, and provenance
(checkpoint sha256, robosandbox git rev, lerobot/mujoco/torch versions,
full CLI args). The JSON shape is the same for one MuJoCo trial or
1024 parallel Newton trials, which is what makes two evals actually
comparable.

```bash
robo-sandbox eval \
    --task pick_cube_franka_random \
    --policy outputs/act_franka_pick/checkpoints/050000/pretrained_model \
    --sim-backend mujoco \
    --n-trials 64 --action-repeat 6 --settle-steps 60 \
    --output outputs/eval_50k.json
```

The full IL recipe is in
[Train ACT and eval it](tutorials/train-act-and-eval.md): 200
randomized scripted demos, an external `lerobot train` for the ACT
policy, and a 64-trial eval. On `pick_cube_franka_random`, the 50k
checkpoint scores 28/64 (43.8%) with a Wilson CI of [32.3, 55.9]. The
schema and the seven enforced invariants behind the contract are in
[The eval contract](concepts/the-eval-contract.md).

The only policy adapter we ship is `LeRobotPolicyAdapter`. The
`Policy` protocol is framework-agnostic, but we do not ship adapters
for Diffusion Policy, Octo, π0, or other frameworks; those need a
thin user-side wrapper. We also do not ship a real-robot driver:
`RealRobotBackend` is a Protocol with a software skeleton, and the
SO-101 backend tracks commanded state in memory rather than driving
hardware.

The eval CLI, the Newton GPU backend, the cross-sim transfer harness,
and the experimental world-model backend live on the
`experimental/newton-eval` branch today and land on `main` with the
eval-and-recording-hygiene PR. Reproducing any of the deeper results
needs that branch checked out.

## Get started

**Open the browser viewer** — no API key, no model download.

```bash
uv run robo-sandbox viewer
# → open http://localhost:8000
```

Pick a task, type a command like `pick up the red cube`, click **Run**.
The arm plans and executes while frames stream to your browser. Hit
**Record** before running to save the episode to disk for training.

The built-in planner understands a small grammar — pick, place, push,
pour, stack, open/close drawer, go home — which is enough to exercise
the full [agent loop](concepts/skills-and-agents.md). See the
**[Quickstart](quickstart.md)** for the install steps and the
record → export → train flow.

**Want richer natural language?** Plug in a VLM for free-form commands
and visual scene reasoning:

=== "Ollama (local, no API key)"

    ```bash
    ollama pull llama3.2-vision && ollama serve &
    uv run robo-sandbox run --vlm-provider ollama \
      "pick up the blue cube and put it on the green cube"
    ```

=== "OpenAI (hosted)"

    ```bash
    export OPENAI_API_KEY=sk-...
    uv run robo-sandbox run --vlm-provider openai \
      "stack all three cubes by colour — red on green on blue"
    ```

Both use the same agent loop — only the planner changes.

## Where to go next

- **[Quickstart](quickstart.md)** — install, open the viewer, record an
  episode. 5 minutes end-to-end.
- **Start here if you want the product thesis first** —
  [Why RoboSandbox exists](concepts/why-robosandbox-exists.md) explains
  the robot loop, where modern model families fit, and what problem
  RoboSandbox is actually trying to solve.
- **Where things are** — [a short map](where-things-are.md) of which
  page (and which branch) covers each thing, especially useful for
  features split between `main` and `experimental/newton-eval`.
- **Compared to other tools** — [a side-by-side](comparisons.md) with
  LeRobot, IsaacLab, RoboCasa, and robosuite, including what we don't
  do.
- **Concepts** — [Scenes & objects](concepts/scenes.md), [Skills & agents](concepts/skills-and-agents.md), [Perception & grasping](concepts/perception-and-grasping.md), [Recording & export](concepts/recording-and-export.md), [Real-robot bridge](concepts/real-robot.md), [The eval contract](concepts/the-eval-contract.md), [Eval for world models](concepts/eval-for-world-models.md), [The RL track](concepts/rl-track.md).
- **Tutorials** — [Custom arm](tutorials/custom-arm.md), [Custom task](tutorials/custom-task.md), [Custom skill](tutorials/custom-skill.md), [Policy replay](tutorials/policy-replay.md), [Train ACT and eval it](tutorials/train-act-and-eval.md), [World model as a sim backend](tutorials/world-model-as-sim-backend.md).
- **Reference** — [CLI](reference/cli.md), [API](reference/api.md),
  [roadmap](reference/roadmap.md).

## What ships in v0.1

- MuJoCo physics backend + built-in 6-DOF arm + bundled Franka Panda
  (URDF import path).
- 9 skills: `pick`, `place_on`, `push`, `home`, `pour`, `tap`,
  `open_drawer`, `close_drawer`, `stack`.
- 8 default benchmark tasks + 1 experimental, including a long-horizon
  `pour_can_into_bowl` and an articulated-drawer primitive.
- 10 bundled YCB objects, drop-in via `@ycb:<id>`.
- Browser live viewer with record + keyboard teleop.
- LeRobot v3 parquet export + policy replay loop.
- Real-robot bridge stub — subclass, fill the hardware driver;
  observation+step skills (`Home`, teleop, policy rollouts) carry
  over unchanged. Motion-planning skills (`Pick`, `PlaceOn`, `Push`)
  still depend on MuJoCo kinematics — see the
  [sim-to-real handoff tutorial](tutorials/sim-to-real-handoff.md).

The eval CLI, the Newton GPU backend, the cross-sim transfer harness,
and the experimental world-model backend live on
`experimental/newton-eval` and ride along with the
eval-and-recording-hygiene PR onto `main`. The RL training pieces (a
`train` CLI, `rl/` module, eval-log lineage, `results` subcommands)
stay on the experimental branch — training models is intentionally
out of scope for the project.

See the [roadmap](reference/roadmap.md) for what is coming next.
