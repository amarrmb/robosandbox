# RoboSandbox

> Score manipulation policies against a reproducible eval contract.
> Today: any LeRobot-compatible checkpoint, MuJoCo + Newton backends.

Point a checkpoint at a task. Get back a JSON: success rate, Wilson 95%
CI, per-position breakdown of where it failed, full provenance
(checkpoint sha, git rev, library versions). Same contract whether you
ran one trial in MuJoCo or a thousand parallel trials in Newton.

```bash
robo-sandbox eval \
    --task pick_cube_franka_random \
    --policy outputs/act_franka_pick/checkpoints/050000/pretrained_model \
    --sim-backend mujoco \
    --n-trials 64 --action-repeat 6 --settle-steps 60 \
    --output outputs/eval_50k.json
# → 28/64 (43.8%), CI [32.3, 55.9], spatial_breakdown by cube x/y
```

Start with **[What you can do](what-you-can-do.md)** for three concrete
scenarios, or **[Quickstart](quickstart.md)** to install and record
your first episode in 5 minutes.

## What works today

| | Result | Where |
|---|---|---|
| ACT policy on `pick_cube_franka_random` (n=64) | 43.8%, CI [32.3, 55.9] | `main` — see [Train ACT and eval it](tutorials/train-act-and-eval.md) |
| Reach, 1024 parallel Newton worlds | 100% (1024/1024) | branch `experimental/newton-eval` |
| USB-A 1.5mm insertion, 64 worlds | 100% (64/64) | branch `experimental/newton-eval` |
| USB-A 1.5mm insertion, 1024 worlds | 96.5%, CI [95.2, 97.4] | branch `experimental/newton-eval` |
| Same insertion policy, classical MuJoCo (zero code change) | 100% (32/32), CI [89.3, 100.0] | branch `experimental/newton-eval` |

The cross-sim row is the headline. A policy trained on Newton runs in
classical MuJoCo at full success without re-training. That's the
precondition for trusting any sim number at all — without it, every
sim demo is just sim theater.

<video controls preload="metadata" playsinline loop muted style="width: 100%; border-radius: 12px; margin: 1rem 0;">
  <source src="assets/demos/usb_a_insertion_zoom.mp4" type="video/mp4">
</video>

*Four Franka arms, four different port positions, USB-A spec (1.5mm
clearance), all four insertions complete. Same policy ran in classical
MuJoCo at 32/32 with zero code change.*

!!! info "Reproducing the branch rows"
    The four `experimental/newton-eval` rows above need that branch
    checked out — the `train` CLI, Newton parallel backend, and
    cross-sim harness live there. The `eval` CLI itself lands on
    `main` with the eval-and-recording-hygiene PR. Until that ships,
    the IL row (`pick_cube_franka_random`) is reproducible only on
    the branch as well; afterward it'll work straight from `main`.

The RL track that produced the bottom four rows is intentionally
off-thesis — see [The RL track](concepts/rl-track.md). The IL track is
the recommended on-ramp.

## Scope

What this **is**:

- A reproducible eval contract for manipulation policies
- A workflow for recording demos and exporting them to LeRobot v3
- A pluggable sim layer (MuJoCo single-world, Newton parallel-world)

What this is **not**:

- A training framework. Train elsewhere (e.g. `lerobot train`) and
  bring the checkpoint here.
- A photorealistic simulator. MuJoCo + Newton, no rendering tricks.
- A drop-in for arbitrary policy frameworks. The only adapter shipped
  today is `LeRobotPolicyAdapter`. Other frameworks need a thin
  wrapper around the `Policy` protocol.

If you outgrow this and move to Isaac Sim or your team's internal
stack, that's success.

!!! info "Platform support"
    **macOS** (Apple Silicon / Intel): works out of the box.
    **Linux** (Ubuntu 22.04/24.04): one `apt-get` line for headless GL
    (see [Quickstart](quickstart.md)). CI-tested.
    **Windows**: WSL2 + Ubuntu 22.04 only.

## How a single eval flows

```
checkpoint + task YAML
       │
       ▼
 sim backend (MuJoCo single-world OR Newton 1–1024 worlds)
       │
       ▼
 policy adapter (LeRobotPolicyAdapter today)
       │
       ▼
 run_policy / run_eval_parallel
       │
       ▼
 eval JSON: rate, Wilson CI, spatial_breakdown, per-trial, provenance
```

## Where to go next

- **[What you can do](what-you-can-do.md)** — three concrete scenarios
  (score a checkpoint, compare two checkpoints, find the failing
  workspace slice). One command + one artifact each.
- **[The eval contract](concepts/the-eval-contract.md)** — the schema +
  the seven invariants that make two evals comparable.
- **[Compared to other tools](comparisons.md)** — table vs LeRobot,
  IsaacLab, RoboCasa, robosuite. Honest about losses.
- **[Eval for world models](concepts/eval-for-world-models.md)** —
  where this fits relative to World Labs, AMI Labs, NVIDIA Cosmos,
  and V-JEPA-2. Includes a working `--sim-backend world_model` slot.
- **[Quickstart](quickstart.md)** — install, open the viewer, record
  one episode. 5 minutes.
- **[Train ACT and eval it](tutorials/train-act-and-eval.md)** — full
  IL recipe: 200 demos → external `lerobot train` → `robo-sandbox eval`.
- **[Where things are](where-things-are.md)** — map of which page (and
  which branch) covers each thing.
- **[Why this exists](concepts/why-robosandbox-exists.md)** — the
  longer essay if you want the framing, not the commands.
- **[The RL track](concepts/rl-track.md)** — what the
  `experimental/newton-eval` branch ships, what works, what's open.

## What's in the box on `main`

- MuJoCo physics, built-in 6-DOF arm, bundled Franka Panda (URDF).
- 9 skills (`pick`, `place_on`, `push`, `home`, `pour`, `tap`,
  `open_drawer`, `close_drawer`, `stack`) and 8 default benchmark tasks.
- 10 pre-decomposed YCB objects; drop into a task with `@ycb:<id>`.
- Browser live viewer with record + keyboard teleop.
- LeRobot v3 parquet export + `LeRobotPolicyAdapter` for policy replay.
- Real-robot bridge **stub** (`RealRobotBackend` Protocol). Observation
  + teleop + policy rollouts carry over; `Pick` / `PlaceOn` / `Push`
  still depend on MuJoCo kinematics. See [sim-to-real
  handoff](tutorials/sim-to-real-handoff.md) for the current state.

See the [roadmap](reference/roadmap.md) for what's coming next.
