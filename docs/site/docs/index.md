# RoboSandbox

Two checkpoints, same task. Which one actually got better?

In sim today, you can't tell. Different teams settle physics differently, run different numbers of trials, score success differently. The numbers don't compose. Read a 43% pick rate in one paper and a 51% in another and you're comparing weather reports from different planets.

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

That's the whole product. [What you can do](what-you-can-do.md) walks the three things you'll actually use it for. [Quickstart](quickstart.md) installs and records your first episode in five minutes.

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

<video controls preload="metadata" playsinline loop muted style="width: 100%; border-radius: 12px; margin: 1rem 0;">
  <source src="assets/demos/usb_a_insertion_zoom.mp4" type="video/mp4">
</video>

Four Franka arms, four different port positions, USB-A spec (1.5mm clearance), all four insertions complete. Same policy ran in classical MuJoCo at 32/32 with zero code change.

The `eval` CLI lives on `experimental/newton-eval` today and lands on `main` with the eval-and-recording-hygiene PR. Until that ships, reproducing any of the rows above means checking out the branch.

## What this isn't

A training framework. Train your policy elsewhere — `lerobot train`, your own RL loop, whatever — and bring the checkpoint here.

A photorealistic simulator. MuJoCo + Newton, no rendering tricks.

A drop-in for arbitrary policy frameworks. The `LeRobotPolicyAdapter` is what's shipped. Other frameworks need their own wrapper around the `Policy` protocol.

If you outgrow this and move to Isaac Sim or your team's internal stack, that's success.

## Where to go next

- [What you can do](what-you-can-do.md) — three concrete uses, one command + one artifact each.
- [Quickstart](quickstart.md) — install and record your first episode in five minutes.
- [Train ACT and eval it](tutorials/train-act-and-eval.md) — full IL recipe from 200 demos to a scored checkpoint.
- [The eval contract](concepts/the-eval-contract.md) — the schema and the seven invariants in code.
- [Compared to other tools](comparisons.md) — when to pick this vs LeRobot, IsaacLab, RoboCasa, robosuite.
- [Eval for world models](concepts/eval-for-world-models.md) — where this fits as the world-model wave (World Labs, AMI, Cosmos, V-JEPA-2) ships learned simulators.
- [Where things are](where-things-are.md) — map of which page and which branch covers each thing.
- [The RL track](concepts/rl-track.md) — what the experimental branch ships, what works, what's open.
- [Why this exists](concepts/why-robosandbox-exists.md) — the longer essay.

## What's in the box on `main`

MuJoCo physics, built-in 6-DOF arm, bundled Franka Panda (URDF). Nine skills (`pick`, `place_on`, `push`, `home`, `pour`, `tap`, `open_drawer`, `close_drawer`, `stack`). Eight default benchmark tasks plus one experimental. Ten pre-decomposed YCB objects, drop into a task with `@ycb:<id>`. Browser live viewer with record + keyboard teleop. LeRobot v3 parquet export and `LeRobotPolicyAdapter` for policy replay.

Real-robot bridge is a stub today (`RealRobotBackend` Protocol). Observation, teleop, and policy rollouts carry over; `Pick`, `PlaceOn`, and `Push` still depend on MuJoCo kinematics. Status and the integration shape are in [Sim-to-real handoff](tutorials/sim-to-real-handoff.md).

See the [roadmap](reference/roadmap.md) for what's coming next.

## Platforms

macOS (Apple Silicon or Intel) works out of the box. Linux (Ubuntu 22.04 / 24.04) is the CI-tested platform — headless GL needs one apt-get line, see [Quickstart](quickstart.md). Windows isn't directly supported; WSL2 + Ubuntu 22.04 works.
