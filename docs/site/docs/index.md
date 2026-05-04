# RoboSandbox

RoboSandbox is an evaluation harness for manipulation policies. It runs a LeRobot-compatible checkpoint against a task in MuJoCo or Newton and writes a JSON.

The JSON contains:

- success rate
- Wilson 95% CI on the rate
- per-trial details (initial object pose, peak lift, EE-object min distance, success step)
- per-position breakdown of where the policy failed
- provenance (checkpoint sha256, robosandbox git rev, lerobot/mujoco/torch versions, full CLI args)

The shape is the same for a single MuJoCo trial or 1024 parallel Newton trials.

```bash
robo-sandbox eval \
    --task pick_cube_franka_random \
    --policy outputs/act_franka_pick/checkpoints/050000/pretrained_model \
    --sim-backend mujoco \
    --n-trials 64 --action-repeat 6 --settle-steps 60 \
    --output outputs/eval_50k.json
# 28/64 (43.8%), CI [32.3, 55.9], spatial_breakdown by cube x/y
```

[What you can do](what-you-can-do.md) walks the three concrete uses. [Quickstart](quickstart.md) installs and records the first episode in five minutes.

## What's scored today

Policy support is `LeRobotPolicyAdapter`. The `Policy` protocol is framework-agnostic. Wrappers for Diffusion Policy, Octo, and π0 are a few hours of glue each, but they aren't shipped.

Numbers from real runs on this branch:

| Run | Result |
|---|---|
| ACT 50k on `pick_cube_franka_random`, n=64, MuJoCo | 28/64 (43.8%), CI [32.3, 55.9] |
| Reach, 1024 parallel Newton worlds | 1024/1024 (100%) |
| USB-A 1.5mm insertion, 64 Newton worlds | 64/64 (100%) |
| USB-A 1.5mm insertion, 1024 Newton worlds | 988/1024 (96.5%), CI [95.2, 97.4] |
| Same insertion policy, replayed in classical MuJoCo, n=32 | 32/32 (100%), CI [89.3, 100.0] |

The cross-sim row is the load-bearing one. A policy trained in Newton runs in classical MuJoCo at full success without re-training or code changes. Without sim-to-sim transfer, sim numbers in isolation don't tell you much.

<video controls preload="metadata" playsinline loop muted style="width: 100%; border-radius: 12px; margin: 1rem 0;">
  <source src="assets/demos/usb_a_insertion_zoom.mp4" type="video/mp4">
</video>

The video shows four Franka arms picking USB-A connectors (1.5mm clearance) from four different port positions. All four insertions complete. The same policy then ran in classical MuJoCo at 32/32 with no code changes.

The `eval` CLI lives on `experimental/newton-eval` today. It lands on `main` with the eval-and-recording-hygiene PR. Reproducing the rows above needs that branch checked out.

## What this isn't

- **A training framework.** Train policies elsewhere (`lerobot train`, your own RL loop, whatever). Bring the checkpoint here.
- **A photorealistic simulator.** MuJoCo + Newton, no rendering tricks. Wrong tool if the policy needs realistic textures or lighting.
- **A drop-in for arbitrary policy frameworks.** Only `LeRobotPolicyAdapter` ships. Other frameworks need their own wrapper around the `Policy` protocol.
- **A multi-robot stack.** Single robot per scene.

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
