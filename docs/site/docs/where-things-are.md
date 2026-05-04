# Where things are

Map of which page (and which branch) covers each thing. There are two tracks: an **IL track** (record demos, train ACT externally, evaluate) and an experimental **RL track** (PPO + Newton parallel worlds). They live in different places.

## By task

| Task | Page | Branch |
|---|---|---|
| Open the browser viewer + run a scripted task | [Quickstart](quickstart.md) | `main` |
| Record demos + export to LeRobot v3 | [LeRobot export](tutorials/lerobot-export.md) | `main` |
| Train an ACT policy and score it | [Train ACT and eval it](tutorials/train-act-and-eval.md) | `main` (eval CLI lands with eval-and-recording-hygiene PR) |
| Replay a public LeRobot checkpoint (cross-embodiment) | [LeRobot policy replay](tutorials/lerobot-policy-replay.md) | `main` |
| Iterate on a policy using `spatial_breakdown` | [Iterating on a policy](guides/iterating-on-a-policy.md) | `main` (eval CLI required) |
| Take a sim-validated skill to real hardware | [Sim-to-real handoff](tutorials/sim-to-real-handoff.md) | `main` (skeleton, see status note on page) |
| Bring your own arm | [Bring your own robot](guides/bring-your-own-robot.md) | `main` |
| Bring your own task | [Bring your own task](guides/bring-your-own-task.md) | `main` |
| Run RL (PPO + Newton parallel worlds) | [The RL track](concepts/rl-track.md) | `experimental/newton-eval` |
| Score against a learned world model | [World model as a sim backend](tutorials/world-model-as-sim-backend.md), [Eval for world models](concepts/eval-for-world-models.md) | `experimental/newton-eval` |
| Distill an ACT policy into a state-only MLP | `scripts/distill_act_to_mlp.py` | `experimental/newton-eval` |
| Run cross-sim transfer (Newton → MuJoCo) | `scripts/eval_insert_1024.py`, `scripts/cross_sim_*` | `experimental/newton-eval` |

## On the IL ↔ RL bridge

The natural mental model is *"train ACT → distill → warm-start PPO → fine-tune."* That path is wired but currently fails on contact-rich pick. Newton's mujoco_warp solver settles arm joints under gravity at ~22 mrad off the IK target; classical MuJoCo settles much closer. Gravity feedforward via `mj_rne` cuts that to ~12 mrad, but the residual is still ~6 mm at the end-effector — beyond grasp tolerance for a 24 mm cube.

What this means for the two tracks:

- **IL track ends at `robo-sandbox eval` in MuJoCo.** That works.
- **RL track is from-scratch curriculum PPO on tasks Newton solves cleanly** — reach (1024 worlds, 100% success) and connector insertion (4mm → 2mm → 1.5mm USB-A curriculum, 100% at 64 worlds, 96.5% at 1024 worlds, 100% cross-sim on classical MuJoCo at n=32).
- **They're sibling tracks, not a sequential pipeline.** Don't expect to chain them through warm-start on contact-rich pick today.

For the structural cause and reproduction commands, see [The RL track](concepts/rl-track.md). The diagnostics live in `scripts/probe_*` on `experimental/newton-eval`.

## On branches

`main` is the public, stable surface. The eval-and-recording-hygiene subset of `experimental/newton-eval` ports the eval CLI and Newton-as-a-backend onto `main` as a single PR. The RL training pieces (`rl/` module, `train` CLI, eval-log lineage, `results` subcommands) stay on `experimental/newton-eval`. Pages tagged `experimental/newton-eval` above need that branch checked out to reproduce.
