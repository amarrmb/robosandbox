# Where things are

A short map of which page (and which branch) covers each thing you
might want to do. RoboSandbox has both an **IL track** (record demos,
train ACT, eval) and an experimental **RL track** (PPO, Newton parallel
worlds). They live in different places.

## By task

| If you want to… | Read | Branch |
|---|---|---|
| Open the browser viewer + run a scripted task | [Quickstart](quickstart.md) | `main` |
| Record demos + export to LeRobot v3 | [LeRobot export](tutorials/lerobot-export.md) | `main` |
| Train an ACT policy on those demos and score it | [Train ACT and eval it](tutorials/train-act-and-eval.md) | `main` (eval CLI lands with eval-and-recording-hygiene PR) |
| Replay a *public* LeRobot checkpoint (cross-embodiment) | [LeRobot policy replay](tutorials/lerobot-policy-replay.md) | `main` |
| Iterate on a policy using `spatial_breakdown` | [Iterating on a policy](guides/iterating-on-a-policy.md) | `main` (eval CLI required) |
| Take a sim-validated skill to real hardware | [Sim-to-real handoff](tutorials/sim-to-real-handoff.md) | `main` |
| Bring your own arm | [Bring your own robot](guides/bring-your-own-robot.md) | `main` |
| Bring your own task | [Bring your own task](guides/bring-your-own-task.md) | `main` |
| Run RL (PPO + Newton parallel worlds) | [The RL track](concepts/rl-track.md) | `experimental/newton-eval` |
| Score against a learned world model | [World model as a sim backend](tutorials/world-model-as-sim-backend.md) + [Eval for world models](concepts/eval-for-world-models.md) | `experimental/newton-eval` (lands on main with the eval-and-recording-hygiene PR) |
| Distill an ACT policy into a state-only MLP | `scripts/distill_act_to_mlp.py` | `experimental/newton-eval` |
| Run cross-sim transfer (Newton → MuJoCo) | `scripts/eval_insert_1024.py`, `scripts/cross_sim_*` | `experimental/newton-eval` |

## On the IL ↔ RL bridge

The natural mental model is "train ACT → distill → warm-start PPO →
fine-tune." That path is **wired but currently fails on contact-rich
pick** because of a structural gravity-settling difference between
Newton and classical MuJoCo (~12 mrad steady-state residual at the
joint, ~6 mm at the end-effector — beyond grasp tolerance for a 24 mm
cube). The distilled MLP gets 34% in MuJoCo and 0% in Newton; PPO
fine-tuning then has no positive reward signal and wipes the prior.

Concretely:

- The **IL track** ends at `robo-sandbox eval` in MuJoCo. That works.
- The **RL track** (PPO from scratch with curriculum on Newton) works
  for tasks Newton's solver handles well — reach (1024 worlds, 100%
  success) and connector insertion (4mm → 2mm → 1.5mm USB-A
  curriculum, 100% at 64 worlds, 96.5% at 1024 worlds, 100% cross-sim
  on classical MuJoCo at n=32).
- They are **sibling tracks**, not a sequential pipeline. Don't expect
  to chain them through warm-start on contact-rich pick today.

If you're researching the gap, see [The RL track](concepts/rl-track.md)
for the structural cause + reproduction commands, and the
`scripts/probe_*` diagnostics on `experimental/newton-eval`.

## On branches

`main` is the public, stable surface. The eval-and-recording-hygiene
subset of the experimental branch lands on `main` as a single PR; the
RL track stays on `experimental/newton-eval` until further notice.
Pages above that say *"branch:* `experimental/newton-eval`*"* are not
yet on the public release.
