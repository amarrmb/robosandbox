# The RL Track (Experimental)

RoboSandbox has an experimental reinforcement-learning track that
lives on the `experimental/newton-eval` branch. It is a sibling to
the IL track, not a step that follows it. This page is the place to
decide whether to look at the branch at all, and what to expect when
you do.

## Off-Thesis on Purpose

The product thesis, spelled out in
[Why RoboSandbox Exists](why-robosandbox-exists.md), is that
RoboSandbox does not train models. The RL track intentionally
violates that thesis to integrate Newton's GPU parallel-world
simulator. It stays on the experimental branch and out of `main`
until either the thesis changes or the work matures into a separable
package. We are not currently planning to merge it.

The reason it exists at all is that Newton parallel sim is genuinely
useful for evaluating policies at scale (1024 worlds in one run), and
the easiest way to validate the integration was to actually train a
policy in it from scratch. The training code is the proof that the
backend works end-to-end. It isn't the product.

## What's on the Branch

The branch ships PPO with end-effector-space Cartesian actions via
DLS pseudoinverse, in `packages/robosandbox-core/src/robosandbox/rl/`.
It targets state-only observations and runs on Newton. There's a
`robo-sandbox train` CLI subcommand, with curriculum support via
`--warm-start <prior-actor>` for chaining policies through clearance
or scale increments.

The Newton parallel-worlds backend supports `world_count` from 1 to
1024+, with optional batched RGB via `SensorTiledCamera` (off by
default to keep state-only RL fast). An eval-log lineage system
(`policies.jsonl`, `evals.jsonl`) and `robo-sandbox results
slice|compare|regression` subcommands sit on top, plus a cross-sim
transfer harness that re-runs a Newton-trained policy in classical
MuJoCo without code changes.

## What Works

These numbers were recorded on the experimental branch on a single
DGX Spark, single GB10. Each one is reproducible from the commands
later on this page.

| Task | Backend | Workers | Result |
|---|---|---|---|
| `reach_target_franka` | Newton | 1024 | 100% (1024/1024) |
| `insert_connector_franka` 4 mm | Newton | 64 | 98.4% (63/64) |
| `insert_connector_franka` 2 mm | Newton | 64 | 100% (64/64) |
| `insert_connector_franka` 1.5 mm USB-A | Newton | 64 | 100% (64/64) |
| `insert_connector_franka` 1.5 mm USB-A | Newton | 1024 | 96.5% sustained, CI [95.2, 97.4] |
| `insert_connector_franka` 1.5 mm USB-A | classical MuJoCo (transfer) | 32 | 100% (32/32), CI [89.3, 100.0] |

The insertion curriculum is 4 mm → 2 mm → 1.5 mm, each stage
warm-started from the previous. The cross-sim row is the load-bearing
one: a Newton-trained policy runs in classical MuJoCo at full success
without re-training, which is the precondition for the sim numbers
to mean anything outside Newton.

## What's Open: The IL ↔ RL Bridge

The natural mental model for combining the two tracks is *train ACT
in MuJoCo, distill to a state-only MLP, warm-start PPO in Newton,
fine-tune.* That path is wired but currently fails on contact-rich
pick (`pick_cube_franka`).

The distilled state-only MLP from a 50k-step ACT scores 34% in MuJoCo
and 0% in Newton. A 200k env-step PPO fine-tune of that warm-start
in Newton wipes the prior — there is no positive reward signal in
Newton, so the policy drifts randomly and the MuJoCo skill is
destroyed. The result is 0% in MuJoCo and 0% in Newton after the
fine-tune.

The cause is structural, not a hyperparameter problem. Newton's
mujoco_warp solver settles arm joints under gravity at about 22 mrad
off the IK target, while classical MuJoCo settles much closer.
Gravity feedforward via `mj_rne` cuts the residual to about 12 mrad
— a real improvement, not a hack — but that's still about 6 mm at
the end-effector, which is beyond grasp tolerance for a 24 mm cube.
The remaining residual is Newton's inertial parameters not being
bit-identical to MuJoCo's, so MuJoCo-computed feedforward
undershoots Newton's actual gravity by about 5%.

The working RL results above are therefore from-scratch curriculum
PPO on tasks Newton's solver handles well. They are not warm-started
from IL. If you are planning to use the RL track to fine-tune a
behaviorally cloned policy on contact-rich pick today, the honest
answer is that this doesn't work yet, and the sim-to-sim gap is the
gating issue.

If you are researching that gap, the diagnostics are in
`scripts/probe_*.py` on the experimental branch, and the longer notes
are in `CLAUDE.md` under the sim-to-sim findings section.

## Reproducing the Working Numbers

Reach (about 25 minutes on a single GB10):

```bash
robo-sandbox train --task reach_target_franka --sim-backend newton \
    --world-count 1024 --total-steps 40_000_000 \
    --output outputs/reach_ppo_1024
```

The insertion curriculum, with each stage warm-starting the next:

```bash
# 4 mm
robo-sandbox train --task insert_connector_franka --sim-backend newton \
    --world-count 64 --output outputs/insert_v4c_hold

# 2 mm, warm-started from 4 mm
robo-sandbox train --task insert_connector_franka --sim-backend newton \
    --world-count 64 --clearance-m 0.002 \
    --warm-start outputs/insert_v4c_hold \
    --output outputs/insert_v7a_2mm

# 1.5 mm USB-A spec, warm-started from 2 mm
robo-sandbox train --task insert_connector_franka --sim-backend newton \
    --world-count 64 --clearance-m 0.0015 \
    --warm-start outputs/insert_v7a_2mm \
    --output outputs/insert_v7b_1p5mm
```

The cross-sim transfer:

```bash
python3 scripts/eval_v7b_classical_mujoco.py --n-trials 32
```

## When This Lands on `main`

The eval substrate (the `eval` CLI, the structured eval JSON, the
Newton backend as a parallel evaluator) lands on `main` with the
eval-and-recording-hygiene PR. The RL training pieces — the `rl/`
module, the `train` CLI, the eval-log lineage, the `results`
subcommands — stay off `main` per the no-train-in-tree thesis. This
page stays as the canonical pointer to the branch until or unless
that decision changes.
