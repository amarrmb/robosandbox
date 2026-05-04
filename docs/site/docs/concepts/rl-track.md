# The RL track (experimental)

RoboSandbox has an experimental reinforcement-learning track that lives
on the `experimental/newton-eval` branch. It is **a sibling to the IL
track, not a step after it**. Use this page to decide whether to look at
the branch at all.

!!! warning "Off-thesis"
    The product thesis is *RoboSandbox does not train models* — see
    [Why RoboSandbox exists](why-robosandbox-exists.md). The RL track
    intentionally violates that thesis to integrate Newton's GPU
    parallel-world simulator. It stays out of `main` until either the
    thesis changes or the work matures into a separable package.

## What's on the branch

- **PPO with EE-space Cartesian actions** (`packages/robosandbox-core/src/robosandbox/rl/`)
  via DLS pseudoinverse. Targets state-only observations; runs on Newton.
- **`robo-sandbox train`** CLI subcommand. Curriculum support via
  `--warm-start <prior-actor>`.
- **Newton parallel-worlds backend** with `world_count` 1–1024+. Optional
  batched RGB via `SensorTiledCamera` (default off — state-only RL).
- **Eval log + lineage** (`policies.jsonl`, `evals.jsonl`, …) +
  `robo-sandbox results slice|compare|regression` subcommands.
- **Cross-sim transfer harness** for re-running a Newton-trained policy
  in classical MuJoCo without code changes.

## What works (numbers, not promises)

All recorded on `experimental/newton-eval`, single DGX Spark:

| Task | Backend | Workers | Result |
|---|---|---|---|
| `reach_target_franka` | Newton | 1024 | 100% (1024/1024) |
| `insert_connector_franka` 4mm | Newton | 64 | 98.4% (63/64) |
| `insert_connector_franka` 2mm | Newton | 64 | 100% (64/64) |
| `insert_connector_franka` 1.5mm USB-A | Newton | 64 | 100% (64/64) |
| `insert_connector_franka` 1.5mm USB-A | Newton | 1024 | 96.5% sustained, CI [95.2, 97.4] |
| `insert_connector_franka` 1.5mm USB-A | classical MuJoCo (transfer) | 32 | 100% (32/32), CI [89.3, 100.0] |

The insertion curriculum is 4mm → 2mm → 1.5mm, each warm-started from
the previous. The cross-sim row is the headline: a Newton-trained policy
runs in classical MuJoCo with zero code changes and holds 100% at n=32.

## What's open (the IL↔RL bridge)

The natural mental model is *train ACT in MuJoCo → distill to a
state-only MLP → warm-start PPO in Newton → fine-tune*. That path is
**wired but currently fails on contact-rich pick** (`pick_cube_franka`).

- ACT 50k checkpoint distilled to a state-only MLP: **34% in MuJoCo,
  0% in Newton.**
- 200k env-step PPO fine-tune of that warm-start in Newton: **0% in
  MuJoCo, 0% in Newton.** No positive reward signal in Newton →
  drifted randomly → wiped the MuJoCo skill.

The root cause is structural, not a hyperparameter problem: Newton's
mujoco_warp solver settles arm joints under gravity at ~22 mrad off
the IK target, vs MuJoCo's classical solver settling much closer.
Gravity feedforward via `mj_rne` cuts that to ~12 mrad — equivalent
to ~6 mm at the EE, beyond grasp tolerance for a 24 mm cube. The
remaining residual is Newton's inertial parameters not being
bit-identical to MuJoCo's, so MuJoCo-computed FF undershoots Newton's
actual gravity by ~5%.

So the working RL results above (reach, insertion) are **from-scratch
curriculum PPO on tasks Newton's solver handles well**, not warm-started
from IL.

If you want to research closing this gap, see `CLAUDE.md`'s sim-to-sim
findings section and the `scripts/probe_*.py` diagnostics.

## How to use the branch

```bash
git fetch origin experimental/newton-eval
git checkout experimental/newton-eval
# Newton + warp need a CUDA box. On non-GPU machines the import is lazy
# so the package still installs.
```

Reproduce reach (≈25 min on a single GB10):

```bash
robo-sandbox train --task reach_target_franka --sim-backend newton \
    --world-count 1024 --total-steps 40_000_000 \
    --output outputs/reach_ppo_1024
```

Reproduce the insertion curriculum (each stage warm-starts the next):

```bash
# 4mm
robo-sandbox train --task insert_connector_franka --sim-backend newton \
    --world-count 64 --output outputs/insert_v4c_hold

# 2mm, warm-started
robo-sandbox train --task insert_connector_franka --sim-backend newton \
    --world-count 64 --clearance-m 0.002 \
    --warm-start outputs/insert_v4c_hold \
    --output outputs/insert_v7a_2mm

# 1.5mm USB-A spec, warm-started
robo-sandbox train --task insert_connector_franka --sim-backend newton \
    --world-count 64 --clearance-m 0.0015 \
    --warm-start outputs/insert_v7a_2mm \
    --output outputs/insert_v7b_1p5mm
```

Cross-sim transfer:

```bash
python3 scripts/eval_v7b_classical_mujoco.py --n-trials 32
```

## When this lands on `main`

The eval-and-recording-hygiene PR ports the *eval* substrate to `main`
(eval CLI, Wilson CI, structured JSON, provenance, spatial breakdown,
Newton as a backend). The **RL training pieces** (`rl/` module, `train`
CLI, eval-log lineage, `results` subcommands) stay off `main` per the
no-train-in-tree thesis. Treat this page as the canonical pointer
until that decision changes.
