# Tutorial — Train an ACT Policy and Eval It in MuJoCo

End-to-end pipeline that produces a trained ACT checkpoint and a
reproducible n=64 success rate on a single workstation. Covers the
data-generation, export, training, and evaluation steps, and writes
down every gotcha that bit us during the first run-through so the
next person doesn't repeat them.

You walk away with a working ACT-style vision policy trained on 200
randomized scripted demos, evaluated against `pick_cube_franka_random`
with a Wilson 95% CI and a per-position spatial breakdown of where it
failed. The actual training step is `lerobot train`; RoboSandbox
doesn't train models. This tutorial is the data pipeline plus the
eval harness.

The eval CLI used in step 4 (`robo-sandbox eval`, the structured JSON,
the Wilson CI, the spatial breakdown) lives on the
`experimental/newton-eval` branch and lands on `main` with the
eval-and-recording-hygiene PR. Steps 1 through 3 work on `main`
today; step 4 needs that branch or that PR. The only policy adapter
shipped is `LeRobotPolicyAdapter`. Other frameworks (Diffusion Policy,
Octo, π0) need their own wrapper around the `Policy` protocol.

## Hardware

Validated on the configurations below. The whole recipe runs end to
end on the desktop tier; the laptop alone is fine if you're willing
to wait for a slower train.

| Tier | Train | Eval | Notes |
| --- | --- | --- | --- |
| Laptop CPU | not feasible | ~75 env-steps/s on n=64 | MuJoCo 3.3 + CPU policy inference |
| Desktop RTX 4060 (8 GB) | ~5.15 step/s @ batch=32 | same as laptop | one full 100k-step run is ~5.5 h |
| DGX Spark (GB10, 122 GiB) | ACT training not measured here | parallel: 1024+ worlds | use Newton backend for scale |

## The Four Commands

```bash
# 1. Generate 200 randomized scripted demos (~30 min on 4 CPU jobs)
python scripts/generate_demos.py --task pick_cube_franka_random --n 200 --jobs 4

# 2. Export to LeRobot v3.0 dataset (~30 s)
robo-sandbox export-lerobot \
    runs/demos_franka_pick \
    datasets/demos_franka_pick_lerobot \
    --task pick_cube_franka_random --fps 30

# 3. Train ACT — external; requires lerobot >= 0.4.3
lerobot-train \
    --dataset.repo_id=local/franka_pick \
    --dataset.root=$(pwd)/datasets/demos_franka_pick_lerobot \
    --dataset.video_backend=pyav \
    --policy.type=act --policy.push_to_hub=false --policy.device=cuda \
    --output_dir=$(pwd)/outputs/act_franka_pick \
    --steps=50000 --batch_size=32 --num_workers=4 \
    --save_freq=10000 --log_freq=200 --wandb.enable=false

# 4. Evaluate the checkpoint
robo-sandbox eval \
    --task pick_cube_franka_random \
    --policy outputs/act_franka_pick/checkpoints/050000/pretrained_model \
    --sim-backend mujoco \
    --n-trials 64 --max-steps 900 \
    --action-repeat 6 --settle-steps 60 \
    --output outputs/eval_50k.json
```

50k steps is the sweet spot for a 200-demo dataset. The "Watch the
Loss-vs-Success Curve" section below explains why more steps aren't
better.

Expected output on the 200-demo dataset, 50k checkpoint:

```
[eval] successes:     28 / 64  (43.8%)
[eval] 95% CI (Wilson): [32.3%, 55.9%]
[eval] wall:          ~17 min  (CPU laptop)
```

The output JSON includes per-trial cube pose, peak lift, EE-object
distance, the spatial breakdown by cube x and y, and full provenance
(checkpoint sha256, robosandbox git rev, lerobot, mujoco, torch
versions). The schema is in
`packages/robosandbox-core/src/robosandbox/eval/stats.py`.

## Known Issues — Fix Once, Never Again

These are the seven bugs we hit on the first end-to-end run. Each
costs about 15 minutes to discover and 5 minutes to fix once the
symptom is recognized.

### 1. Don't Record Settle Frames

`scripts/generate_demos.py` runs `sim.step()` about 60 times after
`sim.load()` so physics can settle — the cube falls onto the table,
joint actuators converge. Do not call `recorder.write_frame()` inside
that loop. Settle frames have no commanded action (`sim.last_action()`
is `None`), and the exporter previously filled `action=None` with a
copy of the current state. That copied "action" mixes
gripper-width-in-meters (state) with gripper-command-in-[0,1] (real
actions) in the same column, and teaches the policy "predict your
current pose" for about 65% of training samples. Symptom: every
trial undershoots the demo trajectory by 17–60° per joint.

The exporter now refuses `action=None` (`_coerce_action` raises) so
this regression cannot silently come back.

### 2. lerobot 0.4 Splits the Normalizer Out of the Policy

`ACTPolicy.select_action` returns normalized actions and does not run
the postprocessor. The model also expects normalized state and
ImageNet-stat-normalized images. `LeRobotPolicyAdapter` must load the
saved `policy_preprocessor.json` and `policy_postprocessor.json` and
apply both, or the model receives out-of-distribution inputs and
emits actions in normalized space that the sim then tries to use as
raw joint angles.

Symptom: predictions look random; joint MAE on training data is
about 30°. With the processors wired in, joint MAE drops to about
0.3°.

### 3. Hold Each Policy Action for `sim_dt / dataset_dt` Sim Steps

The dataset is recorded at 30 fps; sim runs at 200 Hz (sim_dt=0.005).
Calling `policy.act()` every sim step replays the model's 100-action
chunk 6.7× faster than training. Symptom: the gripper closes about
0.3 s into the episode instead of about 2 s, while the arm is still
about 15 cm from the cube.

Pass `--action-repeat=6` (or whatever your `sim_dt / dataset_dt`
ratio is) to the eval CLI. `run_policy` holds each commanded action
for that many sim steps.

### 4. Latch Success on First Match, Not on Final State

A trained policy doesn't know when to stop. Once it succeeds it
keeps emitting actions, often perturbing the scene back into a
"failed" state by `max_steps`. Final-state-only success checking
hides genuinely successful runs as failures — for example, a cube
lifted to +120 mm at step 400 and dropped back to +5 mm by step 900
scores as a fail.

`run_policy` now latches success on the first per-step match
(`success_step` field in the result), which is the
RoboMimic / IsaacLab convention.

### 5. Adapters Need `reset()`

The eval CLI calls `policy.reset()` between trials so each trial
starts at step 0 of the model's action queue. `LeRobotPolicyAdapter`
previously had no `reset()` method, so `getattr(adapter, "reset",
None)` returned `None` and the inner ACT's `_action_queue` carried
partial chunks across trials. Per-trial outcomes depended on trial
order.

`LeRobotPolicyAdapter.reset()` now forwards to `inner.reset()`.

### 6. Settle the Eval the Same Way You Settled the Demos

`generate_demos.py` settles physics for about 60 sim steps before
recording the first frame. The training data therefore reflects a
settled scene — cube resting on the table, actuators converged. If
the eval skips that step, the policy's first observation has the
cube mid-fall, which is out-of-distribution input.

The eval CLI's `--settle-steps` was wired only for the Newton path;
it now runs for the MuJoCo path too. Pass `--settle-steps=60` to
match this recipe's demo generation.

### 7. Reload the Policy Fresh Per Trial

`policy.reset()` clears the documented state (the action queue, the
temporal ensembler), but it doesn't undo BatchNorm running buffers,
register_buffer tensors, RNG state, or any cached compile or
autograd state. Per-trial results under "load once and reset" depend
on trial order.

`--reload-policy` is the default and calls `load_policy()` fresh per
trial, at a cost of about 1–2 seconds per trial. Pass
`--no-reload-policy` only for raw speed when order-dependent results
are acceptable.

## Watch the Loss-vs-Success Curve

The last checkpoint is rarely the best. The 100k-step ACT training
on the 200-demo dataset above produced this curve:

| Checkpoint | Train loss | Success / 64 | Wilson 95% CI |
| --- | --- | --- | --- |
| 30k | 0.015 | 8 (12.5%) | [6.5, 22.8] |
| 50k | 0.012 | 28 (43.8%) | [32.3, 55.9] ← peak |
| 60k | 0.011 | 13 (20.3%) | [12.3, 31.7] |
| 80k | 0.010 | 10 (15.6%) | [8.7, 26.4] |
| 100k | 0.010 | 6 (9.4%) | [4.4, 19.0] |

Loss kept dropping by about 30% from 50k to 100k, and the success
rate dropped by 80% in the same interval. 178 effective epochs on
200 demos overfit the model so hard it stopped generalizing.
Evaluate intermediate checkpoints — the eval is fast enough to do
that for free in any training pipeline.

The spatial breakdown emitted by `--output` makes the overfitting
failure mode obvious. At 50k the policy succeeds where the data is
densest (centre of the workspace); at 100k it only succeeds at the
visually distinctive extremes, which is a textbook narrowing of the
model's effective coverage.

```jsonc
// excerpt from outputs/eval_50k.json
"spatial_breakdown": {
    "by_object_x": [
        {"lo": 0.350, "hi": 0.370, "n": 11, "successes": 2,  "rate": 0.18},
        {"lo": 0.370, "hi": 0.390, "n": 15, "successes": 9,  "rate": 0.60},
        {"lo": 0.390, "hi": 0.410, "n": 18, "successes": 11, "rate": 0.61},
        {"lo": 0.410, "hi": 0.430, "n": 10, "successes": 3,  "rate": 0.30},
        {"lo": 0.430, "hi": 0.450, "n": 10, "successes": 3,  "rate": 0.30}
    ]
}
```

## When to Use What

Single MuJoCo eval (this recipe) is for correctness and visual
debugging. Use it on any policy you actually want to understand.

Newton parallel eval (`--sim-backend=newton --world-count=N`) is for
scaling up to thousands of trials and tightening the Wilson CI. The
Newton parallel path is state-only — vision policies fall back to
zero-image frames, so don't run a vision policy through Newton.

Comparison runs (`robo-sandbox compare a.json b.json`) are how you
check whether a rate difference is significant once you have multiple
eval JSONs. The `provenance` block in each JSON tells you whether
the comparison is apples-to-apples — matching `checkpoint_sha256`
and `robosandbox_git_rev` mean yes, mismatching either field means
no.

## Where This Fits

This is the IL recipe end to end. The data path that feeds it is in
[LeRobot Export](./lerobot-export.md). The pre-trained checkpoint
alternative — wire a public ACT checkpoint instead of training your
own — is in [LeRobot Policy Replay](./lerobot-policy-replay.md). The
real-arm story for the resulting policy is in
[Sim-to-Real Handoff](./sim-to-real-handoff.md).

## Reproducibility

Everything in the eval JSON is enough to reproduce the result:

- `policy.path` and `provenance.checkpoint_sha256` — exact checkpoint
- `provenance.robosandbox_git_rev` — exact eval code
- `provenance.lerobot_version`, `mujoco_version`, `torch_version` — runtime
- The CLI args echoed in the eval log — task, n_trials, max_steps, action_repeat, settle_steps, reload_policy, seed

If the provenance has a `-dirty` git rev, the code has uncommitted
changes. Commit before running eval if you want a citable result.
