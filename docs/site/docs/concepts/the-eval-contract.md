# The eval contract

When two people report a "pick success rate" on the same task, you
should be able to compare those numbers. Most of the time you can't —
different settle behaviour, different action timing, different success
criteria, different stats. The eval contract is the small set of rules
that make those numbers actually comparable.

It's not marketing. It's seven invariants enforced in code, plus one
JSON schema. This page documents both.

## The schema

Source of truth: `EvalSummary` in
[`packages/robosandbox-core/src/robosandbox/eval/stats.py`](https://github.com/amarrmb/robosandbox/blob/main/packages/robosandbox-core/src/robosandbox/eval/stats.py).
Schema version is `2`. Every JSON written by `robo-sandbox eval`
carries this shape:

```jsonc
{
  "schema_version": 2,
  "task": "pick_cube_franka_random",
  "policy": ".../050000/pretrained_model",
  "sim_backend": "mujoco",
  "n_trials": 64,
  "successes": 28,
  "rate": 0.4375,
  "ci_low": 0.323,
  "ci_high": 0.559,
  "ci_level": 0.95,
  "success_per_trial": [true, false, ...],
  "per_trial_details": [
    {"object_initial_xyz": [0.41, -0.07, 0.06],
     "peak_lift_mm": 142.0, "ee_object_min_dist_mm": 8.3,
     "success": true, "success_step": 387}, ...
  ],
  "spatial_breakdown": {
    "by_object_x": [{"lo": 0.35, "hi": 0.37, "n": 11, "successes": 2, "rate": 0.18}, ...],
    "by_object_y": [...]
  },
  "provenance": {
    "checkpoint_sha256": "...",
    "robosandbox_git_rev": "abc1234",
    "lerobot_version": "0.4.3",
    "mujoco_version": "3.3.0",
    "torch_version": "2.4.0",
    "cli_args": {"n_trials": 64, "action_repeat": 6, "settle_steps": 60, ...}
  }
}
```

Three blocks worth understanding:

**The headline number** — `rate`, `ci_low`, `ci_high`. The CI is
[Wilson](https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval#Wilson_score_interval),
not Wald. Wald collapses to zero width at 0% and 100% and undercovers
for small `n`; Wilson stays well-defined and conservative across the
whole range. Default `ci_level` is 0.95.

**Spatial breakdown** — `spatial_breakdown.by_object_x` and `by_object_y`
bucket the trials by the tracked object's initial position and report
the per-bin success rate. A gradient across the bins is the signature
of a generalization gap. This is derived at JSON-write time from
`per_trial_details`, so it's free.

**Provenance** — checkpoint sha256, robosandbox git rev,
lerobot/mujoco/torch versions, full CLI args. Two JSONs with matching
provenance are guaranteed-comparable. Mismatching provenance means at
least one of code/checkpoint/sim changed and any apples-to-apples
comparison is suspect — `robo-sandbox compare` checks this before
printing a delta.

## The seven invariants

These are the rules that turn "ran an eval" into "ran *the* eval."
Each one has a real failure mode behind it; each is enforced in code,
not folklore.

### 1. No settled-frame contamination in training data

The exporter refuses `action=None` in `events.jsonl`
(`_coerce_action` raises). Settle frames have no commanded action; if
you record them and the exporter silently fills the column with
`observation.state`, you teach the policy "predict your current pose"
for a third of training. Symptom: every trial undershoots by 17–60°
per joint.

### 2. Normalizer is part of the policy adapter

`ACTPolicy.select_action` returns *normalized* actions and expects
*normalized* state + *ImageNet-normalized* images.
`LeRobotPolicyAdapter` loads `policy_preprocessor.json` +
`policy_postprocessor.json` and applies both. Without this the model
sees out-of-distribution inputs and emits actions in normalized space
the sim then misuses as raw joint angles. Symptom: predictions look
random.

### 3. Action repeat matches `sim_dt / dataset_dt`

The dataset is recorded at 30 fps; the sim runs at 200 Hz. Calling
`policy.act()` every sim step replays the model's chunk 6.7× faster
than training. `run_policy` holds each commanded action for
`--action-repeat` sim steps. For the bundled Franka recipe that's `6`.
Wrong number = wrong gripper-close timing.

### 4. Success latches on first match

A trained policy doesn't know when to stop. Once it succeeds it keeps
emitting actions and often perturbs the scene back into a "failed"
state by `max_steps`. Final-state-only success checking hides genuine
successes as failures. `run_policy` records `success_step` — the first
per-step match — and uses that, not the final state. RoboMimic / IsaacLab
convention.

### 5. Adapters implement `reset()`

`LeRobotPolicyAdapter.reset()` forwards to the inner policy's
`reset()`. Without it, ACT's `_action_queue` carries partial chunks
across trials and per-trial outcomes depend on trial order. The eval
CLI calls `policy.reset()` between trials by contract.

### 6. Settle parity between training data and eval

`generate_demos.py` settles the sim ~60 steps before recording the
first frame. Training data therefore reflects a settled scene. If the
eval skips that step the policy's first observation has the cube
mid-fall — out-of-distribution. Pass `--settle-steps=60` to match.
This is wired for both MuJoCo and Newton paths.

### 7. Fresh policy per trial (default)

`policy.reset()` clears documented state but doesn't undo BatchNorm
running buffers, register_buffers, RNG state, or any cached
compile/autograd state. Per-trial results under "load once + reset"
depend on trial order. `--reload-policy` (default `true`) calls
`load_policy()` fresh per trial at ~1–2s per trial cost.
`--no-reload-policy` exists for raw speed when you accept order
dependence.

## What "comparable" means in practice

`robo-sandbox compare a.json b.json` enforces the contract at compare
time:

- Same `task` — different tasks are not comparable. Hard fail.
- Same `robosandbox_git_rev` — different eval code is not comparable.
  Soft fail (warn + refuse delta).
- Same `cli_args` (n_trials, action_repeat, settle_steps,
  reload_policy) — different harness settings are not comparable.
  Soft fail.
- Different `checkpoint_sha256` is the *expected* difference; that's
  the whole point of comparing.

When the gates pass, you get a rate delta with significance from the
two-proportion z-test (`proportion_z_test` in `stats.py`). When they
don't, `compare` tells you *which* field broke comparability. That's
the value: a number you can't reproduce isn't a number.

## What this contract isn't

- It isn't a real-robot eval contract. Sim numbers are sim numbers;
  see [Real-robot bridge](real-robot.md) for the (still-open) story
  on transfer.
- It isn't an OOD-detector. There's no claim that a policy with high
  in-distribution success rate generalizes to a different task or
  embodiment.
- It isn't a leaderboard. Two checkpoints with comparable provenance
  give you a defensible per-task rate delta. Aggregating across tasks
  is up to you.

## Where this is enforced

| Invariant | Where in code |
|---|---|
| Schema | `eval/stats.py` — `EvalSummary`, `summarise_eval`, `_spatial_breakdown` |
| Wilson CI | `eval/stats.py` — `wilson_ci` |
| Significance | `eval/stats.py` — `proportion_z_test` |
| Action=None reject | `recorder/lerobot_export.py` — `_coerce_action` |
| Normalizer wiring | `policy/lerobot_adapter.py` — `LeRobotPolicyAdapter.__init__` |
| Action repeat | `policy/__init__.py` — `run_policy(..., action_repeat=...)` |
| Success latch | `policy/__init__.py` — `run_policy` (`success_step`) |
| Reset | `policy/lerobot_adapter.py` — `LeRobotPolicyAdapter.reset` |
| Settle parity | `cli.py` — `--settle-steps` (both backends) |
| Reload per trial | `cli.py` — `--reload-policy` (default true) |

If you change any of those files, you're changing the contract. Bump
`schema_version`, document the change, accept that old JSONs no longer
compare.
