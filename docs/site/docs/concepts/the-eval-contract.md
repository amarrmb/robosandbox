# The Eval Contract

When two people report a "pick success rate" on the same task, you
want those two numbers to mean the same thing. They usually don't.
Different teams settle physics for different durations, hold actions
for different numbers of sim steps, score success on the final state
or on the first match, and report rates with different statistical
intervals. The numbers don't compose because the rules behind them
don't compose.

The eval contract is the small set of rules that make two RoboSandbox
evals actually comparable. It's a JSON schema and seven invariants
that the code enforces. This page documents both.

## The Schema

The source of truth is `EvalSummary` in
[`packages/robosandbox-core/src/robosandbox/eval/stats.py`](https://github.com/amarrmb/robosandbox/blob/main/packages/robosandbox-core/src/robosandbox/eval/stats.py),
schema version 2. Every JSON written by `robo-sandbox eval` has this
shape:

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

The headline number is the `rate` together with `ci_low` and
`ci_high`, which are the Wilson 95% confidence interval bounds on the
rate. The Wilson interval is preferred over the textbook Wald
interval because Wald collapses to zero width at 0% and 100% and
undercovers for small `n`. Wilson stays well-defined and conservative
across the whole range. The default confidence level is 95%.

The `spatial_breakdown` block buckets trials by the tracked object's
initial position and reports the per-bin success rate. A gradient
across the bins is the signature of a generalization gap. The
breakdown is derived at JSON-write time from `per_trial_details`, so
it costs nothing extra to emit.

The `provenance` block is what makes two JSONs comparable in the
first place. It carries the checkpoint sha256, the robosandbox git
revision, the lerobot, mujoco, and torch versions, and the full CLI
arguments. `robo-sandbox compare` checks the provenance before
printing a delta; mismatching fields are reported and the delta is
withheld, because at least one of code, checkpoint, or sim changed
and the comparison isn't apples-to-apples.

## The Seven Invariants

These are the rules that turn "ran an eval" into "ran *the* eval".
Each one has a real failure mode behind it, and each one is enforced
in code rather than left to convention.

### 1. No Settled-Frame Contamination in Training Data

The exporter refuses `action=None` rows in `events.jsonl`
(`_coerce_action` raises). Settle frames have no commanded action;
if you record them and the exporter silently fills the column with
`observation.state`, you teach the policy "predict your current pose"
for a third of training. The symptom is every trial undershooting
the demo trajectory by 17–60° per joint. The exporter check is what
keeps that regression from coming back silently.

### 2. The Normalizer Is Part of the Policy Adapter

`ACTPolicy.select_action` returns normalized actions and expects
normalized state and ImageNet-normalized images.
`LeRobotPolicyAdapter` loads the saved
`policy_preprocessor.json` and `policy_postprocessor.json` and
applies both. Without that wiring the model receives
out-of-distribution inputs and emits actions in normalized space
that the sim then misuses as raw joint angles. Predictions look
random until the processors are in place; with them, joint MAE on
training data drops from ~30° to ~0.3°.

### 3. Action Repeat Matches `sim_dt / dataset_dt`

The dataset is recorded at 30 fps. The sim runs at 200 Hz. Calling
`policy.act()` every sim step replays the model's chunk 6.7× faster
than training. `run_policy` holds each commanded action for
`--action-repeat` sim steps. For the bundled Franka recipe that
ratio is 6. The wrong number gives wrong gripper-close timing and
the policy closes the fingers on empty air.

### 4. Success Latches on First Match

A trained policy doesn't know when to stop. Once it succeeds it
keeps emitting actions, often perturbing the scene back into a
"failed" state by `max_steps`. Final-state-only success checking
hides genuine successes as failures — a cube lifted to +120 mm at
step 400 and dropped back to +5 mm by step 900 scores as a fail
under final-state checking even though the task was solved.
`run_policy` records `success_step` (the first per-step match) and
uses that, not the final state. This is the RoboMimic / IsaacLab
convention.

### 5. Adapters Implement `reset()`

The eval CLI calls `policy.reset()` between trials so each trial
starts at step 0 of the model's action queue.
`LeRobotPolicyAdapter.reset()` forwards to the inner policy's
`reset()`. Without it, ACT's `_action_queue` carries partial chunks
across trials and per-trial outcomes depend on trial order — which
is the same bug, in user-space, that makes one team's "30%" not
match another team's "30%".

### 6. Settle Parity Between Demos and Eval

`generate_demos.py` settles the sim for ~60 steps before recording
the first frame. The training data therefore reflects a settled
scene. If the eval skips that step, the policy's first observation
has the cube mid-fall, which is out-of-distribution. The eval CLI's
`--settle-steps` is wired for both the MuJoCo and Newton paths;
passing `--settle-steps=60` matches the recipe's demo generation.

### 7. Fresh Policy Per Trial

`policy.reset()` clears the documented state (the action queue, the
temporal ensembler), but it doesn't undo BatchNorm running buffers,
register_buffers, RNG state, or any cached compile or autograd
state. Per-trial results under "load once and reset" depend on
trial order. `--reload-policy` (the default) calls `load_policy()`
fresh per trial, at a cost of about one to two seconds per trial.
`--no-reload-policy` exists for raw speed when order-dependence is
acceptable.

## What "Comparable" Means in Practice

`robo-sandbox compare a.json b.json` enforces the contract at
compare time. Same `task` is mandatory; different tasks aren't
comparable and the tool refuses to print a delta. Same
`robosandbox_git_rev` is a soft check; different eval code can
silently change behaviour and a warning fires if the revisions
don't match. Same `cli_args` (n_trials, action_repeat, settle_steps,
reload_policy) is also a soft check. Different `checkpoint_sha256`
is the *expected* difference — that is the whole point of comparing.

When the gates pass you get a per-checkpoint rate, the Wilson CI on
each, a delta in percentage points, and a two-proportion z-test
significance flag. When they don't, the tool prints which field
broke comparability and exits without a number. A number you can't
reproduce isn't a number, and the contract's job is to keep that
property mechanical instead of cultural.

## What This Contract Isn't

It isn't a real-robot eval contract. Sim numbers are sim numbers. The
real-arm story is in [the real-robot bridge concept page](real-robot.md)
and the [sim-to-real handoff tutorial](../tutorials/sim-to-real-handoff.md);
neither closes the transfer claim today. There is no eval contract
that survives a hardware delta automatically.

It isn't an OOD detector. A high in-distribution success rate doesn't
imply generalization to a different task or a different embodiment.
The spatial breakdown shows where the policy fails *in distribution*,
which is a different question.

It isn't a leaderboard. Two checkpoints with comparable provenance
give a defensible per-task rate delta. Aggregating across tasks is
left to the user and to the downstream tooling.

## Where Each Invariant Lives in Code

| Invariant | File and symbol |
|---|---|
| Schema | `eval/stats.py` — `EvalSummary`, `summarise_eval`, `_spatial_breakdown` |
| Wilson CI | `eval/stats.py` — `wilson_ci` |
| Two-proportion z-test | `eval/stats.py` — `proportion_z_test` |
| Action=None reject | `recorder/lerobot_export.py` — `_coerce_action` |
| Normalizer wiring | `policy/lerobot_adapter.py` — `LeRobotPolicyAdapter.__init__` |
| Action repeat | `policy/__init__.py` — `run_policy(..., action_repeat=...)` |
| Success latching | `policy/__init__.py` — `run_policy` (`success_step`) |
| Reset forwarding | `policy/lerobot_adapter.py` — `LeRobotPolicyAdapter.reset` |
| Settle parity | `cli.py` — `--settle-steps` (both backends) |
| Reload per trial | `cli.py` — `--reload-policy` (default true) |

Changing any of those files changes the contract. The honest move is
to bump `schema_version`, document the change, and accept that older
JSONs no longer compare against newer ones.
