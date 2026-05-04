# What you can do

Three things, in order of how often you'll use them. Each one is a
single command and a single artifact.

## 1. Score a checkpoint

Point `robo-sandbox eval` at a checkpoint and a task. Get a JSON.

```bash
robo-sandbox eval \
    --task pick_cube_franka_random \
    --policy outputs/act_franka_pick/checkpoints/050000/pretrained_model \
    --sim-backend mujoco \
    --n-trials 64 --action-repeat 6 --settle-steps 60 \
    --output outputs/eval_50k.json
```

What you get back:

```jsonc
{
  "schema_version": 2,
  "task": "pick_cube_franka_random",
  "policy": {"path": ".../050000/pretrained_model", "kind": "lerobot_act"},
  "sim_backend": "mujoco",
  "n_trials": 64,
  "successes": 28,
  "rate": 0.4375,
  "ci_low": 0.323, "ci_high": 0.559,    // Wilson 95%
  "spatial_breakdown": { "by_object_x": [...], "by_object_y": [...] },
  "trials": [ /* per-trial cube pose, peak lift, EE-object distance */ ],
  "provenance": {
    "checkpoint_sha256": "...",
    "robosandbox_git_rev": "...",
    "lerobot_version": "0.4.3",
    "mujoco_version": "3.3.0"
  }
}
```

The headline number is `rate` + `ci_low`/`ci_high`. The interesting
number is `spatial_breakdown` — it's how you find the bug, not the
score.

Today's policy support: `LeRobotPolicyAdapter` (any LeRobot-compatible
checkpoint). Other frameworks need a thin wrapper around the
[`Policy` protocol](reference/api.md).

## 2. Compare two checkpoints fairly

Train your policy twice (different seed, different hyperparams,
different demo set). Score both. Diff them.

```bash
robo-sandbox eval --policy outputs/act_50k --output outputs/eval_50k.json   ...
robo-sandbox eval --policy outputs/act_60k --output outputs/eval_60k.json   ...

robo-sandbox compare outputs/eval_50k.json outputs/eval_60k.json
```

`compare` checks the `provenance` blocks first. If the task,
robosandbox git rev, or eval CLI args differ between the two runs,
it tells you the comparison isn't apples-to-apples and refuses to
print a delta. That's the whole point of the contract — a number you
can't reproduce isn't a number.

When the provenance lines up, you get the rate delta with significance:

```
50k: 28/64  43.8%  CI [32.3, 55.9]
60k: 13/64  20.3%  CI [12.3, 31.7]
delta: -23.4 pp  (overlapping CIs — significant at p<0.05? no)
```

That single command is what stops you from chasing noise.

## 3. Find where your policy fails

`spatial_breakdown` buckets every trial by cube x and y. Open the JSON.

```jsonc
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

Read it like a histogram of success rate by position. Above: works in
the centre (~60%), collapses in the workspace extremes (~25%). That's
a coverage problem in the demos, not a hyperparameter problem. Record
more demos in the failure zone, retrain, re-eval, diff. Loop closes.

The full loop is in [Iterating on a policy](guides/iterating-on-a-policy.md).

## What this gets you

You can answer three questions you usually can't answer:

- *Did this checkpoint actually get better, or did I get lucky on the
  test seed?* → CI on the rate.
- *Is this comparison meaningful?* → provenance match.
- *Where is my policy failing?* → spatial breakdown.

That's it. That's the product.

For the schema, the seven enforced invariants behind the contract, and
where each one lives in code, see
**[The eval contract](concepts/the-eval-contract.md)**.
