# Iterating on a Policy

Once you have a trained checkpoint and an eval JSON, the question
shifts from "what's the score" to "where is it failing, and did the
next iteration help that slice?" This guide walks one full cycle of
that loop using `robo-sandbox eval`'s `spatial_breakdown` block as
the artifact that drives where new demos should go.

The loop is small on purpose. One eval JSON per checkpoint, one
breakdown to read, one place to add demos. No lineage database, no
intermediate distillation step required. The full IL recipe that
produces the first checkpoint is in
[Train ACT and eval it](../tutorials/train-act-and-eval.md). This
guide picks up at "you have a checkpoint, now what."

```text
   train ACT  →  eval (n=64, MuJoCo)  →  read spatial_breakdown
       ↑                                        |
       |                                        ↓
   add demos   ←——  which x/y bucket failed?  ——
   in failure
   zone
```

## 1. Train (External)

Out of scope for this guide. Bring the checkpoint; RoboSandbox brings
the eval substrate. The full single-workstation recipe with
`lerobot train` is in
[Train ACT and eval it](../tutorials/train-act-and-eval.md). Any
other training framework works as long as the result is a
LeRobot-compatible checkpoint, since `LeRobotPolicyAdapter` is the
only adapter shipped today.

## 2. Evaluate

```bash
robo-sandbox eval \
    --task pick_cube_franka_random \
    --policy outputs/act_franka_pick/checkpoints/050000/pretrained_model \
    --sim-backend mujoco \
    --n-trials 64 --max-steps 900 \
    --action-repeat 6 --settle-steps 60 \
    --output outputs/eval_50k_v1.json
```

The output JSON has the headline rate, the Wilson 95% CI, per-trial
cube pose, peak lift, EE-object distance, and a `spatial_breakdown`
block keyed by cube x and y. The schema is in
`packages/robosandbox-core/src/robosandbox/eval/stats.py`. The full
schema and the seven invariants behind it are in
[The eval contract](../concepts/the-eval-contract.md).

## 3. Find the Failing Slice

Open the JSON and look at `spatial_breakdown.by_object_x` and
`by_object_y`:

```jsonc
// excerpt from outputs/eval_50k_v1.json
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

The bucket with high `n` and low `rate` is the failure zone. In the
excerpt above, the policy works in the centre of the workspace
(around 60%) and collapses in the extremes (around 25 to 30%). That
is almost always a coverage problem in the demos rather than a
training-step problem — more steps on the same data won't fix
positions the model never saw.

## 4. Add Demos in the Failure Zone

Record more demonstrations targeting that bucket. The recording flow
is unchanged — see
[LeRobot export](../tutorials/lerobot-export.md) for how a recorded
run becomes a LeRobot v3 dataset row. For scripted regeneration, the
existing demo generator accepts the same randomization config you
used for the original 200 demos; bias the seed range or the
randomize block toward the failure zone instead of writing new
trajectory code.

## 5. Re-train and Re-eval

Run training again with the larger demo set, then re-run step 2
against the new checkpoint. Save the new JSON next to the old one,
typically as `outputs/eval_50k_v1.json` and `outputs/eval_50k_v2.json`,
so the comparison in step 6 has both files at hand.

## 6. Compare the Two Runs

```bash
robo-sandbox compare outputs/eval_50k_v1.json outputs/eval_50k_v2.json
```

`compare` checks the `provenance` blocks first. If the task,
robosandbox git rev, or eval CLI args don't match between the two
runs, the tool prints which field broke and refuses to print a
delta. When provenance lines up you get a per-checkpoint rate, the
Wilson CI on each, the delta in percentage points, and a
two-proportion z-test significance flag.

A win is the bucket you targeted moving up *and* nothing else moving
down. If the centre regresses while the extremes improve, the new
demos overweighted the edges; re-balance and try the loop again.

## What's Not in This Guide

A few iteration aids exist on the `experimental/newton-eval` branch
but have not landed on `main`. These are part of the experimental RL
track, not the IL track, and they should be treated as research code
until the corresponding subset merges:

- `scripts/distill_act_to_mlp.py` — distill an ACT vision policy
  into a state-only MLP for fast Newton-side eval.
- `policies.jsonl` and `evals.jsonl` lineage logs, plus
  `robo-sandbox results slice|compare|regression` subcommands.
- Warm-started PPO fine-tuning via `robo-sandbox train --warm-start`.

The natural mental model of *"train ACT, distill, warm-start PPO,
fine-tune"* currently fails on contact-rich pick because of a
structural Newton-vs-classical-MuJoCo gravity-settling gap. The
distilled MLP gets 34% in MuJoCo and 0% in Newton; PPO fine-tuning
of that warm-start in Newton wipes the prior. The structural cause
and the working RL results that come from a different angle (reach
and connector insertion, both from-scratch curriculum PPO) are in
[The RL track](../concepts/rl-track.md). This guide stops at
MuJoCo eval, which is where the IL loop is honest.
