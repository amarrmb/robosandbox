# Iterating on a policy

!!! warning "This guide is for the IL track only"
    *"Train ACT → distill → warm-start PPO → fine-tune"* is a tempting
    next step but **currently fails on contact-rich pick** because of a
    structural Newton↔MuJoCo gravity-settling gap (~12 mrad joint
    residual ≈ ~6 mm at the EE — beyond grasp tolerance for a 24 mm
    cube). See [The RL track](../concepts/rl-track.md) for what works,
    what doesn't, and why. This guide stops at MuJoCo eval.

The IL loop only works if you can see *where* a policy is failing and
*whether* the next iteration helped that slice. `robo-sandbox eval`
writes a JSON with a `spatial_breakdown` block — that's the artifact
this guide uses to drive iteration.

```text
   train ACT  -->  eval (n=64, MuJoCo)  -->  read spatial_breakdown
       ^                                          |
       |                                          v
   add demos   <-----  which x/y bucket failed?   |
   in failure                                     |
   zone        <-----------------------------------
```

## 1. Train ACT (external)

Out of scope for this guide. Bring the checkpoint; RoboSandbox brings
the eval substrate. See the
[Train ACT and eval it tutorial](../tutorials/train-act-and-eval.md)
for a full single-workstation pipeline, or use your own training
framework.

## 2. Evaluate

```bash
robo-sandbox eval \
    --task pick_cube_franka_random \
    --policy outputs/act_franka_pick/checkpoints/050000/pretrained_model \
    --sim-backend mujoco \
    --n-trials 64 --max-steps 900 \
    --action-repeat 6 --settle-steps 60 \
    --output outputs/eval_50k.json
```

The output JSON includes the headline rate + Wilson 95% CI, per-trial
cube pose, peak lift, EE-object distance, and a `spatial_breakdown`
block keyed by cube x and y. Schema lives in
`packages/robosandbox-core/src/robosandbox/eval/stats.py`.

## 3. Find the failing slice

Open the JSON and look at `spatial_breakdown.by_object_x` and
`by_object_y`:

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

The bucket with high `n` and low `rate` is the failure zone. Above:
the policy works in the centre (~60%) but collapses in the workspace
extremes (~25–30%). That's a coverage problem in the demos, not a
training-step problem.

## 4. Add demos in the failure zone

Record more demonstrations targeting that bucket. The recording flow is
unchanged — see [LeRobot export](../tutorials/lerobot-export.md). For
scripted regeneration, the existing demo generator accepts the same
randomization config you used for the original 200 demos; bias the seed
range or the randomize block toward the failure zone.

## 5. Re-train and re-eval

Run training again with the larger demo set, then re-run step 2 against
the new checkpoint. Save the new JSON next to the old one
(`outputs/eval_50k_v1.json`, `outputs/eval_50k_v2.json`).

## 6. Compare the two runs

Diff the two `spatial_breakdown` blocks by hand, or use the
`robo-sandbox compare` subcommand if it's available on your branch:

```bash
robo-sandbox compare outputs/eval_50k_v1.json outputs/eval_50k_v2.json
```

A win is the bucket you targeted moving up *and* nothing else moving
down. If the workspace centre regresses while the extremes improve, the
new demos overweighted the edges — re-balance and try again.

That's the loop. It's deliberately small: one eval JSON per checkpoint,
one breakdown to read, one place to add demos. No lineage database
required.

## What's not in this guide (yet)

A few iteration aids exist on the `experimental/newton-eval` branch but
haven't landed on `main`:

- `scripts/distill_act_to_mlp.py` — distill an ACT vision policy into a
  state-only MLP for fast Newton-side eval.
- `policies.jsonl` / `evals.jsonl` lineage logs and `robo-sandbox
  results slice|compare|regression` subcommands.
- Warm-started PPO fine-tuning (`robo-sandbox train --warm-start ...`).

These are part of the experimental RL track, not the IL track. Treat
them as research code until the corresponding subset merges to `main`.
