# What You Can Do

Three things, in the order you'll most often use them.

## 1. Score a Checkpoint

`robo-sandbox eval` takes a checkpoint and a task and writes a JSON.
That JSON is the artifact: every downstream comparison, slicing, and
regression check reads from it.

```bash
robo-sandbox eval \
    --task pick_cube_franka_random \
    --policy outputs/act_franka_pick/checkpoints/050000/pretrained_model \
    --sim-backend mujoco \
    --n-trials 64 --action-repeat 6 --settle-steps 60 \
    --output outputs/eval_50k.json
```

The output JSON carries the headline number, a Wilson 95% CI on the
rate, per-trial details (initial object pose, peak lift, EE-object
min distance, success step), a spatial breakdown of where the policy
failed, and a provenance block (checkpoint sha256, robosandbox git
rev, lerobot/mujoco/torch versions, full CLI args). The Wilson CI
matters more than the rate alone because Wald collapses to zero width
at 0% and 100% and undercovers for small `n`. The provenance block
matters because it lets a second run check, mechanically, whether
the comparison is apples-to-apples.

Today's policy support is `LeRobotPolicyAdapter`. The `Policy`
protocol is framework-agnostic, but we do not ship adapters for
Diffusion Policy, Octo, or π0 — those need a thin user-side wrapper
around the `Policy` protocol. See
[the API reference](reference/api.md) for the protocol shape.

## 2. Compare Two Checkpoints Fairly

Train your policy twice, score both, and diff them. `robo-sandbox
compare` checks the `provenance` blocks before printing a delta, so
two runs that differ on task, robosandbox git rev, or eval CLI args
are caught instead of silently compared.

```bash
robo-sandbox eval --policy outputs/act_50k --output outputs/eval_50k.json   ...
robo-sandbox eval --policy outputs/act_60k --output outputs/eval_60k.json   ...

robo-sandbox compare outputs/eval_50k.json outputs/eval_60k.json
```

When provenance lines up, the output is a per-checkpoint rate, the
Wilson CI on each, and a delta with a significance test:

```
50k: 28/64  43.8%  CI [32.3, 55.9]
60k: 13/64  20.3%  CI [12.3, 31.7]
delta: -23.4 pp  (overlapping CIs — significant at p<0.05? no)
```

## 3. Find Where a Policy Fails

The eval JSON's `spatial_breakdown` block buckets every trial by the
tracked object's initial position. Open the JSON and read that block
like a histogram — the bucket with high `n` and low `rate` is the
failure zone.

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

That excerpt shows a policy that works in the centre of the
workspace (around 60%) and collapses in the extremes (around 25%).
The fix is usually more demos in the failure bucket, not more
training steps; the demo distribution drove the coverage gap. The
full demo-add-retrain-re-eval loop is in
[Iterating on a policy](guides/iterating-on-a-policy.md).

## See also

- [The eval contract](concepts/the-eval-contract.md) — the schema
  and the seven invariants enforced in code.
- [Train ACT and eval it](tutorials/train-act-and-eval.md) — the
  full IL recipe end-to-end.
- [Where things are](where-things-are.md) — which page (and which
  branch) covers each part of the workflow.
