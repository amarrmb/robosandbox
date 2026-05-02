# Iterating on a policy

The IM+RL loop only works if you can see *where* a policy is failing
and *whether* the next iteration helped that slice. The eval log makes
that loop concrete. This guide walks through one full cycle.

```text
   train ACT   -->   distill to MLP   -->   eval (32 trials)
       ^                                          |
       |                                          v
   add demos in  <---- which slice failed? <----  results slice
   failure zone                                   results compare
                                                  results regression
```

## 1. Train ACT (external)

Out of scope for this guide. You bring the checkpoint, RoboSandbox
brings the eval substrate. See LeRobot or your training framework of
choice.

## 2. Distill to a state-only MLP

```bash
python3 scripts/distill_act_to_mlp.py \
    --act-policy outputs/act_ckpt_50k \
    --task pick_cube_franka_random \
    --n-episodes 30 --epochs 200 \
    --out outputs/distilled_v1
```

The script writes a `policy.json` and emits a `policies.jsonl` row with
`parent_policy_id` pointing at the ACT checkpoint.

## 3. Evaluate

```bash
robo-sandbox eval --task pick_cube_franka_random \
    --policy outputs/distilled_v1 \
    --n-trials 32 --action-repeat 6
```

`eval` auto-emits one row to `evals.jsonl` and 32 rows to
`eval_runs.jsonl`, each tagged with the slice axes that the task
exposes (`target_xy_bucket`, `cube_size`, ...) and the per-trial OOD
score against the policy's training distribution.

## 4. Find the failing slice

```bash
robo-sandbox results slice distilled_v1 --axis target_xy_bucket
```

Output is a small table — `target_xy_bucket`, `n_trials`,
`success_rate`. The cell with high `n` and low rate is your bug.

## 5. Add demos in the failure zone

Record more demonstrations targeting the slice that failed. `LocalRecorder`
emits one `demos.jsonl` row per episode automatically, with the slice
axes captured at recording time. You don't need to hand-track which
demos you added for what — the log already knows.

## 6. Re-train, distill, re-eval

Re-run steps 2 and 3 with the larger demo set. The new policy gets a
new `policy_id`, with `parent_policy_id` pointing at `distilled_v1`.

## 7. Did the failing slice shrink?

```bash
robo-sandbox results compare distilled_v1 distilled_v2 \
    --task pick_cube_franka_random
```

One line per policy plus a `delta_pp`. Combine with another
`results slice` on `v2` to confirm the bucket you targeted moved.

## 8. Did anything else regress?

```bash
robo-sandbox results regression distilled_v2
```

This walks one step up the lineage to `distilled_v1` and prints
slice-by-slice deltas with an arrow per direction. A win is a slice
you targeted moving up *and* nothing else moving down.

That's the loop. Each step writes to the log; each query reads from it.
No spreadsheets, no remembered numbers.
