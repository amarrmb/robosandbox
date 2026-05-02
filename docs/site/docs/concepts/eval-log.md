# The eval log

When you iterate on a policy you are really asking the same question
over and over: *did this change help, and where?* Without somewhere to
record the answer, the loop runs blind — you remember the headline
success number from a week ago, not which slice failed and by how much.

The eval log is that somewhere.

## What it is

An append-only event store at `runs/eval_log/`. Four JSONL files:

- `policies.jsonl` — one row per checkpoint, with `parent_policy_id`
- `demos.jsonl` — one row per recorded demonstration episode
- `evals.jsonl` — one row per eval invocation (a batch of trials)
- `eval_runs.jsonl` — one row per *trial* inside an eval

Every row carries a `schema_version` so the format can grow without
migrations. Reads go through DuckDB views — embedded, no server.

## What's stored vs what's derived

Stored rows are the small, durable facts: this trial succeeded, this
checkpoint descends from that one, this policy's training demos came
from these recordings. Everything you actually look at — bucketed
slice tables, regression deltas vs the lineage parent, lineage chains
— is derived at query time. That keeps the log honest: nothing to
back-fill when you change how you slice.

## Slice axes and OOD

Each trial carries an open-ended `slice_axes` dict — `target_xy_bucket`,
`lighting`, `cube_size`, whatever you measure. Two keys are reserved:
`ood_score` and `is_ood`. They come from a Mahalanobis fit over the
policy's training distribution, computed at log-write time. Below 30
training samples there is not enough data to fit, so both are null —
the system fails to *insufficient*, never to *false-confident*.

## Lineage as a first-class graph

`parent_policy_id` chains BC → distill → PPO → fine-tune through the
log. `walk_lineage(root, policy_id)` traverses the chain with cycle
detection. `regression_check` uses it to compare any policy to its
parent slice-by-slice — which is the only honest way to ask "did this
fine-tune help?".

## Spec

For the full schema and the design rationale, see
[`docs/design/2026-05-01-eval-log-design.md`](https://github.com/amarrmb/robosandbox/blob/main/docs/design/2026-05-01-eval-log-design.md).
