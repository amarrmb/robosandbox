# Replan loop

When a skill fails, the agent can hand that failure back to the planner
and ask for a new plan instead of stopping immediately.

![replan loop](../assets/demos/replan_loop.gif){ loading=lazy }

That's three `PLAN` → `EXECUTE` cycles. Each one hits the same
`unreachable` failure (cube placed 1.2 m from the Franka, reach ~0.85
m). On the fourth attempt the agent gives up with `replan_exhausted`
and surfaces the root cause: `IK did not converge in 400 iters
(pos_err=0.7052m > tol=0.001m)`.

## Why replan at all

The planner picks actions from a static image + object list. Things go
wrong for reasons the plan didn't anticipate:

- perception located the wrong object;
- the grasp slipped;
- the object moved between perception and the grasp;
- the target was unreachable.

Rather than expecting the planner to predict every failure in advance,
the agent reacts to what actually happened and plans again.

## The loop

```
IDLE
  │
  ▼
PLAN ◄──────────────────────────────────┐
  │                                      │
  ▼                                      │
EXECUTE skill                            │
  │                                      │
  ├─► success? ──► next skill in plan    │
  │                                      │
  └─► failure? ──► append to prior_attempts
                   ├─► replans < max? ───┘
                   └─► replans = max? ──► FAILED (replan_exhausted)
```

Exact flow in [`agent/agent.py`](https://github.com/amarrmb/robosandbox/blob/main/packages/robosandbox-core/src/robosandbox/agent/agent.py):

```python
failed_step = ...
prior_attempts.append({
    "step_idx": len(ep.steps),
    "skill":    failed_step.skill,
    "args":     failed_step.args,
    "reason":   failed_step.result.reason,
    "reason_detail": failed_step.result.reason_detail,
})
if replans >= self._max_replans:
    ep.final_reason = "replan_exhausted"
    return ep
replans += 1
```

## What the planner sees on replan

`VLMPlanner` appends one more message block on replan iterations:

```json
{
  "role": "user",
  "content": [
    {"type": "text", "text": "Previously-failed steps — do NOT repeat them unchanged:"},
    {"type": "text", "text": "[{\"step_idx\": 1, \"skill\": \"pick\", \"args\": {\"object\": \"red_cube\"}, \"reason\": \"object_not_found\", \"reason_detail\": \"perception returned zero matches\"}]"}
  ]
}
```

The planner gets a summary of the failed step and can choose a
different object, change the arguments, or give up.

`StubPlanner` ignores `prior_attempts`, so it will exhaust replans on
every deterministic failure. A real planner should use that feedback.

## Run the demo

```bash
uv run python examples/replan_demo.py
```

`examples/replan_demo.py` loads a task whose cube is placed 1.2 m from
the Franka's base — well past reach. With `max_replans=2` you get 3
attempts and a final verdict:

```
PLAN: task='pick up the red cube' replan=0
EXECUTE: pick({'object': 'red_cube'})
PLAN: task='pick up the red cube' replan=1
EXECUTE: pick({'object': 'red_cube'})
PLAN: task='pick up the red cube' replan=2
EXECUTE: pick({'object': 'red_cube'})

FINAL VERDICT
  success:     False
  replans:     2
  steps:       3
  final reason: replan_exhausted
  detail:      pick failed: unreachable — IK did not converge in 400 iters (pos_err=0.7052m > tol=0.001m)
  wall:        0.5s
```

Three fields in `result.json` matter here:

- **`replans`** — how many times the agent re-entered PLAN
- **`final_reason`** — `plan_complete`, `replan_exhausted`, `vlm_transport`, etc.
- **`final_detail`** — the inner skill's `reason_detail`; what actually broke

## When the loop helps and when it does not

**Replan helps** when the failure is:

- recoverable by a different action (`object_not_found` → try a
  different object name);
- transient (`grasp_slipped` → re-plan from fresh observation);
- under-specified (`bad_arguments` → planner was missing context).

**Replan spins wastefully** when the failure is:

- structural (object physically out of reach);
- `bad_arguments` from a planner that always emits the same args;
- `vlm_transport` (no point retrying a rate-limited API from the same key).

For structural failures the loop just burns through `max_replans` and
stops. For transient issues, it often recovers in one or two tries.

## Tuning `max_replans`

```python
Agent(ctx, skills=[...], planner=..., max_replans=3)
```

Default is `3`. Raise it for agents with high-variance perception
(`VLMPointer` on a small VLM). Lower it for deterministic perception
(`GroundTruthPerception` for bench evaluations — no point retrying
a deterministic failure).

## Skill-side contract

For replanning to help, skills need to return useful failure reasons:

```python
# GOOD — gives planner something to work with
return SkillResult(
    success=False,
    reason="object_not_found",
    reason_detail=f"perception returned zero matches for '{object}'",
)

# BAD — planner sees nothing actionable
return SkillResult(success=False, reason="error")
```

`reason` is the short machine-readable slug. `reason_detail` is the
human-readable context that lands in `result.json` and gets fed back
into the next plan.

## Troubleshooting a stuck replan loop

If you see the same step failing across every replan, check:

1. **Planner output is deterministic.** `StubPlanner` always returns
   the same plan — make sure you're using `VLMPlanner` (or a planner
   that reads `prior_attempts`) for any task where the planner should
   adapt.
2. **Skill's `reason` is distinct per failure mode.** "unreachable"
   vs "verification_failed" mean different things; a planner can only
   react if you tell it which one.
3. **Observation actually changes between attempts.** If perception
   and sim state are identical, a deterministic planner will emit the
   same plan. Consider resetting sim state between attempts for
   non-stateful skills.

## What's next

- [Running the agent](./agent-runs.md) — where `final_reason` values come from.
- [VLM tool-calling](./vlm-tool-calling.md) — how `prior_attempts` is serialized to the model.
- [Add a skill](./add-a-skill.md) — wiring up good `reason`/`reason_detail` in your own skills.
