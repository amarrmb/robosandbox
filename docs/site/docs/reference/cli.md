# CLI reference

Two console scripts ship:

- `robo-sandbox` — subcommand entry point (`demo`, `viewer`, `run`,
  `export-lerobot`, `results`).
- `robo-sandbox-bench` — the benchmark runner.

## `robo-sandbox`

```
robo-sandbox {demo | viewer | run | export-lerobot | results} [options]
```

### `demo`

```bash
robo-sandbox demo
```

Scripted pick with the built-in arm. No VLM, no API key. Useful for
"does MuJoCo work on my box?".

### `viewer`

```bash
robo-sandbox viewer [--host HOST] [--port PORT] [--task TASK] [--runs-dir DIR]
```

Starts the FastAPI + WebSocket live viewer. Needs the `viewer` extra:
`uv pip install -e 'packages/robosandbox-core[viewer]'`.

| Flag | Default | Meaning |
|---|---|---|
| `--host` | `127.0.0.1` | Interface to bind. Use `0.0.0.0` to expose on LAN. |
| `--port` | `8000` | Port. |
| `--task` | `pick_cube_franka` | Built-in task to preload in the dropdown. |
| `--runs-dir` | `runs` | Where the sidebar Record button writes episodes. |

Open `http://<host>:<port>` in a browser. Sidebar: task dropdown,
Run/Reset, Record toggle, Teleop toggle.

Teleop bindings (when Teleop is on): **W/S** = +x/-x, **A/D** =
+y/-y, **Q/E** = +z/-z, **Space** = toggle gripper. 1.5 cm per
keystroke; unreachable poses are safely ignored.

### `run`

```bash
robo-sandbox run [task] [--policy PATH] [options]
```

Two modes:

#### Planner mode (default)

```bash
robo-sandbox run "pick up the red cube"
robo-sandbox run --vlm-provider ollama "stack red on green"
```

| Flag | Default | Meaning |
|---|---|---|
| `task` (positional) | — | Natural-language task. Required unless `--policy` is set. |
| `--vlm-provider` | `stub` | One of `stub`, `openai`, `ollama`, `custom`. |
| `--model` | provider default | Override model name (e.g. `gpt-4o`). |
| `--base-url` | provider default | OpenAI-compatible endpoint URL. |
| `--api-key-env` | `OPENAI_API_KEY` for `openai`, unused for `ollama` | Env var name holding the key. |
| `--perception` | auto | `ground_truth` or `vlm`. Defaults depend on provider. |
| `--max-replans` | `3` | Replan cap before giving up. |
| `--max-steps` | `1000` | Upper bound on skill steps per episode. |
| `--log-level` | `INFO` | Python logging level. |

Provider defaults:

- `stub` — no model; StubPlanner.
- `openai` — `gpt-4o-mini`, `https://api.openai.com/v1`.
- `ollama` — `llama3.2-vision`, `http://localhost:11434/v1`.
- `custom` — you supply `--model` + `--base-url`.

#### <a id="run-policy"></a>Policy mode

```bash
robo-sandbox run --policy PATH --task TASK_NAME [--max-steps N]
```

| Flag | Meaning |
|---|---|
| `--policy PATH` | Directory containing `policy.json` (+ trajectory) or `events.jsonl`. Bypasses the planner. |
| `--task TASK_NAME` | Built-in task name (e.g. `pick_cube_franka`). Required with `--policy`. |
| `--max-steps N` | Loop cap. Default `1000`. |

Bypasses the agent/planner entirely and drives the sim via `run_policy`.
See [policy replay tutorial](../tutorials/policy-replay.md).

### <a id="export-lerobot"></a>`export-lerobot`

```bash
robo-sandbox export-lerobot SRC DST [--task TASK] [--fps N]
```

Convert one recorded episode directory (`runs/<ts>-<id>/`) to a
LeRobot v3 dataset.

| Flag | Meaning |
|---|---|
| `SRC` (positional) | Source episode directory (from `LocalRecorder`). |
| `DST` (positional) | Destination dataset directory (created). |
| `--task` | Override the task string in the exported metadata. |
| `--fps` | Video framerate. Default `30`. |

Needs `uv pip install -e 'packages/robosandbox-core[lerobot]'`. See
[recording & export](../concepts/recording-and-export.md) for the
dataset layout.

### `results`

```bash
robo-sandbox results {list | slice | lineage | compare | regression | repro | sql | migrate} [args]
```

Query the eval log under `runs/eval_log/`. The log is populated as a
side effect of `robo-sandbox eval` and recording episodes — see
[the eval log concept](../concepts/eval-log.md). All subcommands print
to stdout; nothing is mutated.

#### `results list`

```bash
robo-sandbox results list
```

Lists every policy in the log with its kind, parent, and total eval
count. Example row:

```
distilled_v2                 ppo_neural             distilled_v1                       4
```

#### `results slice`

```bash
robo-sandbox results slice POLICY_ID --axis AXIS [--axis AXIS ...]
```

Per-slice success-rate breakdown for one policy along the given axes.
Pass `--axis` once per dimension (e.g. `--axis target_xy_bucket`).
Output is a tab-separated table — `axis_value`, `n_trials`, `success_rate`.

```bash
robo-sandbox results slice distilled_v1 --axis target_xy_bucket
# target_xy_bucket   n   success_rate
# back_left          8   0.125
# front_center       8   0.875
```

#### `results lineage`

```bash
robo-sandbox results lineage POLICY_ID
```

Prints the lineage chain root-to-leaf, indented per generation, with
the `lineage_op` in brackets when present.

```bash
robo-sandbox results lineage distilled_v2
# act_50k (lerobot_act)
#   distilled_v1 (ppo_neural) [distill]
#     distilled_v2 (ppo_neural) [fine_tune]
```

#### `results compare`

```bash
robo-sandbox results compare POLICY_A POLICY_B [--task TASK_ID]
```

Two-policy success-rate diff, optionally restricted to one task.
Output is two rates and a delta in percentage points.

```bash
robo-sandbox results compare distilled_v1 distilled_v2 --task pick_cube_franka_random
# distilled_v1: 34.4%
# distilled_v2: 50.0%
# delta:        +15.62pp
```

#### `results regression`

```bash
robo-sandbox results regression POLICY_ID
```

Walks one step up the lineage and prints per-task deltas vs the
parent. Each line ends with `^` for an improvement and `v` for a
regression — a quick scan tells you whether a fine-tune was a net win.

```bash
robo-sandbox results regression distilled_v2
# vs parent distilled_v1
#   pick_cube_franka_random: parent 34.4% -> child 50.0% (+15.62pp) ^
```

#### `results repro`

```bash
robo-sandbox results repro EVAL_ID
```

Prints the `git_sha` and full `command_line` recorded for one eval, so
you can reproduce it byte-for-byte.

```bash
robo-sandbox results repro eval_a1b2c3d4
# git_sha:  a9e6341
# command:  robo-sandbox eval --task pick_cube_franka_random --policy outputs/distilled_v1 --n-trials 32
```

#### `results sql`

```bash
robo-sandbox results sql "QUERY"
```

Runs raw DuckDB SQL against the JSONL views. Tables: `policies`,
`demos`, `evals`, `eval_runs`. Use this for one-off questions the
canned subcommands don't cover.

```bash
robo-sandbox results sql "SELECT policy_id, COUNT(*) FROM eval_runs WHERE success GROUP BY 1"
```

#### `results migrate`

```bash
robo-sandbox results migrate RUNS_DIR
```

One-shot import of legacy `runs/<ts>-<id>/result.json` directories
into the eval log. Idempotent — re-running skips already-imported
runs. Prints `imported N legacy runs`.

## `robo-sandbox-bench`

```bash
robo-sandbox-bench [--tasks NAME ...] [options]
```

Runs the built-in task suite. Without `--tasks`, runs every
non-experimental task (files under
`robosandbox/tasks/definitions/*.yaml` not prefixed with `_`).

| Flag | Default | Meaning |
|---|---|---|
| `--tasks NAME ...` | all | Subset of task names to run. |
| `--seeds N` | `1` | Seeds per task. Seed 0 is deterministic; seeds ≥ 1 sample the task's `randomize` block. |
| `--vlm-provider` | `stub` | Planner provider — same options as `run`. |
| `--model` | provider default | — |
| `--base-url` | provider default | — |
| `--api-key-env` | provider default | — |
| `--max-replans` | `3` | — |
| `--settle-steps` | `140` | Sim steps to run for gravity settle before agent starts. |
| `--out PATH` | `benchmark_results.json` | JSON results file. |
| `--log-level` | `WARNING` | — |

Output on stdout is a one-row-per-seed table with `OK` / `FAIL`,
wall seconds, replan count, and detail fields.

Examples:

```bash
uv run robo-sandbox-bench                                  # deterministic run
uv run robo-sandbox-bench --seeds 50                       # randomized + aggregated
uv run robo-sandbox-bench --tasks pick_cube_franka         # one task
uv run robo-sandbox-bench --tasks pick_ycb_mug --vlm-provider ollama
```

Experimental tasks (filename prefix `_experimental_`) are skipped by
default — pass them via `--tasks _experimental_stack_two` to opt in.

## Env vars honored by CLI-less paths

These fall back into the SDK and matter for the `examples/llm_guided.py` script:

- `OPENAI_API_KEY` — used by `--vlm-provider openai` and the OpenAI
  defaults in the examples.
- `ROBOSANDBOX_VLM_BASE_URL` — override the base URL in
  `examples/llm_guided.py`.
- `ROBOSANDBOX_VLM_MODEL` — override the model in
  `examples/llm_guided.py`.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | Success. |
| 1 | Run ran to completion but did not meet success criterion (policy mode). |
| 2 | Argument error (e.g. missing `--task` with `--policy`, task not found, policy failed to load). |

## Related

- [Quickstart](../quickstart.md) — what every subcommand does in
  context.
- [Tutorials](../tutorials/policy-replay.md) for end-to-end flows.
