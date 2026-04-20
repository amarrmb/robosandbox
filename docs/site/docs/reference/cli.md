# CLI reference

Two console scripts ship:

- `robo-sandbox` — subcommand entry point (`demo`, `viewer`, `run`,
  `export-lerobot`).
- `robo-sandbox-bench` — the benchmark runner.

## `robo-sandbox`

```
robo-sandbox {demo | viewer | run | export-lerobot} [options]
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
