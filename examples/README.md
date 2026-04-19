# Examples

Runnable scripts that exercise specific RoboSandbox features. Every
file is standalone — read the top docstring for usage. Assumes:

```bash
uv sync --package robosandbox --extra dev --extra viewer --extra meshes --extra lerobot
```

## Core features

| Script | What it shows |
|---|---|
| `list_ycb.py` | Print the bundled YCB catalog (ids, masses, hull counts). |
| `spawn_ycb_scene.py` | Spawn three YCB objects on the Franka, render one frame. |
| `custom_robot.py` | Load any URDF + sidecar via `Scene(robot_urdf=..., robot_config=...)`. |
| `custom_task.py` | Author a task in YAML, load it, run with the stub planner. |
| `custom_skill.py` | Implement and register a new skill in the agent loop. |
| `procedural_scene.py` | `tabletop_clutter(n_objects, seed)` — reproducible randomized scene. |
| `record_demo.py` | Record one episode (MP4 + events.jsonl + result.json). |
| `headless_eval.py` | Run the built-in benchmark tasks programmatically, print JSON. |
| `llm_guided.py` | Drive the agent with `VLMPlanner` (OpenAI or Ollama). |

## Agent + replan loop

| Script | What it shows |
|---|---|
| `run_custom_task.py` | Run any user-authored task YAML through the same agent loop the benchmark uses. |
| `replan_demo.py` | Trace the ReAct replan loop on a deliberately unreachable task. |
| `vlm_tool_calling_walkthrough.py` | Print the exact batch `VLMPlanner` sends to an OpenAI-compatible endpoint, with a mock client (no API key needed). |

## LeRobot workflow

Companions to `docs/site/docs/tutorials/lerobot-export.md` and
`docs/site/docs/tutorials/lerobot-policy-replay.md`.

| Script | What it shows |
|---|---|
| `inspect_lerobot_dataset.py` | Read the parquet + metadata produced by `robo-sandbox export-lerobot`. |
| `so_arm100/smoke_test.py` | Non-bundled URDF import flow end-to-end on the SO-ARM100. |
| `so_arm100/probe_hub_schemas.py` | Classify a list of public SO-100 ACT checkpoints by config schema (legacy / current). |
| `so_arm100/migrate_lerobot_config.py` | Rewrite a pre-lerobot-0.5 `config.json` into the current schema. |
| `so_arm100/run_so100_policy.py` | Run a public ACT checkpoint against the SO-ARM100 via `LeRobotPolicyAdapter` + `run_policy`. |

## Sim-to-real handoff

Companion to `docs/site/docs/tutorials/sim-to-real-handoff.md`.

| Script | What it shows |
|---|---|
| `real_robot_swap.py` | Subclass `RealRobotBackend` to satisfy the `SimBackend` Protocol; documents what carries over vs what doesn't. |
| `so101_handoff/so101_backend.py` | Concrete SO-101 skeleton with `_TODO(real)` blocks where a real driver replaces the stub. |
| `so101_handoff/run_home_skill.py` | The `Home` skill runs unchanged against the real-backend subclass. |

## Supporting data

- `so_arm100/` — non-bundled SO-ARM100 URDF + meshes + sidecar YAML (Apache-2.0, from `google-deepmind/mujoco_menagerie`).
- `tasks/` — example user-authored task YAMLs consumed by `run_custom_task.py` and `replan_demo.py`.
