# Examples

Runnable scripts that exercise specific RoboSandbox features. Each file
is standalone and self-contained — read the top docstring for usage.

Assumes `uv sync --package robosandbox --extra dev --extra viewer
--extra meshes` has been run once.

| Script | What it shows |
|---|---|
| `list_ycb.py` | Print the bundled YCB catalog (ids, masses, hull counts). |
| `spawn_ycb_scene.py` | Spawn three YCB objects on the Franka, render one frame. |
| `custom_robot.py` | Load any URDF + sidecar via `Scene(robot_urdf=..., robot_config=...)`. |
| `custom_task.py` | Author a task in YAML, load it, run with the stub planner. |
| `custom_skill.py` | Implement and register a new skill (`tap`) in the agent loop. |
| `record_demo.py` | Record one episode (MP4 + events.jsonl + result.json). |
| `headless_eval.py` | Run the 6 built-in benchmarks programmatically, print JSON. |
| `llm_guided.py` | Drive the agent with VLMPlanner (OpenAI or Ollama). |
| `procedural_scene.py` | `tabletop_clutter(n_objects, seed)` — reproducible randomized scene. |

All 9 examples ship with Sprint A. Coming in future slices:
policy-in-the-loop replay (4.3), LeRobot export (4.2).
