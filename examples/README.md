# Examples

Runnable scripts that exercise specific RoboSandbox features. Each section
shows the exact command and what to expect in the terminal.

## Prerequisites

```bash
uv sync --package robosandbox --extra dev --extra viewer --extra meshes --extra lerobot
```

Scripts that render or step the sim need a headless GL backend:

```bash
export MUJOCO_GL=osmesa    # or egl if your machine has it
```

---

## Core features

### list_ycb.py

YCB (Yale-Carnegie Mellon-Berkeley) is a benchmark set of everyday household
objects — cans, bottles, tools — photographed and 3D-scanned for robotics
research. RoboSandbox ships 10 pre-decomposed YCB objects with their collision
hulls ready to use. This script lists all of them with their mass, hull count,
and on-disk size.

```bash
uv run python examples/list_ycb.py
```

Prints a table of all 10 bundled objects:

```
10 bundled YCB objects:

  id                            mass   hulls   bytes(visual+hulls)
  -------------------------     ----   -----   --------------------
  003_cracker_box            0.411kg       1      578.9 KB
  005_tomato_soup_can        0.349kg       1      716.2 KB
  006_mustard_bottle         0.603kg       1      715.5 KB
  011_banana                 0.066kg       2      798.7 KB
  013_apple                  0.068kg       2      860.8 KB
  024_bowl                   0.147kg      11      868.9 KB
  025_mug                    0.118kg      15      998.5 KB
  035_power_drill            0.895kg       3      839.4 KB
  042_adjustable_wrench      0.252kg       2      583.5 KB
  055_baseball               0.148kg       1      793.3 KB
```

### spawn_ycb_scene.py

Spawn three YCB objects (apple, soup can, banana) alongside the Franka arm,
run physics to let them settle, and save a rendered frame as a PNG. Shows the
`@ycb:` mesh resolver and the full mesh-import pipeline from scene definition
to MuJoCo render.

```bash
uv run python examples/spawn_ycb_scene.py --out examples/out.png
```

Compiles the scene, steps physics for 120 ticks, then writes the image:

```
compiled: 15 bodies, 19 meshes, 33 geoms
wrote examples/out.png
```

### custom_robot.py

Load any URDF or MJCF robot by pointing `Scene` at a file path and a small
sidecar YAML that describes joints, actuators, and gripper geometry. This
example uses the bundled Franka Panda; swap the paths to use your own robot.

```bash
uv run python examples/custom_robot.py
```

Prints the parsed robot spec loaded from the URDF and sidecar:

```
Loaded robot from panda.xml
  arm joints:       ('joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7')
  arm actuators:    ('actuator1', 'actuator2', 'actuator3', 'actuator4', 'actuator5', 'actuator6', 'actuator7')
  gripper primary:  finger_joint1
  end-effector site: robosandbox_ee_site
  home qpos:        (0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853)

Ready: Scene(robot_urdf=panda.xml, robot_config=panda.robosandbox.yaml)
Objects: 0 (add SceneObject(...) tuples to populate)
```

### custom_task.py

Author a task as a YAML string (or file), load it at runtime, and run it
through the stub planner. Task YAMLs bundle a scene description, a
natural-language prompt, and a declarative success criterion — no executable
check code, so they are diffable and versionable.

```bash
uv run python examples/custom_task.py
```

Runs one pick episode against a mini apple-pick task and reports success:

```
loaded task: 'my_pick_apple' — 'pick up the apple'
scene has 1 objects

episode success: True
  plan: ['pick']
  steps: 1
    pick({'object': 'apple'}) -> SkillResult(OK)
```

### custom_skill.py

Implement a new skill as a plain Python class that satisfies the `Skill`
protocol (`name`, `description`, `parameters_schema`, `__call__`), then hand
it to `Agent`. No registration, no base class required. This example adds a
`tap` primitive that touches the top of a named object.

```bash
uv run python examples/custom_skill.py
```

The agent dispatches the tap skill and reports success:

```
tap result: SkillResult(OK)
agent episode success=True, steps=1
```

### procedural_scene.py

Generate a reproducible tabletop scene by calling `tabletop_clutter(n, seed)`.
Each seed produces a distinct, non-overlapping arrangement of YCB distractors.
Useful for building randomized benchmarks without hand-placed object
definitions.

```bash
uv run python examples/procedural_scene.py
uv run python examples/procedural_scene.py --n 7 --seed 3 --out examples/scene.png
```

Prints the initial spawn poses and the settled poses after physics steps:

```
tabletop_clutter(n=5, seed=0):
  mug                   at (0.473, -0.041, 0.070)
  baseball              at (0.436, 0.043, 0.070)
  cracker_box           at (0.376, -0.084, 0.070)
  mustard_bottle        at (0.347, 0.097, 0.070)
  apple                 at (0.321, -0.011, 0.070)

settled poses:
  mug                   at (0.473, -0.041, 0.040)
  baseball              at (0.436, 0.043, 0.039)
  cracker_box           at (0.380, -0.086, 0.039)
  mustard_bottle        at (0.347, 0.097, 0.040)
  apple                 at (0.307, -0.008, 0.039)
```

### record_demo.py

Run one pick episode with `LocalRecorder` attached. The recorder writes an
MP4 video, a per-step `events.jsonl`, and a `result.json` summary to a
timestamped directory under `--out-dir`.

```bash
uv run python examples/record_demo.py --out-dir runs
```

Picks the apple, then lists the files written:

```
recorded episode d8828fc6
  runs/20260421-085535-d8828fc6/
    episode.json  (157 B)
    events.jsonl  (58566 B)
    result.json   (131 B)
    video.mp4     (76521 B)
```

### headless_eval.py

Run the built-in benchmark tasks programmatically and print JSON results.
Wraps `robo-sandbox-bench` at the library level so you can embed it in CI
or a notebook without the CLI.

```bash
uv run python examples/headless_eval.py
uv run python examples/headless_eval.py --tasks pick_cube,home
```

Prints a JSON summary for each task:

```json
{
  "n_tasks": 2,
  "n_success": 2,
  "results": [
    {"task": "pick_cube", "success": true, "plan": ["pick"], "steps": 1, "replans": 0, "final_reason": "plan_complete"},
    {"task": "home",      "success": true, "plan": ["home"], "steps": 1, "replans": 0, "final_reason": "plan_complete"}
  ]
}
```

### llm_guided.py

Drive the agent with `VLMPlanner` against an OpenAI-compatible endpoint.
Set `OPENAI_API_KEY` for the OpenAI API, or pass `--vlm-provider ollama`
to point at a local Ollama server. Without a key the script exits early
with a clear message.

```bash
# OpenAI
OPENAI_API_KEY=sk-... uv run python examples/llm_guided.py "pick the apple"

# Ollama — no key needed
# ollama pull llama3.2-vision && ollama serve
uv run python examples/llm_guided.py --vlm-provider ollama "pick the apple"
```

The VLMPlanner calls the model, builds a plan, then executes it — output
matches the structure of `replan_demo.py` but with live model responses in
the PLAN lines.

---

## Agent + replan loop

### run_custom_task.py

Load any task YAML from an arbitrary path and run it through the same agent
loop the benchmark uses, optionally across multiple randomized seeds.

```bash
uv run python examples/run_custom_task.py examples/tasks/pick_yellow_cube.yaml
uv run python examples/run_custom_task.py examples/tasks/pick_yellow_cube.yaml --seeds 5
```

Reports per-seed success and a final summary:

```
Task: pick_yellow_cube
Prompt: pick up the yellow cube
Success: lifted (object=yellow_cube)
Seeds: 1

  seed=0  OK   wall=24.2s  reason=plan_complete

SUMMARY: 1/1 successful
```

### replan_demo.py

Trace the ReAct replan loop on a deliberately unreachable task. The cube
is placed outside the Franka's reach envelope so every pick attempt fails;
the agent replans until `max_replans` is exhausted. Shows how
`prior_attempts` feed back into the planner on each cycle.

```bash
uv run python examples/replan_demo.py
```

Logs each plan/execute cycle, then prints the final verdict:

```
08:54:33  robosandbox.agent     PLAN: task='pick up the red cube' replan=0
08:54:34  robosandbox.agent     EXECUTE: pick({'object': 'red_cube'})
08:54:35  robosandbox.agent     PLAN: task='pick up the red cube' replan=1
08:54:35  robosandbox.agent     EXECUTE: pick({'object': 'red_cube'})
08:54:35  robosandbox.agent     PLAN: task='pick up the red cube' replan=2
08:54:36  robosandbox.agent     EXECUTE: pick({'object': 'red_cube'})

FINAL VERDICT
  success:     False
  replans:     2
  final reason: replan_exhausted
  detail:      pick failed: unreachable — IK did not converge (pos_err=0.705m)
```

### vlm_tool_calling_walkthrough.py

Print the exact request `VLMPlanner` sends to an OpenAI-compatible endpoint
and the response it parses, using a mock client with a canned response. No
API key needed. Shows how skills become tool definitions and how
`tool_calls` in the model response map back to `SkillCall` objects.

```bash
uv run python examples/vlm_tool_calling_walkthrough.py
```

Prints four labeled sections — tool definitions, the chat request (image
data-url truncated), the canned model response, and the parsed `SkillCall`
list:

```
──────────────────────────────────────────────────────────────────────
  1. SKILLS → OPENAI TOOL DEFINITIONS
──────────────────────────────────────────────────────────────────────
[
  {"type": "function", "function": {"name": "done", ...}},
  {"type": "function", "function": {"name": "pick", ...}},
  ...plus 2 more tools (place_on, home).
]

──────────────────────────────────────────────────────────────────────
  4. PARSED SkillCall's — what the Agent executes
──────────────────────────────────────────────────────────────────────
  SkillCall(name='pick'     arguments={'object': 'red_cube'}   tool_call_id='call_abc123')
  SkillCall(name='place_on' arguments={'target': 'green_cube'} tool_call_id='call_def456')

Total VLM calls made: 1
```

---

## LeRobot workflow

Companions to `docs/site/docs/tutorials/lerobot-export.md` and
`docs/site/docs/tutorials/lerobot-policy-replay.md`.

### inspect_lerobot_dataset.py

Read the parquet frame table and metadata JSON that `robo-sandbox
export-lerobot` produces. Prints episode counts, state/action
dimensionality, joint names, and a per-episode task list. Use as a
smoke-check after exporting a dataset.

```bash
uv run python examples/inspect_lerobot_dataset.py datasets/pick_demo
```

Prints a summary of the dataset structure:

```
Dataset:        datasets/pick_demo
LeRobot:        v3.0
Episodes:       1
Total frames:   89
fps:            30

State dim:      8
Action dim:     8
State names:    ['joint_0', 'joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6', 'gripper']
Video key:      observation.images.scene  (h264, 30 fps)

Tasks (1):
  [0] 'pick up the apple'

Episodes (1):
  episode_000000  length=89  tasks=['pick up the apple']
```

### so_arm100/smoke_test.py

Load the non-bundled SO-ARM100 URDF from `examples/so_arm100/`, run the
reachability pre-flight, step the sim, and verify joint names, DoF count,
and gripper convention. Exercises the full bring-your-own-robot flow from
`docs/site/docs/guides/bring-your-own-robot.md`.

```bash
uv run python examples/so_arm100/smoke_test.py
```

Prints the reachability result and a sim sanity check:

```
== reachability pre-flight ==
reachability: all objects look reachable

== sim load + settle ==
n_dof:       5
joint_names: ['Rotation', 'Pitch', 'Elbow', 'Wrist_Pitch', 'Wrist_Roll']
ee_pose:     (-0.200, -0.419, +0.225)
gripper_w:   3.0000  (open_qpos → large; closed_qpos → ~0)

== gripper sanity — command closed, then back to open ==
closed:      0.3500
open:        0.6670
```

### so_arm100/probe_hub_schemas.py

Fetch a list of public SO-100 ACT checkpoints from the Hugging Face Hub and
classify each by config schema — legacy (pre-lerobot 0.5) vs current.
Grounds the claim in the LeRobot policy replay tutorial that most public
SO-100 checkpoints ship the old config shape.

```bash
# needs network + huggingface_hub
uv run python examples/so_arm100/probe_hub_schemas.py
```

Prints a verdict for each checkpoint:

```
cadene/act_so100_5_lego_test_080000        legacy (no type/input_features)
satvikahuja/act_so100_test                 legacy (no type/input_features)
koenvanwijk/act_so100_test                 legacy (no type/input_features)
Chojins/so100_test20                       legacy (no type/input_features)
pingev/lerobot-so100-1                     legacy (no type/input_features)
maximilienroberti/act_so100_lego_red_box   legacy (no type/input_features)
```

### so_arm100/migrate_lerobot_config.py

Rewrite a pre-`lerobot 0.5` `config.json` in place to the schema that
`LeRobotPolicyAdapter` expects. If you download a public SO-100 checkpoint
that was trained before lerobot 0.5, run this before loading it.

```bash
uv run python examples/so_arm100/migrate_lerobot_config.py /path/to/ckpt/config.json
```

Reads the legacy `input_shapes` / `output_shapes` fields and writes an
updated config with `input_features`, `output_features`, and a `type` field.
Prints a confirmation on completion:

```
migrated /path/to/ckpt/config.json  (wrote N keys, dropped 4 legacy keys)
```

### so_arm100/run_so100_policy.py

Download a public SO-100 ACT checkpoint, migrate its config schema, load it
via `LeRobotPolicyAdapter`, and run a closed-loop rollout in sim with
recording. Exercises the full sim-to-policy pipeline end to end.

```bash
# needs [lerobot] extra + ~1 GB checkpoint download
uv run python examples/so_arm100/run_so100_policy.py
```

Downloads the checkpoint once (cached to `~/.cache/huggingface/`), migrates
the config, runs 200 rollout steps, and writes a recording:

```
downloading satvikahuja/act_so100_test ...
migrating config schema (legacy → current) ...
policy loaded: ACT  state_dim=7  action_dim=7
running 200-step rollout ...
recorded episode <id> → runs/<timestamp>-<id>/
```

---

## Sim-to-real handoff

Companion to `docs/site/docs/tutorials/sim-to-real-handoff.md`.

### real_robot_swap.py

Subclass `RealRobotBackend` with a fake hardware bridge that prints what it
would do instead of driving motors. Shows that any code consuming the
`SimBackend` protocol — observation+step skills, `LocalRecorder`,
`LeRobotPolicyAdapter` — runs unchanged against a real-backend subclass.

```bash
uv run python examples/real_robot_swap.py
```

The fake backend satisfies the protocol and confirms it end to end:

```
[fake_so101] load scene with 0 objects
[fake_so101] reset to home
[example] sim-or-real run ok.  n_dof=6  t=0.025
```

### so101_handoff/so101_backend.py

A concrete SO-101 backend skeleton with `_TODO(real)` blocks marking every
place where a real hardware driver replaces the stub. The stubs track
commanded joint state in software so every observation+step skill runs
unchanged against it.

This file is a starting point, not a runner. Open it, read the `_TODO(real)`
annotations, and fill them in with calls to your serial or SDK driver. See
`docs/site/docs/tutorials/sim-to-real-handoff.md` for the first-run safety
checklist before touching real hardware.

### so101_handoff/run_home_skill.py

Run the `Home` skill against the SO-101 skeleton backend without any
hardware. Proves the sim-to-real interface holds for observation+step skills:
the same `AgentContext` and `Home` call that work in sim work here unchanged.

```bash
uv run python examples/so101_handoff/run_home_skill.py
```

The skeleton tracks joint motion in software and confirms zero error at home:

```
before home: [ 0.5 -0.5  0.5  0.3 -0.3]
after home:  [ 0.  -1.4  1.4  0.   0. ]
result:      success=True reason='homed'
home error:  0.00mm-equivalent joint norm

HANDOFF VERDICT: the Home skill ran against a RealRobotBackend subclass
with no changes. Observation+step skills transfer for free; see the
tutorial for the motion-planning caveat.
```

---

## Supporting data

- `tasks/` — two example task YAMLs: `pick_yellow_cube.yaml` (reachable,
  randomized) used by `run_custom_task.py`, and `pick_unreachable_cube.yaml`
  (cube placed outside the Franka's reach envelope) used by `replan_demo.py`.
- `so_arm100/assets/` — SO-ARM100 URDF, STL meshes, and sidecar YAML
  (Apache-2.0, from `google-deepmind/mujoco_menagerie`). Used by
  `smoke_test.py` and `run_so100_policy.py`.
