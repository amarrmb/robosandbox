---
name: Examples README improvements
description: Rewrite examples/README.md to replace summary tables with per-section docs that each have a run command and expected output
type: project
---

# Design: Examples README — run instructions + expected output

## Goal

Replace the current table-of-contents format in `examples/README.md` with
per-example sections that each contain:
- 1–2 sentence description (what it exercises, when to reach for it)
- `Run:` code block with the exact command (including required env vars)
- `Expect:` prose sentence + trimmed real terminal output

Also add a clear definition of YCB (Yale-Carnegie Mellon-Berkeley Object and
Model Set) on first use.

## Structure

```
# Examples

<intro paragraph>

## Prerequisites

<uv sync block>

## Core features
### list_ycb.py
### spawn_ycb_scene.py
### custom_robot.py
### custom_task.py
### custom_skill.py
### procedural_scene.py
### record_demo.py
### headless_eval.py
### llm_guided.py

## Agent + replan loop
### run_custom_task.py
### replan_demo.py
### vlm_tool_calling_walkthrough.py

## LeRobot workflow
### inspect_lerobot_dataset.py
### so_arm100/smoke_test.py
### so_arm100/probe_hub_schemas.py
### so_arm100/migrate_lerobot_config.py
### so_arm100/run_so100_policy.py

## Sim-to-real handoff
### real_robot_swap.py
### so101_handoff/so101_backend.py
### so101_handoff/run_home_skill.py

## Supporting data
<brief note about tasks/ YAMLs and so_arm100/assets/>
```

## Per-section template

```markdown
### <script>.py

<1–2 sentences: what the script exercises and when to reach for it>

```bash
uv run python examples/<script>.py [args]
```

<1 sentence describing what happens.> Expected output:

```
<trimmed real terminal output>
```
```

For scripts needing an API key or optional dep, the Run block opens with
`# needs ...` comment — matching the main README's convention.

For `so101_handoff/so101_backend.py` (a skeleton, not a runner): describe
what to fill in and link to the tutorial. No Run block.

## YCB definition placement

In `list_ycb.py`'s section (first use), expand the acronym:

> YCB (Yale-Carnegie Mellon-Berkeley) is a benchmark set of 77 everyday
> household objects — cans, bottles, tools — photographed and 3D-scanned
> for robotics research. RoboSandbox ships 10 pre-decomposed YCB objects
> with their collision hulls ready to use.

Subsequent sections use "YCB" without definition.

## Real output captured

All output blocks below are from actual runs and should be used verbatim
(trimmed for brevity where noted).

### list_ycb.py
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
Source inspection (renders, no text beyond filenames):
```
compiled: N bodies, M meshes, K geoms
wrote examples/out.png
```

### custom_robot.py
```
Loaded robot from panda.xml
  arm joints:       ('joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7')
  arm actuators:    ('actuator1', 'actuator2', ...)
  gripper primary:  finger_joint1
  end-effector site: robosandbox_ee_site
  home qpos:        (0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853)

Ready: Scene(robot_urdf=panda.xml, robot_config=panda.robosandbox.yaml)
```

### custom_task.py
```
loaded task: 'my_pick_apple' — 'pick up the apple'
scene has 1 objects

episode success: True
  plan: ['pick']
  steps: 1
    pick({'object': 'apple'}) -> SkillResult(OK)
```

### custom_skill.py
```
tap result: SkillResult(OK)
agent episode success=True, steps=1
```

### procedural_scene.py
```
tabletop_clutter(n=5, seed=0):
  mug                   at (0.473, -0.041, 0.070)
  baseball              at (0.436, 0.043, 0.070)
  cracker_box           at (0.376, -0.084, 0.070)
  mustard_bottle        at (0.347, 0.097, 0.070)
  apple                 at (0.321, -0.011, 0.070)

settled poses:
  mug                   at (0.473, -0.041, 0.040)
  ...
```

### record_demo.py
Source inspection (writes MP4 + JSONL + result.json, prints file listing):
```
recorded episode <id>
  runs/<timestamp>-<id>/
    events.jsonl (N B)
    result.json (M B)
    video.mp4 (K B)
```

### headless_eval.py (two tasks shown)
```json
{
  "n_tasks": 2,
  "n_success": 2,
  "results": [
    {"task": "pick_cube",  "success": true, "plan": ["pick"], "steps": 1, "replans": 0, "final_reason": "plan_complete"},
    {"task": "home",       "success": true, "plan": ["home"], "steps": 1, "replans": 0, "final_reason": "plan_complete"}
  ]
}
```

### run_custom_task.py
```
Task: pick_yellow_cube
Prompt: pick up the yellow cube
Success: lifted (object=yellow_cube)
Seeds: 1

  seed=0  OK   wall=24.2s  reason=plan_complete

SUMMARY: 1/1 successful
```

### replan_demo.py
```
08:54:33  robosandbox.agent     PLAN: task='pick up the red cube' replan=0
08:54:34  robosandbox.agent     EXECUTE: pick({'object': 'red_cube'})
08:54:35  robosandbox.agent     PLAN: task='pick up the red cube' replan=1
...

FINAL VERDICT
  success:     False
  replans:     2
  final reason: replan_exhausted
  detail:      pick failed: unreachable — IK did not converge (pos_err=0.705m)
```

### vlm_tool_calling_walkthrough.py
(trimmed — show all 4 sections with key lines)

### inspect_lerobot_dataset.py
```
Dataset:        datasets/pick_demo
LeRobot:        v3.0
Episodes:       1
Total frames:   89
fps:            30

State dim:      8
Action dim:     8
State names:    ['joint_0', 'joint_1', ..., 'gripper']
Video key:      observation.images.scene  (h264, 30 fps)

Tasks (1):
  [0] 'pick up the apple'
```

### so_arm100/smoke_test.py
```
== reachability pre-flight ==
reachability: all objects look reachable

== sim load + settle ==
n_dof:       5
joint_names: ['Rotation', 'Pitch', 'Elbow', 'Wrist_Pitch', 'Wrist_Roll']
ee_pose:     (-0.200, -0.419, +0.225)
```

### real_robot_swap.py
```
[fake_so101] load scene with 0 objects
[fake_so101] reset to home
[example] sim-or-real run ok.  n_dof=6  t=0.025
```

### so101_handoff/run_home_skill.py
```
before home: [ 0.5 -0.5  0.5  0.3 -0.3]
after home:  [ 0.  -1.4  1.4  0.   0. ]
result:      success=True reason='homed'
home error:  0.00mm-equivalent joint norm
```

## Scripts needing external deps / network

- `llm_guided.py` — needs `OPENAI_API_KEY` or Ollama running locally
- `so_arm100/probe_hub_schemas.py` — needs network + `huggingface_hub`; show sample output from docstring
- `so_arm100/run_so100_policy.py` — needs `[lerobot]` extra + downloads from HF Hub; note this clearly
- `so_arm100/migrate_lerobot_config.py` — takes a config.json path argument; show usage pattern

## What does NOT change

- Group headers (Core features / Agent + replan loop / LeRobot workflow / Sim-to-real handoff)
- The prerequisite `uv sync` command at the top
- The Supporting data section (brief, stays as prose)
