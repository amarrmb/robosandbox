# Running the agent

One command, one pick, three artifacts on disk. Then replay them.

![franka pick](../assets/demos/franka_pick.gif){ loading=lazy }

## The three ways to run

| Entry point | When to reach for it | What it writes |
|---|---|---|
| **`robo-sandbox-bench`** | Reproducible test runs with success tracking | `benchmark_results.json` |
| **`robo-sandbox viewer`** | Interactive exploration + recording on demand | `runs/<id>/{video.mp4, events.jsonl, result.json}` *(when Record is on)* |
| **`robo-sandbox run`** | One-off scripted run from the CLI | `runs/<id>/{video.mp4, events.jsonl, result.json}` |

All three share the same `Agent` loop and every skill. The difference
is in the *scene source* and what gets written out.

## The fastest successful pick

```bash
uv run robo-sandbox-bench --tasks pick_cube_franka --seeds 1
```

Produces (the `franka_bench` GIF from the [URDF guide](./bring-your-own-robot.md)):

```
TASK               SEED  RESULT   SECS  REPLANS DETAIL
------------------------------------------------------------------------
pick_cube_franka   0     OK        1.6        0 dz_mm=166.905, min_mm=50.000

SUMMARY: 1/1 successful
```

Bench runs are headless — no window opens, no video written. If you
want the video + event log, use the **viewer** or the **`run`** CLI.

## Running from the viewer (recommended first time)

```bash
uv run robo-sandbox viewer --task pick_cube_franka
# open http://localhost:8000
```

1. Click **Record** (saves MP4 + events.jsonl when the episode ends).
2. Click **Run**.
3. Watch the pick.
4. Scrub the **Inspector** slider to step back through every frame.

## What's in `runs/<id>/`

![run artifacts](../assets/demos/run_artifacts.gif){ loading=lazy }

```
runs/20260418-230155-4471d060/
├── episode.json      # 4 lines: episode_id, task, started_at, sim_dt
├── events.jsonl      # one line per sim step: joints, ee_pose, objects, gripper
├── result.json       # verdict: success, frames, wall, reason
└── video.mp4         # 30 fps render of the agent's camera
```

**`result.json`** is the line most humans read:

```json
{
  "episode_id": "4471d060",
  "success": true,
  "ended_at": "2026-04-18T23:02:30.623",
  "frames": 1656,
  "task": "pick_cube_franka",
  "wall": 26.29,
  "reason": "plan_complete"
}
```

**`events.jsonl`** is the line policies are trained from. One object per sim tick (`dt=0.005`s ≈ 200 Hz) with:

- `t` — sim time in seconds
- `frame_idx` — zero-based
- `robot_joints` — full DoF vector
- `ee_pose` — `{xyz, quat_xyzw}` in world frame
- `gripper_width` — meters between fingertips
- `objects` — every scene object's current pose
- `action` — what the skill commanded at that tick (if any)

This is the raw stream. For training off-policy, convert to a LeRobot
dataset with:

```bash
uv run robo-sandbox export-lerobot runs/<id> datasets/mypolicy
```

## Swapping the planner

`robo-sandbox run` and the viewer both support `--vlm-provider`:

```bash
# regex planner, zero deps (default)
robo-sandbox run "pick up the red cube" --vlm-provider stub

# local Ollama with a vision model
ollama pull llama3.2-vision
ollama serve &
robo-sandbox run "pick up the red cube" --vlm-provider ollama

# OpenAI
export OPENAI_API_KEY=sk-...
robo-sandbox run "pick up the red cube" --vlm-provider openai --model gpt-4o-mini

# any OpenAI-compatible endpoint (vLLM, together.ai, groq, ...)
robo-sandbox run "..." --vlm-provider custom --base-url https://... --api-key-env TOGETHER_API_KEY
```

The `Agent` loop is identical across all four. Only the `Planner`
instance differs. See the [VLM tool-calling guide](./vlm-tool-calling.md) for the wire-level view of what each provider sees.

## Watching the phases

`PLAN`/`EXECUTE` log lines surface the ReAct loop:

![phase logs](../assets/demos/phases_log.gif){ loading=lazy }

```
PLAN:    task='pick up the red cube' replan=0
EXECUTE: pick({'object': 'red_cube'})
TASK               SEED  RESULT   SECS  REPLANS DETAIL
---------------------------------------------------------
pick_cube_franka   0     OK        1.1        0 dz_mm=166.905
```

One `PLAN` → one `EXECUTE` → success. On a failed skill, you'd see
`PLAN` again with a higher `replan=N` counter. That's the recovery
loop; see [Replan loop](./replan-loop.md) for what prior_attempts feedback looks like.

## Reading `result.json` programmatically

```python
import json
from pathlib import Path

for run_dir in sorted(Path("runs").iterdir()):
    r = json.loads((run_dir / "result.json").read_text())
    if r.get("success"):
        print(f"{run_dir.name}  {r['task']}  {r['wall']:.1f}s  {r['frames']} frames")
```

Common reasons surfaced in `result.json.reason`:

| reason | meaning |
|---|---|
| `plan_complete` | every skill in the plan succeeded |
| `already_done` | planner returned empty plan on the first call |
| `replan_exhausted` | `max_replans` hit; see the final skill's detail |
| `vlm_transport` | VLM API error (timeout, auth, rate limit) |
| `stopped_by_user` | viewer Record stopped mid-episode |

## What's next

- [Bring your own task](./bring-your-own-task.md) — author a YAML, run it through the same path.
- [Replan loop](./replan-loop.md) — trace a deliberately failing run.
- [VLM tool-calling](./vlm-tool-calling.md) — what each provider actually sees.
