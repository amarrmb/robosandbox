# Recording & export

Every run in RoboSandbox can be written to disk. From there you can
inspect it directly or export it to LeRobot v3 for training.

## `LocalRecorder`

Writes one directory per episode under `runs/` (or any `--runs-dir`
path):

```
runs/20260418-094533-1a2b3c4d/
├── episode.json    # task, started_at, sim_dt, metadata you passed in
├── events.jsonl    # one line per recorded frame (see schema below)
├── result.json     # success, ended_at, frames, reason
└── video.mp4       # subsampled RGB at video_fps (default 30)
```

`LocalRecorder` implements the `RecordSink` protocol:

```python
class RecordSink(Protocol):
    def start_episode(self, task: str, metadata: dict) -> str: ...
    def write_frame(self, obs: Observation, action: dict | None = None) -> None: ...
    def end_episode(self, success: bool, result: dict) -> None: ...
```

If you pass `sim_dt` in `metadata`, `LocalRecorder` subsamples frames so
the video stays in wall time even if the sim is running faster than
real time.

## `events.jsonl`

One JSON object per line:

```json
{
  "t": 0.016667,
  "frame_idx": 0,
  "robot_joints": [0.0, -0.4, 0.8, ...],
  "ee_pose": {"xyz": [0.3, 0.0, 0.3], "quat_xyzw": [1, 0, 0, 0]},
  "gripper_width": 0.07,
  "objects": {
    "red_cube": {"xyz": [0.4, 0.0, 0.05], "quat_xyzw": [0, 0, 0, 1]}
  },
  "action": {"joints": [...], "gripper": 0.0}
}
```

`action` is whatever the caller passes to `write_frame(obs, action)`.
The recorder does not inspect it. The only convention today is that
`events.jsonl` is the input to `ReplayTrajectoryPolicy`, which reads
either `joints`/`gripper` or falls back to `robot_joints`/normalised
`gripper_width`.

## Wiring it in

### Scripted (headless)

```python
from robosandbox.recorder.local import LocalRecorder

recorder = LocalRecorder(root="runs")
recorder.start_episode(
    task="pick up the apple",
    metadata={"sim_dt": sim.model.opt.timestep},
)
# stream frames from the sim
ctx = AgentContext(
    sim=sim, perception=..., grasp=..., motion=...,
    recorder=recorder,
    on_step=lambda: recorder.write_frame(sim.observe()),
)
result = Pick()(ctx, object="apple")
recorder.end_episode(success=result.success, result={"reason": result.reason})
```

For a runnable version, see `examples/record_demo.py`.

### Viewer

Toggle **Record** in the viewer sidebar before hitting Run. Episodes
land in `./runs/` by default; override with
`robo-sandbox viewer --runs-dir /path/to/elsewhere`.

## CLI shortcut

```bash
# scripted demo that records
uv run python examples/record_demo.py --out-dir runs
```

Any CLI path that drives the agent loop can also plug a recorder into
`AgentContext`.

## Export to LeRobot v3

```bash
uv pip install -e 'packages/robosandbox-core[lerobot]'
robo-sandbox export-lerobot runs/20260418-094533-1a2b3c4d /tmp/my_dataset
```

This writes:

```
/tmp/my_dataset/
├── meta/
│   ├── info.json
│   ├── tasks.jsonl
│   └── episodes.jsonl
├── data/chunk-000/episode_000000.parquet
└── videos/chunk-000/observation.images.scene/episode_000000.mp4
```

The export makes a few schema choices that are worth being explicit
about:

- `observation.state` = `concat(robot_joints, [gripper_width])`
  float32. Convention most LeRobot datasets (Aloha, Koch, SO-100)
  follow.
- `action` = the recorded `action` field when numeric; otherwise a
  copy of `observation.state` (standard teleop-less fallback).
- `observation.images.scene` is stored as a `VideoFrame` reference,
  not inlined bytes.

Each export is single-episode. If you want a multi-episode dataset, run
`export-lerobot` multiple times and merge the results yourself.

From Python:

```python
from robosandbox.export.lerobot import export_episode
from pathlib import Path

out = export_episode(
    Path("runs/20260418-094533-1a2b3c4d"),
    Path("/tmp/my_dataset"),
    task=None,    # None → read from episode.json
    fps=30,
)
```

## Still deferred

- An MCAP recorder (protocol-stable, will drop in behind the same
  `RecordSink` interface).
- Trajectory scrubber in the viewer — requires in-RAM episode buffer.
  See [roadmap](../reference/roadmap.md).

## Related

- [Policy replay tutorial](../tutorials/policy-replay.md) — turn a
  recording into a policy and run it back in sim.
- [CLI: `export-lerobot`](../reference/cli.md#export-lerobot)
