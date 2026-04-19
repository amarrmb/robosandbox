"""Export a recorded episode (`runs/<ts>-<id>/`) to LeRobot v3 dataset layout.

Consumes the artefacts written by :class:`robosandbox.recorder.local.LocalRecorder`
(`episode.json`, `events.jsonl`, `video.mp4`) and produces a LeRobot v3 dataset::

    <dst>/
      meta/
        info.json
        tasks.jsonl
        episodes.jsonl
      data/chunk-000/episode_000000.parquet
      videos/chunk-000/observation.images.scene/episode_000000.mp4

Schema choices (documented here because LeRobot v3's spec has wiggle room):

* ``observation.state`` = concat(``robot_joints``, ``[gripper_width]``) as a single
  float32 vector of length ``n_dof + 1``. This is the convention most
  open-source LeRobot datasets (Aloha, Koch, SO-100) follow.
* ``action`` = frame's recorded action vector if present and numeric; otherwise
  a copy of ``observation.state`` (the standard fallback for teleop-less
  scripted demonstrations — action == next-state).
* ``observation.images.scene`` is stored as a video reference (not inlined
  bytes). LeRobot's ``VideoFrame`` feature type handles this.
* Single-episode exports only: ``episode_index = 0``, one task, one chunk.
  Multi-episode stitching is a separate concern.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

_CHUNK = "chunk-000"
_EPISODE_ID = 0
_VIDEO_KEY = "observation.images.scene"


def _require_pyarrow():  # pragma: no cover - import guard
    try:
        import pyarrow
        import pyarrow.parquet  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "pyarrow is required for LeRobot export. Install with: "
            "pip install 'robosandbox[lerobot]'  (or pip install pyarrow>=15)"
        ) from e


def _coerce_action(action: Any, fallback: list[float]) -> list[float]:
    """Turn a recorded `action` field into a flat float vector.

    Falls back to `fallback` (typically the current state) if the action is
    missing or not a numeric sequence.
    """
    if action is None:
        return list(fallback)
    # Common shapes: {"joints": [...]}, {"qpos_target": [...]}, plain list.
    if isinstance(action, list):
        seq = action
    elif isinstance(action, dict):
        for key in ("joints", "qpos_target", "q_target", "target", "vector"):
            val = action.get(key)
            if isinstance(val, list):
                seq = val
                break
        else:
            return list(fallback)
    else:
        return list(fallback)
    try:
        return [float(x) for x in seq]
    except (TypeError, ValueError):
        return list(fallback)


def _read_events(events_path: Path) -> list[dict]:
    if not events_path.exists():
        raise FileNotFoundError(f"events.jsonl not found: {events_path}")
    events: list[dict] = []
    with events_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            events.append(json.loads(line))
    if not events:
        raise ValueError(f"events.jsonl is empty: {events_path}")
    return events


def export_episode(
    src_dir: Path,
    dst_dir: Path,
    *,
    task: str | None = None,
    fps: int = 30,
) -> Path:
    """Convert a single recorded episode directory to LeRobot v3 format.

    Parameters
    ----------
    src_dir:
        Directory produced by :class:`LocalRecorder` — must contain
        ``events.jsonl`` and ``episode.json``; ``video.mp4`` is optional but
        strongly recommended.
    dst_dir:
        Output LeRobot dataset root. Created if missing. Existing files with
        matching paths are overwritten.
    task:
        Task string to record in ``meta/tasks.jsonl``. Falls back to
        ``episode.json[task]`` then ``"unknown"``.
    fps:
        Recording rate advertised in ``meta/info.json``. Should match the rate
        used by the recorder (default 30).

    Returns
    -------
    Path
        The dataset root (``dst_dir``).
    """
    _require_pyarrow()
    import pyarrow as pa
    import pyarrow.parquet as pq

    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    if not src_dir.exists() or not src_dir.is_dir():
        raise FileNotFoundError(f"source episode directory does not exist: {src_dir}")

    events = _read_events(src_dir / "events.jsonl")

    episode_meta: dict = {}
    ep_json = src_dir / "episode.json"
    if ep_json.exists():
        episode_meta = json.loads(ep_json.read_text())
    task_str = task or episode_meta.get("task") or "unknown"

    # --- build column arrays --------------------------------------------------
    n_frames = len(events)
    state_dim = len(events[0]["robot_joints"]) + 1  # + gripper_width

    states: list[list[float]] = []
    actions: list[list[float]] = []
    timestamps: list[float] = []
    frame_indices: list[int] = []

    t0 = float(events[0].get("t", 0.0))
    for i, ev in enumerate(events):
        joints = [float(x) for x in ev["robot_joints"]]
        gripper = float(ev.get("gripper_width", 0.0))
        state = joints + [gripper]
        if len(state) != state_dim:
            raise ValueError(
                f"inconsistent state dim at frame {i}: got {len(state)}, expected {state_dim}"
            )
        states.append(state)
        actions.append(_coerce_action(ev.get("action"), fallback=state))
        timestamps.append(float(ev.get("t", i / float(fps))) - t0)
        frame_indices.append(int(ev.get("frame_idx", i)))

    episode_index = [_EPISODE_ID] * n_frames
    task_index = [0] * n_frames
    # LeRobot v3 uses an `index` column = global (multi-episode) frame index.
    # For a single-episode export this coincides with frame_index.
    global_index = list(range(n_frames))

    table = pa.table(
        {
            "observation.state": pa.array(states, type=pa.list_(pa.float32())),
            "action": pa.array(actions, type=pa.list_(pa.float32())),
            "timestamp": pa.array(timestamps, type=pa.float32()),
            "frame_index": pa.array(frame_indices, type=pa.int64()),
            "episode_index": pa.array(episode_index, type=pa.int64()),
            "index": pa.array(global_index, type=pa.int64()),
            "task_index": pa.array(task_index, type=pa.int64()),
        }
    )

    # --- lay out directories --------------------------------------------------
    meta_dir = dst_dir / "meta"
    data_dir = dst_dir / "data" / _CHUNK
    video_dir = dst_dir / "videos" / _CHUNK / _VIDEO_KEY
    for d in (meta_dir, data_dir, video_dir):
        d.mkdir(parents=True, exist_ok=True)

    parquet_path = data_dir / "episode_000000.parquet"
    pq.write_table(table, parquet_path)

    # --- copy video (if any) --------------------------------------------------
    src_video = src_dir / "video.mp4"
    dst_video = video_dir / "episode_000000.mp4"
    if src_video.exists():
        shutil.copyfile(src_video, dst_video)

    # --- metadata -------------------------------------------------------------
    info = {
        "codebase_version": "v3.0",
        "robot_type": episode_meta.get("robot_type", "unknown"),
        "total_episodes": 1,
        "total_frames": n_frames,
        "total_tasks": 1,
        "total_videos": 1 if src_video.exists() else 0,
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": int(fps),
        "splits": {"train": f"0:{n_frames}"},
        "data_path": "data/{episode_chunk:s}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/{episode_chunk:s}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.state": {
                "dtype": "float32",
                "shape": [state_dim],
                "names": [f"joint_{i}" for i in range(state_dim - 1)] + ["gripper"],
            },
            "action": {
                "dtype": "float32",
                "shape": [state_dim],
                "names": [f"joint_{i}" for i in range(state_dim - 1)] + ["gripper"],
            },
            "observation.images.scene": {
                "dtype": "video",
                "shape": [0, 0, 3],  # filled by consumer / unknown at export time
                "names": ["height", "width", "channels"],
                "video_info": {"video.fps": float(fps), "video.codec": "h264"},
            },
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }
    (meta_dir / "info.json").write_text(json.dumps(info, indent=2))

    with (meta_dir / "tasks.jsonl").open("w") as f:
        f.write(json.dumps({"task_index": 0, "task": task_str}) + "\n")

    episode_record = {
        "episode_index": _EPISODE_ID,
        "tasks": [task_str],
        "length": n_frames,
    }
    with (meta_dir / "episodes.jsonl").open("w") as f:
        f.write(json.dumps(episode_record) + "\n")

    return dst_dir
