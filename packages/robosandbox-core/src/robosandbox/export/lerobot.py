"""Export recorded episodes (``runs/<ts>-<id>/``) to a LeRobot v3.0 dataset.

Consumes the artefacts written by :class:`robosandbox.recorder.local.LocalRecorder`
(``episode.json``, ``events.jsonl``, ``video.mp4``) and produces the layout that
``lerobot`` >= 0.4 reads::

    <dst>/
      meta/
        info.json
        tasks.parquet
        episodes/chunk-000/file-000.parquet
      data/chunk-000/file-000.parquet
      data/chunk-000/file-001.parquet
      ...
      videos/observation.images.scene/chunk-000/file-000.mp4
      ...

Key schema decisions:

* ``observation.state`` = ``concat(robot_joints, [gripper_width])`` (float32).
  This matches the Aloha / Koch / SO-100 convention.
* ``action`` = the frame's recorded action if numeric, otherwise a copy of
  ``observation.state`` (scripted demos have no teleop → action == next-state).
  Short recorded actions are padded to ``state_dim`` so policies see a
  consistent action/state dim.
* Per-episode file layout: one parquet + one mp4 per episode, with
  ``file_index == episode_index`` inside chunk 0. Fits the v3.0 chunk-file
  contract (``DEFAULT_CHUNK_SIZE = 1000``) for any dataset up to 1000
  episodes — enough headroom for robosandbox's demo-gen use case.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

_CHUNK_INDEX = 0
_CHUNK_DIR = f"chunk-{_CHUNK_INDEX:03d}"
_VIDEO_KEY = "observation.images.scene"

# LeRobot v3 templates (mirror lerobot.datasets.utils.DEFAULT_*).
_DATA_PATH_TMPL = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
_VIDEO_PATH_TMPL = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"


def _require_pyarrow():  # pragma: no cover - import guard
    try:
        import pyarrow  # noqa: F401
        import pyarrow.parquet  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "pyarrow is required for LeRobot export. Install with: "
            "pip install 'robosandbox[lerobot]'  (or pip install pyarrow>=15)"
        ) from e


def _probe_video_shape(video_path: Path) -> tuple[int, int, int]:
    """Return (height, width, 3) of the first video stream, or (0, 0, 3) on failure."""
    if not video_path.exists():
        return (0, 0, 3)
    try:
        out = subprocess.check_output(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height",
                "-of", "csv=p=0",
                str(video_path),
            ],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        w_s, h_s = out.split(",")[:2]
        return (int(h_s), int(w_s), 3)
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        return (0, 0, 3)


def _coerce_action(
    action: Any,
    fallback: list[float],
    target_dim: int | None = None,
) -> list[float]:
    """Turn a recorded ``action`` field into a flat float vector.

    Raises ``ValueError`` when ``action`` is missing or non-numeric — the
    recorder must not emit frames without a commanded action (otherwise
    settle/idle frames poison the dataset with state-as-action pairs).

    Padding from ``fallback`` only fixes dim alignment when the recorded
    action has joints but no gripper field — it is NOT a missing-data
    fallback.
    """
    if action is None:
        raise ValueError(
            "recorded event has action=None — cannot export. The recorder "
            "should not write frames during settle / idle phases. If you "
            "deliberately want a no-op action, record it explicitly as "
            "{'joints': [...current targets...], 'gripper': <command>}."
        )
    seq: list | None = None
    gripper: float | None = None
    if isinstance(action, list):
        seq = action
    elif isinstance(action, dict):
        for key in ("joints", "qpos_target", "q_target", "target", "vector"):
            val = action.get(key)
            if isinstance(val, list):
                seq = val
                break
        if "gripper" in action:
            try:
                gripper = float(action["gripper"])
            except (TypeError, ValueError):
                gripper = None
    if seq is None:
        raise ValueError(
            f"recorded action has no numeric joint sequence: {action!r}"
        )
    try:
        out = [float(x) for x in seq]
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"recorded action joint sequence is not all numeric: {seq!r}"
        ) from e
    if gripper is not None:
        out.append(gripper)
    if target_dim is not None and len(out) < target_dim:
        out = out + list(fallback[len(out):target_dim])
    return out


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


def export_episodes(
    src_dirs: list[Path],
    dst_dir: Path,
    *,
    task: str | None = None,
    fps: int = 30,
) -> Path:
    """Convert recorded episodes into a single LeRobot v3.0 dataset.

    Each ``src_dirs`` entry must be a directory produced by
    :class:`LocalRecorder` (containing ``events.jsonl`` and ideally
    ``video.mp4``). The output uses the v3.0 chunk/file layout so ``lerobot
    train`` can load it directly.
    """
    _require_pyarrow()
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    if not src_dirs:
        raise ValueError("export_episodes requires at least one source episode")
    src_dirs = [Path(s) for s in src_dirs]
    for s in src_dirs:
        if not s.exists() or not s.is_dir():
            raise FileNotFoundError(f"source episode directory does not exist: {s}")

    if len(src_dirs) > 1000:
        # One file per episode in a single chunk; chunk size = 1000 in v3.0.
        # Sharding across chunks would require more bookkeeping than this
        # exporter currently does.
        raise NotImplementedError(
            f"more than 1000 episodes ({len(src_dirs)}) is not yet supported — "
            "multi-chunk output not implemented"
        )

    dst_dir = Path(dst_dir)
    meta_dir = dst_dir / "meta"
    data_dir = dst_dir / "data" / _CHUNK_DIR
    video_dir = dst_dir / "videos" / _VIDEO_KEY / _CHUNK_DIR
    episodes_meta_dir = meta_dir / "episodes" / _CHUNK_DIR
    for d in (meta_dir, data_dir, video_dir, episodes_meta_dir):
        d.mkdir(parents=True, exist_ok=True)

    state_dim: int | None = None
    total_frames = 0
    total_videos = 0
    task_str = task or "unknown"
    robot_type: str | None = None
    first_video_shape: tuple[int, int, int] | None = None

    episode_rows: list[dict] = []

    # Running stats for observation.state and action. lerobot's dataloader
    # factory reads meta/stats.json and crashes on missing keys — we must
    # emit at least mean/std/min/max/count per non-visual feature.
    state_stack: list[list[float]] = []
    action_stack: list[list[float]] = []

    for ep_idx, src in enumerate(src_dirs):
        events = _read_events(src / "events.jsonl")
        ep_meta: dict = {}
        ep_json = src / "episode.json"
        if ep_json.exists():
            ep_meta = json.loads(ep_json.read_text())
        if task is None and ep_meta.get("task"):
            task_str = ep_meta["task"]
        if robot_type is None and ep_meta.get("robot_type"):
            robot_type = ep_meta["robot_type"]

        n_frames = len(events)
        ep_state_dim = len(events[0]["robot_joints"]) + 1
        if state_dim is None:
            state_dim = ep_state_dim
        elif state_dim != ep_state_dim:
            raise ValueError(
                f"episode {src} state_dim={ep_state_dim} != dataset state_dim={state_dim}"
            )

        states: list[list[float]] = []
        actions: list[list[float]] = []
        timestamps: list[float] = []
        frame_indices: list[int] = []
        # Timestamps MUST match the video's encoded rate, not the simulator's
        # wall-clock. lerobot's dataloader queries frames by timestamp with a
        # 0.1 ms default tolerance, and our video is muxed at exactly `fps` —
        # so frame i sits at i/fps. Storing event `t` here would drift out of
        # tolerance within a few seconds.
        for i, ev in enumerate(events):
            joints = [float(x) for x in ev["robot_joints"]]
            gripper = float(ev.get("gripper_width", 0.0))
            state = joints + [gripper]
            states.append(state)
            actions.append(_coerce_action(ev.get("action"), fallback=state, target_dim=state_dim))
            timestamps.append(i / float(fps))
            frame_indices.append(int(ev.get("frame_idx", i)))

        global_indices = list(range(total_frames, total_frames + n_frames))
        episode_index_col = [ep_idx] * n_frames
        task_index_col = [0] * n_frames
        table = pa.table(
            {
                "observation.state": pa.array(states, type=pa.list_(pa.float32())),
                "action": pa.array(actions, type=pa.list_(pa.float32())),
                "timestamp": pa.array(timestamps, type=pa.float32()),
                "frame_index": pa.array(frame_indices, type=pa.int64()),
                "episode_index": pa.array(episode_index_col, type=pa.int64()),
                "index": pa.array(global_indices, type=pa.int64()),
                "task_index": pa.array(task_index_col, type=pa.int64()),
            }
        )
        file_idx = ep_idx
        pq.write_table(table, data_dir / f"file-{file_idx:03d}.parquet")

        src_video = src / "video.mp4"
        if src_video.exists():
            dst_video = video_dir / f"file-{file_idx:03d}.mp4"
            shutil.copyfile(src_video, dst_video)
            total_videos += 1
            if first_video_shape is None:
                first_video_shape = _probe_video_shape(dst_video)

        # In the packed v3.0 layout multiple episodes share one mp4 and each
        # episode records its [from_timestamp, to_timestamp) slice. We emit
        # one mp4 per episode, so from=0 and to=length/fps.
        episode_rows.append(
            {
                "episode_index": ep_idx,
                "tasks": [task_str],
                "length": n_frames,
                "dataset_from_index": total_frames,
                "dataset_to_index": total_frames + n_frames,
                "data/chunk_index": _CHUNK_INDEX,
                "data/file_index": file_idx,
                f"videos/{_VIDEO_KEY}/chunk_index": _CHUNK_INDEX,
                f"videos/{_VIDEO_KEY}/file_index": file_idx,
                f"videos/{_VIDEO_KEY}/from_timestamp": 0.0,
                f"videos/{_VIDEO_KEY}/to_timestamp": float(n_frames) / float(fps),
            }
        )
        total_frames += n_frames
        state_stack.extend(states)
        action_stack.extend(actions)

    assert state_dim is not None  # at least one episode by precondition

    img_shape = list(first_video_shape) if first_video_shape is not None else [0, 0, 3]

    info = {
        "codebase_version": "v3.0",
        "robot_type": robot_type or "unknown",
        "total_episodes": len(src_dirs),
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_videos": total_videos,
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": int(fps),
        "splits": {"train": f"0:{total_frames}"},
        "data_path": _DATA_PATH_TMPL,
        "video_path": _VIDEO_PATH_TMPL,
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
                "shape": img_shape,
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

    tasks_df = pd.DataFrame([{"task_index": 0, "task": task_str}])
    tasks_df.to_parquet(meta_dir / "tasks.parquet", index=False)

    episodes_df = pd.DataFrame(episode_rows)
    episodes_df.to_parquet(episodes_meta_dir / "file-000.parquet", index=False)

    # meta/stats.json — required by lerobot's factory.py (crashes on None).
    # For state/action we write real stats; for the video feature we write
    # per-channel placeholders in [0,1] range (lerobot overwrites these with
    # IMAGENET_STATS when dataset.use_imagenet_stats=True, which is the
    # default for ACT and Diffusion policies).
    import numpy as np
    state_arr = np.asarray(state_stack, dtype=np.float64)
    action_arr = np.asarray(action_stack, dtype=np.float64)

    def _stats(x: np.ndarray) -> dict[str, list[float] | int]:
        return {
            "mean": x.mean(axis=0).astype(np.float32).tolist(),
            "std": (x.std(axis=0) + 1e-8).astype(np.float32).tolist(),
            "min": x.min(axis=0).astype(np.float32).tolist(),
            "max": x.max(axis=0).astype(np.float32).tolist(),
            "count": [int(x.shape[0])],
        }

    stats = {
        "observation.state": _stats(state_arr),
        "action": _stats(action_arr),
        _VIDEO_KEY: {
            # Placeholder per-channel stats in [0,1] range, shape (3,1,1) as
            # lerobot expects channel-first broadcast-compatible stats.
            "mean": [[[0.5]], [[0.5]], [[0.5]]],
            "std": [[[0.25]], [[0.25]], [[0.25]]],
            "min": [[[0.0]], [[0.0]], [[0.0]]],
            "max": [[[1.0]], [[1.0]], [[1.0]]],
            "count": [total_frames],
        },
    }
    (meta_dir / "stats.json").write_text(json.dumps(stats, indent=2))

    return dst_dir


def export_episode(
    src_dir: Path,
    dst_dir: Path,
    *,
    task: str | None = None,
    fps: int = 30,
) -> Path:
    """Convert a single recorded episode directory to LeRobot v3.0 format.

    Thin wrapper over :func:`export_episodes` for the single-episode case.
    """
    return export_episodes([Path(src_dir)], Path(dst_dir), task=task, fps=fps)
