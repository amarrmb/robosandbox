"""LocalRecorder: writes MP4 + JSONL per episode under `runs/<episode_id>/`.

Deliberately simple for v0.1. A proper MCAP writer lives in v0.2 where it
will replace the body of `write_frame` and `end_episode` — the protocol
surface stays the same.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np

from robosandbox.types import Observation


class LocalRecorder:
    name = "local"

    def __init__(
        self,
        root: str | Path = "runs",
        video_fps: int = 30,
        subsample_to_fps: bool = True,
    ) -> None:
        self._root = Path(root)
        self._video_fps = video_fps
        self._subsample = subsample_to_fps
        self._episode_id: str | None = None
        self._episode_dir: Path | None = None
        self._rgb_frames: list[np.ndarray] = []
        self._events_fh = None
        self._frame_counter = 0
        self._last_recorded_t: float = -1e9
        self._sim_dt: float | None = None

    # ---- RecordSink protocol -------------------------------------------
    def start_episode(self, task: str, metadata: dict) -> str:
        self._episode_id = metadata.get("episode_id") or str(uuid.uuid4())[:8]
        self._episode_dir = self._root / f"{datetime.now():%Y%m%d-%H%M%S}-{self._episode_id}"
        self._episode_dir.mkdir(parents=True, exist_ok=True)
        self._rgb_frames = []
        self._frame_counter = 0
        self._last_recorded_t = -1e9
        self._sim_dt = metadata.get("sim_dt")

        info = {
            "episode_id": self._episode_id,
            "task": task,
            "started_at": datetime.now().isoformat(),
            **metadata,
        }
        (self._episode_dir / "episode.json").write_text(json.dumps(info, indent=2, default=str))
        self._events_fh = (self._episode_dir / "events.jsonl").open("w")
        return self._episode_id

    def write_frame(self, obs: Observation, action: dict | None = None) -> None:
        if self._episode_dir is None:
            return
        # Subsample to target fps if sim_dt was provided and requested.
        if self._subsample and self._sim_dt is not None:
            target_period = 1.0 / float(self._video_fps)
            if obs.timestamp - self._last_recorded_t < target_period - 1e-6:
                return
            self._last_recorded_t = obs.timestamp

        self._rgb_frames.append(obs.rgb.copy())
        if self._events_fh is not None:
            ev = {
                "t": obs.timestamp,
                "frame_idx": self._frame_counter,
                "robot_joints": obs.robot_joints.tolist(),
                "ee_pose": {
                    "xyz": obs.ee_pose.xyz,
                    "quat_xyzw": obs.ee_pose.quat_xyzw,
                },
                "gripper_width": obs.gripper_width,
                "objects": {
                    k: {"xyz": v.xyz, "quat_xyzw": v.quat_xyzw}
                    for k, v in obs.scene_objects.items()
                },
                "action": action,
            }
            self._events_fh.write(json.dumps(ev) + "\n")
        self._frame_counter += 1

    def end_episode(self, success: bool, result: dict[str, Any]) -> None:
        if self._episode_dir is None:
            return
        if self._events_fh is not None:
            self._events_fh.close()
            self._events_fh = None

        if self._rgb_frames:
            video_path = self._episode_dir / "video.mp4"
            iio.imwrite(
                video_path,
                np.stack(self._rgb_frames, axis=0),
                fps=self._video_fps,
                macro_block_size=1,
            )

        summary = {
            "episode_id": self._episode_id,
            "success": success,
            "ended_at": datetime.now().isoformat(),
            "frames": self._frame_counter,
            **result,
        }
        (self._episode_dir / "result.json").write_text(json.dumps(summary, indent=2, default=str))
        self._rgb_frames = []
        self._episode_id = None
        self._episode_dir = None
