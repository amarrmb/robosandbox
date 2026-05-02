import json
from pathlib import Path
from robosandbox.recorder.local import LocalRecorder
from robosandbox.types import Observation, Pose
import numpy as np


def test_end_episode_writes_demo_row(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rec = LocalRecorder(root=Path("runs"), video_fps=30)
    rec.start_episode(task="pick the red cube", metadata={"sim_backend": "mujoco"})
    obs = Observation(
        rgb=np.zeros((1, 1, 3), dtype=np.uint8), depth=None,
        robot_joints=np.zeros(7, dtype=np.float32),
        ee_pose=Pose(xyz=(0, 0, 0), quat_xyzw=(0, 0, 0, 1)),
        gripper_width=0.04, scene_objects={},
    )
    rec.write_frame(obs)
    rec.end_episode(success=True, result={"reason": "picked"})
    log = tmp_path / "runs" / "eval_log" / "demos.jsonl"
    assert log.exists(), "demos.jsonl should be created"
    rows = [json.loads(l) for l in log.read_text().splitlines() if l.strip()]
    assert len(rows) == 1
    assert rows[0]["outcome_label"] == "success"
    assert rows[0]["sim_backend"] == "mujoco"


def test_end_episode_failure_label(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rec = LocalRecorder(root=Path("runs"), video_fps=30)
    rec.start_episode(task="pick the red cube", metadata={"sim_backend": "mujoco"})
    obs = Observation(
        rgb=np.zeros((1, 1, 3), dtype=np.uint8), depth=None,
        robot_joints=np.zeros(7, dtype=np.float32),
        ee_pose=Pose(xyz=(0, 0, 0), quat_xyzw=(0, 0, 0, 1)),
        gripper_width=0.04, scene_objects={},
    )
    rec.write_frame(obs)
    rec.end_episode(success=False, result={"reason": "fail"})
    rows = [json.loads(l) for l in (tmp_path / "runs" / "eval_log" / "demos.jsonl").read_text().splitlines() if l.strip()]
    assert rows[0]["outcome_label"] == "failure"
