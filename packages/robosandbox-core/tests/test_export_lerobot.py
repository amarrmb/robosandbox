"""Tests for the LeRobot v3 exporter."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from robosandbox.export.lerobot import export_episode  # noqa: E402 — guarded by importorskip above


@pytest.fixture
def fake_episode(tmp_path: Path) -> Path:
    """Synthesize a minimal runs/<ts>-<id>/ directory with 30 frames."""
    ep_dir = tmp_path / "20260101-120000-cafebabe"
    ep_dir.mkdir()

    n_frames = 30
    n_dof = 7
    dt = 1.0 / 30.0

    (ep_dir / "episode.json").write_text(
        json.dumps(
            {
                "episode_id": "cafebabe",
                "task": "pick_cube",
                "robot_type": "franka",
                "started_at": "2026-01-01T12:00:00",
                "sim_dt": 0.005,
            }
        )
    )

    with (ep_dir / "events.jsonl").open("w") as f:
        for i in range(n_frames):
            ev = {
                "t": i * dt,
                "frame_idx": i,
                "robot_joints": [0.1 * j + 0.001 * i for j in range(n_dof)],
                "ee_pose": {"xyz": [0.3, 0.0, 0.2], "quat_xyzw": [0, 0, 0, 1]},
                "gripper_width": 0.04,
                "objects": {},
                # Every frame carries a real commanded action. The exporter
                # rejects action=None (would otherwise silently fall back to
                # action=state and poison ~65% of training samples — see
                # _coerce_action). The settle/idle case is covered by
                # test_export_rejects_action_none below.
                "action": {"joints": [0.2 * j for j in range(n_dof)]},
            }
            f.write(json.dumps(ev) + "\n")

    # Tiny 3-frame MP4 via imageio (non-fatal if ffmpeg unavailable).
    try:
        import imageio.v3 as iio

        frames = np.zeros((3, 16, 16, 3), dtype=np.uint8)
        iio.imwrite(ep_dir / "video.mp4", frames, fps=30, macro_block_size=1)
    except Exception:  # pragma: no cover - ffmpeg missing in CI
        pass

    return ep_dir


def test_export_produces_valid_parquet(fake_episode: Path, tmp_path: Path) -> None:
    dst = tmp_path / "dataset"
    out = export_episode(fake_episode, dst, task="pick_cube", fps=30)
    assert out == dst

    # Parquet file exists at LeRobot v3.0 path (file-NNN, not episode-NNN).
    parquet_path = dst / "data" / "chunk-000" / "file-000.parquet"
    assert parquet_path.exists(), parquet_path

    table = pq.read_table(parquet_path)
    cols = set(table.column_names)
    for required in {
        "observation.state",
        "action",
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
    }:
        assert required in cols, f"missing column: {required}"

    assert table.num_rows == 30

    # State dim = 7 joints + 1 gripper.
    state_first = table.column("observation.state")[0].as_py()
    assert len(state_first) == 8
    # Each recorded action has 7 joints (no gripper key in this fixture). The
    # exporter pads short actions up to state_dim (8 = 7 joints + gripper)
    # using state[7] so policies trained on the dataset see consistent
    # action/state dims — without this, ACT/Diffusion would learn 7-D
    # actions while observing 8-D states (silent gripper-open-by-default
    # bias). Pad value MUST come from state, not zero, because the dataset's
    # state[7] is the actual gripper width at that frame.
    action_first = table.column("action")[0].as_py()
    assert len(action_first) == 8
    action_second = table.column("action")[1].as_py()
    assert len(action_second) == 8


def test_export_writes_meta_files(fake_episode: Path, tmp_path: Path) -> None:
    dst = tmp_path / "dataset"
    export_episode(fake_episode, dst, fps=30)

    info = json.loads((dst / "meta" / "info.json").read_text())
    assert info["codebase_version"] == "v3.0"
    assert info["total_frames"] == 30
    assert info["fps"] == 30
    assert "observation.state" in info["features"]
    # Path templates must use v3.0 chunk_index/file_index variables so lerobot
    # resolves data and video files from the episodes.parquet pointers.
    assert "{chunk_index" in info["data_path"]
    assert "{file_index" in info["data_path"]

    tasks = pq.read_table(dst / "meta" / "tasks.parquet").to_pylist()
    assert tasks == [{"task_index": 0, "task": "pick_cube"}]

    episodes = pq.read_table(
        dst / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    ).to_pylist()
    assert episodes[0]["episode_index"] == 0
    assert episodes[0]["length"] == 30
    assert episodes[0]["dataset_from_index"] == 0
    assert episodes[0]["dataset_to_index"] == 30
    assert episodes[0]["data/chunk_index"] == 0
    assert episodes[0]["data/file_index"] == 0


def test_export_task_falls_back_to_episode_json(fake_episode: Path, tmp_path: Path) -> None:
    dst = tmp_path / "dataset"
    export_episode(fake_episode, dst)  # no `task=` override
    tasks = pq.read_table(dst / "meta" / "tasks.parquet").to_pylist()
    assert tasks[0]["task"] == "pick_cube"


def test_export_rejects_action_none(tmp_path: Path) -> None:
    """Regression: action=None used to be silently filled with state. That
    poisoned ~65% of training samples (the settle frames) and pulled all
    skill-action targets toward the home baseline, so the trained policy
    consistently undershot demo trajectories. The exporter now refuses
    action=None to force the recorder to drop those frames upstream.
    """
    ep_dir = tmp_path / "ep-with-none-action"
    ep_dir.mkdir()
    (ep_dir / "episode.json").write_text(json.dumps({"task": "x"}))
    n_dof = 7
    with (ep_dir / "events.jsonl").open("w") as f:
        for i in range(5):
            ev = {
                "t": i / 30.0,
                "frame_idx": i,
                "robot_joints": [0.0] * n_dof,
                "ee_pose": {"xyz": [0, 0, 0], "quat_xyzw": [0, 0, 0, 1]},
                "gripper_width": 0.04,
                "objects": {},
                "action": None,  # the bug-triggering case
            }
            f.write(json.dumps(ev) + "\n")
    with pytest.raises(ValueError, match="action=None"):
        export_episode(ep_dir, tmp_path / "out", task="x")


def test_export_missing_src_raises(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError, match="source episode directory"):
        export_episode(missing, tmp_path / "out")


def test_export_missing_events_raises(tmp_path: Path) -> None:
    ep_dir = tmp_path / "empty-ep"
    ep_dir.mkdir()
    (ep_dir / "episode.json").write_text("{}")
    with pytest.raises(FileNotFoundError, match="events.jsonl"):
        export_episode(ep_dir, tmp_path / "out")
