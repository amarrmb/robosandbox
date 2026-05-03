"""CI smoke test for the eval pipeline.

Builds a tiny open-loop replay episode (Franka holding home pose) and
runs `robo-sandbox eval --task pick_cube_franka --sim-backend mujoco
--n-trials 2 --output result.json` as a subprocess. Replay won't lift
the cube — that's expected — so we accept exit code 0 OR 1. Exit code
2+ means the eval pipeline itself broke (load, scene, runner, recorder),
which is what this test is here to catch.

Asserts the output JSON conforms to schema_version=2 (the contract in
robosandbox.eval.stats.summarise_eval).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HOME_QPOS = [0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853]
GRIPPER_OPEN = 0.07  # m


def _write_replay_episode(out_dir: Path, n_steps: int = 20) -> None:
    """Emit episode.json + events.jsonl in the LocalRecorder format."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "episode.json").write_text(json.dumps({
        "episode_id": "smoke",
        "task": "pick up the red cube",
        "started_at": "1970-01-01T00:00:00Z",
        "sim_dt": 0.005,
        "source": "test_eval_smoke",
    }))
    with (out_dir / "events.jsonl").open("w") as fh:
        for i in range(n_steps):
            fh.write(json.dumps({
                "t": 0.005 * i,
                "robot_joints": HOME_QPOS,
                "ee_pose": [0.4, 0.0, 0.6, 0.0, 0.0, 0.0, 1.0],
                "gripper_width": GRIPPER_OPEN,
                "objects": {"red_cube": [0.4, 0.0, 0.06, 0.0, 0.0, 0.0, 1.0]},
            }) + "\n")


def _which_robo_sandbox() -> str | None:
    """Find a callable robo-sandbox CLI; falls back to `python -m`."""
    return shutil.which("robo-sandbox")


@pytest.mark.skipif(
    os.environ.get("ROBOSANDBOX_SKIP_EVAL_SMOKE") == "1",
    reason="set when MuJoCo isn't installed in the env",
)
def test_eval_pipeline_round_trip(tmp_path: Path) -> None:
    try:
        import mujoco  # noqa: F401
    except ImportError:
        pytest.skip("mujoco not installed")

    episode_dir = tmp_path / "replay"
    _write_replay_episode(episode_dir)

    out_json = tmp_path / "result.json"

    cli = _which_robo_sandbox()
    if cli:
        cmd = [cli]
    else:
        cmd = [sys.executable, "-m", "robosandbox.cli"]
    cmd += [
        "eval",
        "--task", "pick_cube_franka",
        "--policy", str(episode_dir),
        "--sim-backend", "mujoco",
        "--n-trials", "2",
        "--max-steps", "30",
        "--settle-steps", "10",
        "--output", str(out_json),
    ]
    env = {**os.environ, "MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl"}
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=180)

    # Exit 0 = success, 1 = task not solved (expected for hold-pose replay).
    # 2+ = pipeline error (the failure mode this test is here to catch).
    assert proc.returncode in (0, 1), (
        f"eval pipeline broken: rc={proc.returncode}\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )

    assert out_json.exists(), f"--output JSON not written\nstderr:\n{proc.stderr}"
    payload = json.loads(out_json.read_text())

    # Lock the v2 schema. summarise_eval is the source of truth.
    for key in (
        "schema_version", "task", "policy", "sim_backend",
        "n_trials", "successes", "rate", "ci_low", "ci_high",
    ):
        assert key in payload, f"missing key {key!r} in result JSON: {payload!r}"

    assert payload["schema_version"] == 2
    assert payload["task"] == "pick_cube_franka"
    assert payload["sim_backend"] == "mujoco"
    assert payload["n_trials"] == 2
    assert 0 <= payload["successes"] <= payload["n_trials"]
    assert 0.0 <= payload["rate"] <= 1.0
    assert 0.0 <= payload["ci_low"] <= payload["ci_high"] <= 1.0
