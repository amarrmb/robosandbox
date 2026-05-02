"""Integration test: eval CLI emits eval_log rows alongside result.json.

Skips automatically when MUJOCO_GL is not set or when no checkpoint is
available at the expected path.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.skipif(os.environ.get("MUJOCO_GL") is None, reason="needs MUJOCO_GL")
def test_eval_emits_log_rows(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    policy_path = "/home/amar/robosandbox/outputs/reach_wide"
    if not Path(policy_path).exists():
        pytest.skip("no checkpoint to eval against")
    cmd = [
        sys.executable, "-m", "robosandbox.cli", "eval",
        "--task", "reach_target_franka",
        "--policy", policy_path,
        "--sim-backend", "mujoco",
        "--world-count", "1", "--n-trials", "4", "--max-steps", "30",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode in (0, 1), r.stderr  # exit 1 = task unsolved, still success
    log = tmp_path / "runs" / "eval_log"
    assert (log / "evals.jsonl").exists()
    n_lines = len((log / "evals.jsonl").read_text().splitlines())
    assert n_lines == 4
    assert (log / "eval_runs.jsonl").exists()
    assert (log / "policies.jsonl").exists()
