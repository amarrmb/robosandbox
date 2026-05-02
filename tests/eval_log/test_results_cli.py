import json
import os
import subprocess
import sys
from pathlib import Path

from robosandbox.eval_log import EvalLogStore, EvalRow, EvalRunRow, PolicyRow


def _seed(root: Path):
    s = EvalLogStore(root)
    s.append_policy(PolicyRow("p1", "ppo_neural", None, None, None, 0, None,
                              "2026-05-01T00:00:00Z", "out/p1"))
    s.append_eval_run(EvalRunRow("e1", "p1", "task1", 4,
                                 "2026-05-01T01:00:00Z", "2026-05-01T01:00:30Z",
                                 "abc", "cfg", "cpu", "robo-sandbox eval ..."))
    for i in range(4):
        s.append_eval(EvalRow("e1", f"{i:05d}", "p1", "task1", "h", "newton", i,
                              {"target_xy_bucket": "left" if i < 2 else "right"},
                              {"success": i < 3}, 0.1, "cpu", "2026-05-01T01:00:00Z"))


def test_results_list(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _seed(tmp_path / "runs" / "eval_log")
    r = subprocess.run([sys.executable, "-m", "robosandbox.cli", "results", "list"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "p1" in r.stdout


def test_results_slice(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _seed(tmp_path / "runs" / "eval_log")
    r = subprocess.run([sys.executable, "-m", "robosandbox.cli", "results", "slice",
                        "p1", "--axis", "target_xy_bucket"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "left" in r.stdout and "right" in r.stdout


def test_results_compare(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _seed(tmp_path / "runs" / "eval_log")
    s = EvalLogStore(tmp_path / "runs" / "eval_log")
    s.append_policy(PolicyRow("p2", "ppo_neural", "p1", "warm_start_ppo", None, 0, None,
                              "2026-05-01T02:00:00Z", "out/p2"))
    s.append_eval_run(EvalRunRow("e2", "p2", "task1", 4,
                                 "2026-05-01T02:00:00Z", "2026-05-01T02:00:30Z",
                                 "abc", "cfg", "cpu", "robo-sandbox eval ..."))
    for i in range(4):
        s.append_eval(EvalRow("e2", f"{i:05d}", "p2", "task1", "h", "newton", i,
                              {}, {"success": True}, 0.1, "cpu", "2026-05-01T02:00:00Z"))
    r = subprocess.run([sys.executable, "-m", "robosandbox.cli", "results",
                        "compare", "p1", "p2", "--task", "task1"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "p2" in r.stdout


def test_results_lineage(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _seed(tmp_path / "runs" / "eval_log")
    s = EvalLogStore(tmp_path / "runs" / "eval_log")
    s.append_policy(PolicyRow("p2", "ppo_neural", "p1", "warm_start_ppo", None, 0, None,
                              "2026-05-01T02:00:00Z", "out/p2"))
    r = subprocess.run([sys.executable, "-m", "robosandbox.cli", "results", "lineage", "p2"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "p2" in r.stdout and "p1" in r.stdout


def test_results_empty_log(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    r = subprocess.run([sys.executable, "-m", "robosandbox.cli", "results", "list"],
                       capture_output=True, text=True)
    assert r.returncode == 0
    assert "no data" in r.stdout.lower() or r.stdout.strip() == ""
