"""Cross-backend agreement check.

The product claim "we run your policy in MuJoCo or Newton" is only
meaningful if both backends agree on what the policy *did*. This
runs the same policy through both backends (MuJoCo single-world,
Newton world_count=1) and reports whether they agree on:

1. Final task success outcome — the load-bearing claim.
2. End-of-episode joint positions — bounded by ``joint_tol_rad``.
3. End-of-episode object positions — bounded by ``xy_tol_m``.

Per-step state divergence is tolerated by design: MuJoCo's CG/Newton
solvers and Warp-based Newton are entirely different numerical
methods, so mid-trajectory drift is expected (especially during
contact). Outcome agreement at the end is what matters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from robosandbox.policy import load_policy, run_eval_parallel, run_policy
from robosandbox.tasks.loader import load_builtin_task
from robosandbox.types import Observation


@dataclass
class BackendOutcome:
    backend: str
    success: bool | None
    steps: int
    final_joints: np.ndarray
    final_objects: dict[str, np.ndarray]  # id -> xyz (3,)


@dataclass
class SimCheckReport:
    schema_version: int = 1
    task: str = ""
    policy: str = ""
    tolerances: dict[str, float] = field(default_factory=dict)
    mujoco: dict[str, Any] = field(default_factory=dict)
    newton: dict[str, Any] = field(default_factory=dict)
    outcome_match: bool = False
    joint_max_abs_diff_rad: float = float("nan")
    object_max_xy_diff_m: float = float("nan")
    object_diffs: dict[str, float] = field(default_factory=dict)
    verdict: str = "UNKNOWN"  # PASS | OUTCOME_MISMATCH | STATE_DRIFT | RUN_FAILED


def run_sim_check(
    *,
    task_name: str,
    policy_path: str,
    max_steps: int = 600,
    settle_steps: int = 100,
    joint_tol_rad: float = 0.087,  # ~5 deg
    xy_tol_m: float = 0.005,       # 5 mm
    device: str = "cuda:0",
) -> SimCheckReport:
    """Drive the same policy through MuJoCo and Newton; report agreement."""
    from robosandbox.sim import create_sim_backend

    task = load_builtin_task(task_name)
    rep = SimCheckReport(
        task=task_name,
        policy=policy_path,
        tolerances={"joint_rad": joint_tol_rad, "xy_m": xy_tol_m},
    )

    # ---- MuJoCo ---------------------------------------------------------
    try:
        policy_m = load_policy(policy_path)
        sim_m = create_sim_backend("mujoco", render_size=(240, 320), camera="scene")
        sim_m.load(task.scene)
        try:
            res_m = run_policy(sim_m, policy_m, max_steps=max_steps, success=task.success)
            outcome_m = _capture_outcome("mujoco", res_m["final_obs"], res_m)
        finally:
            sim_m.close()
    except Exception as e:
        rep.mujoco = {"error": f"{type(e).__name__}: {e}"}
        rep.verdict = "RUN_FAILED"
        return rep

    # ---- Newton (world_count=1 to apple-to-apple compare) --------------
    try:
        policy_n = load_policy(policy_path)
        sim_n = create_sim_backend(
            "newton", viewer="null", device=device, world_count=1,
            render_size=(240, 320), camera="scene",
        )
        sim_n.load(task.scene)
        try:
            res_n = run_eval_parallel(
                sim_n, policy_n, max_steps=max_steps, success=task.success,
                settle_steps=settle_steps,
            )
            obs_n = sim_n.observe_all()[0]
            outcome_n = _capture_outcome(
                "newton", obs_n,
                {"steps": res_n["steps"],
                 "success": (res_n["success_per_world"][0] if res_n["success_per_world"] else None)},
            )
        finally:
            sim_n.close()
    except Exception as e:
        rep.newton = {"error": f"{type(e).__name__}: {e}"}
        rep.mujoco = _outcome_to_dict(outcome_m)
        rep.verdict = "RUN_FAILED"
        return rep

    rep.mujoco = _outcome_to_dict(outcome_m)
    rep.newton = _outcome_to_dict(outcome_n)
    rep.outcome_match = (bool(outcome_m.success) == bool(outcome_n.success))

    # Joint diff: max absolute element-wise difference across the n_dof joints.
    if outcome_m.final_joints.shape == outcome_n.final_joints.shape:
        rep.joint_max_abs_diff_rad = float(
            np.max(np.abs(outcome_m.final_joints - outcome_n.final_joints))
        )
    else:
        rep.joint_max_abs_diff_rad = float("nan")

    # Per-object xy diff. Only report objects present in both.
    common_ids = sorted(set(outcome_m.final_objects) & set(outcome_n.final_objects))
    diffs: dict[str, float] = {}
    for oid in common_ids:
        a = outcome_m.final_objects[oid][:2]
        b = outcome_n.final_objects[oid][:2]
        diffs[oid] = float(np.linalg.norm(a - b))
    rep.object_diffs = diffs
    rep.object_max_xy_diff_m = max(diffs.values()) if diffs else 0.0

    if not rep.outcome_match:
        rep.verdict = "OUTCOME_MISMATCH"
    elif (
        rep.joint_max_abs_diff_rad == rep.joint_max_abs_diff_rad  # not NaN
        and rep.joint_max_abs_diff_rad > joint_tol_rad
    ) or rep.object_max_xy_diff_m > xy_tol_m:
        rep.verdict = "STATE_DRIFT"
    else:
        rep.verdict = "PASS"
    return rep


def _capture_outcome(backend: str, obs: Observation, res: dict) -> BackendOutcome:
    final_objects = {
        oid: np.asarray(p.xyz, dtype=np.float64) for oid, p in obs.scene_objects.items()
    }
    return BackendOutcome(
        backend=backend,
        success=res.get("success"),
        steps=int(res.get("steps", 0)),
        final_joints=np.asarray(obs.robot_joints, dtype=np.float64).copy(),
        final_objects=final_objects,
    )


def _outcome_to_dict(o: BackendOutcome) -> dict[str, Any]:
    return {
        "backend": o.backend,
        "success": o.success,
        "steps": o.steps,
        "final_joints": o.final_joints.tolist(),
        "final_objects": {k: v.tolist() for k, v in o.final_objects.items()},
    }


def report_to_dict(rep: SimCheckReport) -> dict[str, Any]:
    """Serialize a SimCheckReport for `--output result.json`."""
    return {
        "schema_version": rep.schema_version,
        "task": rep.task,
        "policy": rep.policy,
        "tolerances": rep.tolerances,
        "mujoco": rep.mujoco,
        "newton": rep.newton,
        "agreement": {
            "outcome_match": rep.outcome_match,
            "joint_max_abs_diff_rad": rep.joint_max_abs_diff_rad,
            "object_max_xy_diff_m": rep.object_max_xy_diff_m,
            "object_diffs": rep.object_diffs,
        },
        "verdict": rep.verdict,
    }
