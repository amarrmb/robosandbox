"""Shaped reward functions derived from declarative SuccessCriteria."""

from __future__ import annotations

import numpy as np

from robosandbox.tasks.loader import SuccessCriterion
from robosandbox.types import Observation


def compute_shaped_reward(
    criterion: SuccessCriterion,
    initial_obs: Observation,
    current_obs: Observation,
) -> float:
    """Dense reward ∈ [0, 2] built from the task's success criterion.

    Components (task-specific):
    - Progress signal proportional to how close we are to the goal (0–1)
    - Approach bonus proportional to EE proximity to relevant object (0–0.1)
    - Success bonus +1 when criterion is met
    """
    return _reward_check(criterion.data, initial_obs, current_obs)


def _reward_check(check: dict, initial: Observation, current: Observation) -> float:
    kind = check.get("kind")

    if kind == "lifted":
        oid = check["object"]
        min_mm = float(check.get("min_mm", 50.0))
        z0 = initial.scene_objects.get(oid)
        zf = current.scene_objects.get(oid)
        if z0 is None or zf is None:
            return 0.0
        dz_mm = (zf.xyz[2] - z0.xyz[2]) * 1000.0
        lift_r = float(np.clip(dz_mm / min_mm, 0.0, 1.0))
        # Approach: reward when EE is close to the object
        ee = np.array(current.ee_pose.xyz)
        obj = np.array(zf.xyz)
        dist = float(np.linalg.norm(ee - obj))
        approach_r = float(np.clip(1.0 - dist / 0.4, 0.0, 1.0)) * 0.1
        success_r = 1.0 if dz_mm >= min_mm else 0.0
        return lift_r + approach_r + success_r

    if kind == "moved_above":
        oid = check["object"]
        tid = check["target"]
        xy_tol = float(check.get("xy_tol", 0.03))
        min_dz = float(check.get("min_dz", 0.01))
        o = current.scene_objects.get(oid)
        t = current.scene_objects.get(tid)
        if o is None or t is None:
            return 0.0
        xy = float(np.linalg.norm(np.array(o.xyz[:2]) - np.array(t.xyz[:2])))
        dz = o.xyz[2] - t.xyz[2]
        ok = xy <= xy_tol and dz >= min_dz
        xy_progress = float(np.clip(1.0 - xy / (xy_tol * 3), 0.0, 1.0))
        return (1.0 + xy_progress) if ok else xy_progress

    if kind == "displaced":
        oid = check["object"]
        direction = str(check["direction"]).lower()
        min_mm = float(check.get("min_mm", 30.0))
        vec_map = {
            "forward": (1.0, 0.0), "back": (-1.0, 0.0), "backward": (-1.0, 0.0),
            "left": (0.0, -1.0), "right": (0.0, 1.0),
        }
        dx, dy = vec_map.get(direction, (0.0, 0.0))
        o0 = initial.scene_objects.get(oid)
        of = current.scene_objects.get(oid)
        if o0 is None or of is None:
            return 0.0
        disp_mm = float(np.dot(
            [(of.xyz[0] - o0.xyz[0]) * 1000, (of.xyz[1] - o0.xyz[1]) * 1000],
            [dx, dy],
        ))
        return float(np.clip(disp_mm / min_mm, 0.0, 1.0))

    if kind == "all":
        checks = check.get("checks", [])
        if not checks:
            return 1.0
        return sum(_reward_check(c, initial, current) for c in checks) / len(checks)

    if kind == "any":
        checks = check.get("checks", [])
        if not checks:
            return 0.0
        return max(_reward_check(c, initial, current) for c in checks)

    return 0.0
