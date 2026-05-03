"""Shaped reward functions derived from declarative SuccessCriteria.

Reward shape can be customized per task via the optional ``shape`` block:

    success:
      kind: ee_near
      object: target_marker
      threshold_m: 0.04
      shape:
        kind: exp           # linear | exp | inverse
        decay: 0.05         # for exp: e^(-dist/decay)
        # k: 10.0           # for inverse: 1/(1 + k*dist)
        # max_dist: 0.6     # for linear: clip(1 - dist/max_dist, 0, 1)
        bonus: 1.0          # added when within threshold

The ``shape`` block is optional; the default for each criterion preserves
the original behaviour. The reward inspector UI in the viewer
(``/reward/curve``, ``/reward/heatmap``) plots these shapes interactively.
"""

from __future__ import annotations

import numpy as np

from robosandbox.tasks.loader import SuccessCriterion
from robosandbox.types import Observation


def shape_progress_batch(dist: np.ndarray, shape: dict) -> np.ndarray:
    """Vectorized version of shape_progress over an array of distances."""
    kind = str(shape.get("kind", "linear"))
    if kind == "linear":
        max_d = float(shape.get("max_dist", 0.6))
        return np.clip(1.0 - dist / max_d, 0.0, 1.0)
    if kind == "exp":
        decay = float(shape.get("decay", 0.05))
        return np.exp(-dist / decay)
    if kind == "inverse":
        k = float(shape.get("k", 10.0))
        return 1.0 / (1.0 + k * dist)
    return np.zeros_like(dist)


def compute_shaped_reward_batch(
    criterion: SuccessCriterion,
    ee_xyz_batch: np.ndarray,
    tgt_xyz_batch: np.ndarray,
) -> np.ndarray:
    """Batched reward for ee_near criterion. Returns (N,) array.

    Caller responsibility: extract ``ee_xyz_batch`` (N,3) and ``tgt_xyz_batch``
    (N,3) once per step, then pass arrays in. ~100x faster than calling
    compute_shaped_reward in a Python loop over N=1024 worlds.

    Falls back to None for criteria we don't have a batched path for; caller
    should use the scalar version then.
    """
    check = criterion.data
    kind = check.get("kind")
    if kind == "ee_near":
        threshold_m = float(check.get("threshold_m", 0.05))
        shape = check.get("shape") or {
            "kind": "linear",
            "max_dist": float(check.get("max_dist_m", 0.6)),
        }
        bonus = float(shape.get("bonus", 1.0))
        dist = np.linalg.norm(ee_xyz_batch - tgt_xyz_batch, axis=1)  # (N,)
        progress = shape_progress_batch(dist, shape)
        success = np.where(dist <= threshold_m, bonus, 0.0)
        return progress + success
    return None


def compute_inserted_reward_batch(
    criterion: SuccessCriterion,
    ee_xyz_batch: np.ndarray,    # (N, 3)
    ee_quat_batch: np.ndarray,   # (N, 4) xyzw
    port_xyz_batch: np.ndarray,  # (N, 3)
    port_quat_batch: np.ndarray | None = None,  # (N, 4) xyzw, required for yaw mode
    inserted_streak: np.ndarray | None = None,  # (N,) consecutive inserted-step count, for hold bonus
    last_action: np.ndarray | None = None,      # (N, act_dim) last action, for action penalty
) -> np.ndarray:
    """Vectorized insertion reward over N parallel worlds.

    Used by the PPO training loop when the task's success kind is
    ``inserted``. Batched form is ~100× faster than calling
    compute_shaped_reward in a Python loop.
    """
    check = criterion.data
    port_axis = np.asarray(check.get("port_axis", [0.0, 0.0, 1.0]), dtype=np.float64)
    port_axis = port_axis / max(float(np.linalg.norm(port_axis)), 1e-9)
    plane_z = float(check.get("port_plane_z", 0.0))
    offset_ee = np.asarray(
        check.get("plug_tip_offset_ee", [0.0, 0.0, 0.05]), dtype=np.float64
    )
    min_depth = float(check.get("min_depth_m", 0.005))
    axis_tol_deg = float(check.get("axis_tol_deg", 5.0))
    shape = check.get("shape") or {}
    approach_decay = float(shape.get("approach_decay", 0.20))
    align_weight = float(shape.get("align_weight", 0.5))
    align_gate_m = float(shape.get("align_gate_m", 0.04))
    insertion_weight = float(shape.get("insertion_weight", 1.0))
    bonus = float(shape.get("success_bonus", 1.0))

    qx = ee_quat_batch[:, 0]
    qy = ee_quat_batch[:, 1]
    qz = ee_quat_batch[:, 2]
    qw = ee_quat_batch[:, 3]
    x2, y2, z2 = qx + qx, qy + qy, qz + qz
    wx, wy, wz = qw * x2, qw * y2, qw * z2
    xx, xy_, xz = qx * x2, qx * y2, qx * z2
    yy, yz, zz = qy * y2, qy * z2, qz * z2
    # Rotation matrix per-world: shape (N, 3, 3).
    R = np.empty((ee_quat_batch.shape[0], 3, 3), dtype=np.float64)
    R[:, 0, 0] = 1.0 - (yy + zz); R[:, 0, 1] = xy_ - wz;        R[:, 0, 2] = xz + wy
    R[:, 1, 0] = xy_ + wz;        R[:, 1, 1] = 1.0 - (xx + zz); R[:, 1, 2] = yz - wx
    R[:, 2, 0] = xz - wy;         R[:, 2, 1] = yz + wx;         R[:, 2, 2] = 1.0 - (xx + yy)

    # tip = ee + R @ offset
    tip = ee_xyz_batch + np.einsum("nij,j->ni", R, offset_ee)
    plug_axis = np.einsum("nij,j->ni", R, np.array([0.0, 0.0, 1.0]))
    delta = tip - port_xyz_batch
    dist = np.linalg.norm(delta, axis=1)
    approach = np.exp(-dist / approach_decay)

    horizontal_dist = np.linalg.norm(delta[:, :2], axis=1)
    cos_align = np.einsum("ni,i->n", plug_axis, -port_axis)
    cos_align = np.clip(cos_align, -1.0, 1.0)
    align_r = np.where(
        horizontal_dist <= align_gate_m,
        align_weight * np.maximum(cos_align, 0.0),
        0.0,
    )
    depth = plane_z - np.einsum("ni,i->n", tip, port_axis)
    insertion_r = insertion_weight * np.clip(depth / min_depth, 0.0, 2.0)
    align_deg = np.degrees(np.arccos(cos_align))

    # Yaw matching (vectorized): plug long-axis vs port long-axis modulo 180°.
    # Reward shape: (1 - yaw_err_deg/90)^2 — peaks at 1.0 when aligned, falls
    # smoothly to 0 at 90°. UNGATED (unlike axial align): the policy needs
    # global yaw signal to learn rotation-conditioning, not just close-range.
    # Stronger shape than the older `cos_yaw` which had weak gradient near 90°.
    yaw_tol_deg = float(check.get("yaw_tol_deg", 0.0))
    yaw_weight = float(shape.get("yaw_weight", 1.0))
    yaw_r = np.zeros_like(approach)
    yaw_ok = np.ones_like(approach, dtype=bool)
    if yaw_tol_deg > 0.0 and port_quat_batch is not None:
        plug_x = np.einsum("nij,j->ni", R, np.array([1.0, 0.0, 0.0]))
        # Build R_port per-world from port_quat_batch
        pqx = port_quat_batch[:, 0]; pqy = port_quat_batch[:, 1]
        pqz = port_quat_batch[:, 2]; pqw = port_quat_batch[:, 3]
        x2p, y2p, z2p = pqx + pqx, pqy + pqy, pqz + pqz
        wxp, wyp, wzp = pqw * x2p, pqw * y2p, pqw * z2p
        xxp, xyp_, xzp = pqx * x2p, pqx * y2p, pqx * z2p
        yyp, yzp, zzp = pqy * y2p, pqy * z2p, pqz * z2p
        Rp = np.empty_like(R)
        Rp[:, 0, 0] = 1.0 - (yyp + zzp); Rp[:, 0, 1] = xyp_ - wzp;        Rp[:, 0, 2] = xzp + wyp
        Rp[:, 1, 0] = xyp_ + wzp;        Rp[:, 1, 1] = 1.0 - (xxp + zzp); Rp[:, 1, 2] = yzp - wxp
        Rp[:, 2, 0] = xzp - wyp;         Rp[:, 2, 1] = yzp + wxp;         Rp[:, 2, 2] = 1.0 - (xxp + yyp)
        port_x = np.einsum("nij,j->ni", Rp, np.array([1.0, 0.0, 0.0]))
        # Project both to plane perpendicular to port_axis
        plug_x_proj = plug_x - np.einsum("ni,i->n", plug_x, port_axis)[:, None] * port_axis[None, :]
        port_x_proj = port_x - np.einsum("ni,i->n", port_x, port_axis)[:, None] * port_axis[None, :]
        n1 = np.linalg.norm(plug_x_proj, axis=1)
        n2 = np.linalg.norm(port_x_proj, axis=1)
        denom = np.maximum(n1 * n2, 1e-9)
        cos_yaw = np.abs(np.einsum("ni,ni->n", plug_x_proj, port_x_proj) / denom)
        cos_yaw = np.clip(cos_yaw, 0.0, 1.0)
        yaw_err_deg = np.degrees(np.arccos(cos_yaw))
        # Ungated, smooth-falloff yaw reward
        yaw_progress = np.clip(1.0 - yaw_err_deg / 90.0, 0.0, 1.0) ** 2
        yaw_r = yaw_weight * yaw_progress
        yaw_ok = yaw_err_deg <= yaw_tol_deg

    # Optional yaw-gated rewards: when yaw_gate_depth is set in shape, the
    # depth/insertion/success reward is zeroed for worlds that aren't within
    # `yaw_gate_deg` of the port yaw. Forces the policy to learn yaw FIRST
    # before getting credit for depth — without this, yaw is a small side
    # signal vs a big depth/bonus reward and the policy ignores it.
    yaw_gate_depth = bool(shape.get("yaw_gate_depth", False))
    yaw_gate_deg = float(shape.get("yaw_gate_deg", 30.0))
    if yaw_gate_depth and yaw_tol_deg > 0.0 and port_quat_batch is not None:
        yaw_gate_ok = (yaw_err_deg <= yaw_gate_deg)
        insertion_r = np.where(yaw_gate_ok, insertion_r, 0.0)

    success = (depth >= min_depth) & (align_deg <= axis_tol_deg) & yaw_ok

    # Hold-still bonus: when inserted, reward grows linearly with the
    # consecutive-inserted-step streak (capped). Pays the policy to STAY
    # in the inserted state instead of drift out. Without this, PPO's
    # stochastic exploration pushes the EE off-target after peak insertion.
    hold_bonus_max = float(shape.get("hold_still_bonus", 0.0))
    hold_bonus_steps = int(shape.get("hold_still_target_steps", 50))
    hold_r = np.zeros_like(approach)
    if hold_bonus_max > 0.0 and inserted_streak is not None:
        streak_norm = np.minimum(inserted_streak.astype(np.float64) / max(hold_bonus_steps, 1), 1.0)
        hold_r = np.where(success, hold_bonus_max * streak_norm, 0.0)

    # Action-magnitude penalty when inserted: small negative reward proportional
    # to action L2 norm, ONLY when in inserted state. Encourages the policy to
    # zero its actions once aligned (don't keep wiggling). Off when not inserted
    # so the policy is free to move during approach.
    action_penalty_w = float(shape.get("action_penalty_when_inserted", 0.0))
    action_pen = np.zeros_like(approach)
    if action_penalty_w > 0.0 and last_action is not None:
        a_l2 = np.sum(last_action ** 2, axis=1)
        action_pen = np.where(success, action_penalty_w * a_l2, 0.0)

    return approach + align_r + yaw_r + insertion_r + np.where(success, bonus, 0.0) + hold_r - action_pen


def shape_progress(dist: float, shape: dict) -> float:
    """Map distance → progress reward in [0, 1] using the given shape config.

    Pure function — no observation dependency. Used both by training loops
    and by the viewer's reward-inspector endpoints to plot curves/heatmaps.
    """
    kind = str(shape.get("kind", "linear"))
    if kind == "linear":
        max_d = float(shape.get("max_dist", 0.6))
        return float(np.clip(1.0 - dist / max_d, 0.0, 1.0))
    if kind == "exp":
        decay = float(shape.get("decay", 0.05))
        return float(np.exp(-dist / decay))
    if kind == "inverse":
        k = float(shape.get("k", 10.0))
        return float(1.0 / (1.0 + k * dist))
    return 0.0


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

    if kind == "ee_near":
        # Reach to a target object's xyz. Reward shape is configurable via
        # check["shape"]; default is linear over a 0.6m workspace.
        oid = check["object"]
        threshold_m = float(check.get("threshold_m", 0.05))
        shape = check.get("shape") or {
            "kind": "linear",
            "max_dist": float(check.get("max_dist_m", 0.6)),
        }
        bonus = float(shape.get("bonus", 1.0))
        t = current.scene_objects.get(oid)
        if t is None:
            return 0.0
        ee = np.asarray(current.ee_pose.xyz, dtype=np.float64)
        tgt = np.asarray(t.xyz, dtype=np.float64)
        dist = float(np.linalg.norm(ee - tgt))
        progress = shape_progress(dist, shape)
        success = bonus if dist <= threshold_m else 0.0
        return progress + success

    if kind == "inserted":
        # Insertion reward: stage 1 approach (tip→port distance), stage 2
        # alignment (gated on horizontal proximity), stage 3 insertion
        # depth past port plane, plus a success bonus on full criterion.
        port_id = check.get("port") or check.get("object")
        port_axis = np.asarray(check.get("port_axis", [0.0, 0.0, 1.0]), dtype=np.float64)
        port_axis = port_axis / max(float(np.linalg.norm(port_axis)), 1e-9)
        plane_z = float(check.get("port_plane_z", 0.0))
        offset_ee = np.asarray(
            check.get("plug_tip_offset_ee", [0.0, 0.0, 0.05]), dtype=np.float64
        )
        min_depth = float(check.get("min_depth_m", 0.005))
        axis_tol_deg = float(check.get("axis_tol_deg", 5.0))
        shape = check.get("shape") or {}
        approach_decay = float(shape.get("approach_decay", 0.20))
        align_weight = float(shape.get("align_weight", 0.5))
        align_gate_m = float(shape.get("align_gate_m", 0.04))
        insertion_weight = float(shape.get("insertion_weight", 1.0))
        bonus = float(shape.get("success_bonus", 1.0))
        port = current.scene_objects.get(port_id)
        if port is None:
            return 0.0
        ee = np.asarray(current.ee_pose.xyz, dtype=np.float64)
        qx, qy, qz, qw = current.ee_pose.quat_xyzw
        x2, y2, z2 = qx + qx, qy + qy, qz + qz
        wx, wy, wz = qw * x2, qw * y2, qw * z2
        xx, xy_, xz = qx * x2, qx * y2, qx * z2
        yy, yz, zz = qy * y2, qy * z2, qz * z2
        R = np.array([
            [1.0 - (yy + zz), xy_ - wz,        xz + wy],
            [xy_ + wz,        1.0 - (xx + zz), yz - wx],
            [xz - wy,         yz + wx,         1.0 - (xx + yy)],
        ])
        tip = ee + R @ offset_ee
        plug_axis = R @ np.array([0.0, 0.0, 1.0])
        port_xyz = np.asarray(port.xyz, dtype=np.float64)
        # Approach: exp decay on tip-to-port distance
        dist = float(np.linalg.norm(tip - port_xyz))
        approach = float(np.exp(-dist / approach_decay))
        # Alignment: cos(plug_axis · -port_axis), gated on horizontal proximity
        # so the policy chases position first and only rotates when close.
        horizontal_dist = float(np.linalg.norm((tip - port_xyz)[:2]))
        if horizontal_dist <= align_gate_m:
            cos_align = float(np.dot(plug_axis, -port_axis))
            cos_align = max(-1.0, min(1.0, cos_align))
            align_r = align_weight * max(0.0, cos_align)
        else:
            align_r = 0.0
        # Optional yaw match (when the success criterion declares yaw_tol_deg):
        # plug long-axis must match port long-axis modulo 180°. Same gate as
        # alignment so we don't push yaw before position is close.
        yaw_tol_deg = float(check.get("yaw_tol_deg", 0.0))
        yaw_weight = float(shape.get("yaw_weight", 0.5))
        yaw_r = 0.0
        if yaw_tol_deg > 0.0 and horizontal_dist <= align_gate_m:
            plug_x = R @ np.array([1.0, 0.0, 0.0])
            pqx, pqy, pqz, pqw = port.quat_xyzw
            x2p, y2p, z2p = pqx + pqx, pqy + pqy, pqz + pqz
            wxp, wyp, wzp = pqw * x2p, pqw * y2p, pqw * z2p
            xxp, xyp, xzp = pqx * x2p, pqx * y2p, pqx * z2p
            yyp, yzp, zzp = pqy * y2p, pqy * z2p, pqz * z2p
            R_port = np.array([
                [1.0 - (yyp + zzp), xyp - wzp,         xzp + wyp],
                [xyp + wzp,         1.0 - (xxp + zzp), yzp - wxp],
                [xzp - wyp,         yzp + wxp,         1.0 - (xxp + yyp)],
            ])
            port_x = R_port @ np.array([1.0, 0.0, 0.0])
            plug_x_proj = plug_x - np.dot(plug_x, port_axis) * port_axis
            port_x_proj = port_x - np.dot(port_x, port_axis) * port_axis
            n1 = float(np.linalg.norm(plug_x_proj))
            n2 = float(np.linalg.norm(port_x_proj))
            if n1 > 1e-6 and n2 > 1e-6:
                cos_yaw = abs(float(np.dot(plug_x_proj, port_x_proj) / (n1 * n2)))
                yaw_r = yaw_weight * cos_yaw
        # Insertion: depth past plane, clamped to [0, 2*min_depth]/min_depth
        depth = float(plane_z - np.dot(port_axis, tip))
        insertion_r = insertion_weight * float(np.clip(depth / min_depth, 0.0, 2.0))
        # Success bonus inlined (avoids per-world _eval_check call cost):
        cos_align_full = float(np.dot(plug_axis, -port_axis))
        cos_align_full = max(-1.0, min(1.0, cos_align_full))
        align_deg = float(np.degrees(np.arccos(cos_align_full)))
        if yaw_tol_deg > 0.0:
            # yaw_r above only computed when horizontal_dist <= align_gate_m
            # AND yaw_tol_deg > 0; recompute the yaw error here for the
            # success gate so it's evaluated even far from the port.
            plug_x_full = R @ np.array([1.0, 0.0, 0.0])
            pqx, pqy, pqz, pqw = port.quat_xyzw
            x2p, y2p, z2p = pqx + pqx, pqy + pqy, pqz + pqz
            wxp, wyp, wzp = pqw * x2p, pqw * y2p, pqw * z2p
            xxp, xyp, xzp = pqx * x2p, pqx * y2p, pqx * z2p
            yyp, yzp, zzp = pqy * y2p, pqy * z2p, pqz * z2p
            R_port_full = np.array([
                [1.0 - (yyp + zzp), xyp - wzp,         xzp + wyp],
                [xyp + wzp,         1.0 - (xxp + zzp), yzp - wxp],
                [xzp - wyp,         yzp + wxp,         1.0 - (xxp + yyp)],
            ])
            port_x_full = R_port_full @ np.array([1.0, 0.0, 0.0])
            plug_x_proj_full = plug_x_full - np.dot(plug_x_full, port_axis) * port_axis
            port_x_proj_full = port_x_full - np.dot(port_x_full, port_axis) * port_axis
            n1f = float(np.linalg.norm(plug_x_proj_full))
            n2f = float(np.linalg.norm(port_x_proj_full))
            if n1f > 1e-6 and n2f > 1e-6:
                cos_yaw_full = abs(float(np.dot(plug_x_proj_full, port_x_proj_full) / (n1f * n2f)))
                yaw_err = float(np.degrees(np.arccos(min(1.0, cos_yaw_full))))
            else:
                yaw_err = 0.0
            yaw_ok = yaw_err <= yaw_tol_deg
        else:
            yaw_ok = True
        success = (depth >= min_depth) and (align_deg <= axis_tol_deg) and yaw_ok
        return approach + align_r + yaw_r + insertion_r + (bonus if success else 0.0)

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
