"""Damped-least-squares Jacobian IK using MuJoCo's built-in kinematics.

Zero extra deps — this keeps the v0.1 core honest with its promise of
a 60-second `pip install && run` experience. When someone wants proper
collision-aware planning, the `robosandbox-curobo` plugin drops in.
"""

from __future__ import annotations

from typing import Any, Literal

import mujoco
import numpy as np

from robosandbox.types import JointTrajectory, Pose

OrientationMode = Literal["full", "none", "z_down", "z_up"]


class UnreachableError(RuntimeError):
    """Raised when IK fails to converge within the iteration budget."""


def _quat_xyzw_to_wxyz(q: tuple[float, float, float, float]) -> np.ndarray:
    x, y, z, w = q
    return np.array([w, x, y, z], dtype=np.float64)


def _quat_err(q_current_wxyz: np.ndarray, q_target_wxyz: np.ndarray) -> np.ndarray:
    """3D rotation-vector error such that applying it to current reaches target."""
    q_err = np.empty(4, dtype=np.float64)
    q_neg = np.empty(4, dtype=np.float64)
    mujoco.mju_negQuat(q_neg, q_current_wxyz)
    mujoco.mju_mulQuat(q_err, q_target_wxyz, q_neg)
    rot_vec = np.empty(3, dtype=np.float64)
    mujoco.mju_quat2Vel(rot_vec, q_err, 1.0)
    return rot_vec


def solve_ik(
    sim: Any,
    target_pose: Pose,
    *,
    max_iters: int = 200,
    pos_tol: float = 1e-3,
    rot_tol: float = 1e-2,
    step_size: float = 0.4,
    damping: float = 1e-2,
    seed_joints: np.ndarray | None = None,
    orientation: OrientationMode = "full",
) -> np.ndarray:
    """Solve IK for the sim's end-effector site. Returns joint solution.

    The `sim` is expected to be a MuJoCoBackend-like object exposing
    `model`, `data`, `ee_site_id`, and `arm_qpos_adr`.

    `orientation` modes:
    - "full":   match the target quaternion exactly (3D rotation error)
    - "none":   ignore orientation entirely
    - "z_down": constrain the end-effector's +Z axis to point along world -Z
                (the canonical top-down grasp). 2D rotation error — leaves
                wrist roll free which makes IK much easier.
    - "z_up":   constrain +Z axis to point along world +Z.
    """
    model: mujoco.MjModel = sim.model
    data: mujoco.MjData = sim.data
    site_id = int(sim.ee_site_id)
    qpos_adr = list(sim.arm_qpos_adr)
    n = len(qpos_adr)

    if seed_joints is not None:
        seed = np.asarray(seed_joints, dtype=np.float64).ravel()
        if seed.shape != (n,):
            raise ValueError(f"seed_joints must have shape ({n},), got {seed.shape}")
        for adr, q in zip(qpos_adr, seed):
            data.qpos[adr] = q
        mujoco.mj_forward(model, data)

    target_xyz = np.asarray(target_pose.xyz, dtype=np.float64)
    target_quat_wxyz = _quat_xyzw_to_wxyz(target_pose.quat_xyzw)

    # Joint limits for the arm joints.
    lower = np.array([model.jnt_range[model.joint(f"j{i}").id][0] for i in range(1, n + 1)])
    upper = np.array([model.jnt_range[model.joint(f"j{i}").id][1] for i in range(1, n + 1)])

    jacp = np.zeros((3, model.nv), dtype=np.float64)
    jacr = np.zeros((3, model.nv), dtype=np.float64)

    # Map qpos_adr → qvel dof indices. For hinge/slide joints, dofadr==qposadr
    # is NOT guaranteed, so look it up via the joint API.
    dof_adrs = []
    for i in range(1, n + 1):
        j = model.joint(f"j{i}")
        dof_adrs.append(int(j.dofadr[0]))

    # Pre-compute the world-frame target axis for z_down / z_up modes.
    axis_world_target: np.ndarray | None = None
    if orientation == "z_down":
        axis_world_target = np.array([0.0, 0.0, -1.0])
    elif orientation == "z_up":
        axis_world_target = np.array([0.0, 0.0, 1.0])

    for it in range(max_iters):
        mujoco.mj_forward(model, data)

        current_xyz = np.asarray(data.site_xpos[site_id], dtype=np.float64)
        current_mat = np.asarray(data.site_xmat[site_id]).reshape(3, 3)

        pos_err = target_xyz - current_xyz

        if orientation == "none":
            rot_err = np.zeros(3)
            use_rot = False
        elif orientation == "full":
            current_quat_wxyz = np.empty(4, dtype=np.float64)
            mujoco.mju_mat2Quat(current_quat_wxyz, current_mat.ravel())
            rot_err = _quat_err(current_quat_wxyz, target_quat_wxyz)
            use_rot = True
        else:  # z_down | z_up
            current_z_axis = current_mat[:, 2]  # ee's +Z expressed in world
            assert axis_world_target is not None
            # Small-angle rotation vector that takes current_z onto target.
            cross = np.cross(current_z_axis, axis_world_target)
            dot = float(np.dot(current_z_axis, axis_world_target))
            # For near-antipodal (180° flip) cases, cross collapses to ~0;
            # inject a perturbation perpendicular to the target axis to
            # break the symmetry and let the iteration escape.
            if dot < -0.95 and np.linalg.norm(cross) < 1e-3:
                # pick the world axis most perpendicular to the target
                e = np.array([1.0, 0.0, 0.0])
                if abs(axis_world_target[0]) > 0.9:
                    e = np.array([0.0, 1.0, 0.0])
                cross = np.cross(current_z_axis, e)
            rot_err = cross
            use_rot = True

        pos_ok = np.linalg.norm(pos_err) < pos_tol
        rot_ok = (not use_rot) or np.linalg.norm(rot_err) < rot_tol
        if pos_ok and rot_ok:
            break

        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
        J_cols = np.array(dof_adrs)
        if use_rot:
            J = np.vstack([jacp[:, J_cols], jacr[:, J_cols]])
            err = np.concatenate([pos_err, rot_err])
        else:
            J = jacp[:, J_cols]
            err = pos_err

        # DLS: dq = J^T (J J^T + λ^2 I)^-1 err
        JJt = J @ J.T
        dq = J.T @ np.linalg.solve(JJt + (damping**2) * np.eye(JJt.shape[0]), err)
        dq = step_size * dq

        for adr, idx in zip(qpos_adr, range(n)):
            data.qpos[adr] = float(np.clip(data.qpos[adr] + dq[idx], lower[idx], upper[idx]))
    else:
        mujoco.mj_forward(model, data)
        final_err = np.linalg.norm(target_xyz - np.asarray(data.site_xpos[site_id]))
        raise UnreachableError(
            f"IK ({orientation}) did not converge in {max_iters} iters "
            f"(pos_err={final_err:.4f}m > tol={pos_tol}m)"
        )

    return np.array([data.qpos[adr] for adr in qpos_adr], dtype=np.float64)


class DLSMotionPlanner:
    """MotionPlanner that linearly interpolates between start and IK goal.

    Adequate for v0.1 tabletop tasks. Collision avoidance is delegated to
    future `robosandbox-curobo` plugin.
    """

    name = "dls"

    def __init__(self, n_waypoints: int = 200, dt: float = 0.005) -> None:
        self._n = n_waypoints
        self._dt = dt

    def plan(
        self,
        sim: Any,
        start_joints: np.ndarray,
        target_pose: Pose,
        constraints: dict[str, Any] | None = None,
    ) -> JointTrajectory:
        constraints = constraints or {}
        orientation: OrientationMode = constraints.get("orientation", "full")
        # Snapshot qpos so IK's modifications don't leak into caller.
        model: mujoco.MjModel = sim.model
        data: mujoco.MjData = sim.data
        qpos_backup = data.qpos.copy()

        # ---- multi-seed escape from singular configurations ----------
        # The built-in arm is prone to wrist-flip singularities from
        # neutral. DLS from a single seed can oscillate and fail to
        # converge. We try a few semi-deterministic seeds:
        #   0. current joints (fastest path for sequential skills)
        #   1. warm_seed — j1 rotated toward target (escapes vertical
        #      plane singularity for targets away from the base axis)
        #   2,3. warm_seed perturbed ±0.3 rad on j1 (breaks the
        #      degeneracy when target_y is near 0)
        #   4. same with j2/j4 biased to a bent-elbow pose
        base_xyz = np.asarray(data.body("base").xpos)
        dx = float(target_pose.xyz[0] - base_xyz[0])
        dy = float(target_pose.xyz[1] - base_xyz[1])
        j1_hint = float(np.arctan2(dy, dx))

        start = np.asarray(start_joints, dtype=np.float64)
        warm_seed = start.copy()
        warm_seed[0] = j1_hint

        def _bump(seed: np.ndarray, d_j1: float = 0.0, bent: bool = False) -> np.ndarray:
            s = seed.copy()
            s[0] = float(np.clip(s[0] + d_j1, -3.0, 3.0))
            if bent:
                s[1] = 0.3
                s[2] = 1.0
                s[3] = -0.5
            return s

        seed_candidates = [
            start,
            warm_seed,
            _bump(warm_seed, 0.3),
            _bump(warm_seed, -0.3),
            _bump(warm_seed, 0.0, bent=True),
        ]
        try:
            max_iters = int(constraints.get("max_iters", 400))
            damping = float(constraints.get("damping", 3e-2))
            step_size = float(constraints.get("step_size", 0.4))
            # For axis-constrained modes, warm-start from a position-only
            # solve. Full-orientation DLS gets trapped in local minima when
            # it tries to satisfy both at once from a bad seed.
            goal: np.ndarray | None = None
            last_err: UnreachableError | None = None
            for seed in seed_candidates:
                try:
                    if orientation in ("z_down", "z_up"):
                        pos_seed = solve_ik(
                            sim,
                            target_pose,
                            seed_joints=seed,
                            orientation="none",
                            max_iters=max_iters,
                            damping=damping,
                            step_size=step_size,
                        )
                        goal = solve_ik(
                            sim,
                            target_pose,
                            seed_joints=pos_seed,
                            orientation=orientation,
                            max_iters=max_iters,
                            damping=damping,
                            step_size=step_size,
                        )
                    else:
                        goal = solve_ik(
                            sim,
                            target_pose,
                            seed_joints=seed,
                            orientation=orientation,
                            max_iters=max_iters,
                            damping=damping,
                            step_size=step_size,
                        )
                    break
                except UnreachableError as err:
                    last_err = err
                    # restore and retry from the next seed
                    data.qpos[:] = qpos_backup
                    mujoco.mj_forward(model, data)
            if goal is None:
                assert last_err is not None
                raise last_err
        finally:
            data.qpos[:] = qpos_backup
            mujoco.mj_forward(model, data)

        # Linear interpolation from start to goal.
        t = np.linspace(0.0, 1.0, self._n).reshape(-1, 1)
        waypoints = (1.0 - t) * start_joints.reshape(1, -1) + t * goal.reshape(1, -1)
        return JointTrajectory(waypoints=waypoints, dt=self._dt)
