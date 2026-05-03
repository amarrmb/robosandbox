"""Lightweight per-world Jacobian + DLS pseudoinverse for EE-space PPO.

Holds a single CPU MuJoCo model loaded from the task scene. Per inner
step, sets joint qpos for each world, calls mj_kinematics + mj_jacBody,
collects an (N, 3, n_arm) Jacobian batch, and converts an EE delta
(N, 3) into a joint delta (N, n_arm) via damped least-squares.

Why per-world serial: ``mj_jacBody`` is very fast (~10–30 µs per call)
and runs in C; even at N=1024 the Python loop adds ~10–30 ms per
inner step — negligible next to Newton's GPU sim cost.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class FrankaEEKine:
    """Per-world EE Jacobian + IK helper using a CPU MuJoCo model."""

    def __init__(
        self,
        scene: Any,
        ee_body: str,
        arm_joint_names: list[str],
        damping: float = 0.05,
    ):
        import mujoco
        from robosandbox.scene.robot_loader import load_and_compile

        self._mj = mujoco
        self._model, _spec = load_and_compile(scene)
        self._data = mujoco.MjData(self._model)
        self._ee_body_id = self._model.body(ee_body).id
        # qpos address (where to write joint angle) and dof column index
        # (which J column it maps to — same for 1-DOF revolute joints).
        self._qpos_addrs = []
        self._dof_cols = []
        for n in arm_joint_names:
            jid = self._model.joint(n).id
            self._qpos_addrs.append(int(self._model.jnt_qposadr[jid]))
            self._dof_cols.append(int(self._model.jnt_dofadr[jid]))
        self._n_arm = len(arm_joint_names)
        self._damping = float(damping)
        self._dof_cols_arr = np.asarray(self._dof_cols, dtype=np.int64)

    @property
    def n_arm(self) -> int:
        return self._n_arm

    def jacobian_batch(self, joints_batch: np.ndarray) -> np.ndarray:
        """Return (N, 3, n_arm) translational Jacobian per world."""
        N = joints_batch.shape[0]
        out = np.zeros((N, 3, self._n_arm), dtype=np.float64)
        jacp = np.zeros((3, self._model.nv), dtype=np.float64)
        jacr = np.zeros((3, self._model.nv), dtype=np.float64)
        for w in range(N):
            for i, addr in enumerate(self._qpos_addrs):
                self._data.qpos[addr] = float(joints_batch[w, i])
            self._mj.mj_kinematics(self._model, self._data)
            self._mj.mj_comPos(self._model, self._data)
            self._mj.mj_jacBody(
                self._model, self._data, jacp, jacr, self._ee_body_id
            )
            out[w] = jacp[:, self._dof_cols_arr]
        return out

    def jacobian_batch_full(self, joints_batch: np.ndarray) -> np.ndarray:
        """Return (N, 6, n_arm) stacked translational + rotational Jacobian per world.

        Top 3 rows are the translational Jacobian (jacp), bottom 3 are
        rotational (jacr). Used by the 6-DoF (ee_xyz_rpy) action mode where
        the policy outputs both linear and angular EE deltas.
        """
        N = joints_batch.shape[0]
        out = np.zeros((N, 6, self._n_arm), dtype=np.float64)
        jacp = np.zeros((3, self._model.nv), dtype=np.float64)
        jacr = np.zeros((3, self._model.nv), dtype=np.float64)
        for w in range(N):
            for i, addr in enumerate(self._qpos_addrs):
                self._data.qpos[addr] = float(joints_batch[w, i])
            self._mj.mj_kinematics(self._model, self._data)
            self._mj.mj_comPos(self._model, self._data)
            self._mj.mj_jacBody(
                self._model, self._data, jacp, jacr, self._ee_body_id
            )
            out[w, :3, :] = jacp[:, self._dof_cols_arr]
            out[w, 3:, :] = jacr[:, self._dof_cols_arr]
        return out

    def delta_q_from_delta_ee(
        self,
        joints_batch: np.ndarray,
        delta_ee_batch: np.ndarray,
        lam: float | None = None,
    ) -> np.ndarray:
        """Damped least-squares: Δq = J^T (J J^T + λ²I)⁻¹ Δx.

        ``joints_batch``: (N, n_arm). ``delta_ee_batch``: (N, 3). Returns (N, n_arm).
        """
        if lam is None:
            lam = self._damping
        J = self.jacobian_batch(joints_batch)                       # (N, 3, n_arm)
        Jt = np.swapaxes(J, 1, 2)                                   # (N, n_arm, 3)
        JJt = J @ Jt                                                # (N, 3, 3)
        damped = JJt + (lam ** 2) * np.eye(3)[None, :, :]           # (N, 3, 3)
        rhs = np.linalg.solve(damped, delta_ee_batch[..., None])    # (N, 3, 1)
        return (Jt @ rhs).squeeze(-1)                               # (N, n_arm)

    def delta_q_from_delta_ee_6d(
        self,
        joints_batch: np.ndarray,
        delta_ee_batch: np.ndarray,
        lam: float | None = None,
    ) -> np.ndarray:
        """6-DoF damped least-squares: Δq = J^T (J J^T + λ²I)⁻¹ [Δx; Δω].

        ``joints_batch``: (N, n_arm). ``delta_ee_batch``: (N, 6) — first 3
        translational delta in m, last 3 angular delta in rad. Returns (N, n_arm).
        """
        if lam is None:
            lam = self._damping
        J = self.jacobian_batch_full(joints_batch)                  # (N, 6, n_arm)
        Jt = np.swapaxes(J, 1, 2)                                   # (N, n_arm, 6)
        JJt = J @ Jt                                                # (N, 6, 6)
        damped = JJt + (lam ** 2) * np.eye(6)[None, :, :]           # (N, 6, 6)
        rhs = np.linalg.solve(damped, delta_ee_batch[..., None])    # (N, 6, 1)
        return (Jt @ rhs).squeeze(-1)                               # (N, n_arm)
