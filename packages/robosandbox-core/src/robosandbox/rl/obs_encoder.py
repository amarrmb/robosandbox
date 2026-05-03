"""Flatten an Observation into a fixed-length numpy vector for RL policies."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from robosandbox.types import Observation


class ObsEncoder:
    """Observation → flat float32 vector with online Welford normalization.

    Layout (in order):
      robot_joints  (n_dof,)
      ee_pose       (7,)   — xyz + quat_xyzw
      gripper_width (1,)
      per object    (7,)   — xyz + quat_xyzw, zeros if object missing
    """

    def __init__(
        self,
        object_ids: list[str],
        n_dof: int = 7,
        include_ee_relative: bool = True,
    ) -> None:
        self.object_ids = list(object_ids)
        self.n_dof = n_dof
        self.include_ee_relative = include_ee_relative
        # 7 per object (xyz + quat) + optional 3 per object (xyz - ee_xyz)
        per_obj = 7 + (3 if include_ee_relative else 0)
        self.obs_dim = n_dof + 7 + 1 + per_obj * len(object_ids)

        self._count = 0
        self._mean = np.zeros(self.obs_dim, dtype=np.float64)
        self._m2 = np.ones(self.obs_dim, dtype=np.float64)

    def encode(self, obs: Observation) -> np.ndarray:
        ee_xyz_arr = np.asarray(obs.ee_pose.xyz, dtype=np.float64)
        parts: list[np.ndarray] = [
            np.asarray(obs.robot_joints, dtype=np.float64).ravel(),
            np.concatenate([ee_xyz_arr, np.asarray(obs.ee_pose.quat_xyzw, dtype=np.float64)]),
            np.array([obs.gripper_width], dtype=np.float64),
        ]
        for oid in self.object_ids:
            pose = obs.scene_objects.get(oid)
            if pose is not None:
                obj_xyz = np.asarray(pose.xyz, dtype=np.float64)
                parts.append(np.concatenate([obj_xyz, np.asarray(pose.quat_xyzw, dtype=np.float64)]))
                if self.include_ee_relative:
                    parts.append(obj_xyz - ee_xyz_arr)
            else:
                parts.append(np.zeros(7, dtype=np.float64))
                if self.include_ee_relative:
                    parts.append(np.zeros(3, dtype=np.float64))
        return np.concatenate(parts)

    def encode_batch(self, obs_list: list[Observation]) -> np.ndarray:
        """Encode N observations → (N, obs_dim) float32 array.

        Vectorized fast path: extract each field as a (N, k) array via
        single np.array() calls, then concatenate. ~5-10x faster than a
        Python loop over self.encode() at N=1024.
        """
        if not obs_list:
            return np.zeros((0, self.obs_dim), dtype=np.float32)
        N = len(obs_list)
        joints = np.array([o.robot_joints for o in obs_list], dtype=np.float64)  # (N, n_dof)
        ee_xyz = np.array([o.ee_pose.xyz for o in obs_list], dtype=np.float64)
        ee_quat = np.array([o.ee_pose.quat_xyzw for o in obs_list], dtype=np.float64)
        gripper = np.array([o.gripper_width for o in obs_list], dtype=np.float64).reshape(N, 1)
        parts = [joints, ee_xyz, ee_quat, gripper]
        for oid in self.object_ids:
            obj_pose = np.zeros((N, 7), dtype=np.float64)
            for i, o in enumerate(obs_list):
                p = o.scene_objects.get(oid)
                if p is not None:
                    obj_pose[i, :3] = p.xyz
                    obj_pose[i, 3:] = p.quat_xyzw
            parts.append(obj_pose)
            if self.include_ee_relative:
                parts.append(obj_pose[:, :3] - ee_xyz)
        return np.concatenate(parts, axis=1).astype(np.float32)

    def encode_arrays(self, arrays: dict) -> np.ndarray:
        """Encode pre-batched arrays from a backend's observe_all_arrays().

        ``arrays`` keys: 'joints' (N, n_dof), 'ee_xyz' (N, 3), 'ee_quat' (N, 4),
        'gripper_width' (N,), 'obj_xyz' (dict), 'obj_quat' (dict).

        Bypasses the per-Observation Python overhead in encode_batch.
        Returns (N, obs_dim) float32, identical layout to encode_batch.
        """
        joints = np.asarray(arrays["joints"], dtype=np.float64)
        ee_xyz = np.asarray(arrays["ee_xyz"], dtype=np.float64)
        ee_quat = np.asarray(arrays["ee_quat"], dtype=np.float64)
        gripper = np.asarray(arrays["gripper_width"], dtype=np.float64).reshape(-1, 1)
        N = joints.shape[0]
        parts = [joints, ee_xyz, ee_quat, gripper]
        obj_xyz = arrays.get("obj_xyz", {})
        obj_quat = arrays.get("obj_quat", {})
        for oid in self.object_ids:
            xyz = obj_xyz.get(oid)
            quat = obj_quat.get(oid)
            if xyz is not None and quat is not None:
                parts.append(np.concatenate([xyz, quat], axis=1))
                if self.include_ee_relative:
                    parts.append(xyz - ee_xyz)
            else:
                parts.append(np.zeros((N, 7), dtype=np.float64))
                if self.include_ee_relative:
                    parts.append(np.zeros((N, 3), dtype=np.float64))
        return np.concatenate(parts, axis=1).astype(np.float32)

    def update_stats(self, vec: np.ndarray) -> None:
        """Welford online mean/variance update."""
        self._count += 1
        delta = vec - self._mean
        self._mean += delta / self._count
        self._m2 += delta * (vec - self._mean)

    def update_stats_batch(self, batch: np.ndarray) -> None:
        """Welford-style mean/variance update over a batch of N observations.

        Vectorized: a single arithmetic pass over the batch instead of a
        Python loop calling update_stats N times. Numerically equivalent
        for online stats over independent draws.
        """
        if batch.size == 0:
            return
        n_batch = batch.shape[0]
        b64 = batch.astype(np.float64, copy=False)
        new_count = self._count + n_batch
        # Combined mean & m2 (Welford parallel-merge form, simplified for
        # weight 1 batches).
        batch_mean = b64.mean(axis=0)
        batch_m2 = ((b64 - batch_mean) ** 2).sum(axis=0)
        delta = batch_mean - self._mean
        self._mean += delta * (n_batch / max(new_count, 1))
        self._m2 += batch_m2 + (delta ** 2) * (self._count * n_batch / max(new_count, 1))
        self._count = new_count

    def normalize(self, vec: np.ndarray) -> np.ndarray:
        if self._count < 2:
            return vec.astype(np.float32)
        std = np.sqrt(self._m2 / max(self._count - 1, 1) + 1e-8)
        return ((vec - self._mean) / std).astype(np.float32)

    def normalize_batch(self, batch: np.ndarray) -> np.ndarray:
        if self._count < 2:
            return batch.astype(np.float32)
        std = np.sqrt(self._m2 / max(self._count - 1, 1) + 1e-8)
        return ((batch - self._mean) / std).astype(np.float32)

    def to_dict(self) -> dict:
        return {
            "object_ids": self.object_ids,
            "n_dof": self.n_dof,
            "obs_dim": self.obs_dim,
            "include_ee_relative": self.include_ee_relative,
            "count": int(self._count),
            "mean": self._mean.tolist(),
            "m2": self._m2.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> ObsEncoder:
        enc = cls(
            d["object_ids"],
            n_dof=int(d.get("n_dof", 7)),
            include_ee_relative=bool(d.get("include_ee_relative", True)),
        )
        enc._count = int(d.get("count", 0))
        enc._mean = np.array(d.get("mean", [0.0] * enc.obs_dim), dtype=np.float64)
        enc._m2 = np.array(d.get("m2", [1.0] * enc.obs_dim), dtype=np.float64)
        return enc

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path) -> ObsEncoder:
        return cls.from_dict(json.loads(Path(path).read_text()))
