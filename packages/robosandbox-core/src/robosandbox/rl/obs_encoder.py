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

    def __init__(self, object_ids: list[str], n_dof: int = 7) -> None:
        self.object_ids = list(object_ids)
        self.n_dof = n_dof
        self.obs_dim = n_dof + 7 + 1 + 7 * len(object_ids)

        self._count = 0
        self._mean = np.zeros(self.obs_dim, dtype=np.float64)
        self._m2 = np.ones(self.obs_dim, dtype=np.float64)

    def encode(self, obs: Observation) -> np.ndarray:
        parts: list[np.ndarray] = [
            np.asarray(obs.robot_joints, dtype=np.float64).ravel(),
            np.array([*obs.ee_pose.xyz, *obs.ee_pose.quat_xyzw], dtype=np.float64),
            np.array([obs.gripper_width], dtype=np.float64),
        ]
        for oid in self.object_ids:
            pose = obs.scene_objects.get(oid)
            if pose is not None:
                parts.append(np.array([*pose.xyz, *pose.quat_xyzw], dtype=np.float64))
            else:
                parts.append(np.zeros(7, dtype=np.float64))
        return np.concatenate(parts)

    def encode_batch(self, obs_list: list[Observation]) -> np.ndarray:
        """Encode N observations → (N, obs_dim) float32 array."""
        return np.array([self.encode(o) for o in obs_list], dtype=np.float32)

    def update_stats(self, vec: np.ndarray) -> None:
        """Welford online mean/variance update."""
        self._count += 1
        delta = vec - self._mean
        self._mean += delta / self._count
        self._m2 += delta * (vec - self._mean)

    def update_stats_batch(self, batch: np.ndarray) -> None:
        for row in batch:
            self.update_stats(row)

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
            "count": int(self._count),
            "mean": self._mean.tolist(),
            "m2": self._m2.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> ObsEncoder:
        enc = cls(d["object_ids"], n_dof=int(d.get("n_dof", 7)))
        enc._count = int(d.get("count", 0))
        enc._mean = np.array(d.get("mean", [0.0] * enc.obs_dim), dtype=np.float64)
        enc._m2 = np.array(d.get("m2", [1.0] * enc.obs_dim), dtype=np.float64)
        return enc

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path) -> ObsEncoder:
        return cls.from_dict(json.loads(Path(path).read_text()))
