"""RoboSandbox: sim-first agentic manipulation sandbox."""

from __future__ import annotations

import os
from pathlib import Path

from robosandbox.types import (
    DetectedObject,
    Grasp,
    JointTrajectory,
    Observation,
    Pose,
    Scene,
    SceneObject,
    SkillResult,
)

__version__ = "0.1.0"


def _cache_root(subdir: str) -> Path:
    """Shared resolver for RoboSandbox disk caches.

    Honors ``ROBOSANDBOX_CACHE`` if set (useful for CI, ephemeral envs,
    or redirecting to a scratch disk), otherwise falls back to
    ``~/.cache/robosandbox/``. Each caller passes its own subdir name
    (e.g. ``"mesh_hulls"``, ``"franka_visuals"``) so paths stay scoped.
    """
    root = os.environ.get("ROBOSANDBOX_CACHE")
    base = Path(root) if root else Path.home() / ".cache" / "robosandbox"
    return base / subdir

__all__ = [
    "DetectedObject",
    "Grasp",
    "JointTrajectory",
    "Observation",
    "Pose",
    "Scene",
    "SceneObject",
    "SkillResult",
    "__version__",
]
