"""Backends — sim and real robot implementations of the ``SimBackend`` protocol.

The rest of the RoboSandbox stack (agent, skills, motion, perception,
recorder) consumes the ``SimBackend`` protocol defined in ``protocols.py``.
Swapping from sim to real is mechanically a constructor swap — no changes
to skills or agent code are required.

Submodules:
  - :mod:`robosandbox.sim.mujoco_backend` — the shipped MuJoCo sim impl.
  - :mod:`robosandbox.backends.real` — ``RealRobotBackend`` stub +
    integration notes. Users subclass and fill in the hardware driver.
"""

from __future__ import annotations

from robosandbox.backends.real import RealRobotBackend, RealRobotBackendConfig

__all__ = ["RealRobotBackend", "RealRobotBackendConfig"]
