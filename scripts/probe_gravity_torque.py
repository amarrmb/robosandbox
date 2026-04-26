"""Sanity check the gravity-FF torques returned by MuJoCo for the panda."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "robosandbox-core" / "src"))


def main() -> int:
    from robosandbox.sim.mujoco_backend import MuJoCoBackend
    from robosandbox.tasks.loader import load_builtin_task

    task = load_builtin_task("pick_cube_franka_random")
    mb = MuJoCoBackend(render_size=(240, 320), camera="scene")
    mb.load(task.scene)

    home = np.array(mb._robot.home_qpos, dtype=np.float64)
    print(f"[probe] home_qpos = {home}")
    tau = mb.compute_gravity_torque(home, gripper_qpos=0.04)
    print(f"[probe] gravity FF torque @ home = {tau}")

    # Try a few stretched configurations
    j = home.copy()
    j[1] = 0.3   # shoulder forward
    j[3] = -0.5  # elbow more open
    print(f"[probe] joints = {j}")
    tau = mb.compute_gravity_torque(j, gripper_qpos=0.04)
    print(f"[probe] gravity FF torque = {tau}")

    j[1] = 0.8
    j[3] = -0.2
    print(f"[probe] stretched joints = {j}")
    tau = mb.compute_gravity_torque(j, gripper_qpos=0.04)
    print(f"[probe] gravity FF torque = {tau}")

    mb.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
