"""Probe per-backend state agreement after load + settle.

Loads the same scene (same seed) into both Newton and MuJoCo, settles,
then prints cube position, robot joints, and EE pose from each. Used
to isolate whether divergence is in the kinematic oracle or in Newton's
PD tracking.
"""
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
    from robosandbox.sim.newton_backend import NewtonBackend
    from robosandbox.tasks.loader import load_builtin_task
    from robosandbox.tasks.randomize import jitter_scene

    task = load_builtin_task("pick_cube_franka_random")
    scene = jitter_scene(task.scene, task.randomize, seed=1)

    print(f"[scene] cube target xyz = {scene.objects[0].pose.xyz}")

    nb = NewtonBackend(world_count=1, render_size=(240, 320))
    nb.load(scene)
    for _ in range(60):
        nb.step()

    mb = MuJoCoBackend(render_size=(240, 320), camera="scene")
    mb.load(scene)
    for _ in range(60):
        mb.step()

    no = nb.observe()
    mo = mb.observe()

    print()
    print(f"[newton]  joints     = {no.robot_joints}")
    print(f"[mujoco]  joints     = {mo.robot_joints}")
    print(f"          delta      = {no.robot_joints - mo.robot_joints}")
    print(f"          max |dj|   = {np.max(np.abs(no.robot_joints - mo.robot_joints)):.6f} rad")
    print()
    print(f"[newton]  ee xyz     = {no.ee_pose.xyz}")
    print(f"[mujoco]  ee xyz     = {mo.ee_pose.xyz}")
    delta = np.array(no.ee_pose.xyz) - np.array(mo.ee_pose.xyz)
    print(f"          delta xyz  = {delta}")
    print(f"          delta xyz mm = {np.linalg.norm(delta) * 1000:.1f}")
    print()
    print(f"[newton]  cube xyz   = {no.scene_objects['red_cube'].xyz}")
    print(f"[mujoco]  cube xyz   = {mo.scene_objects['red_cube'].xyz}")
    cd = np.array(no.scene_objects['red_cube'].xyz) - np.array(mo.scene_objects['red_cube'].xyz)
    print(f"          delta xyz mm = {np.linalg.norm(cd) * 1000:.1f}")

    nb.close()
    mb.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
