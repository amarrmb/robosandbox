"""Run the distilled MLP closed-loop in both backends; log per-step state.

Same seed in both. Logs gripper width, cube z, EE z+xy distance to cube.
Reveals whether policy commands diverge (different obs → different action),
or actions are similar but physics responds differently.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "robosandbox-core" / "src"))


def main() -> int:
    from robosandbox.policy import load_policy
    from robosandbox.sim.mujoco_backend import MuJoCoBackend
    from robosandbox.sim.newton_backend import NewtonBackend
    from robosandbox.tasks.loader import load_builtin_task
    from robosandbox.tasks.randomize import jitter_scene

    SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    POLICY = sys.argv[2] if len(sys.argv) > 2 else "/home/amar/robosandbox/outputs/act_50k_distilled"
    ACTION_REPEAT = 6
    MAX_STEPS = 200  # action steps

    task = load_builtin_task("pick_cube_franka_random")
    scene = jitter_scene(task.scene, task.randomize, seed=SEED)
    z0 = float(scene.objects[0].pose.xyz[2])
    cube_xyz0 = scene.objects[0].pose.xyz
    print(f"[seed={SEED}] cube xyz = {cube_xyz0}  z0={z0*1000:.1f}mm")

    nb = NewtonBackend(world_count=1, render_size=(64, 64), dt=0.005)
    nb.load(scene)
    mb = MuJoCoBackend(render_size=(64, 64), camera="scene")
    mb.load(scene)
    nb.set_gravity_compensation(mb.compute_gravity_torque)

    # Pre-settle (matches eval pipeline)
    for _ in range(100):
        mb.step()
        nb.step()

    # Two independent policy instances so per-step state isn't shared
    pol_mj = load_policy(Path(POLICY))
    pol_nw = load_policy(Path(POLICY))
    if hasattr(pol_mj, "reset"): pol_mj.reset()
    if hasattr(pol_nw, "reset"): pol_nw.reset()

    print()
    print(f"{'step':>4} | "
          f"{'mj_grip':>7} {'mj_dxy':>7} {'mj_dz':>7} {'mj_lift':>8} | "
          f"{'nw_grip':>7} {'nw_dxy':>7} {'nw_dz':>7} {'nw_lift':>8} | "
          f"{'a_diff':>6}")
    print("-" * 100)

    log_every = 5
    mj_peak = 0.0
    nw_peak = 0.0
    mj_lifted = False
    nw_lifted = False

    for step in range(MAX_STEPS):
        mj_obs = mb.observe()
        nw_obs = nb.observe()
        # call policies
        mj_act = np.asarray(pol_mj.act(mj_obs), dtype=np.float64).ravel()
        nw_act = np.asarray(pol_nw.act(nw_obs), dtype=np.float64).ravel()
        # Policy returns [j1..j7, gripper]
        mj_arm, mj_grip_cmd = mj_act[:7], float(mj_act[7])
        nw_arm, nw_grip_cmd = nw_act[:7], float(nw_act[7])

        for _ in range(ACTION_REPEAT):
            mb.step(target_joints=mj_arm, gripper=mj_grip_cmd)
            nb.step(target_joints=nw_arm, gripper=nw_grip_cmd)

        if step % log_every == 0 or step == MAX_STEPS - 1:
            mo = mb.observe()
            no = nb.observe()
            mj_grip_w = (mo.gripper_width or 0.0) * 1000.0
            nw_grip_w = (no.gripper_width or 0.0) * 1000.0
            mj_cube = mo.scene_objects["red_cube"].xyz
            nw_cube = no.scene_objects["red_cube"].xyz
            mj_lift = (mj_cube[2] - z0) * 1000.0
            nw_lift = (nw_cube[2] - z0) * 1000.0
            mj_peak = max(mj_peak, mj_lift); nw_peak = max(nw_peak, nw_lift)
            if mj_lift >= 50.0 and not mj_lifted:
                print(f"  [! mj solved at step {step}]")
                mj_lifted = True
            if nw_lift >= 50.0 and not nw_lifted:
                print(f"  [! nw solved at step {step}]")
                nw_lifted = True
            mj_dxy = float(np.hypot(mo.ee_pose.xyz[0] - mj_cube[0], mo.ee_pose.xyz[1] - mj_cube[1])) * 1000.0
            mj_dz = (mo.ee_pose.xyz[2] - mj_cube[2]) * 1000.0
            nw_dxy = float(np.hypot(no.ee_pose.xyz[0] - nw_cube[0], no.ee_pose.xyz[1] - nw_cube[1])) * 1000.0
            nw_dz = (no.ee_pose.xyz[2] - nw_cube[2]) * 1000.0
            a_diff = float(np.linalg.norm(mj_arm - nw_arm))  # rad
            print(f"{step:>4d} | "
                  f"{mj_grip_w:>7.1f} {mj_dxy:>7.1f} {mj_dz:>7.1f} {mj_lift:>8.2f} | "
                  f"{nw_grip_w:>7.1f} {nw_dxy:>7.1f} {nw_dz:>7.1f} {nw_lift:>8.2f} | "
                  f"{a_diff:>6.4f}")

    print()
    print(f"[summary] mj_peak_lift={mj_peak:.1f}mm  nw_peak_lift={nw_peak:.1f}mm  "
          f"mj_solved={mj_lifted}  nw_solved={nw_lifted}")

    nb.close()
    mb.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
