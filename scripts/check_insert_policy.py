"""Run the trained insert_v1_fixed policy with EXACTLY the same rollout
shape as training (1024 worlds, settle 50, run 128 steps) and check
end_ins. This isolates whether the 63% from the training log holds up
when the policy is loaded fresh, vs whether something in the eval CLI
mismatches training conditions.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
sys.path.insert(0, "/home/amar/robosandbox/packages/robosandbox-core/src")


def main() -> int:
    from robosandbox.tasks.loader import load_builtin_task
    from robosandbox.sim.newton_backend import NewtonBackend
    from robosandbox.policy import load_policy

    task = load_builtin_task("insert_connector_franka")
    policy = load_policy(Path("/home/amar/robosandbox/outputs/insert_v1_fixed"),
                         scene=task.scene)

    sim = NewtonBackend(world_count=1024, render_size=(120, 160))
    sim.load(task.scene)
    print(f"loaded sim with {sim.n_worlds} worlds")

    # Reset + settle
    sim.reset()
    for _ in range(50):
        sim.step()

    # Run for 128 policy steps (same as training rollout). Use observe_all
    # because NeuralPolicy.act() takes an Observation, not arrays.
    for step in range(128):
        obs_all = sim.observe_all()
        targets = np.zeros((sim.n_worlds, sim.n_dof), dtype=np.float64)
        grippers = np.zeros(sim.n_worlds, dtype=np.float64)
        for w in range(sim.n_worlds):
            action = policy.act(obs_all[w])  # (n_dof + 1,)
            targets[w, :sim.n_dof] = action[:sim.n_dof]
            grippers[w] = action[sim.n_dof]
        sim.step_all(targets, grippers)

    # Final state — compute end_ins exactly as the training log does
    arr = sim.observe_all_arrays()
    check = task.success.data
    port_axis = np.asarray(check["port_axis"], dtype=np.float64)
    plane_z = float(check["port_plane_z"])
    offset_ee = np.asarray(check["plug_tip_offset_ee"], dtype=np.float64)
    min_depth = float(check["min_depth_m"])
    axis_tol = float(check["axis_tol_deg"])

    ee_xyz = arr["ee_xyz"]
    ee_quat = arr["ee_quat"]
    port_xyz = arr["obj_xyz"]["port_target"]

    qx, qy, qz, qw = ee_quat[:, 0], ee_quat[:, 1], ee_quat[:, 2], ee_quat[:, 3]
    x2, y2, z2 = qx + qx, qy + qy, qz + qz
    wx, wy, wz = qw * x2, qw * y2, qw * z2
    xx, xy_, xz = qx * x2, qx * y2, qx * z2
    yy, yz, zz = qy * y2, qy * z2, qz * z2
    R = np.empty((sim.n_worlds, 3, 3))
    R[:, 0, 0] = 1.0 - (yy + zz); R[:, 0, 1] = xy_ - wz;        R[:, 0, 2] = xz + wy
    R[:, 1, 0] = xy_ + wz;        R[:, 1, 1] = 1.0 - (xx + zz); R[:, 1, 2] = yz - wx
    R[:, 2, 0] = xz - wy;         R[:, 2, 1] = yz + wx;         R[:, 2, 2] = 1.0 - (xx + yy)
    tip = ee_xyz + np.einsum("nij,j->ni", R, offset_ee)
    plug_axis = np.einsum("nij,j->ni", R, np.array([0.0, 0.0, 1.0]))
    cos_a = np.clip(np.einsum("ni,i->n", plug_axis, -port_axis), -1.0, 1.0)
    align_deg = np.degrees(np.arccos(cos_a))
    depth = plane_z - np.einsum("ni,i->n", tip, port_axis)

    end_aligned = align_deg <= axis_tol
    end_inserted = (depth >= min_depth) & end_aligned

    n_aligned = int(end_aligned.sum())
    n_inserted = int(end_inserted.sum())
    print(f"world_count={sim.n_worlds}")
    print(f"  end_aligned: {n_aligned}/{sim.n_worlds}  ({100.0*n_aligned/sim.n_worlds:.1f}%)")
    print(f"  end_inserted: {n_inserted}/{sim.n_worlds}  ({100.0*n_inserted/sim.n_worlds:.1f}%)")
    print(f"  mean tip-port dist: {np.linalg.norm(tip - port_xyz, axis=1).mean()*100:.2f} cm")
    print(f"  mean align deg: {align_deg.mean():.2f}")
    print(f"  mean depth mm: {depth.mean()*1000:.2f}")
    print(f"  inserted depth distribution (mm):")
    if n_inserted > 0:
        ins_depth_mm = depth[end_inserted] * 1000.0
        print(f"    min={ins_depth_mm.min():.2f}  median={np.median(ins_depth_mm):.2f}  max={ins_depth_mm.max():.2f}")
    sim.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
