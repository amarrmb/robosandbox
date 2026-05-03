"""Scripted-IK oracle baseline for insert_connector_franka.

Phase 1 / task #57. Verifies the task is solvable with perfect state.
If the oracle can't insert reliably, no policy can.

Approach:
  * Load the task in classical MuJoCo.
  * For each trial, draw the port pose from the task's randomize block
    (Phase 1: fixed pose).
  * Drive the EE to a sequence of poses via DLS IK (FrankaEEKine):
      1. Approach: 5 cm above the port plane, axis aligned with -port_axis.
      2. Pre-insert: 1 mm above port plane, same alignment.
      3. Insert: 8 mm past port plane.
  * After holding insertion for ~50 steps, evaluate the success criterion.
  * Report success rate over n_trials.

Usage:
    MUJOCO_GL=egl python3 scripts/oracle_insert_baseline.py --n-trials 16
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "robosandbox-core" / "src"))


def quat_to_R(q_xyzw: np.ndarray) -> np.ndarray:
    qx, qy, qz, qw = q_xyzw
    x2, y2, z2 = qx + qx, qy + qy, qz + qz
    wx, wy, wz = qw * x2, qw * y2, qw * z2
    xx, xy_, xz = qx * x2, qx * y2, qx * z2
    yy, yz, zz = qy * y2, qy * z2, qz * z2
    return np.array([
        [1.0 - (yy + zz), xy_ - wz,        xz + wy],
        [xy_ + wz,        1.0 - (xx + zz), yz - wx],
        [xz - wy,         yz + wx,         1.0 - (xx + yy)],
    ])


def axis_angle_log(R: np.ndarray) -> np.ndarray:
    """Rotation matrix → axis-angle vector (Lie algebra so(3))."""
    cos_t = (R.trace() - 1.0) * 0.5
    cos_t = max(-1.0, min(1.0, cos_t))
    theta = float(np.arccos(cos_t))
    if theta < 1e-6:
        return np.zeros(3)
    return theta / (2.0 * np.sin(theta)) * np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=16)
    parser.add_argument("--task", type=str, default="insert_connector_franka")
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--gain", type=float, default=0.4,
                        help="Step-fraction toward goal each tick")
    args = parser.parse_args()

    from robosandbox.sim.mujoco_backend import MuJoCoBackend
    from robosandbox.tasks.loader import load_builtin_task
    from robosandbox.tasks.runner import _eval_check
    from robosandbox.rl.ee_ik import FrankaEEKine

    task = load_builtin_task(args.task)
    arm = [f"joint{i}" for i in range(1, 8)]
    kine = FrankaEEKine(scene=task.scene, ee_body="hand", arm_joint_names=arm)

    check = task.success.data
    port_id = check.get("port") or check["object"]
    port_axis = np.asarray(check.get("port_axis", [0.0, 0.0, 1.0]), dtype=np.float64)
    port_axis /= max(np.linalg.norm(port_axis), 1e-9)
    plane_z = float(check.get("port_plane_z", 0.0))
    offset_ee = np.asarray(check.get("plug_tip_offset_ee", [0.0, 0.0, 0.05]), dtype=np.float64)
    min_depth = float(check.get("min_depth_m", 0.005))

    n_success = 0
    detail_log = []
    for trial in range(args.n_trials):
        backend = MuJoCoBackend(render_size=(120, 160), camera="scene")
        backend.load(task.scene)
        for _ in range(50):
            backend.step()
        initial = backend.observe()
        port_xyz = np.asarray(initial.scene_objects[port_id].xyz, dtype=np.float64)

        # Three waypoints, expressed as (target_tip_world, target_R)
        # Plug axis points along EE -z when in the canonical "down" gripper pose
        # (i.e. plug_axis_world = R @ [0,0,1] should equal -port_axis).
        # We define R_target such that R_target @ [0,0,1] = -port_axis.
        # For port_axis = [0, 0, 1], we want plug_axis = [0, 0, -1], which means
        # the EE z-axis points down. The Franka home pose already has this.
        # So we keep the home orientation and only translate.
        ee_R_home = quat_to_R(np.asarray(initial.ee_pose.quat_xyzw, dtype=np.float64))

        # Plug-tip world target = ee_xyz + R @ offset, so given target tip,
        # ee_xyz_target = tip - R @ offset.
        approach_tip = port_xyz + port_axis * 0.05  # 5cm above plane (plane is below by definition)
        # Wait: port_axis = [0,0,1] points UP, so "above plane" is +z. But plane
        # is at z=0.11 and tip should be there. "Approach" = 5cm above plane = z=0.16.
        approach_tip = np.array([port_xyz[0], port_xyz[1], plane_z + 0.05])
        pre_insert_tip = np.array([port_xyz[0], port_xyz[1], plane_z + 0.001])
        insert_tip = np.array([port_xyz[0], port_xyz[1], plane_z - 0.008])

        waypoints = [approach_tip, pre_insert_tip, insert_tip]
        wp_idx = 0
        wp_hold = 0  # consecutive ticks within tolerance at current waypoint

        for step in range(args.max_steps):
            obs = backend.observe()
            ee_xyz = np.asarray(obs.ee_pose.xyz, dtype=np.float64)
            ee_R = quat_to_R(np.asarray(obs.ee_pose.quat_xyzw, dtype=np.float64))
            current_q = np.asarray(obs.robot_joints, dtype=np.float64)[:7]
            tip = ee_xyz + ee_R @ offset_ee
            target_tip = waypoints[wp_idx]
            tip_err = target_tip - tip
            R_err = ee_R_home @ ee_R.T  # rotate current frame TO home
            ang_err = axis_angle_log(R_err)
            # Compose 6-DoF delta. Note the Jacobian is in WORLD frame (jacBody),
            # so use world-frame translation and rotation deltas.
            delta_ee = np.zeros(6)
            delta_ee[:3] = args.gain * tip_err
            delta_ee[3:] = 0.1 * args.gain * ang_err  # weak orientation correction
            # Clamp per-step to safe magnitude
            t_norm = np.linalg.norm(delta_ee[:3])
            if t_norm > 0.04:
                delta_ee[:3] *= 0.04 / t_norm
            r_norm = np.linalg.norm(delta_ee[3:])
            if r_norm > 0.05:
                delta_ee[3:] *= 0.05 / r_norm
            dq = kine.delta_q_from_delta_ee_6d(current_q[None, :], delta_ee[None, :])[0]
            new_q = current_q + dq
            # Step. MuJoCo backend takes (target_joints, gripper) separately;
            # use closed gripper to avoid contact noise from open fingers.
            # Repeat 10 sim ticks per IK update so the PD has time to track
            # the new setpoint before we re-plan.
            for _ in range(10):
                backend.step(new_q, gripper=0.0)

            # Advance through waypoints once close enough
            if np.linalg.norm(tip_err) < 0.005:
                wp_hold += 1
                if wp_hold > 20 and wp_idx < len(waypoints) - 1:
                    wp_idx += 1
                    wp_hold = 0
            else:
                wp_hold = 0

        final = backend.observe()
        ok, det = _eval_check(check, initial, final)
        n_success += int(ok)
        detail_log.append({"trial": trial, "ok": ok, **det})
        backend.close()
        print(f"  trial {trial:>3}: ok={ok}  depth={det.get('depth_m', 0)*1000:+.2f}mm  "
              f"align={det.get('align_deg', 0):.2f}°")

    print()
    print(f"Oracle baseline ({args.task}, classical MuJoCo): "
          f"{n_success}/{args.n_trials} = {100.0*n_success/args.n_trials:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
