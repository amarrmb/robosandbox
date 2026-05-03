"""Closed-loop scripted oracle for insert_connector_franka.

Replaces the teleport-oracle (set qpos directly) with a real closed-loop
controller. At each step: observe → solve_ik to next waypoint pose →
drive PD setpoint → repeat. If this hits 100%, the task is genuinely
closed-loop controllable; teleporting was not enough proof.

Uses motion/ik.py::solve_ik with `z_down` orientation mode (wrist roll
free) which converges reliably for the Franka.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
sys.path.insert(0, "/home/amar/robosandbox/packages/robosandbox-core/src")


class _IKSimAdapter:
    """Minimal sim shim exposing the attrs solve_ik() needs."""

    def __init__(self, backend):
        self.model = backend._model
        self.data = backend._data
        self.ee_site_id = backend._ee_site_id
        self.arm_qpos_adr = backend._arm_qpos_adr
        self.joint_names = list(backend._robot.arm_joint_names)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=8)
    parser.add_argument("--task", type=str, default="insert_connector_franka")
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--substeps-per-plan", type=int, default=5,
                        help="MuJoCo sim ticks between IK re-plans")
    args = parser.parse_args()

    import mujoco
    from robosandbox.sim.mujoco_backend import MuJoCoBackend
    from robosandbox.tasks.loader import load_builtin_task
    from robosandbox.tasks.randomize import jitter_scene
    from robosandbox.tasks.runner import _eval_check
    from robosandbox.motion.ik import solve_ik
    from robosandbox.types import Pose

    task = load_builtin_task(args.task)
    check = task.success.data
    port_id = check.get("port") or check["object"]
    plane_z = float(check.get("port_plane_z", 0.0))
    offset_ee = np.asarray(check.get("plug_tip_offset_ee", [0.0, 0.0, 0.05]))

    n_success = 0
    for trial in range(args.n_trials):
        # Per-trial random port pose so we test the closed-loop oracle on
        # the actual deployment distribution (not a single fixed pose).
        scene = jitter_scene(task.scene, task.randomize, seed=trial + 1)

        backend = MuJoCoBackend(render_size=(120, 160), camera="scene")
        backend.load(scene)
        for _ in range(50):
            backend.step()
        initial = backend.observe()
        port_xyz = np.asarray(initial.scene_objects[port_id].xyz)

        # Plug tip world target = ee_xyz + R_ee @ offset_ee. With z_down,
        # R_ee @ [0,0,0.05] = [0,0,-0.05], so ee_z_target = tip_z + 0.05.
        # Three waypoints: approach (5cm above), pre-insert (1mm above),
        # insert (8mm past plane).
        approach_ee = np.array([port_xyz[0], port_xyz[1], plane_z + 0.05 + 0.05])
        pre_insert_ee = np.array([port_xyz[0], port_xyz[1], plane_z + 0.001 + 0.05])
        insert_ee = np.array([port_xyz[0], port_xyz[1], plane_z - 0.008 + 0.05])
        target_quat = (1.0, 0.0, 0.0, 0.0)  # not used by z_down mode

        ik_sim = _IKSimAdapter(backend)
        wp_idx = 0
        wp_hold = 0
        steps_in_phase = 0
        waypoints = [approach_ee, pre_insert_ee, insert_ee]
        # Steps to allow per waypoint before forcing advance (PD takes time
        # to settle; without this we hop too fast through approach and never
        # actually reach it).
        max_steps_per_waypoint = [120, 60, 80]

        seed_q = None  # warm-start IK from previous solution to keep stable

        for step in range(args.max_steps):
            obs = backend.observe()
            ee_xyz = np.asarray(obs.ee_pose.xyz)
            current_q = np.asarray(obs.robot_joints)[:7]
            target_ee = waypoints[wp_idx]
            tip_err = np.linalg.norm(target_ee - ee_xyz)

            # Re-plan every substeps_per_plan ticks (or first iter)
            if step % args.substeps_per_plan == 0:
                try:
                    target_pose = Pose(xyz=tuple(target_ee), quat_xyzw=target_quat)
                    target_q = solve_ik(
                        ik_sim, target_pose,
                        seed_joints=seed_q if seed_q is not None else current_q,
                        orientation="z_down",
                        max_iters=80, damping=1e-2, step_size=0.5,
                        pos_tol=2e-3,
                    )
                    seed_q = target_q
                except Exception as e:
                    target_q = current_q
            backend.step(target_q, gripper=0.0)

            # Phase advance: when EE close to current waypoint OR phase budget exhausted
            steps_in_phase += 1
            if (tip_err < 0.005 or steps_in_phase >= max_steps_per_waypoint[wp_idx]):
                if wp_idx < len(waypoints) - 1:
                    wp_idx += 1
                    steps_in_phase = 0

        final = backend.observe()
        ok, det = _eval_check(check, initial, final)
        n_success += int(ok)
        port_show = (round(port_xyz[0], 3), round(port_xyz[1], 3))
        print(f"  trial {trial:>2}: ok={ok}  port={port_show}  "
              f"depth={det.get('depth_m', 0)*1000:+.2f}mm  "
              f"align={det.get('align_deg', 0):.2f}°")
        backend.close()

    pct = 100.0 * n_success / args.n_trials
    print()
    print(f"Closed-loop oracle ({args.task}, classical MuJoCo, random port pose): "
          f"{n_success}/{args.n_trials} = {pct:.1f}%")
    return 0 if n_success > 0 else 2


if __name__ == "__main__":
    sys.exit(main())
