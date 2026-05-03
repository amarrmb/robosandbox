"""Teleport-based oracle for insert_connector_franka.

Phase 1 / task #57. Solves IK once for the canonical "plug inserted"
EE pose, sets the sim qpos directly to that solution, settles, and
checks the success criterion. This is a "is the task structurally
solvable" sanity check — not a closed-loop policy demonstration.

If this fails (e.g., target unreachable, or success criterion mis-set),
no PPO can succeed. If it passes, the task is well-formed.
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


def solve_ik_to_pose(model, data, ee_site_id, arm_qpos_adr, target_xyz,
                      target_z_axis_world, max_iters=200, damping=0.05,
                      pos_tol=1e-3, rot_tol=1e-2):
    """Iterative DLS IK with z-axis lock. Returns final joint qpos array."""
    import mujoco
    n = len(arm_qpos_adr)
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    dof_adrs = []
    for adr in arm_qpos_adr:
        # qposadr → corresponding dof. For 1-DoF hinge/slide, dofadr = jnt_dofadr[joint_id].
        jid = int(np.where(model.jnt_qposadr == adr)[0][0])
        dof_adrs.append(int(model.jnt_dofadr[jid]))
    for it in range(max_iters):
        mujoco.mj_forward(model, data)
        cur_xyz = np.asarray(data.site_xpos[ee_site_id])
        cur_R = np.asarray(data.site_xmat[ee_site_id]).reshape(3, 3)
        pos_err = target_xyz - cur_xyz
        cur_z = cur_R[:, 2]
        cross = np.cross(cur_z, target_z_axis_world)
        dot = float(np.dot(cur_z, target_z_axis_world))
        if dot < -0.95 and np.linalg.norm(cross) < 1e-3:
            cross = np.cross(cur_z, np.array([1.0, 0.0, 0.0]))
        rot_err = cross
        if np.linalg.norm(pos_err) < pos_tol and np.linalg.norm(rot_err) < rot_tol:
            return np.array([data.qpos[a] for a in arm_qpos_adr])
        mujoco.mj_jacSite(model, data, jacp, jacr, ee_site_id)
        cols = np.array(dof_adrs)
        J = np.vstack([jacp[:, cols], jacr[:, cols]])
        err = np.concatenate([pos_err, rot_err])
        JJt = J @ J.T
        dq = J.T @ np.linalg.solve(JJt + (damping ** 2) * np.eye(6), err)
        dq = 0.5 * dq
        for i, a in enumerate(arm_qpos_adr):
            data.qpos[a] += dq[i]
    return np.array([data.qpos[a] for a in arm_qpos_adr])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=4)
    parser.add_argument("--task", type=str, default="insert_connector_franka")
    args = parser.parse_args()

    import mujoco
    from robosandbox.sim.mujoco_backend import MuJoCoBackend
    from robosandbox.tasks.loader import load_builtin_task
    from robosandbox.tasks.runner import _eval_check

    task = load_builtin_task(args.task)
    check = task.success.data
    port_id = check.get("port") or check["object"]
    plane_z = float(check.get("port_plane_z", 0.0))
    offset_ee = np.asarray(check.get("plug_tip_offset_ee", [0.0, 0.0, 0.05]))

    n_success = 0
    for trial in range(args.n_trials):
        backend = MuJoCoBackend(render_size=(120, 160), camera="scene")
        backend.load(task.scene)
        for _ in range(50):
            backend.step()
        initial = backend.observe()
        port_xyz = np.asarray(initial.scene_objects[port_id].xyz)

        # We want plug tip at port_xy with z = plane_z - 0.008 (8mm past plane).
        # Plug tip world = ee_xyz + R_ee @ offset_ee. With z_down orientation,
        # R_ee @ [0,0,0.05] = [0,0,-0.05] (EE z points along world -z).
        # So target_ee_xyz = target_tip_xyz - [0,0,-0.05] = target_tip + [0,0,0.05].
        target_tip = np.array([port_xyz[0], port_xyz[1], plane_z - 0.008])
        target_ee = target_tip + np.array([0.0, 0.0, 0.05])
        # Solve IK in a fresh data clone so the running sim's PD control
        # doesn't drift while we plan.
        m = backend._model
        d = mujoco.MjData(m)
        # Seed at home pose
        for adr, q in zip(backend._arm_qpos_adr, backend._robot.home_qpos):
            d.qpos[adr] = q
        target_q = solve_ik_to_pose(
            m, d, backend._ee_site_id, backend._arm_qpos_adr,
            target_ee, target_z_axis_world=np.array([0.0, 0.0, -1.0]),
        )
        # Teleport: set the running sim's qpos directly to the IK solution
        # and step a few times to settle (PD will hold it there).
        for adr, q in zip(backend._arm_qpos_adr, target_q):
            backend._data.qpos[adr] = float(q)
        # Set actuator targets to match
        target_q_arr = np.asarray(target_q, dtype=np.float64)
        for _ in range(200):
            backend.step(target_q_arr, gripper=0.0)

        final = backend.observe()
        ok, det = _eval_check(check, initial, final)
        n_success += int(ok)
        print(f"  trial {trial:>2}: ok={ok}  depth={det.get('depth_m', 0)*1000:+.2f}mm  "
              f"align={det.get('align_deg', 0):.2f}°")
        backend.close()

    print()
    print(f"Teleport oracle ({args.task}, classical MuJoCo): "
          f"{n_success}/{args.n_trials} = {100.0*n_success/args.n_trials:.1f}%")
    return 0 if n_success > 0 else 2


if __name__ == "__main__":
    sys.exit(main())
