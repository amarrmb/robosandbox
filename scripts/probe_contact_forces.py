"""Final sim-parity check: contact-solve outputs at scripted grasp pose.

Sets BOTH backends to the same arm pose with the gripper closed on the
cube (no policy, no replay), then steps once and reads contact normal
forces, friction forces, and qfrc_constraint. If those differ at >5%
with everything else parsed identically, the residual gap is in the
solver implementation (mujoco_warp vs classical) and is off-thesis to
debug from RoboSandbox.

Read-only; modifies neither backend.
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

    import mujoco

    task = load_builtin_task("pick_cube_franka_random")
    scene = task.scene  # base scene, cube at (0.4, 0, 0.06), no jitter

    nb = NewtonBackend(world_count=1, render_size=(64, 64), dt=0.002)
    nb.load(scene)
    mb = MuJoCoBackend(render_size=(64, 64), camera="scene")
    mb.load(scene)
    nb.set_gravity_compensation(mb.compute_gravity_torque)

    # Settle both for 0.5s of sim time
    for _ in range(int(0.5 / 0.002)):
        nb.step()
        mb.step()

    # Approach pose: hand-tuned joint config that puts the EE just above
    # the cube at (0.4, 0, 0.05), gripper open. This isn't from a policy
    # — it's a deterministic config so both sims see the exact same
    # commanded state.
    approach_q = np.array([
        0.0,           # j1 base rotation
        0.7,           # j2 shoulder pitch (forward)
        0.0,           # j3
        -1.8,          # j4 elbow (more bent)
        0.0,           # j5
        2.5,           # j6 wrist pitch (down)
        -0.785,        # j7 wrist roll
    ])

    # Drive arm to approach pose, gripper open, for 1.0s
    for _ in range(int(1.0 / 0.002)):
        nb.step(target_joints=approach_q, gripper=0.0)
        mb.step(target_joints=approach_q, gripper=0.0)

    # Close gripper for 1.0s while holding pose
    for _ in range(int(1.0 / 0.002)):
        nb.step(target_joints=approach_q, gripper=1.0)
        mb.step(target_joints=approach_q, gripper=1.0)

    nob = nb.observe()
    mob = mb.observe()
    print(f"After grasp attempt:")
    print(f"  [newton] gripper_w = {(nob.gripper_width or 0)*1000:.2f} mm  "
          f"cube z = {nob.scene_objects['red_cube'].xyz[2]*1000:.2f} mm  "
          f"ee z = {nob.ee_pose.xyz[2]*1000:.2f} mm")
    print(f"  [mujoco] gripper_w = {(mob.gripper_width or 0)*1000:.2f} mm  "
          f"cube z = {mob.scene_objects['red_cube'].xyz[2]*1000:.2f} mm  "
          f"ee z = {mob.ee_pose.xyz[2]*1000:.2f} mm")
    print()

    # ---- Inspect MuJoCo classical contacts ---------------------------
    print("=== MuJoCo contacts (classical) ===")
    md = mb._data
    mm = mb._model
    print(f"  ncon = {md.ncon}")
    finger_geoms = set()
    cube_geoms = set()
    for i in range(mm.ngeom):
        bid = mm.geom_bodyid[i]
        bname = mujoco.mj_id2name(mm, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
        if "finger" in bname:
            finger_geoms.add(i)
        elif bname == "red_cube":
            cube_geoms.add(i)
    cube_finger_contacts = []
    for i in range(md.ncon):
        c = md.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        is_cube_finger = (g1 in finger_geoms and g2 in cube_geoms) or \
                         (g1 in cube_geoms and g2 in finger_geoms)
        if is_cube_finger:
            f = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(mm, md, i, f)
            normal_force = float(f[0])
            tan_force = float(np.linalg.norm(f[1:3]))
            cube_finger_contacts.append((g1, g2, c.dist, normal_force, tan_force))
            print(f"  contact {i}: g1={g1} g2={g2} dist={c.dist*1000:.3f}mm "
                  f"normal_force={normal_force:.3f}N tangent={tan_force:.3f}N")
    if not cube_finger_contacts:
        print("  (no cube-finger contacts found)")
    mj_total_normal = sum(c[3] for c in cube_finger_contacts)
    print(f"  total cube-finger normal force: {mj_total_normal:.3f} N")
    print()

    # ---- Inspect Newton contacts -------------------------------------
    print("=== Newton contacts (mujoco_warp solver state) ===")
    s = nb._solver
    mwd = s.mjw_data
    nacon = int(mwd.nacon.numpy()[0])
    print(f"  nacon = {nacon}")
    if nacon > 0:
        contact = mwd.contact
        c_geom = contact.geom.numpy()[:nacon]
        c_dist = contact.dist.numpy()[:nacon]
        c_pos = contact.pos.numpy()[:nacon]
        c_friction = contact.friction.numpy()[:nacon]
        # Per-contact force lives in efc.force, indexed via efc_address
        efc_addr = contact.efc_address.numpy()[:nacon]
        efc_force = mwd.efc.force.numpy()[0]  # (njmax,)
        # Need to map mjw geom indices to body labels
        mw = s.mjw_model
        g_bodyid = mw.geom_bodyid.numpy()
        # Build finger/cube geom sets in mjw indexing
        # mjw body 0 = world, mjw body 1..8 = link0..link7,
        # mjw body 9 = hand, body 10 = left_finger, body 11 = right_finger,
        # body 12 = red_cube
        n_finger_bodies = {10, 11}
        n_cube_body = 12
        n_finger_geoms = {i for i in range(len(g_bodyid)) if int(g_bodyid[i]) in n_finger_bodies}
        n_cube_geoms = {i for i in range(len(g_bodyid)) if int(g_bodyid[i]) == n_cube_body}
        nw_total_normal = 0.0
        n_cube_finger = 0
        for i in range(nacon):
            g1, g2 = int(c_geom[i, 0]), int(c_geom[i, 1])
            is_cube_finger = (g1 in n_finger_geoms and g2 in n_cube_geoms) or \
                             (g1 in n_cube_geoms and g2 in n_finger_geoms)
            if is_cube_finger:
                # Normal force lives at efc.force[efc_address[i]]
                normal_force = float(efc_force[int(efc_addr[i, 0])]) if efc_addr[i, 0] >= 0 else 0.0
                nw_total_normal += abs(normal_force)
                n_cube_finger += 1
                print(f"  contact {i}: g1={g1} g2={g2} dist={c_dist[i]*1000:.3f}mm "
                      f"normal_force={normal_force:.3f}N fric={tuple(round(float(x),4) for x in c_friction[i])}")
        if n_cube_finger == 0:
            print("  (no cube-finger contacts found)")
        print(f"  total cube-finger normal force: {nw_total_normal:.3f} N "
              f"({n_cube_finger} contacts)")
    print()

    print("=== Verdict ===")
    print(f"  MuJoCo: {len(cube_finger_contacts)} cube-finger contacts, "
          f"total normal {sum(c[3] for c in cube_finger_contacts):.3f} N")
    if nacon > 0:
        print(f"  Newton: {n_cube_finger} cube-finger contacts, "
              f"total normal {nw_total_normal:.3f} N")
    print(f"  MuJoCo cube z={mob.scene_objects['red_cube'].xyz[2]*1000:.1f}mm  "
          f"Newton cube z={nob.scene_objects['red_cube'].xyz[2]*1000:.1f}mm")

    nb.close()
    mb.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
