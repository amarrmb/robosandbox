"""Walk the Franka kinematic chain in both Newton and MuJoCo, body by body.

Find the first link in the chain (link0 → link7 → hand) where the two
backends report different world poses for the same joint config. That
isolates whether the FK mismatch comes from a single body or accumulates
across multiple.

Output: per-body world xyz + quat side by side, with the per-link delta.
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

    task = load_builtin_task("pick_cube_franka_random")
    scene = task.scene  # base scene, no jitter

    nb = NewtonBackend(world_count=1, render_size=(240, 320), dt=0.005)
    nb.load(scene)

    mb = MuJoCoBackend(render_size=(240, 320), camera="scene")
    mb.load(scene)

    # Set both to exactly home_qpos (no settle drift)
    home = nb._robot.home_qpos
    print(f"[home_qpos] {home}")

    # Newton: set joint_q directly
    import warp as wp
    nb._ensure_runtime()
    full_q = nb._state_0.joint_q.numpy()
    for local_q, q in zip(nb._w_arm_q, home):
        full_q[local_q] = float(q)
    # Also set finger joints to open
    for local_q in nb._w_gripper_q:
        full_q[local_q] = nb._robot.gripper_open_qpos
    arr = wp.array(full_q, dtype=nb._state_0.joint_q.dtype)
    wp.copy(nb._state_0.joint_q, arr)
    # Run forward kinematics
    import newton
    newton.eval_fk(nb._model, nb._state_0.joint_q, nb._state_0.joint_qd, nb._state_0)

    # MuJoCo: set qpos directly. Set BOTH finger qposes (equality constraint
    # only fires under mj_step, not mj_forward).
    import mujoco
    for adr, q in zip(mb._arm_qpos_adr, home):
        mb._data.qpos[adr] = float(q)
    # Set every finger joint by name
    for finger_name in nb._robot.gripper_joint_names:
        try:
            adr = int(mb._model.joint(finger_name).qposadr[0])
            mb._data.qpos[adr] = 0.04
        except Exception as e:
            print(f"[probe] WARN: couldn't set {finger_name}: {e}")
    mujoco.mj_forward(mb._model, mb._data)

    # Walk Newton bodies
    n_body_q = nb._state_0.body_q.numpy()
    n_labels = list(nb._model.body_label)
    print(f"\n[newton] {len(n_labels)} bodies")

    # MuJoCo body names
    m_labels = []
    for i in range(mb._model.nbody):
        name = mujoco.mj_id2name(mb._model, mujoco.mjtObj.mjOBJ_BODY, i) or f"body_{i}"
        m_labels.append(name)
    print(f"[mujoco] {len(m_labels)} bodies")

    # Side-by-side pairs
    print(f"\n{'name':<30} {'newton xyz':>32}  {'mujoco xyz':>32}  delta_mm")
    print("-" * 130)
    # Match by suffix (newton labels often "panda/link0", mujoco "link0")
    for i_n, n_label in enumerate(n_labels):
        # short name
        n_short = n_label.split("/")[-1]
        # find matching mujoco body
        i_m = None
        for j, m_label in enumerate(m_labels):
            if m_label == n_short:
                i_m = j
                break
        if i_m is None:
            continue
        n_xyz = n_body_q[i_n][:3]
        n_quat = n_body_q[i_n][3:7]
        m_xyz = mb._data.xpos[i_m]
        m_quat_wxyz = mb._data.xquat[i_m]
        m_quat_xyzw = (m_quat_wxyz[1], m_quat_wxyz[2], m_quat_wxyz[3], m_quat_wxyz[0])
        delta = np.linalg.norm(np.asarray(n_xyz) - np.asarray(m_xyz)) * 1000
        # compute quat delta (angular)
        nq = np.asarray(n_quat)
        mq = np.asarray(m_quat_xyzw)
        dot = float(abs(np.dot(nq, mq)))
        ang_deg = float(np.degrees(2 * np.arccos(min(1.0, dot)))) if dot < 1.0 else 0.0
        print(f"{n_short:<30} {str(tuple(round(float(x),4) for x in n_xyz)):>32}  {str(tuple(round(float(x),4) for x in m_xyz)):>32}  {delta:6.1f}mm {ang_deg:5.2f}°")

    nb.close()
    mb.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
