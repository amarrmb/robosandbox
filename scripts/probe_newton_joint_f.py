"""Probe what Newton's joint_q layout is + what happens when joint_f is set."""
from __future__ import annotations
import os, sys
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
    sim = NewtonBackend(world_count=1, render_size=(240, 320), dt=0.005)
    sim.load(task.scene)
    kin = MuJoCoBackend(render_size=(240, 320), camera="scene")
    kin.load(task.scene)

    print(f"[probe] _w_arm_q = {sim._w_arm_q}")
    print(f"[probe] _w_gripper_q = {sim._w_gripper_q}")
    print(f"[probe] _dof_per_world = {sim._dof_per_world}")
    print(f"[probe] _actuators_per_world = {sim._actuators_per_world}")
    print(f"[probe] joint_q shape = {sim._state_0.joint_q.numpy().shape}")
    print(f"[probe] joint_q values = {sim._state_0.joint_q.numpy()}")

    # Try one manual step with FF set
    home = np.array(sim._robot.home_qpos, dtype=np.float64)
    tau = kin.compute_gravity_torque(home, gripper_qpos=0.04)
    print(f"\n[probe] tau from MuJoCo = {tau}")
    print(f"[probe] tau shape = {tau.shape}")

    # Register and step many times — settle under FF
    sim.set_gravity_compensation(kin.compute_gravity_torque)
    print(f"\n[probe] settling 200 steps under FF...")
    for i in range(200):
        sim.step()
        q = sim._state_0.joint_q.numpy()
        if np.any(np.isnan(q)):
            print(f"[probe] NaN at step {i}")
            print(f"  joint_q: {q}")
            print(f"  joint_f: {sim._control.joint_f.numpy()}")
            break
    else:
        q = sim._state_0.joint_q.numpy()
        print(f"[probe] OK after 200 steps")
        print(f"  joint_q[0:9] = {q[:9]}")
        target = sim._control.joint_target_pos.numpy()
        print(f"  joint_target_pos[0:9] = {target[:9]}")
        # Compare arm joints to home_qpos
        home = np.array(sim._robot.home_qpos)
        actual_arm = np.array([q[i] for i in sim._w_arm_q])
        print(f"  arm error vs home: {actual_arm - home}")
        print(f"  max |err|: {np.max(np.abs(actual_arm - home)):.6f} rad")

    sim.close()
    kin.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
