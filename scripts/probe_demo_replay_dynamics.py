"""Open-loop replay of a successful scripted demo in both backends.

Same seed, same scene jitter, same recorded action sequence. Logs cube z,
gripper width, and EE z each step. Removes policy variance from the
sim-to-sim comparison: any divergence is pure physics (PD tracking
under load, contact friction, solver behavior).

Usage:
    python scripts/probe_demo_replay_dynamics.py [demo_dir]

Expected outcome before our investigation: MuJoCo lifts the cube; Newton
fails to. That isolates *which physics behavior* breaks the lift.
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


def _load_demo_actions(demo_dir: Path) -> tuple[int, list[np.ndarray], list[float], np.ndarray]:
    """Returns (seed, arm_action_sequence, gripper_action_sequence, initial_robot_joints)."""
    ep = json.loads((demo_dir / "episode.json").read_text())
    seed = int(ep.get("seed", 0))
    actions_arm: list[np.ndarray] = []
    actions_grip: list[float] = []
    initial_q: np.ndarray | None = None
    with (demo_dir / "events.jsonl").open() as f:
        for line in f:
            ev = json.loads(line)
            if initial_q is None:
                initial_q = np.array(ev["robot_joints"], dtype=np.float64)
            a = ev.get("action")
            if a is None:
                continue
            actions_arm.append(np.array(a["joints"], dtype=np.float64))
            actions_grip.append(float(a.get("gripper", 0.0)))
    assert initial_q is not None
    return seed, actions_arm, actions_grip, initial_q


def _initial_cube_z(scene) -> float:
    return float(scene.objects[0].pose.xyz[2])


def main() -> int:
    from robosandbox.sim.mujoco_backend import MuJoCoBackend
    from robosandbox.sim.newton_backend import NewtonBackend
    from robosandbox.tasks.loader import load_builtin_task
    from robosandbox.tasks.randomize import jitter_scene

    demo_dir = Path(
        sys.argv[1] if len(sys.argv) > 1
        else "/home/amar/robosandbox/runs/demos_franka_pick/20260424-070730-2f89ef67"
    )
    seed, arm_acts, grip_acts, initial_q = _load_demo_actions(demo_dir)
    n_steps = len(arm_acts)
    print(f"[demo] {demo_dir.name}  seed={seed}  n_action_steps={n_steps}")

    task = load_builtin_task("pick_cube_franka_random")
    scene = jitter_scene(task.scene, task.randomize, seed=seed)
    z0 = _initial_cube_z(scene)
    print(f"[scene] cube initial xyz = {scene.objects[0].pose.xyz}  (z0={z0*1000:.1f} mm)")

    # action_repeat=6: actions are at ~30Hz, sim is at 200Hz -> 6 sim steps/action
    ACTION_REPEAT = 6

    nb = NewtonBackend(world_count=1, render_size=(64, 64), dt=0.005)
    nb.load(scene)
    mb = MuJoCoBackend(render_size=(64, 64), camera="scene")
    mb.load(scene)
    # Hook gravity FF: NewtonBackend uses MuJoCo as IK oracle for FF computation
    nb.set_gravity_compensation(mb.compute_gravity_torque)

    # Settle (match the recorder's settle_steps before its first frame)
    SETTLE = 100
    for _ in range(SETTLE):
        nb.step()
        mb.step()

    # Header
    print()
    print(f"{'step':>5} {'time(s)':>7} | "
          f"{'mj_grip(mm)':>11} {'mj_cube_z(mm)':>14} {'mj_lift(mm)':>11} | "
          f"{'nw_grip(mm)':>11} {'nw_cube_z(mm)':>14} {'nw_lift(mm)':>11} | "
          f"{'d_lift(mm)':>10}")
    print("-" * 130)

    n_arm = 7
    mj_peak_lift = 0.0
    nw_peak_lift = 0.0
    log_every = max(1, n_steps // 30)

    for i in range(n_steps):
        target = arm_acts[i][:n_arm]
        grip = grip_acts[i]
        # Run ACTION_REPEAT sim steps per action (matches recorder cadence)
        for _ in range(ACTION_REPEAT):
            mb.step(target_joints=target, gripper=grip)
            nb.step(target_joints=target, gripper=grip)

        if i % log_every == 0 or i == n_steps - 1:
            mo = mb.observe()
            no = nb.observe()
            mj_grip_mm = (mo.gripper_width or 0.0) * 1000.0
            nw_grip_mm = (no.gripper_width or 0.0) * 1000.0
            mj_cube_z = mo.scene_objects["red_cube"].xyz[2]
            nw_cube_z = no.scene_objects["red_cube"].xyz[2]
            mj_lift_mm = (mj_cube_z - z0) * 1000.0
            nw_lift_mm = (nw_cube_z - z0) * 1000.0
            mj_peak_lift = max(mj_peak_lift, mj_lift_mm)
            nw_peak_lift = max(nw_peak_lift, nw_lift_mm)
            print(
                f"{i:>5d} {nb._t:>7.3f} | "
                f"{mj_grip_mm:>11.1f} {mj_cube_z*1000:>14.2f} {mj_lift_mm:>11.2f} | "
                f"{nw_grip_mm:>11.1f} {nw_cube_z*1000:>14.2f} {nw_lift_mm:>11.2f} | "
                f"{(mj_lift_mm - nw_lift_mm):>10.2f}"
            )

    print()
    print(f"[summary]  MuJoCo peak lift: {mj_peak_lift:.1f} mm   "
          f"Newton peak lift: {nw_peak_lift:.1f} mm   "
          f"(success threshold: 50.0 mm)")
    print(f"[summary]  MuJoCo success: {mj_peak_lift >= 50.0}    "
          f"Newton success: {nw_peak_lift >= 50.0}")

    nb.close()
    mb.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
