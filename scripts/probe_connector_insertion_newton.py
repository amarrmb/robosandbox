"""Newton variant of the connector-insertion probe.

Runs the same MJCF as `probe_connector_insertion.py` through Newton's
mujoco_warp solver and measures whether the chamfered-port physics holds
(no wall penetration, plug settles at commanded depth).

Designed to run on DGX (newton+warp installed). Locally raises ImportError.

Usage on DGX:
    /home/amar/newton/.venv/bin/python3 \
        scripts/probe_connector_insertion_newton.py --clearance-mm 4.0
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Reuse MJCF builder and constants from the classical probe
sys.path.insert(0, str(Path(__file__).parent))
from probe_connector_insertion import (  # noqa: E402
    build_mjcf,
    PORT_BASE_Z,
    WALL_H,
    PLUG_HZ,
    CARRIER_HOME_Z,
)


@dataclass
class NewtonInsertionResult:
    clearance_mm: float
    settled_z_mm: float
    commanded_z_mm: float
    max_penetration_mm: float
    oscillation_mm: float
    success: bool
    raw_trace: np.ndarray | None = None


def run_newton(clearance_mm: float, with_chamfer: bool = True,
               n_settle: int = 200, n_descend: int = 600,
               n_final_settle: int = 200,
               substep_count: int = 4) -> NewtonInsertionResult:
    """Run probe through Newton's mujoco_warp solver."""
    import warp as wp
    import newton
    from newton import JointTargetMode
    from newton.solvers import SolverMuJoCo

    wp.init()

    # Save MJCF to a temp file (newton.add_mjcf takes a file path)
    xml = build_mjcf(clearance_mm, with_chamfer=with_chamfer)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
        f.write(xml)
        mjcf_path = f.name

    builder = newton.ModelBuilder()
    builder.add_mjcf(
        mjcf_path,
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        floating=False,
        enable_self_collisions=False,
        parse_mujoco_options=True,
    )
    print(f"[builder] body_count={builder.body_count}  joint_count={builder.joint_count}  "
          f"shape_count={builder.shape_count}")
    print(f"[builder] joint_target_pos len={len(builder.joint_target_pos)}")
    print(f"[builder] joint_q len={len(builder.joint_q)}")
    if hasattr(builder, "joint_name"):
        print(f"[builder] joint_name={list(builder.joint_name)}")
    builder.add_ground_plane()

    # 6 carrier joints: cx, cy, cz, cyaw, cpitch, croll. Newton's add_mjcf
    # creates joint_target_pos with one fewer entry than joint_count
    # (the implicit world-attach joint has no target). Iterate over whichever
    # is shorter to avoid IndexError, and pad PD gains accordingly.
    n_targets = len(builder.joint_target_pos)
    pd_kp = [500.0, 500.0, 1000.0, 100.0, 100.0, 100.0]
    pd_kd = [50.0, 50.0, 100.0, 10.0, 10.0, 10.0]
    # If Newton dropped a target entry at the front, our 6 carrier joints
    # occupy indices [n_targets-6 : n_targets].
    offset = max(0, n_targets - 6)
    for i in range(min(6, n_targets)):
        idx = offset + i
        builder.joint_target_pos[idx] = 0.0
        builder.joint_target_ke[idx] = pd_kp[i]
        builder.joint_target_kd[idx] = pd_kd[i]
        builder.joint_target_mode[idx] = int(JointTargetMode.POSITION)
    n_joints = 6  # carrier dimensionality
    target_offset = offset

    model = builder.finalize()
    solver = SolverMuJoCo(model)
    state_in = model.state()
    state_out = model.state()
    control = model.control()

    # Helper: read joint q (positions) for cx, cy, cz, cyaw, cpitch, croll
    # Newton stores joint_q in a flat array. For 6 1-dof joints this is 6 entries.
    def joint_q() -> np.ndarray:
        return state_in.joint_q.numpy()[:n_joints].copy()

    # We need plug-tip world position. The plug is the rigid body at the end
    # of the 6-DoF chain (index 5 if intermediate bodies are real, else 0).
    # Newton stores body transforms in body_q. The plug body is the last one
    # added by add_mjcf. We'll find it by iterating after finalize.
    # Simplest: read body_q[plug_body_idx]; plug_body_idx = body_count - 1
    # (assuming the order in add_mjcf preserves XML body order).
    plug_body_idx = model.body_count - 1
    print(f"[newton] body_count={model.body_count}, plug_body_idx={plug_body_idx}")

    def plug_tip_world_z() -> float:
        # body_q is a wp.array of wp.transform. transform = (px,py,pz, qx,qy,qz,qw)
        body_q = state_in.body_q.numpy()
        pz = float(body_q[plug_body_idx][2])
        # Tip is plug_center_z - PLUG_HZ (in body local frame, but body frame
        # is aligned with world at home pose since cyaw/cpitch/croll = 0).
        # For descent test where we don't rotate, this is fine.
        return pz - PLUG_HZ

    port_top_z_world = PORT_BASE_Z + 2 * WALL_H
    target_tip_world_z = port_top_z_world - 0.008  # 8mm past port
    target_carrier_z = target_tip_world_z + PLUG_HZ - CARRIER_HOME_Z
    target_carrier_z = max(target_carrier_z, -0.20)
    commanded_z_past_plane_mm = (target_tip_world_z - port_top_z_world) * 1000.0

    sim_dt = 0.002 / substep_count

    def step(target_z: float) -> None:
        # Set joint target for cz (index 2)
        target = control.joint_act.numpy()
        target[0] = 0.0
        target[1] = 0.0
        target[2] = target_z
        target[3] = 0.0
        target[4] = 0.0
        target[5] = 0.0
        control.joint_act.assign(target)
        for _ in range(substep_count):
            solver.step(state_in, state_out, control, None, sim_dt)
            state_in, state_out_local = state_out, state_in
        # Sloppy alias swap (Python doesn't let us reassign in this scope cleanly)
        # We'll re-fetch via state_in next iteration. Instead, swap by ref:
        state_in_ref[0], state_out_ref[0] = state_out_ref[0], state_in_ref[0]

    # Easier: use list to allow rebinding state_in/state_out across closures
    state_in_ref = [state_in]
    state_out_ref = [state_out]

    def step2(target_z: float) -> None:
        s_in = state_in_ref[0]
        s_out = state_out_ref[0]
        target = control.joint_target_pos.numpy()
        target[:] = 0.0
        target[target_offset + 2] = target_z  # cz is the 3rd of the 6 carrier joints
        target_wp = wp.array(target, dtype=control.joint_target_pos.dtype)
        wp.copy(control.joint_target_pos, target_wp)
        for _ in range(substep_count):
            solver.step(s_in, s_out, control, None, sim_dt)
            s_in, s_out = s_out, s_in
        state_in_ref[0] = s_in
        state_out_ref[0] = s_out

    # Phase 1: settle at home
    for _ in range(n_settle):
        step2(0.0)

    # Phase 2: ramp descent
    trace = []
    for k in range(n_descend):
        alpha = (k + 1) / n_descend
        step2(alpha * target_carrier_z)
        body_q = state_in_ref[0].body_q.numpy()
        # Plug body world transform: extract xyz
        pz = float(body_q[plug_body_idx][2])
        px = float(body_q[plug_body_idx][0])
        py = float(body_q[plug_body_idx][1])
        trace.append([px, py, pz - PLUG_HZ])

    # Phase 3: hold at target
    final_zs = []
    for _ in range(n_final_settle):
        step2(target_carrier_z)
        body_q = state_in_ref[0].body_q.numpy()
        pz = float(body_q[plug_body_idx][2])
        px = float(body_q[plug_body_idx][0])
        py = float(body_q[plug_body_idx][1])
        tip_z = pz - PLUG_HZ
        trace.append([px, py, tip_z])
        final_zs.append(tip_z)

    final_zs = np.array(final_zs)
    settled_z_mm = (final_zs[-50:].mean() - port_top_z_world) * 1000.0
    osc_mm = (final_zs[-100:].max() - final_zs[-100:].min()) * 1000.0

    trace_arr = np.array(trace)
    hole_hx_local = 0.020 + (clearance_mm / 1000.0) / 2.0  # PLUG_HX
    hole_hy_local = 0.005 + (clearance_mm / 1000.0) / 2.0  # PLUG_HY
    below_top_mask = trace_arr[:, 2] < port_top_z_world
    outside_hole_mask = ((np.abs(trace_arr[:, 0]) > hole_hx_local + 0.001) |
                         (np.abs(trace_arr[:, 1]) > hole_hy_local + 0.001))
    wall_pen_mask = below_top_mask & outside_hole_mask
    if wall_pen_mask.any():
        max_pen_mm = (port_top_z_world - trace_arr[wall_pen_mask, 2].min()) * 1000.0
    else:
        max_pen_mm = 0.0

    success = (settled_z_mm <= -5.0) and (osc_mm < 1.5) and (max_pen_mm < 0.5)

    os.unlink(mjcf_path)
    return NewtonInsertionResult(
        clearance_mm=clearance_mm,
        settled_z_mm=settled_z_mm,
        commanded_z_mm=commanded_z_past_plane_mm,
        max_penetration_mm=max_pen_mm,
        oscillation_mm=osc_mm,
        success=success,
        raw_trace=trace_arr,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clearance-mm", type=float, default=None)
    parser.add_argument("--no-chamfer", action="store_true")
    args = parser.parse_args()

    clearances = [args.clearance_mm] if args.clearance_mm else [8.0, 4.0, 1.5]
    results = []
    for c in clearances:
        try:
            r = run_newton(c, with_chamfer=not args.no_chamfer)
            results.append(r)
            print(f"[newton] clearance={c:>5.2f}mm  "
                  f"commanded={r.commanded_z_mm:+.1f}mm  "
                  f"settled={r.settled_z_mm:+6.2f}mm  "
                  f"osc={r.oscillation_mm:5.2f}mm  "
                  f"penetration={r.max_penetration_mm:5.2f}mm  "
                  f"success={'YES' if r.success else 'NO '}")
        except Exception as e:
            print(f"[newton] clearance={c}mm  FAILED: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

    print()
    print("Summary (Newton mujoco_warp):")
    for r in results:
        v = "OK" if r.success else "FAIL"
        print(f"  {r.clearance_mm:>5.2f}mm clearance: {v} "
              f"(settled {r.settled_z_mm:+.2f}mm, osc {r.oscillation_mm:.2f}mm, "
              f"penetration {r.max_penetration_mm:.2f}mm)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
