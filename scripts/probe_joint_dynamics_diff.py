"""Joint-level dynamics parameter diff: Newton parser vs MuJoCo parser.

Compares dof_damping, dof_armature, and joint stiffness on the parsed
panda chain. Newton's MuJoCo-Warp solver may parse the panda.xml
defaults (`damping="0.5" armature="0.02"`) into mjw_model differently
than classical MuJoCo, leaving a residual PD steady-state error after
gravity feedforward (~12 mrad observed at the EE on stretched configs).

Read-only; modifies neither backend.

Output per joint (matched by name):
    name        dof_damping_n / dof_damping_m   dof_armature_n / dof_armature_m
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "robosandbox-core" / "src"))


def _pct_diff(a: float, b: float) -> float:
    denom = max(abs(a), abs(b), 1e-12)
    return 100.0 * abs(a - b) / denom


def main() -> int:
    from robosandbox.sim.mujoco_backend import MuJoCoBackend
    from robosandbox.sim.newton_backend import NewtonBackend
    from robosandbox.tasks.loader import load_builtin_task

    import mujoco

    task = load_builtin_task("pick_cube_franka_random")
    scene = task.scene

    nb = NewtonBackend(world_count=1, render_size=(64, 64), dt=0.005)
    nb.load(scene)
    mb = MuJoCoBackend(render_size=(64, 64), camera="scene")
    mb.load(scene)

    s = nb._solver
    mw = s.mjw_model
    mm = mb._model

    # Newton mjw arrays (per dof)
    n_damping = mw.dof_damping.numpy()                # (1, n_dof)
    n_armature = mw.dof_armature.numpy()              # (1, n_dof)
    # Joint-level (intrinsic stiffness, not actuator PD)
    n_jnt_stiff = mw.jnt_stiffness.numpy()            # (1, n_jnt)
    n_jnt_solref = mw.jnt_solref.numpy()              # (1, n_jnt, 2)
    n_jnt_solimp = mw.jnt_solimp.numpy()              # (1, n_jnt, 5)

    # MuJoCo classical arrays
    m_damping = np.asarray(mm.dof_damping)            # (n_dof,)
    m_armature = np.asarray(mm.dof_armature)          # (n_dof,)
    m_jnt_stiff = np.asarray(mm.jnt_stiffness)        # (n_jnt,)
    m_jnt_solref = np.asarray(mm.jnt_solref)          # (n_jnt, 2)
    m_jnt_solimp = np.asarray(mm.jnt_solimp)          # (n_jnt, 5)

    # Map joint name → newton dof index, mujoco dof index
    nb_jnt_names = []
    for j in range(mw.njnt[0] if hasattr(mw.njnt, "shape") else mw.njnt):
        # Newton mjw doesn't expose joint names directly; use MuJoCo's names
        # since add_mjcf preserves joint ordering.
        try:
            name = mujoco.mj_id2name(mm, mujoco.mjtObj.mjOBJ_JOINT, j) or f"j{j}"
        except Exception:
            name = f"j{j}"
        nb_jnt_names.append(name)

    print(f"[newton] njnt={len(nb_jnt_names)}  ndof={n_damping.shape[1]}")
    print(f"[mujoco] njnt={mm.njnt}  ndof={mm.nv}")
    print()

    print(f"{'idx':>3} {'joint':<22} | "
          f"{'damp_n':>8} {'damp_m':>8} {'%':>5} | "
          f"{'arm_n':>8} {'arm_m':>8} {'%':>5} | "
          f"{'stiff_n':>8} {'stiff_m':>8} {'%':>5} | "
          f"{'solref_n':>14} {'solref_m':>14}")
    print("-" * 140)

    flagged = []
    n_dof_min = min(n_damping.shape[1], len(m_damping))
    n_jnt_min = min(len(n_jnt_stiff[0]), len(m_jnt_stiff))

    for j in range(n_jnt_min):
        name = nb_jnt_names[j] if j < len(nb_jnt_names) else f"j{j}"

        # NOTE: dof index ≠ joint index for ball/free joints, but for the
        # panda's hinge+slide joints there's a 1-1 map per joint. Use j as
        # both indices; if a real free joint slips in, this row will be
        # misaligned and the marker should make it obvious.
        d_n = float(n_damping[0, j]) if j < n_damping.shape[1] else float("nan")
        d_m = float(m_damping[j]) if j < len(m_damping) else float("nan")
        a_n = float(n_armature[0, j]) if j < n_armature.shape[1] else float("nan")
        a_m = float(m_armature[j]) if j < len(m_armature) else float("nan")
        s_n = float(n_jnt_stiff[0, j])
        s_m = float(m_jnt_stiff[j])

        sr_n = tuple(float(x) for x in n_jnt_solref[0, j])
        sr_m = tuple(float(x) for x in m_jnt_solref[j])

        d_pct = _pct_diff(d_n, d_m)
        a_pct = _pct_diff(a_n, a_m)
        s_pct = _pct_diff(s_n, s_m)
        sr_diff = any(_pct_diff(a, b) > 1.0 for a, b in zip(sr_n, sr_m))

        marker = ""
        if d_pct > 5.0: marker += "D"; flagged.append((name, "damping", d_pct))
        if a_pct > 5.0: marker += "A"; flagged.append((name, "armature", a_pct))
        if s_pct > 5.0: marker += "S"; flagged.append((name, "stiffness", s_pct))
        if sr_diff: marker += "R"; flagged.append((name, "solref", 0.0))
        if marker:
            marker = f" <-- {marker}"

        print(f"{j:>3d} {name:<22} | "
              f"{d_n:>8.4f} {d_m:>8.4f} {d_pct:>5.1f} | "
              f"{a_n:>8.4f} {a_m:>8.4f} {a_pct:>5.1f} | "
              f"{s_n:>8.4f} {s_m:>8.4f} {s_pct:>5.1f} | "
              f"{str(sr_n):>14} {str(sr_m):>14}{marker}")

    print()
    print("=== Equality constraints (finger mirror) ===")
    n_neq = mw.eq_active0.shape[0] if hasattr(mw.eq_active0, "shape") else 0
    print(f"[newton] neq parsed:  {n_neq}")
    print(f"[mujoco] neq parsed:  {mm.neq}")
    if mm.neq > 0:
        for i in range(mm.neq):
            print(f"  mujoco eq[{i}]: type={mm.eq_type[i]} obj1={mm.eq_obj1id[i]} "
                  f"obj2={mm.eq_obj2id[i]} solref={tuple(mm.eq_solref[i])} "
                  f"solimp={tuple(mm.eq_solimp[i])}")

    print()
    if flagged:
        print(f"[FLAGGED] {len(flagged)} discrepancies > threshold:")
        for name, kind, val in flagged:
            print(f"  {name:<22} {kind:<10} {val:.2f}%")
    else:
        print("[OK] All joint-level dynamics parameters match.")

    nb.close()
    mb.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
