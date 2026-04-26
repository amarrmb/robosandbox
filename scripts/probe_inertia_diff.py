"""Body-by-body inertia parameter diff: Newton parser vs MuJoCo parser.

Both backends read the same panda.xml. If the parsers disagree on body_mass,
principal moments of inertia, or COM offsets, then `mj_rne`-based gravity
feedforward computed against MuJoCo's params will not exactly cancel
Newton's actual gravity load — leaving a residual PD steady-state error
proportional to the mismatch.

Hypothesis (per "Sim-to-sim findings 2026-04-25"): the residual ~12 mrad
PD error after gravity FF is partly caused by Newton's inertials differing
from MuJoCo's by a few percent on load-bearing links (link2/4/6 + hand).

This probe is read-only; it does not modify either backend.

Output format per body (matched by name):
    name                m(kg)  Inew/Imuj    com_delta(mm)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "robosandbox-core" / "src"))


def _principal_moments(inertia_tensor: np.ndarray) -> np.ndarray:
    """Sorted eigenvalues of a 3x3 inertia tensor (frame-invariant)."""
    w, _ = np.linalg.eigh(inertia_tensor)
    return np.sort(np.asarray(w, dtype=float))


def _pct_diff(a: float, b: float) -> float:
    """Symmetric percent diff. Returns 0 when both ~0."""
    denom = max(abs(a), abs(b), 1e-12)
    return 100.0 * abs(a - b) / denom


def main() -> int:
    from robosandbox.sim.mujoco_backend import MuJoCoBackend
    from robosandbox.sim.newton_backend import NewtonBackend
    from robosandbox.tasks.loader import load_builtin_task

    task = load_builtin_task("pick_cube_franka_random")
    scene = task.scene  # base scene; cube is added but inertia probe focuses on robot

    nb = NewtonBackend(world_count=1, render_size=(64, 64), dt=0.005)
    nb.load(scene)
    mb = MuJoCoBackend(render_size=(64, 64), camera="scene")
    mb.load(scene)

    import mujoco

    # Newton arrays
    n_mass = nb._model.body_mass.numpy()           # (nB,)
    n_inertia = nb._model.body_inertia.numpy()     # (nB, 3, 3)
    n_com = nb._model.body_com.numpy()             # (nB, 3)
    n_labels = list(nb._model.body_label)

    # MuJoCo arrays. Note: body_inertia[i] is principal-moment diagonal
    # in the inertia frame (already eigenvalues, but unsorted).
    m_mass = np.asarray(mb._model.body_mass)              # (nbody,)
    m_inertia_diag = np.asarray(mb._model.body_inertia)   # (nbody, 3)
    m_ipos = np.asarray(mb._model.body_ipos)              # (nbody, 3) COM in body frame

    m_labels = []
    for i in range(mb._model.nbody):
        name = mujoco.mj_id2name(mb._model, mujoco.mjtObj.mjOBJ_BODY, i) or f"body_{i}"
        m_labels.append(name)

    print(f"[newton] {len(n_labels)} bodies  [mujoco] {len(m_labels)} bodies\n")

    header = (
        f"{'body':<22} "
        f"{'m_n(kg)':>9} {'m_m(kg)':>9} {'%dm':>5} | "
        f"{'I_n principal (kg m^2)':>34} | "
        f"{'I_m principal (kg m^2)':>34} | "
        f"{'%dI_max':>7} | "
        f"{'com_n (m)':>22} {'com_m (m)':>22} {'|dcom|mm':>9}"
    )
    print(header)
    print("-" * len(header))

    flagged = []  # (body, what, value)
    summed_n_mass = 0.0
    summed_m_mass = 0.0

    for i_n, n_label in enumerate(n_labels):
        n_short = n_label.split("/")[-1]
        # Match by short name
        try:
            i_m = m_labels.index(n_short)
        except ValueError:
            print(f"{n_short:<22}  (no MuJoCo match)")
            continue

        mn = float(n_mass[i_n]); mm = float(m_mass[i_m])
        dm_pct = _pct_diff(mn, mm)
        summed_n_mass += mn
        summed_m_mass += mm

        I_n = _principal_moments(np.asarray(n_inertia[i_n]))
        I_m = np.sort(np.asarray(m_inertia_diag[i_m], dtype=float))

        # max relative diff across the three principal moments
        if max(np.max(np.abs(I_n)), np.max(np.abs(I_m))) < 1e-10:
            dI_pct_max = 0.0
        else:
            denom = np.maximum(np.maximum(np.abs(I_n), np.abs(I_m)), 1e-12)
            dI_pct_max = float(100.0 * np.max(np.abs(I_n - I_m) / denom))

        com_n = np.asarray(n_com[i_n], dtype=float)
        com_m = np.asarray(m_ipos[i_m], dtype=float)
        dcom_mm = float(np.linalg.norm(com_n - com_m) * 1000.0)

        I_n_str = "(" + ",".join(f"{x:.4g}" for x in I_n) + ")"
        I_m_str = "(" + ",".join(f"{x:.4g}" for x in I_m) + ")"
        com_n_str = "(" + ",".join(f"{x:+.4f}" for x in com_n) + ")"
        com_m_str = "(" + ",".join(f"{x:+.4f}" for x in com_m) + ")"

        marker = ""
        if dm_pct > 5.0:
            flagged.append((n_short, "mass", dm_pct)); marker += "M"
        if dI_pct_max > 5.0:
            flagged.append((n_short, "inertia", dI_pct_max)); marker += "I"
        if dcom_mm > 1.0:
            flagged.append((n_short, "com", dcom_mm)); marker += "C"
        if marker:
            marker = f" <-- {marker}"

        print(
            f"{n_short:<22} "
            f"{mn:>9.4f} {mm:>9.4f} {dm_pct:>5.1f} | "
            f"{I_n_str:>34} | "
            f"{I_m_str:>34} | "
            f"{dI_pct_max:>7.1f} | "
            f"{com_n_str:>22} {com_m_str:>22} {dcom_mm:>9.2f}{marker}"
        )

    print()
    print(f"[total mass] newton={summed_n_mass:.4f} kg  mujoco={summed_m_mass:.4f} kg  "
          f"diff={_pct_diff(summed_n_mass, summed_m_mass):.2f}%")

    print()
    if flagged:
        print(f"[FLAGGED] {len(flagged)} discrepancies > threshold "
              f"(mass>5%, inertia_principal>5%, |dcom|>1mm):")
        for body, kind, val in flagged:
            unit = "%" if kind != "com" else "mm"
            print(f"  {body:<22} {kind:<8} {val:.2f}{unit}")
        print()
        print("Likely impact: gravity FF computed via mj_rne on MuJoCo's params")
        print("will undershoot/overshoot Newton's actual gravity load on these")
        print("links by approximately the relative mass*g_arm error, leaving a")
        print("residual PD steady-state error.")
    else:
        print("[OK] No body exceeds threshold. Inertia parsing matches.")
        print("If PD residual remains ~12 mrad, the cause is NOT inertial-")
        print("parameter parsing. Look elsewhere (contact, solver, damping).")

    nb.close()
    mb.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
