"""Integration test: Franka picks the bundled YCB mug (mesh-import acceptance).

This is the acceptance gate for the mesh-import slice. If this passes,
the pipeline of "bundled per-object sidecar -> pre-decomposed convex
hulls -> MjSpec mesh injection -> MuJoCo contacts -> pick skill" works
end to end on a real concave object.

The 5x smoke test is a cheap guard against the single-shot acceptance
hiding flakiness. Anything below 4/5 is a signal that pose / friction /
decomp threshold needs revisiting before 2.2 (randomized benchmark)
builds on top.
"""

from __future__ import annotations

import pytest

from robosandbox.agent.context import AgentContext
from robosandbox.grasp.analytic import AnalyticTopDown
from robosandbox.motion.ik import DLSMotionPlanner
from robosandbox.perception.ground_truth import GroundTruthPerception
from robosandbox.sim.mujoco_backend import MuJoCoBackend
from robosandbox.skills.pick import Pick
from robosandbox.tasks.loader import load_builtin_task


def _attempt_mug_pick() -> tuple[bool, float]:
    """One attempt at loading the mug task and picking. Returns (success, z_final)."""
    task = load_builtin_task("pick_ycb_mug")
    sim = MuJoCoBackend(render_size=(240, 320))
    sim.load(task.scene)
    try:
        for _ in range(100):
            sim.step()
        ctx = AgentContext(
            sim=sim,
            perception=GroundTruthPerception(),
            grasp=AnalyticTopDown(),
            motion=DLSMotionPlanner(n_waypoints=160, dt=0.005),
        )
        result = Pick()(ctx, object="mug")
        obs_final = sim.observe()
        final_z = obs_final.scene_objects["mug"].xyz[2]
        return (result.success, final_z)
    finally:
        sim.close()


def test_pick_ycb_mug_single_shot() -> None:
    """Single-shot acceptance: mug rises >5 cm on the default seed."""
    success, final_z = _attempt_mug_pick()
    assert success, "Pick skill reported failure"
    # Mug bottom starts at ~0.04; a 5 cm lift puts its origin above 0.09.
    assert final_z > 0.09, f"mug final z={final_z} < 0.09"


def test_pick_ycb_mug_smoke_5x() -> None:
    """Cheap non-flakiness guard: 5 attempts at the same deterministic pose.

    Each attempt creates a fresh sim so any stray warm-state isn't reused.
    Floating-point round-tripping through pickable vs ground-truth
    observations can introduce tiny jitter; >=4/5 is the bar.
    """
    successes = 0
    lifts = []
    for _ in range(5):
        ok, z = _attempt_mug_pick()
        successes += int(ok)
        lifts.append(z)
    assert successes >= 4, (
        f"pick_ycb_mug smoke succeeded {successes}/5 (final z per attempt: {lifts})"
    )
