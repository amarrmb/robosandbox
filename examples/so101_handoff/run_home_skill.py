"""Drive the SO-101 skeleton backend through the ``Home`` skill.

Proves the sim-to-real interface for the class of skills that only use
``observe()`` + ``step()`` (no motion-planning IK): hand the real-
backend subclass to the same ``AgentContext`` the sim uses, call the
same skill, get the same ``SkillResult`` type back.

``Home`` is the cleanest such proof because it needs no perception,
no grasp, no Cartesian IK — just a linear joint-space ramp to the
configured home_qpos. If your hardware wiring can drive one of
these, the RoboSandbox skill abstraction holds for every other
observation+step skill you author (teleop, joint-space interpolations,
open-loop primitives like ``Wave``).

Usage (no hardware required — the skeleton tracks joints in software):

    uv run python examples/so101_handoff/run_home_skill.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow `uv run python examples/so101_handoff/run_home_skill.py` to find
# the sibling so101_backend module without requiring a package install.
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np  # noqa: E402

from so101_backend import SO101Backend  # noqa: E402
from robosandbox.agent.context import AgentContext  # noqa: E402
from robosandbox.skills.home import Home  # noqa: E402
from robosandbox.types import Scene  # noqa: E402


def main() -> int:
    backend = SO101Backend()
    backend.load(Scene())

    # Start somewhere *other* than home so the Home skill has work to do.
    backend._joints = np.array([0.5, -0.5, 0.5, 0.3, -0.3], dtype=np.float64)

    before = backend.observe().robot_joints
    print(f"before home: {np.round(before, 3)}")

    # AgentContext accepts any SimBackend — we pass the real-backend
    # subclass straight through. Other context members (perception,
    # grasp, motion) are unused by Home and can be None.
    ctx = AgentContext(sim=backend, perception=None, grasp=None, motion=None)
    result = Home()(ctx)

    after = backend.observe().robot_joints
    print(f"after home:  {np.round(after, 3)}")
    print(f"result:      success={result.success} reason={result.reason!r}")

    drift = float(np.linalg.norm(after - np.asarray(backend._config.home_qpos)))
    print(f"home error:  {drift*1000:.2f}mm-equivalent joint norm")
    print()
    print(
        "HANDOFF VERDICT: the Home skill ran against a RealRobotBackend "
        "subclass with no changes. Observation+step skills transfer "
        "for free; see the tutorial for the motion-planning caveat."
    )
    backend.close()
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
