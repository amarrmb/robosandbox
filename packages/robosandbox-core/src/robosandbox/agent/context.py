"""Skill execution context.

The agent builds a ``AgentContext`` once per run and passes it to every
skill invocation. Skills read from it; they do not hold references to
any of these objects themselves, so swapping implementations per-skill
is a matter of constructing a new context.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from robosandbox.protocols import (
    GraspPlanner,
    MotionPlanner,
    Perception,
    RecordSink,
    SimBackend,
)


@dataclass
class AgentContext:
    sim: SimBackend
    perception: Perception
    grasp: GraspPlanner
    motion: MotionPlanner
    recorder: RecordSink | None = None
    # Optional hook called after every sim.step — useful for logging,
    # rendering frames to a file, or a live UI. Kept free-form on purpose.
    on_step: Callable[[], None] | None = None
    # Free-form config bag passed through from CLI/YAML.
    config: dict[str, Any] = field(default_factory=dict)
