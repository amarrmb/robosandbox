"""Stack skill: pick N objects and place them on a base, in order.

Reduces to repeated (Pick, PlaceOn) pairs — each source is placed on
the current top of the stack. Exists as its own skill so planners can
emit a single ``stack`` call rather than interleaving many pick + place
primitives (which the agent loop would otherwise replan on error).

The skill fails fast on the first sub-step failure — upstream replan
handles it. Partial success (e.g. first object placed, second failed)
is not retried; the caller should decide whether to rerun the whole
skill or dispatch individual Pick/PlaceOn calls.
"""

from __future__ import annotations

from robosandbox.agent.context import AgentContext
from robosandbox.skills.pick import Pick
from robosandbox.skills.place import PlaceOn
from robosandbox.types import SkillResult


class Stack:
    name = "stack"
    description = (
        "Stack one or more objects on top of a base, in order. Each source "
        "is picked up and placed on top of the previous one."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "sources": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Ordered list of source object names.",
            },
            "base": {
                "type": "string",
                "description": "Base object id that the first source is placed on.",
            },
        },
        "required": ["sources", "base"],
    }

    def __init__(self) -> None:
        self._pick = Pick()
        self._place = PlaceOn()

    def __call__(
        self, ctx: AgentContext, *, sources: list[str], base: str
    ) -> SkillResult:
        if not sources:
            return SkillResult(
                success=False,
                reason="empty_sources",
                reason_detail="Stack requires at least one source object",
            )
        placed: list[str] = []
        top = base
        for src in sources:
            pick_res = self._pick(ctx, object=src)
            if not pick_res.success:
                return SkillResult(
                    success=False,
                    reason="pick_failed",
                    reason_detail=f"pick({src}): {pick_res.reason}/{pick_res.reason_detail}",
                    artifacts={"placed": placed, "failed_at": src},
                )
            place_res = self._place(ctx, target=top)
            if not place_res.success:
                return SkillResult(
                    success=False,
                    reason="place_failed",
                    reason_detail=f"place_on({top}): {place_res.reason}/{place_res.reason_detail}",
                    artifacts={"placed": placed, "failed_at": src},
                )
            placed.append(src)
            top = src

        return SkillResult(
            success=True,
            reason="stacked",
            artifacts={"placed": placed, "base": base},
        )
