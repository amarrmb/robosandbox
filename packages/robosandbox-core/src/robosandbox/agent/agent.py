"""ReAct-style agentic orchestrator.

Plan once via a Planner → execute skills in order → replan from the
failing step on error, up to ``max_replans`` times.

The Agent does not know what kind of Planner it has. Swap in:
- ``VLMPlanner``  — OpenAI-compatible tool-calling (OpenAI, Ollama, ...)
- ``StubPlanner`` — rule-based NLU, zero deps

…and the rest of the loop works the same.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from robosandbox.agent.context import AgentContext
from robosandbox.agent.planner import Planner, SkillCall
from robosandbox.protocols import Skill
from robosandbox.types import SkillResult
from robosandbox.vlm.client import VLMTransportError

log = logging.getLogger("robosandbox.agent")


class AgentState(str, Enum):
    IDLE = "idle"
    PLAN = "plan"
    EXECUTE = "execute"
    EVALUATE = "evaluate"
    REPLAN = "replan"
    DONE = "done"
    FAILED = "failed"


@dataclass
class StepRecord:
    skill: str
    args: dict[str, Any]
    result: SkillResult
    wall_seconds: float


@dataclass
class EpisodeResult:
    success: bool
    task: str
    plan: list[SkillCall] = field(default_factory=list)
    steps: list[StepRecord] = field(default_factory=list)
    replans: int = 0
    vlm_calls: int = 0
    final_reason: str = ""
    final_detail: str = ""


class Agent:
    """Planner-agnostic ReAct loop."""

    def __init__(
        self,
        ctx: AgentContext,
        skills: list[Skill],
        planner: Planner,
        *,
        max_replans: int = 3,
    ) -> None:
        self._ctx = ctx
        self._skills: dict[str, Skill] = {s.name: s for s in skills}
        if not self._skills:
            raise ValueError("Agent requires at least one skill")
        self._planner = planner
        self._max_replans = max_replans

    def run(self, task: str, max_steps: int = 20) -> EpisodeResult:
        ep = EpisodeResult(success=False, task=task)
        prior_attempts: list[dict[str, Any]] = []
        replans = 0

        while True:
            log.info("PLAN: task=%r replan=%d", task, replans)
            try:
                obs = self._ctx.sim.observe()
                plan, calls = self._planner.plan(task, obs, prior_attempts)
            except VLMTransportError as e:
                ep.final_reason = "vlm_transport"
                ep.final_detail = str(e)
                return ep
            ep.vlm_calls += calls
            ep.plan.extend(plan)

            if not plan:
                ep.final_reason = "unrecognized_prompt" if replans == 0 else "no_plan"
                return ep

            failed_step: StepRecord | None = None
            for call in plan:
                if len(ep.steps) >= max_steps:
                    ep.final_reason = "max_steps"
                    return ep

                log.info("EXECUTE: %s(%s)", call.name, call.arguments)
                rec = self._execute(call)
                ep.steps.append(rec)
                if not rec.result.success:
                    failed_step = rec
                    break

            if failed_step is None:
                ep.success = True
                ep.final_reason = "plan_complete"
                return ep

            prior_attempts.append(
                {
                    "step_idx": len(ep.steps),
                    "skill": failed_step.skill,
                    "args": failed_step.args,
                    "reason": failed_step.result.reason,
                    "reason_detail": failed_step.result.reason_detail,
                }
            )
            if replans >= self._max_replans:
                ep.final_reason = "replan_exhausted"
                ep.final_detail = (
                    f"{failed_step.skill} failed: "
                    f"{failed_step.result.reason} — {failed_step.result.reason_detail}"
                )
                return ep
            replans += 1
            ep.replans = replans

    def _execute(self, call: SkillCall) -> StepRecord:
        t_step = time.time()
        try:
            skill_fn = self._skills[call.name]
        except KeyError:
            result = SkillResult(
                success=False,
                reason="unknown_skill",
                reason_detail=f"agent does not know skill {call.name!r}; "
                f"available: {list(self._skills)}",
            )
        else:
            try:
                result = skill_fn(self._ctx, **call.arguments)
            except TypeError as e:
                result = SkillResult(
                    success=False,
                    reason="bad_arguments",
                    reason_detail=f"{call.name}: {e}",
                )
        return StepRecord(call.name, call.arguments, result, time.time() - t_step)
