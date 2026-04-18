"""ReAct-style agentic orchestrator.

States: IDLE → PLAN → EXECUTE → EVALUATE → [next | REPLAN | DONE].
Uses OpenAI-style tool-calling: each Skill's JSON schema is exposed as
a tool; the VLM returns ``tool_calls`` and we dispatch them in order.
"""

from __future__ import annotations

import json
import logging
import textwrap
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from robosandbox.agent.context import AgentContext
from robosandbox.protocols import Skill
from robosandbox.types import Observation, SkillResult
from robosandbox.vlm.client import OpenAIVLMClient, VLMTransportError, rgb_to_data_url

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
class SkillCall:
    name: str
    arguments: dict[str, Any]
    tool_call_id: str | None = None


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


_SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a robotic manipulation agent. Given a natural-language task
    and a current image of the workspace, decompose the task into a
    sequence of skill calls from the available tools. Call each tool
    via the tool-calling interface; do not narrate.

    Rules:
    - One tool call per logical step. Do not chain skills in a single call.
    - Use object names exactly as they appear in the scene; avoid pronouns.
    - If the task is already complete, respond with the `done` tool.
    - Prefer the minimum plan that achieves the goal. Do not add decorative steps.
    """
).strip()


_DONE_TOOL = {
    "type": "function",
    "function": {
        "name": "done",
        "description": "Call this when the task is already complete or no further skills are needed.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}


class Agent:
    def __init__(
        self,
        ctx: AgentContext,
        skills: list[Skill],
        vlm: OpenAIVLMClient,
        *,
        max_replans: int = 3,
        evaluator: str = "sim_state",
    ) -> None:
        self._ctx = ctx
        self._skills: dict[str, Skill] = {s.name: s for s in skills}
        if not self._skills:
            raise ValueError("Agent requires at least one skill")
        self._vlm = vlm
        self._max_replans = max_replans
        self._evaluator = evaluator

    # ---- main loop ------------------------------------------------------
    def run(self, task: str, max_steps: int = 20) -> EpisodeResult:
        """Plan once, execute in order, replan from the failing step on error."""
        ep = EpisodeResult(success=False, task=task)
        prior_attempts: list[dict[str, Any]] = []
        replans = 0

        while True:
            log.info("PLAN: task=%r replan=%d", task, replans)
            try:
                plan, calls = self._plan(task, prior_attempts)
            except VLMTransportError as e:
                ep.final_reason = "vlm_transport"
                ep.final_detail = str(e)
                return ep
            ep.vlm_calls += calls
            ep.plan.extend(plan)

            if not plan:
                ep.success = True
                ep.final_reason = "already_done"
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

            # Failure path → replan or give up.
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

    # ---- single-skill dispatch -----------------------------------------
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

    # ---- plan via VLM ---------------------------------------------------
    def _plan(
        self, task: str, prior_attempts: list[dict[str, Any]]
    ) -> tuple[list[SkillCall], int]:
        obs = self._ctx.sim.observe()
        messages = self._build_plan_messages(task, obs, prior_attempts)
        tools = self._tool_schemas()
        resp = self._vlm.chat(messages, tools=tools, tool_choice="auto")
        vlm_calls = 1
        tcs = resp.get("tool_calls") or []

        plan: list[SkillCall] = []
        for tc in tcs:
            name = tc["name"]
            if name == "done":
                break
            raw_args = tc.get("arguments", "{}") or "{}"
            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError:
                args = {}
            plan.append(SkillCall(name=name, arguments=args, tool_call_id=tc.get("id")))

        # If the VLM returned only prose / no tool calls, retry once with an
        # explicit nudge — common with weaker models.
        if not plan and not any(tc.get("name") == "done" for tc in tcs):
            nudge = {
                "role": "user",
                "content": "Please respond with tool calls only — no prose.",
            }
            resp2 = self._vlm.chat(
                messages + [nudge], tools=tools, tool_choice="auto"
            )
            vlm_calls += 1
            for tc in resp2.get("tool_calls") or []:
                if tc["name"] == "done":
                    break
                try:
                    args = json.loads(tc.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                plan.append(
                    SkillCall(name=tc["name"], arguments=args, tool_call_id=tc.get("id"))
                )

        return plan, vlm_calls

    def _build_plan_messages(
        self, task: str, obs: Observation, prior_attempts: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        scene_summary = {
            k: {"xyz": [round(c, 3) for c in v.xyz]} for k, v in obs.scene_objects.items()
        }
        user_content: list[dict[str, Any]] = [
            {"type": "text", "text": f"Task: {task}"},
            {
                "type": "text",
                "text": "Known objects in the scene (for reference; use their natural names):\n"
                + json.dumps(scene_summary, indent=2),
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": rgb_to_data_url(obs.rgb),
                    "detail": "high",
                },
            },
        ]
        if prior_attempts:
            user_content.append(
                {
                    "type": "text",
                    "text": "Previously-failed steps (for replanning — do NOT repeat them "
                    "unchanged):\n" + json.dumps(prior_attempts, indent=2),
                }
            )

        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    def _tool_schemas(self) -> list[dict[str, Any]]:
        tools = [_DONE_TOOL]
        for name, skill in self._skills.items():
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": skill.description,
                        "parameters": skill.parameters_schema,
                    },
                }
            )
        return tools
