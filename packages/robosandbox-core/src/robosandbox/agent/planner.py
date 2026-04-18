"""Planners: strategies that turn a task + observation into an ordered
list of SkillCalls.

Two implementations ship in core:

- ``VLMPlanner``  — calls an OpenAI-compatible chat endpoint with
  tool-calling + image inputs. Works with OpenAI, Ollama, vLLM, together,
  anything compatible.
- ``StubPlanner`` — rule-based NLU; zero external deps. Handles
  pick / place / stack / home phrasings. Not "AI" — deterministic, fast,
  and enough to prove the agent loop without a model.

A custom planner only needs to match the ``plan`` signature.
"""

from __future__ import annotations

import json
import logging
import re
import textwrap
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from robosandbox.protocols import Skill
from robosandbox.types import Observation
from robosandbox.vlm.client import OpenAIVLMClient, rgb_to_data_url

log = logging.getLogger("robosandbox.planner")


@dataclass
class SkillCall:
    """One planned skill invocation."""

    name: str
    arguments: dict[str, Any]
    tool_call_id: str | None = None


@runtime_checkable
class Planner(Protocol):
    def plan(
        self,
        task: str,
        obs: Observation,
        prior_attempts: list[dict[str, Any]],
    ) -> tuple[list[SkillCall], int]:
        """Return (plan, n_model_calls). Empty plan == 'already done'."""


# ---------------------------------------------------------------------------
# VLMPlanner — OpenAI-compatible tool-calling
# ---------------------------------------------------------------------------

_VLM_SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a robotic manipulation agent. Given a natural-language task
    and a current image of the workspace, decompose the task into a
    sequence of skill calls from the available tools. Call each tool
    via the tool-calling interface; do not narrate.

    Rules:
    - One tool call per logical step. Do not chain skills in a single call.
    - Use object names exactly as they appear in the scene; avoid pronouns.
    - If the task is already complete, respond with the `done` tool.
    - Prefer the minimum plan that achieves the goal. No decorative steps.
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


class VLMPlanner:
    """Plan via VLM tool-calling against an OpenAI-compatible endpoint."""

    def __init__(self, vlm: OpenAIVLMClient, skills: list[Skill]) -> None:
        self._vlm = vlm
        self._skills: dict[str, Skill] = {s.name: s for s in skills}
        if not self._skills:
            raise ValueError("VLMPlanner requires at least one skill")

    def plan(
        self,
        task: str,
        obs: Observation,
        prior_attempts: list[dict[str, Any]],
    ) -> tuple[list[SkillCall], int]:
        messages = self._build_messages(task, obs, prior_attempts)
        tools = self._tool_schemas()
        resp = self._vlm.chat(messages, tools=tools, tool_choice="auto")
        calls = 1
        plan = self._parse_tool_calls(resp.get("tool_calls") or [])

        # Weaker models sometimes emit prose instead of tool calls.
        if not plan and not self._saw_done(resp):
            nudge = {
                "role": "user",
                "content": "Please respond with tool calls only — no prose.",
            }
            resp2 = self._vlm.chat(messages + [nudge], tools=tools, tool_choice="auto")
            calls += 1
            plan = self._parse_tool_calls(resp2.get("tool_calls") or [])

        return plan, calls

    @staticmethod
    def _saw_done(resp: dict[str, Any]) -> bool:
        return any((tc.get("name") == "done") for tc in (resp.get("tool_calls") or []))

    def _parse_tool_calls(self, tcs: list[dict[str, Any]]) -> list[SkillCall]:
        plan: list[SkillCall] = []
        for tc in tcs:
            if tc.get("name") == "done":
                break
            name = tc["name"]
            raw = tc.get("arguments") or "{}"
            try:
                args = json.loads(raw) if isinstance(raw, str) else raw
                if not isinstance(args, dict):
                    args = {}
            except json.JSONDecodeError:
                args = {}
            plan.append(SkillCall(name=name, arguments=args, tool_call_id=tc.get("id")))
        return plan

    def _build_messages(
        self, task: str, obs: Observation, prior_attempts: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        scene_summary = {
            k: {"xyz": [round(c, 3) for c in v.xyz]} for k, v in obs.scene_objects.items()
        }
        user_content: list[dict[str, Any]] = [
            {"type": "text", "text": f"Task: {task}"},
            {
                "type": "text",
                "text": "Known objects in the scene (use their natural names):\n"
                + json.dumps(scene_summary, indent=2),
            },
            {
                "type": "image_url",
                "image_url": {"url": rgb_to_data_url(obs.rgb), "detail": "high"},
            },
        ]
        if prior_attempts:
            user_content.append(
                {
                    "type": "text",
                    "text": "Previously-failed steps — do NOT repeat them unchanged:\n"
                    + json.dumps(prior_attempts, indent=2),
                }
            )
        return [
            {"role": "system", "content": _VLM_SYSTEM_PROMPT},
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


# ---------------------------------------------------------------------------
# StubPlanner — zero-dep rule-based NLU
# ---------------------------------------------------------------------------

_WORD = r"[A-Za-z_][A-Za-z0-9_ ]*?"

# Verbs aliased to pick.
_PICK_VERBS = r"(?:pick(?:\s+up)?|grab|get|lift|take)"
# Verbs aliased to place_on.
_PLACE_VERBS = r"(?:place|put|set|drop)"

# Permissive joiner between the two verbs: comma, "and", "then", ", then", spaces.
_JOIN = r"(?:\s*,\s*then\s+|\s*,\s*|\s+and\s+(?:then\s+)?|\s+then\s+|\s+)+"

_RE_PICK_AND_PLACE = re.compile(
    rf"{_PICK_VERBS}\s+(?:up\s+)?(?:the\s+)?({_WORD}){_JOIN}"
    rf"{_PLACE_VERBS}\s+(?:it\s+)?on(?:\s+top\s+of)?\s+(?:the\s+)?({_WORD})\b",
    re.IGNORECASE,
)
_RE_STACK = re.compile(
    rf"stack\s+(?:the\s+)?({_WORD})\s+on(?:\s+top\s+of)?\s+(?:the\s+)?({_WORD})\b",
    re.IGNORECASE,
)
_RE_PICK = re.compile(
    rf"{_PICK_VERBS}\s+(?:up\s+)?(?:the\s+)?({_WORD})\b",
    re.IGNORECASE,
)
_RE_HOME = re.compile(r"\b(?:go\s+home|home|return\s+home|neutral\s+pose)\b", re.IGNORECASE)

_RE_PUSH = re.compile(
    rf"push\s+(?:the\s+)?({_WORD})\s+(forward|back(?:ward)?|left|right|north|south|east|west)\b",
    re.IGNORECASE,
)

# "pour the mustard into the bowl" / "pour the mug over the apple"
_RE_POUR = re.compile(
    rf"pour\s+(?:the\s+)?({_WORD})\s+(?:into|over|on|onto|in)\s+(?:the\s+)?({_WORD})\b",
    re.IGNORECASE,
)

# "tap the red button" / "press the cube"
_RE_TAP = re.compile(
    rf"(?:tap|press|poke|touch)\s+(?:the\s+)?({_WORD})\b",
    re.IGNORECASE,
)


def _fuzzy_object_match(query: str, candidates: list[str]) -> str | None:
    """Map a natural-language object phrase to a scene_object id.

    Rules (first match wins):
    1. exact case-insensitive match after normalising spaces/underscores
    2. substring match either direction
    3. all query words appear in candidate (word overlap)
    """
    if not query or not candidates:
        return None
    q = query.strip().rstrip(".!?,").lower()
    q_norm = q.replace(" ", "_")
    for c in candidates:
        if c.lower() == q_norm:
            return c
    for c in candidates:
        c_lower = c.lower()
        if q_norm in c_lower or c_lower in q_norm:
            return c
    # word-set overlap (e.g. "the red box" vs "red_cube" -> "red" overlaps)
    q_words = {w for w in re.split(r"[\s_]+", q) if w and w not in {"the", "a", "an"}}
    best: tuple[int, str] | None = None
    for c in candidates:
        c_words = {w for w in re.split(r"[\s_]+", c.lower()) if w}
        overlap = len(q_words & c_words)
        if overlap and (best is None or overlap > best[0]):
            best = (overlap, c)
    return best[1] if best else None


class StubPlanner:
    """Rule-based planner that handles a small useful grammar:

        pick (up) the <obj>
        pick (up) the <obj> (and|,|then) (put|place) (it) on (the) <obj2>
        stack <obj1> on [top of] [the] <obj2>
        (go) home

    Returns an empty plan for anything else (agent treats that as
    "already done", which surfaces as a no-op rather than a crash).
    """

    def __init__(self, skills: list[Skill]) -> None:
        self._available = {s.name for s in skills}

    def plan(
        self,
        task: str,
        obs: Observation,
        prior_attempts: list[dict[str, Any]],
    ) -> tuple[list[SkillCall], int]:
        objs = list(obs.scene_objects.keys())
        t = task.strip()

        m = _RE_PICK_AND_PLACE.search(t) or _RE_STACK.search(t)
        if m:
            o1 = _fuzzy_object_match(m.group(1), objs)
            o2 = _fuzzy_object_match(m.group(2), objs)
            if o1 and o2 and "pick" in self._available and "place_on" in self._available:
                return (
                    [
                        SkillCall("pick", {"object": o1}),
                        SkillCall("place_on", {"target": o2}),
                    ],
                    0,
                )

        m = _RE_PUSH.search(t)
        if m and "push" in self._available:
            o1 = _fuzzy_object_match(m.group(1), objs)
            direction = m.group(2).lower()
            if o1:
                return [SkillCall("push", {"object": o1, "direction": direction})], 0

        # "pour X into Y" decomposes to: pick X, then pour into Y.
        m = _RE_POUR.search(t)
        if m and "pour" in self._available:
            o1 = _fuzzy_object_match(m.group(1), objs)
            o2 = _fuzzy_object_match(m.group(2), objs)
            if o1 and o2:
                plan: list[SkillCall] = []
                if "pick" in self._available:
                    plan.append(SkillCall("pick", {"object": o1}))
                plan.append(SkillCall("pour", {"target": o2}))
                return plan, 0

        m = _RE_TAP.search(t)
        if m and "tap" in self._available:
            o1 = _fuzzy_object_match(m.group(1), objs)
            if o1:
                return [SkillCall("tap", {"object": o1})], 0

        m = _RE_PICK.search(t)
        if m:
            o1 = _fuzzy_object_match(m.group(1), objs)
            if o1 and "pick" in self._available:
                return [SkillCall("pick", {"object": o1})], 0

        if _RE_HOME.search(t) and "home" in self._available:
            return [SkillCall("home", {})], 0

        log.info("StubPlanner found no plan for %r", task)
        return [], 0
