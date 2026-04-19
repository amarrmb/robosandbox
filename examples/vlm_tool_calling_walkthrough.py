"""Walk through how RoboSandbox turns skills into OpenAI tool calls.

Runs with zero external services: a MockVLMClient returns a canned
tool-call response in the exact OpenAI chat-completions format. What
you see printed is what would go over the wire to a real model, and
what would come back.

Usage:
    uv run python examples/vlm_tool_calling_walkthrough.py
"""
from __future__ import annotations

import json
import sys
from typing import Any

from robosandbox.agent.planner import VLMPlanner
from robosandbox.skills.home import Home
from robosandbox.skills.pick import Pick
from robosandbox.skills.place import PlaceOn


class MockVLMClient:
    """Returns a canned OpenAI-shaped tool_calls response, and shows what it saw."""

    def __init__(self) -> None:
        self.last_messages: list[dict[str, Any]] | None = None
        self.last_tools: list[dict[str, Any]] | None = None

    def chat(self, messages, tools=None, **_) -> dict[str, Any]:
        self.last_messages = messages
        self.last_tools = tools
        # Canned response: pick then place.
        return {
            "content": None,
            "tool_calls": [
                {
                    "id": "call_abc123",
                    "name": "pick",
                    "arguments": json.dumps({"object": "red_cube"}),
                },
                {
                    "id": "call_def456",
                    "name": "place_on",
                    "arguments": json.dumps({"target": "green_cube"}),
                },
            ],
            "finish_reason": "tool_calls",
            "raw": None,
        }


def banner(title: str) -> None:
    print("\n" + "─" * 70)
    print(f"  {title}")
    print("─" * 70)


def main() -> int:
    skills = [Pick(), PlaceOn(), Home()]
    mock = MockVLMClient()
    planner = VLMPlanner(mock, skills=skills)  # type: ignore[arg-type]

    banner("1. SKILLS → OPENAI TOOL DEFINITIONS")
    print(
        "Each skill exposes `name`, `description`, `parameters_schema`.\n"
        "VLMPlanner wraps them as OpenAI-format function tools:"
    )
    tools_preview = planner._tool_schemas()  # type: ignore[attr-defined]
    # Show just the first two (done + pick) so the output stays terminal-friendly.
    print(json.dumps(tools_preview[:2], indent=2))
    print(f"...plus {len(tools_preview) - 2} more tools (place_on, home).")

    banner("2. TASK + OBSERVATION → VLM REQUEST")
    import numpy as np

    from robosandbox.types import Pose

    class ObsStub:
        rgb = np.zeros((8, 8, 3), dtype=np.uint8)  # tiny dummy image
        depth = None
        robot_joints = np.zeros(7)
        ee_pose = Pose(xyz=(0.4, 0.0, 0.3))
        gripper_width = 0.08
        scene_objects = {
            "red_cube": Pose(xyz=(0.40, 0.00, 0.06)),
            "green_cube": Pose(xyz=(0.50, 0.10, 0.06)),
        }

    obs = ObsStub()
    plan, n_calls = planner.plan("pick up the red cube and put it on the green cube", obs, [])

    print("The planner built these chat messages (image data-url truncated):")
    msgs_for_print = []
    for m in mock.last_messages or []:
        role = m["role"]
        content = m["content"]
        if isinstance(content, list):
            pretty = []
            for block in content:
                if block.get("type") == "image_url":
                    pretty.append({"type": "image_url", "image_url": {"url": "data:image/png;base64,...truncated..."}})
                else:
                    pretty.append(block)
            msgs_for_print.append({"role": role, "content": pretty})
        else:
            # System message — show truncated.
            snippet = content if len(content) < 160 else content[:157] + "..."
            msgs_for_print.append({"role": role, "content": snippet})
    print(json.dumps(msgs_for_print, indent=2))

    banner("3. MODEL RESPONSE (canned, matches OpenAI shape)")
    print(json.dumps({
        "content": None,
        "tool_calls": [
            {"id": "call_abc123", "name": "pick",
             "arguments": '{"object": "red_cube"}'},
            {"id": "call_def456", "name": "place_on",
             "arguments": '{"target": "green_cube"}'},
        ],
        "finish_reason": "tool_calls",
    }, indent=2))

    banner("4. PARSED SkillCall's — what the Agent executes")
    for sc in plan:
        print(f"  SkillCall(name={sc.name!r:16s} arguments={sc.arguments}  tool_call_id={sc.tool_call_id!r})")

    print(f"\nTotal VLM calls made: {n_calls}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
