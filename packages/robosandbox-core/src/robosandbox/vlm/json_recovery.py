"""JSON-recovery helpers for VLM outputs.

Even models explicitly asked for JSON occasionally emit fenced blocks,
trailing commas, or truncated responses. These helpers do best-effort
parsing and raise a typed exception on unrecoverable failure — callers
can decide whether to retry the VLM call or fail the skill.
"""

from __future__ import annotations

import json
import re
from typing import Any


class VLMOutputError(ValueError):
    """Raised when a VLM response cannot be parsed after all fallbacks."""


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def parse_json_loose(text: str) -> Any:
    """Try progressively more forgiving JSON parses. Raise VLMOutputError on failure."""
    if not text or not text.strip():
        raise VLMOutputError("empty VLM response")

    candidates: list[str] = [text]

    m = _FENCE_RE.search(text)
    if m:
        candidates.insert(0, m.group(1))

    stripped = text.strip()
    for opener, closer in (("{", "}"), ("[", "]")):
        if opener in stripped:
            start = stripped.find(opener)
            end = stripped.rfind(closer)
            if start >= 0 and end > start:
                candidates.append(stripped[start : end + 1])

    for c in candidates:
        try:
            return json.loads(c)
        except json.JSONDecodeError:
            continue

    # Last resort: attempt to close a truncated object / array.
    for c in candidates:
        stripped_c = c.strip().rstrip(",")
        for suffix in ("", "}", "]", "}]", "]}"):
            try:
                return json.loads(stripped_c + suffix)
            except json.JSONDecodeError:
                continue

    raise VLMOutputError(f"could not parse VLM output as JSON: {text[:200]!r}")
