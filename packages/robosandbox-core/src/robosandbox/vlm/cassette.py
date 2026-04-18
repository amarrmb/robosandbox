"""Cassette-based VLM client: record against a real endpoint, replay offline.

The goal is to exercise the *real* VLMPointer code path in CI without
network or API keys, while still allowing a human to record a fresh
cassette against a live OpenAI-compatible endpoint.

Design:

- A cassette is a plain JSON file with shape::

    {
      "_comment": "optional free-text note (e.g. hand-authored vs recorded)",
      "entries": [
        {
          "request": { ...shape-tolerant fingerprint of a chat() call... },
          "response": { "content": str|None, "tool_calls": [...], "finish_reason": str }
        },
        ...
      ]
    }

- Requests are fingerprinted on model + the *text* parts of the messages
  + tool names/descriptions. Image data URLs are normalised to the
  string "<image>" so a cassette recorded on one rendered frame still
  matches a slightly different frame at replay time (pixel noise, JPEG
  vs PNG, resolution tweaks). This is the right trade-off for unit-test
  plumbing — we're verifying that the client/pointer glue works, not
  that the VLM actually saw the same image.

- On `chat()`:
    * record mode (``OPENAI_API_KEY`` set *and*
      ``ROBOSANDBOX_RECORD_CASSETTE=<path>``): delegate to the wrapped
      real client, append the (request_fingerprint, response) to the
      cassette file, return the response.
    * replay mode (default): look up the matching entry in the cassette
      and return its stored response. Raise ``CassetteMissError`` if no
      entry matches.

Recording a new cassette
------------------------

From a shell with the real endpoint reachable::

    export OPENAI_API_KEY=sk-...                 # required
    export ROBOSANDBOX_RECORD_CASSETTE=\\
        packages/robosandbox-core/tests/cassettes/my_case.json
    pytest packages/robosandbox-core/tests/test_vlm_cassette.py -q

Each unique chat() request appends one entry. Re-running with the env
var still set will append again; delete the file first to re-record
from scratch. Without both env vars the same test runs fully offline
against the checked-in cassette.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


class CassetteMissError(RuntimeError):
    """Raised at replay time when no cassette entry matches a chat() call."""


def _normalize_content(content: Any) -> Any:
    """Strip image data URLs from message content so fingerprints are stable."""
    if isinstance(content, list):
        out = []
        for part in content:
            if isinstance(part, dict):
                ptype = part.get("type")
                if ptype == "image_url":
                    out.append({"type": "image_url", "image_url": "<image>"})
                elif ptype == "text":
                    out.append({"type": "text", "text": part.get("text", "")})
                else:
                    # Unknown part type: keep the shape, drop any large blobs.
                    out.append({"type": ptype})
            else:
                out.append(part)
        return out
    return content


def _fingerprint(
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    tool_choice: Any,
) -> dict[str, Any]:
    """Build a shape-tolerant fingerprint of a chat() request."""
    norm_messages = [
        {
            "role": m.get("role"),
            "content": _normalize_content(m.get("content")),
        }
        for m in messages
    ]
    norm_tools: list[dict[str, Any]] | None = None
    if tools:
        norm_tools = [
            {
                "name": (t.get("function") or {}).get("name") or t.get("name"),
                "description": (t.get("function") or {}).get("description"),
            }
            for t in tools
        ]
    return {
        "model": model,
        "messages": norm_messages,
        "tools": norm_tools,
        "tool_choice": tool_choice,
    }


def _hash(fp: dict[str, Any]) -> str:
    blob = json.dumps(fp, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _load_cassette(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"entries": []}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    # Be forgiving on shape.
    if isinstance(data, list):
        data = {"entries": data}
    data.setdefault("entries", [])
    return data


def _save_cassette(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=False)
        f.write("\n")


class CassetteVLMClient:
    """Wraps an OpenAI-compatible client with record/replay semantics.

    The wrapped ``real_client`` (any object with a ``chat(messages, tools,
    tool_choice, **extra)`` method and a ``cfg.model`` attribute) is only
    ever called in record mode; replay mode needs no live client, which
    is why it's optional.
    """

    # Sentinel used to make the record-mode conditional explicit.
    _RECORD_ENV = "ROBOSANDBOX_RECORD_CASSETTE"
    _API_KEY_ENV = "OPENAI_API_KEY"

    def __init__(
        self,
        cassette_path: str | os.PathLike[str],
        *,
        real_client: Any | None = None,
        model: str = "gpt-4o-mini",
    ) -> None:
        self._path = Path(cassette_path)
        self._real = real_client
        # In replay mode we still need *some* model string for the
        # fingerprint; fall back to the wrapped client's cfg if present.
        if real_client is not None and hasattr(real_client, "cfg"):
            self._model = getattr(real_client.cfg, "model", model)
        else:
            self._model = model
        self._cassette = _load_cassette(self._path)
        self._recording = self._should_record()
        if self._recording and self._real is None:
            raise RuntimeError(
                "CassetteVLMClient: recording requested "
                f"({self._RECORD_ENV} set) but no real_client provided"
            )

    def _should_record(self) -> bool:
        record_target = os.environ.get(self._RECORD_ENV)
        if not record_target:
            return False
        # Only record when the target path matches ours (lets multiple
        # cassettes coexist in one test run without cross-contamination).
        try:
            same = Path(record_target).resolve() == self._path.resolve()
        except OSError:
            same = record_target == str(self._path)
        if not same:
            return False
        return bool(os.environ.get(self._API_KEY_ENV))

    # ------------------------------------------------------------------
    # The chat() signature mirrors OpenAIVLMClient so VLMPointer can't
    # tell the difference.
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        **extra: Any,
    ) -> dict[str, Any]:
        fp = _fingerprint(self._model, messages, tools, tool_choice)
        key = _hash(fp)

        if self._recording:
            assert self._real is not None  # guarded in __init__
            resp = self._real.chat(
                messages, tools=tools, tool_choice=tool_choice, **extra
            )
            # Strip unserializable `raw` SDK object; store only the shape
            # VLMPointer consumes.
            stored = {
                "content": resp.get("content"),
                "tool_calls": resp.get("tool_calls") or [],
                "finish_reason": resp.get("finish_reason"),
            }
            self._cassette["entries"].append(
                {"key": key, "request": fp, "response": stored}
            )
            _save_cassette(self._path, self._cassette)
            # Return the original response (including `raw`) so callers
            # mid-record see exactly what the real endpoint returned.
            return resp

        # Replay. Prefer a stored `key`; fall back to rehashing the
        # entry's `request` so hand-authored cassettes (which may omit
        # a precomputed key) still match.
        for entry in self._cassette.get("entries", []):
            entry_key = entry.get("key")
            if entry_key is None and isinstance(entry.get("request"), dict):
                entry_key = _hash(entry["request"])
            if entry_key == key:
                stored = entry.get("response") or {}
                return {
                    "content": stored.get("content"),
                    "tool_calls": stored.get("tool_calls") or [],
                    "finish_reason": stored.get("finish_reason"),
                    "raw": None,
                }
        # Helpful miss message: includes a short digest of what *was*
        # searched for so a human can see why nothing matched.
        summary = {
            "model": fp["model"],
            "num_messages": len(fp["messages"]),
            "first_user_text": next(
                (
                    p.get("text")
                    for m in fp["messages"]
                    if m.get("role") == "user"
                    for p in (m.get("content") or [])
                    if isinstance(p, dict) and p.get("type") == "text"
                ),
                None,
            ),
            "key": key,
        }
        raise CassetteMissError(
            f"no cassette entry matched request {summary!r} in {self._path}"
        )
