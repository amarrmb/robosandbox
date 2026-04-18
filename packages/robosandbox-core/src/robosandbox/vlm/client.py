"""Thin wrapper around the OpenAI SDK.

Works with any OpenAI-compatible endpoint — OpenAI, together.ai, vLLM,
ollama's compat layer — by switching base_url. Images are passed as
data URLs so we don't depend on any upload path.
"""

from __future__ import annotations

import base64
import io
import os
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import openai
    from openai import OpenAI
except ImportError as e:
    raise ImportError(
        "robosandbox requires openai>=1.40 — `pip install openai`"
    ) from e


class VLMTransportError(RuntimeError):
    """Network / auth / rate-limit errors that are external to model outputs."""


@dataclass
class VLMConfig:
    model: str = "gpt-4o-mini"
    base_url: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.0
    timeout_seconds: float = 60.0
    max_retries: int = 2


def rgb_to_data_url(rgb: np.ndarray) -> str:
    """Encode an (H, W, 3) uint8 RGB array as a PNG data URL."""
    try:
        from PIL import Image
    except ImportError as e:
        raise ImportError("`pip install pillow` required for VLM image inputs") from e
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


class OpenAIVLMClient:
    """OpenAI-compatible chat client with tool-calling + image support."""

    def __init__(self, cfg: VLMConfig | None = None) -> None:
        self.cfg = cfg or VLMConfig()
        api_key = os.environ.get(self.cfg.api_key_env)
        if not api_key:
            raise VLMTransportError(
                f"missing API key: set {self.cfg.api_key_env} "
                f"(or change vlm.api_key_env in your config)"
            )
        self._client = OpenAI(
            api_key=api_key,
            base_url=self.cfg.base_url,
            timeout=self.cfg.timeout_seconds,
            max_retries=0,  # we do our own retry loop so transient vs hard fail is visible
        )

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict | None = None,
        **extra: Any,
    ) -> dict[str, Any]:
        """Returns a dict with keys: 'content' (str|None), 'tool_calls' (list), 'raw'."""
        last_exc: Exception | None = None
        for attempt in range(self.cfg.max_retries + 1):
            try:
                resp = self._client.chat.completions.create(
                    model=self.cfg.model,
                    messages=messages,
                    tools=tools,
                    tool_choice=tool_choice,
                    temperature=self.cfg.temperature,
                    **extra,
                )
                choice = resp.choices[0]
                tc_list = []
                for tc in choice.message.tool_calls or []:
                    tc_list.append(
                        {
                            "id": tc.id,
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        }
                    )
                return {
                    "content": choice.message.content,
                    "tool_calls": tc_list,
                    "finish_reason": choice.finish_reason,
                    "raw": resp,
                }
            except (openai.APIConnectionError, openai.APITimeoutError) as e:
                last_exc = e
                if attempt < self.cfg.max_retries:
                    continue
                raise VLMTransportError(f"transport error: {e}") from e
            except openai.RateLimitError as e:
                # Surface immediately; retrying makes things worse.
                raise VLMTransportError(f"rate-limited: {e}") from e
            except openai.AuthenticationError as e:
                raise VLMTransportError(f"auth failed: {e}") from e
        raise VLMTransportError(f"VLM call failed: {last_exc}")
