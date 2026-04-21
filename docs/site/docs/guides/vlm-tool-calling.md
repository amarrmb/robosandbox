# VLM tool-calling

!!! info "What this is — and what it isn't"
    **The browser viewer uses a regex-based planner (StubPlanner) by default. No API key, no VLM, no network call.** It handles everyday phrasings like "pick up the red cube" without any model.

    A VLM (Vision-Language Model) becomes relevant when you want **richer natural-language understanding** — sentences the regex grammar doesn't cover, multi-step descriptions, or reasoning from the camera image. You enable it via the CLI:

    ```bash
    export OPENAI_API_KEY=sk-...
    uv run robo-sandbox run "stack the red cube on the green cube" --vlm-provider openai
    ```

    The VLM sees the task string + a live camera frame, then emits tool calls that drive the same skills the viewer uses. The agent loop is identical — only the planner changes.

This page shows the exact path from a skill definition in Python to a
`SkillCall` coming back from a VLM.

![walkthrough](../assets/demos/vlm_walkthrough.gif){ loading=lazy }

The walkthrough script
[`examples/vlm_tool_calling_walkthrough.py`][walkthrough] prints every
step with a mock VLM client, so you can run it without an API key.

[walkthrough]: https://github.com/amarrmb/robosandbox/blob/main/examples/vlm_tool_calling_walkthrough.py

## The pattern

```
┌──────────┐       schemas           ┌─────────────────┐
│  Skills  │ ─────────────────────► │ Tool definitions│
└──────────┘                          └────────┬────────┘
                                                │ sent with task + image
                                                ▼
                                         ┌──────────────┐
                                         │  VLM (OpenAI │
                                         │  Ollama, ...)│
                                         └──────┬───────┘
                                                │ tool_calls
                                                ▼
                                         ┌──────────────┐
                                         │  SkillCall[] │ → Agent loop executes
                                         └──────────────┘
```

`VLMPlanner` is short enough to read in one sitting:
[`agent/planner.py`](https://github.com/amarrmb/robosandbox/blob/main/packages/robosandbox-core/src/robosandbox/agent/planner.py).
What follows is the whole flow.

## 1. Skills become tool definitions

Every skill exposes `name`, `description`, `parameters_schema` (JSON
schema). `VLMPlanner._tool_schemas()` wraps them in OpenAI's function
format:

```json
{
  "type": "function",
  "function": {
    "name": "pick",
    "description": "Pick up an object from the table. Use when the agent needs to grasp and lift a specific object by name.",
    "parameters": {
      "type": "object",
      "properties": {
        "object": {
          "type": "string",
          "description": "Natural-language name of the object to pick, e.g. 'blue cube'."
        }
      },
      "required": ["object"]
    }
  }
}
```

A synthetic `done` tool is added so the model can say "nothing to do"
without falling back to prose.

The skill author only writes `parameters_schema` as ordinary JSON
schema. The planner wraps it in the provider-specific tool format.

## 2. Task + observation become a VLM request

`VLMPlanner._build_messages()` assembles a 2-message chat:

```json
[
  {"role": "system", "content": "You are a robotic manipulation agent..."},
  {"role": "user", "content": [
    {"type": "text", "text": "Task: pick up the red cube and put it on the green cube"},
    {"type": "text", "text": "Known objects in the scene (use their natural names):\n{\"red_cube\": {\"xyz\": [0.4, 0.0, 0.06]}, ...}"},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
  ]}
]
```

The user message carries three useful pieces of context:

- The **task string** verbatim.
- A **scene summary** from `obs.scene_objects` — gives the model
  exact object ids to use instead of guessing pronouns.
- The **current RGB frame** as a data URL — VLMs see what the arm sees.

On a replan, one more text block is appended:

```json
{"type": "text", "text": "Previously-failed steps — do NOT repeat them unchanged:\n[{\"step_idx\": 1, \"skill\": \"pick\", \"args\": {...}, \"reason\": \"object_not_found\"}]"}
```

That is the ReAct feedback loop in practice: failed steps become
context for the next plan.

## 3. Model response becomes `SkillCall`s

The VLM returns OpenAI-shaped tool calls:

```json
{
  "content": null,
  "tool_calls": [
    {"id": "call_abc123", "name": "pick",
     "arguments": "{\"object\": \"red_cube\"}"},
    {"id": "call_def456", "name": "place_on",
     "arguments": "{\"target\": \"green_cube\"}"}
  ],
  "finish_reason": "tool_calls"
}
```

`_parse_tool_calls()` turns each response into a
`SkillCall(name, arguments, tool_call_id)`. Two details matter:

1. **`arguments` is parsed as JSON.** If it's malformed, the
   SkillCall falls back to `{}` rather than crashing the agent.
2. **If the model emits the synthetic `done` tool first**, parsing
   stops and the planner returns an empty plan — the agent treats
   that as "already done."

## 4. Recovery when the model emits prose

Smaller or local models sometimes ignore the tool interface and answer
in prose. `VLMPlanner.plan()` detects that and retries once with a
nudge:

```json
{"role": "user", "content": "Please respond with tool calls only — no prose."}
```

If the second try also returns prose, the planner gives up and returns
an empty plan.

`n_vlm_calls` tells you whether that extra retry happened, which is
useful when you are comparing models.

## Running against real providers

| Provider | Setup | Command |
|---|---|---|
| **OpenAI** | `export OPENAI_API_KEY=sk-...` | `uv run robo-sandbox run "pick up the red cube" --vlm-provider openai` |
| **Ollama** | `ollama serve &` + `ollama pull llama3.2-vision` | `uv run robo-sandbox run "pick up the red cube" --vlm-provider ollama` |
| **Any OpenAI-compatible** | vLLM, together, groq, etc. | `uv run robo-sandbox run "..." --vlm-provider custom --base-url https://...` |
| **No VLM (regex)** | nothing | `uv run robo-sandbox run "pick up the red cube"` — defaults to `stub` |

All four paths use the same `Planner` interface. Swapping providers is
just a flag change.

## Zero-setup walkthrough

Run the walkthrough yourself — no API keys, no network:

```bash
uv run python examples/vlm_tool_calling_walkthrough.py
```

The output shows the tool definitions, the request payload, the canned
tool-call response, and the parsed `SkillCall`s.

This is the easiest way to understand the pattern before plugging in a
real model.

## Cost & latency notes

- **Per episode**: ~1–2 VLM calls for successful plans, +1 per
  replan. `gpt-4o-mini` ≈ $0.002/episode, `gpt-4o` ≈ $0.02.
- **Latency**: ~1.5 s for `gpt-4o-mini` first token, ~0.5 s on Ollama
  with a warm model. This lives inside the agent loop — between
  `PLAN` and `EXECUTE` log lines.
- **Rate limits**: `OpenAIVLMClient` fails fast on `RateLimitError`
  rather than silently retrying — surfaces the issue instead of
  hiding it in long waits.

## What's next

- [How it works in 3 minutes](./how-it-works.md) — the four-layer loop this plugs into.
- [Add a skill](./add-a-skill.md) — once you add one, `VLMPlanner` auto-exposes it as a new tool.
