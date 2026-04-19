"""Probe a list of public SO-100 ACT checkpoints for config schema.

Grounds the claim in
``docs/site/docs/tutorials/lerobot-policy-replay.md`` that the SO-100
ACT checkpoints we've tried ship a pre-``lerobot 0.5`` config shape.
Anyone can re-run this and get back a table like::

    cadene/act_so100_5_lego_test_080000        legacy (no type/input_features)
    satvikahuja/act_so100_test                  legacy (no type/input_features)
    koenvanwijk/act_so100_test                  legacy (no type/input_features)
    Chojins/so100_test20                        legacy (no type/input_features)
    pingev/lerobot-so100-1                      legacy (no type/input_features)
    maximilienroberti/act_so100_lego_red_box    legacy (no type/input_features)

If new public checkpoints ship in the current schema, add them to the
``_CHECKPOINTS`` list below and rerun. Output is the only source of
truth behind the "all six we probed" line in the tutorial; no
off-repo manual investigation required.

Requires network + the HF Hub client::

    uv pip install huggingface_hub
    uv run python examples/so_arm100/probe_hub_schemas.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


# The specific checkpoints the tutorial's "all six we probed" line
# references. Extend this list when you want to re-ground the claim
# against a new population.
_CHECKPOINTS: tuple[str, ...] = (
    "cadene/act_so100_5_lego_test_080000",
    "satvikahuja/act_so100_test",
    "koenvanwijk/act_so100_test",
    "Chojins/so100_test20",
    "pingev/lerobot-so100-1",
    "maximilienroberti/act_so100_lego_red_box",
)

_LEGACY_KEYS = ("input_shapes", "input_normalization_modes")
_CURRENT_KEYS = ("input_features", "type")


def probe(repo_id: str) -> tuple[str, str]:
    """Download just ``config.json`` for a repo and classify its schema.

    Returns ``(verdict, detail)``. Verdicts:
      - ``"legacy"``      — pre-lerobot-0.5 schema (needs migration)
      - ``"current"``     — lerobot-0.5+ schema (loads directly)
      - ``"mixed"``       — both sets of keys present
      - ``"unknown"``     — neither pattern matched
      - ``"fetch-error"`` — could not retrieve the config file
    """
    from huggingface_hub import hf_hub_download

    try:
        p = Path(hf_hub_download(repo_id, "config.json"))
        cfg = json.loads(p.read_text())
    except Exception as e:
        return "fetch-error", f"{type(e).__name__}: {e}"

    has_legacy = any(k in cfg for k in _LEGACY_KEYS)
    has_current = all(k in cfg for k in _CURRENT_KEYS)
    if has_legacy and has_current:
        return "mixed", "both schemas present"
    if has_legacy:
        missing = [k for k in _CURRENT_KEYS if k not in cfg]
        return "legacy", f"no {', '.join(missing)}"
    if has_current:
        return "current", "ships current schema — loads via from_pretrained"
    return "unknown", f"keys present: {sorted(cfg.keys())[:6]}..."


def main() -> int:
    max_name = max(len(r) for r in _CHECKPOINTS)
    rows = []
    for repo in _CHECKPOINTS:
        verdict, detail = probe(repo)
        rows.append((repo, verdict, detail))
        print(f"{repo:<{max_name}}  {verdict:<12s}  {detail}")
    print()
    counts: dict[str, int] = {}
    for _, v, _ in rows:
        counts[v] = counts.get(v, 0) + 1
    summary = "  ".join(f"{v}={n}" for v, n in sorted(counts.items()))
    print(f"summary: {summary}")
    # Exit non-zero if every probed checkpoint errored — catches network
    # outages in CI without blocking on transient single-repo failures.
    if counts.get("fetch-error", 0) == len(rows):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
