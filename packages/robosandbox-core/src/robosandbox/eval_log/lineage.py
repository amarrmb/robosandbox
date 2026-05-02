"""Walk parent_policy_id chain through policies.jsonl with cycle detection."""
from __future__ import annotations

import json
from pathlib import Path

from robosandbox.eval_log.schema import PolicyRow
from robosandbox.eval_log.store import POLICIES

MAX_DEPTH = 32


class LineageError(Exception):
    pass


def _load_policies_index(root: Path) -> dict[str, PolicyRow]:
    """Read policies.jsonl into a dict keyed by policy_id (last write wins)."""
    out: dict[str, PolicyRow] = {}
    path = root / POLICIES
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        d.pop("schema_version", None)
        try:
            row = PolicyRow(**{k: d.get(k) for k in PolicyRow.__dataclass_fields__})
        except TypeError:
            continue
        out[row.policy_id] = row
    return out


def walk_lineage(root: Path | str, policy_id: str) -> list[PolicyRow]:
    """Return the chain [child, parent, grandparent, ...] up to root or a missing parent.

    Raises ``LineageError`` if depth > MAX_DEPTH (cycle detection).
    Returns ``[]`` if ``policy_id`` is not in the log.
    """
    root = Path(root)
    index = _load_policies_index(root)
    if policy_id not in index:
        return []
    chain: list[PolicyRow] = []
    seen: set[str] = set()
    current_id: str | None = policy_id
    while current_id is not None:
        if current_id in seen or len(chain) >= MAX_DEPTH:
            raise LineageError(
                f"lineage cycle or excessive depth at {current_id!r} "
                f"(depth={len(chain)}, max={MAX_DEPTH})"
            )
        seen.add(current_id)
        node = index.get(current_id)
        if node is None:
            break
        chain.append(node)
        current_id = node.parent_policy_id
    return chain
