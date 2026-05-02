"""Append-only event-log row schemas. Every row carries SCHEMA_VERSION."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PolicyRow:
    policy_id: str
    kind: str  # 'ppo_neural' | 'replay_trajectory' | 'lerobot_act' | 'lerobot_diffusion' | 'foundation_fine_tune'
    parent_policy_id: str | None
    lineage_op: str | None
    training_demo_set_id: str | None
    training_steps: int | None
    training_dist_summary: dict | None  # {"mean": [...], "cov": [[...]], "n_samples": int} or None
    trained_at: str  # ISO8601
    checkpoint_path: str
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        out = asdict(self)
        out["schema_version"] = SCHEMA_VERSION
        return out


@dataclass(frozen=True)
class DemoRow:
    demo_id: str
    demo_set_id: str
    task_id: str
    task_variant_hash: str
    operator: str
    recorded_at: str
    sim_backend: str
    randomize_seed: int | None
    duration_steps: int
    outcome_label: str  # 'success' | 'failure' | 'partial'
    demo_path: str

    def to_dict(self) -> dict:
        out = asdict(self)
        out["schema_version"] = SCHEMA_VERSION
        return out


@dataclass(frozen=True)
class EvalRow:
    eval_id: str
    trial_id: str
    policy_id: str
    task_id: str
    task_variant_hash: str
    sim_backend: str
    trial_seed: int
    slice_axes: dict
    outcome: dict
    wall_seconds: float
    compute_resource: str
    started_at: str

    def to_dict(self) -> dict:
        out = asdict(self)
        out["schema_version"] = SCHEMA_VERSION
        return out


@dataclass(frozen=True)
class EvalRunRow:
    eval_id: str
    policy_id: str
    task_id: str
    n_trials: int
    started_at: str
    completed_at: str
    git_sha: str
    config_hash: str
    compute_resource: str
    command_line: str

    def to_dict(self) -> dict:
        out = asdict(self)
        out["schema_version"] = SCHEMA_VERSION
        return out
