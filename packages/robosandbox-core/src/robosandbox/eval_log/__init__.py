"""Append-only event log + DuckDB queries for slice/lineage-aware eval."""
from robosandbox.eval_log.lineage import LineageError, walk_lineage
from robosandbox.eval_log.migrate import migrate_legacy_runs
from robosandbox.eval_log.ood import (
    MIN_SAMPLES, OOD_THRESHOLD, fit_training_dist_summary, score_trial,
)
from robosandbox.eval_log.query import EvalLogQuery
from robosandbox.eval_log.schema import (
    SCHEMA_VERSION, DemoRow, EvalRow, EvalRunRow, PolicyRow,
)
from robosandbox.eval_log.slicing import bucket_label, quantile_buckets
from robosandbox.eval_log.store import EvalLogStore

__all__ = [
    "DemoRow", "EvalLogQuery", "EvalLogStore", "EvalRow", "EvalRunRow",
    "LineageError", "MIN_SAMPLES", "OOD_THRESHOLD", "PolicyRow", "SCHEMA_VERSION",
    "bucket_label", "fit_training_dist_summary", "migrate_legacy_runs",
    "quantile_buckets", "score_trial", "walk_lineage",
]
