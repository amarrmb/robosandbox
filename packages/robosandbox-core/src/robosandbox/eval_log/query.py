"""DuckDB-backed query layer over runs/eval_log/*.jsonl.

Opens a fresh in-memory DuckDB on each EvalLogQuery instance, registers
view aliases for the four JSONL files (gracefully handling missing files),
and exposes a small set of canned queries plus a free-form ``run_sql``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb

from robosandbox.eval_log.store import DEMOS, EVAL_RUNS, EVALS, POLICIES


class EvalLogQuery:
    def __init__(self, root: str | Path):
        self._root = Path(root)
        self._con = duckdb.connect(":memory:")
        self._register_views()

    def _register_views(self) -> None:
        for name, fname in [
            ("policies", POLICIES),
            ("demos", DEMOS),
            ("evals", EVALS),
            ("eval_runs", EVAL_RUNS),
        ]:
            path = self._root / fname
            if path.exists() and path.stat().st_size > 0:
                self._con.execute(
                    f"CREATE OR REPLACE VIEW {name} AS "
                    f"SELECT * FROM read_json_auto('{path}', union_by_name=true)"
                )
            else:
                self._con.execute(
                    f"CREATE OR REPLACE VIEW {name} AS SELECT NULL WHERE 1=0"
                )

    def list_policies(self) -> list[dict[str, Any]]:
        try:
            return self._con.execute("""
                SELECT policy_id, kind, parent_policy_id, trained_at,
                       (SELECT COUNT(DISTINCT eval_id) FROM evals e WHERE e.policy_id = p.policy_id) AS n_evals
                FROM policies p
                ORDER BY trained_at DESC
            """).df().to_dict("records")
        except duckdb.Error:
            return []

    def slice_breakdown(self, policy_id: str, axes: list[str]) -> list[dict[str, Any]]:
        if not axes:
            return []
        select_cols = ", ".join(
            f"slice_axes->>'$.{a}' AS {a}" for a in axes
        )
        group_cols = ", ".join(axes)
        try:
            return self._con.execute(f"""
                SELECT {select_cols},
                       COUNT(*) AS n_trials,
                       AVG(CAST(outcome.success AS INTEGER))::DOUBLE AS success_rate
                FROM evals
                WHERE policy_id = ?
                GROUP BY {group_cols}
                ORDER BY {group_cols}
            """, [policy_id]).df().to_dict("records")
        except duckdb.Error:
            return []

    def compare_policies(self, policy_a: str, policy_b: str,
                         task_id: str | None = None) -> dict[str, Any]:
        where_task = "AND task_id = ?" if task_id else ""
        params = [policy_a]
        if task_id:
            params.append(task_id)
        try:
            row_a = self._con.execute(f"""
                SELECT AVG(CAST(outcome.success AS INTEGER))::DOUBLE AS rate
                FROM evals WHERE policy_id = ? {where_task}
            """, params).fetchone()
            params = [policy_b]
            if task_id:
                params.append(task_id)
            row_b = self._con.execute(f"""
                SELECT AVG(CAST(outcome.success AS INTEGER))::DOUBLE AS rate
                FROM evals WHERE policy_id = ? {where_task}
            """, params).fetchone()
        except duckdb.Error:
            return {"a_success_rate": None, "b_success_rate": None, "delta_pp": None}
        a = float(row_a[0]) if row_a and row_a[0] is not None else 0.0
        b = float(row_b[0]) if row_b and row_b[0] is not None else 0.0
        return {
            "policy_a": policy_a,
            "policy_b": policy_b,
            "a_success_rate": a,
            "b_success_rate": b,
            "delta_pp": (b - a) * 100.0,
        }

    def regression_check(self, policy_id: str) -> dict[str, Any]:
        from robosandbox.eval_log.lineage import walk_lineage
        chain = walk_lineage(self._root, policy_id)
        if len(chain) < 2:
            return {"parent_policy_id": None, "slices": []}
        parent_id = chain[1].policy_id
        try:
            df = self._con.execute("""
                WITH parent AS (
                    SELECT task_id, AVG(CAST(outcome.success AS INTEGER))::DOUBLE AS r
                    FROM evals WHERE policy_id = ? GROUP BY task_id
                ),
                child AS (
                    SELECT task_id, AVG(CAST(outcome.success AS INTEGER))::DOUBLE AS r
                    FROM evals WHERE policy_id = ? GROUP BY task_id
                )
                SELECT COALESCE(p.task_id, c.task_id) AS task_id,
                       p.r AS parent_rate, c.r AS child_rate
                FROM parent p FULL OUTER JOIN child c USING (task_id)
            """, [parent_id, policy_id]).df()
        except duckdb.Error:
            return {"parent_policy_id": parent_id, "slices": []}
        slices = []
        for row in df.to_dict("records"):
            pr = row["parent_rate"] or 0.0
            cr = row["child_rate"] or 0.0
            slices.append({
                "task_id": row["task_id"],
                "parent_rate": pr,
                "child_rate": cr,
                "delta_pp": (cr - pr) * 100.0,
                "regressed": cr < pr,
            })
        return {"parent_policy_id": parent_id, "slices": slices}

    def repro(self, eval_id: str) -> dict[str, Any] | None:
        try:
            row = self._con.execute(
                "SELECT * FROM eval_runs WHERE eval_id = ?", [eval_id]
            ).fetchone()
            cols = [d[0] for d in self._con.description]
        except duckdb.Error:
            return None
        if row is None:
            return None
        return dict(zip(cols, row))

    def run_sql(self, sql: str) -> list[dict[str, Any]]:
        try:
            return self._con.execute(sql).df().to_dict("records")
        except duckdb.Error as e:
            raise RuntimeError(f"SQL error: {e}") from e
