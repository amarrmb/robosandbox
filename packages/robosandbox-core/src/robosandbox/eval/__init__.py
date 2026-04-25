"""Eval-time utilities: success-rate stats, JSON output schema, comparisons.

Kept out of ``robosandbox.policy`` so the policy module stays focused on
the runtime contract (``act`` / ``run_policy`` / ``run_eval_parallel``)
and doesn't acquire reporting concerns.
"""

from robosandbox.eval.stats import (
    proportion_z_test,
    summarise_eval,
    wilson_ci,
)

__all__ = ["proportion_z_test", "summarise_eval", "wilson_ci"]
