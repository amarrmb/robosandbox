"""Mahalanobis-based OOD scoring for task conditions.

Pluggable: ``score_trial`` uses ``training_dist_summary`` (a JSON dict
produced by ``fit_training_dist_summary``). Future KDE / semantic OOD
implementations can swap by writing summaries with a different ``method``
field; ``score_trial`` dispatches accordingly. v1 only implements
``method='mahalanobis'``.
"""
from __future__ import annotations

import numpy as np

MIN_SAMPLES = 30
OOD_THRESHOLD = 0.5  # score > this → is_ood = True
PERCENTILE_CAP = 99  # for normalization


def fit_training_dist_summary(samples: np.ndarray) -> dict:
    """Fit a (mean, cov, percentile) summary over (n_samples, n_features) samples.

    Returns a JSON-serializable dict suitable for storage in policies.jsonl.
    Returns ``method='insufficient'`` when ``len(samples) < MIN_SAMPLES``.
    """
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim == 1:
        samples = samples.reshape(-1, 1)
    n_samples = int(samples.shape[0])
    if n_samples < MIN_SAMPLES:
        return {"method": "insufficient", "n_samples": n_samples}
    mean = samples.mean(axis=0)
    centered = samples - mean
    cov = (centered.T @ centered) / max(n_samples - 1, 1)
    try:
        cov_inv = np.linalg.pinv(cov + 1e-9 * np.eye(cov.shape[0]))
        d2 = np.einsum("ij,jk,ik->i", centered, cov_inv, centered)
        d = np.sqrt(np.maximum(d2, 0.0))
        cap = float(np.percentile(d, PERCENTILE_CAP))
    except np.linalg.LinAlgError:
        return {"method": "degenerate", "n_samples": n_samples}
    if cap < 1e-9:
        return {"method": "degenerate", "n_samples": n_samples}
    return {
        "method": "mahalanobis",
        "n_samples": n_samples,
        "mean": mean.tolist(),
        "cov": cov.tolist(),
        "cap_dist": cap,
    }


def score_trial(summary: dict | None, trial_features: np.ndarray) -> tuple[float | None, bool | None]:
    """Score one trial → (ood_score in [0,1] or None, is_ood bool or None)."""
    if summary is None:
        return None, None
    method = summary.get("method")
    if method != "mahalanobis":
        return None, None
    mean = np.asarray(summary["mean"], dtype=np.float64)
    cov = np.asarray(summary["cov"], dtype=np.float64)
    cap = float(summary["cap_dist"])
    x = np.asarray(trial_features, dtype=np.float64)
    if x.shape != mean.shape:
        return None, None
    try:
        cov_inv = np.linalg.pinv(cov + 1e-9 * np.eye(cov.shape[0]))
    except np.linalg.LinAlgError:
        return None, None
    diff = x - mean
    d = float(np.sqrt(max(diff @ cov_inv @ diff, 0.0)))
    score = min(d / cap, 1.0) if cap > 0 else 0.0
    return float(score), bool(score > OOD_THRESHOLD)
