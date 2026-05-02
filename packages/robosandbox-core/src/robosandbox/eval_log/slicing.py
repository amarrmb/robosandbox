"""Quantile bucketization for continuous slice values (query-time only)."""
from __future__ import annotations

import numpy as np


def quantile_buckets(values, n_buckets: int = 3) -> list[float]:
    """Return edges (length n_buckets+1) of equal-population quantile buckets.

    For all-same or empty inputs, returns ``[]`` or a degenerate one-bucket
    list; ``bucket_label`` then returns ``"unknown"`` or ``"0"`` accordingly.
    """
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return []
    if np.all(arr == arr[0]):
        return [float(arr[0]), float(arr[0])]
    qs = np.linspace(0.0, 1.0, n_buckets + 1)
    edges = np.quantile(arr, qs).tolist()
    return [float(e) for e in edges]


def bucket_label(value: float, edges: list[float]) -> str:
    """Return the bucket index as a string, or 'unknown' if edges are empty."""
    if not edges:
        return "unknown"
    if len(edges) == 2 and edges[0] == edges[1]:
        return "0"
    v = float(value)
    n = len(edges) - 1
    for i in range(n):
        lo, hi = edges[i], edges[i + 1]
        if i == n - 1:
            if lo <= v <= hi:
                return str(i)
        else:
            if lo <= v < hi:
                return str(i)
    return "0" if v < edges[0] else str(n - 1)
