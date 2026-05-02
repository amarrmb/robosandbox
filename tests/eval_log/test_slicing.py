import numpy as np
from robosandbox.eval_log.slicing import quantile_buckets, bucket_label


def test_terciles_default():
    values = np.linspace(0.0, 1.0, 99)
    edges = quantile_buckets(values, n_buckets=3)
    assert len(edges) == 4  # n_buckets + 1


def test_label_within_bucket():
    edges = quantile_buckets(np.linspace(0.0, 1.0, 99), n_buckets=3)
    assert bucket_label(0.1, edges) == "0"
    assert bucket_label(0.5, edges) == "1"
    assert bucket_label(0.9, edges) == "2"


def test_all_same_value():
    """If all values identical, only one bucket exists; everything labels '0'."""
    edges = quantile_buckets(np.full(50, 0.42), n_buckets=3)
    assert bucket_label(0.42, edges) == "0"
    assert bucket_label(0.99, edges) == "0"


def test_empty_input():
    edges = quantile_buckets(np.array([]), n_buckets=3)
    assert edges == []
    assert bucket_label(0.5, edges) == "unknown"
