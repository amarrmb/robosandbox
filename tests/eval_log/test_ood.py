import numpy as np
import pytest
from robosandbox.eval_log.ood import (
    fit_training_dist_summary, score_trial, MIN_SAMPLES,
)


def test_score_zero_at_distribution_mean():
    samples = np.random.RandomState(0).randn(200, 2) * np.array([0.05, 0.05])
    summary = fit_training_dist_summary(samples)
    score, is_ood = score_trial(summary, np.zeros(2))
    assert 0.0 <= score < 0.5
    assert is_ood is False


def test_score_higher_far_from_mean():
    samples = np.random.RandomState(0).randn(200, 2) * np.array([0.05, 0.05])
    summary = fit_training_dist_summary(samples)
    near_score, _ = score_trial(summary, np.zeros(2))
    far_score, far_ood = score_trial(summary, np.array([2.0, 2.0]))
    assert far_score > near_score
    assert far_ood is True


def test_insufficient_samples_returns_null():
    samples = np.random.RandomState(0).randn(MIN_SAMPLES - 1, 2)
    summary = fit_training_dist_summary(samples)
    score, is_ood = score_trial(summary, np.zeros(2))
    assert score is None
    assert is_ood is None


def test_degenerate_covariance_handled():
    samples = np.zeros((100, 2))
    summary = fit_training_dist_summary(samples)
    score, is_ood = score_trial(summary, np.array([0.5, 0.5]))
    assert score is None or isinstance(score, float)


def test_summary_serializable():
    import json
    samples = np.random.RandomState(0).randn(100, 2)
    summary = fit_training_dist_summary(samples)
    json.dumps(summary)
