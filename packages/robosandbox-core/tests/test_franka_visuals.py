"""Tests for ``robosandbox.assets.franka_visuals`` — cache layout + no network."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from robosandbox.assets.franka_visuals import (
    VISUAL_OBJS,
    default_cache_dir,
    download_all,
)


def test_visual_objs_list_nonempty() -> None:
    # ~56 visual OBJs from menagerie. Exact count is brittle; we just
    # assert it's plausibly a full robot and no empty names slipped in.
    assert 40 < len(VISUAL_OBJS) < 100
    for name in VISUAL_OBJS:
        assert name.endswith(".obj")
        assert "/" not in name and ".." not in name


def test_default_cache_dir_respects_env(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ROBOSANDBOX_CACHE", str(tmp_path))
    d = default_cache_dir()
    assert d == tmp_path / "franka_visuals"


def test_default_cache_dir_falls_back_to_home(monkeypatch) -> None:
    monkeypatch.delenv("ROBOSANDBOX_CACHE", raising=False)
    d = default_cache_dir()
    assert ".cache/robosandbox/franka_visuals" in str(d)


def test_download_all_cached_is_fast(tmp_path: Path) -> None:
    """If every OBJ already exists on disk, download_all returns 'cached'
    for each without making any HTTP calls.
    """
    cache = tmp_path / "cache"
    (cache / "assets").mkdir(parents=True)
    for name in VISUAL_OBJS:
        (cache / "assets" / name).write_bytes(b"stub")

    call_count = {"n": 0}

    def sentinel(url, timeout=30):  # pragma: no cover
        call_count["n"] += 1
        raise RuntimeError("should not be called for fully-cached dir")

    with patch("urllib.request.urlopen", side_effect=sentinel):
        statuses = download_all(cache, verbose=False)

    assert call_count["n"] == 0
    assert set(statuses.values()) == {"cached"}
    assert len(statuses) == len(VISUAL_OBJS)


def test_download_all_missing_returns_status(tmp_path: Path) -> None:
    """404s are reported as status='missing', not raised — keeps the
    CLI output usable even if menagerie's layout drifts.
    """
    import urllib.error

    def always_404(url, timeout=30):
        raise urllib.error.HTTPError(url, 404, "Not Found", hdrs=None, fp=None)

    with patch("urllib.request.urlopen", side_effect=always_404):
        statuses = download_all(tmp_path / "cache", verbose=False)

    assert all(s == "missing" for s in statuses.values())
