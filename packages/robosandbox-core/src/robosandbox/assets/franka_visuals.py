"""Fetch Franka Panda visual meshes from mujoco_menagerie.

The bundled Franka ships with collision-only meshes (~160 KB) to keep
the install lean. This module pulls the full visual set (~33 MB) on
demand and caches it under::

    $ROBOSANDBOX_CACHE/franka_visuals/   (if set)
    ~/.cache/robosandbox/franka_visuals/  (default)

CLI shortcut: ``robo-sandbox download-franka-visuals [--cache-dir PATH]``.

Integration with the robot loader to auto-augment ``panda.xml`` with
visual geoms is a future slice — for now the cache is user-space and
the downloaded OBJs can be used directly (e.g. by a Three.js client
renderer or a user-written MJCF edit).

Source: https://github.com/google-deepmind/mujoco_menagerie (Apache 2.0).
Pinned to a specific commit SHA so a given RoboSandbox release always
fetches the same bytes. Bumping ``_MENAGERIE_REF`` requires re-verifying
``VISUAL_OBJS`` against the new ``franka_emika_panda/assets/`` listing.
"""

from __future__ import annotations

import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

from robosandbox import _cache_root

# Menagerie commit pinned for reproducibility. When bumping, regenerate
# VISUAL_OBJS by diffing a checkout's assets/ directory against this list.
_MENAGERIE_REF = "a03e87bf13502b0b48ebbf2808928fd96ebf9cf3"
_BASE = (
    f"https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/"
    f"{_MENAGERIE_REF}/franka_emika_panda/assets"
)

_FetchStatus = Literal["cached", "downloaded", "missing"]

# Canonical visual mesh set verified against ``_MENAGERIE_REF``. Regenerate
# by listing ``assets/`` in a menagerie checkout and diffing against this
# list when a new menagerie ref is pinned.
VISUAL_OBJS: tuple[str, ...] = (
    "link0_0.obj",
    "link0_1.obj",
    "link0_2.obj",
    "link0_3.obj",
    "link0_4.obj",
    "link0_5.obj",
    "link0_7.obj",
    "link0_8.obj",
    "link0_9.obj",
    "link0_10.obj",
    "link0_11.obj",
    "link1.obj",
    "link2.obj",
    "link3_0.obj",
    "link3_1.obj",
    "link3_2.obj",
    "link3_3.obj",
    "link4_0.obj",
    "link4_1.obj",
    "link4_2.obj",
    "link4_3.obj",
    "link5_0.obj",
    "link5_1.obj",
    "link5_2.obj",
    "link6_0.obj",
    "link6_1.obj",
    "link6_2.obj",
    "link6_3.obj",
    "link6_4.obj",
    "link6_5.obj",
    "link6_6.obj",
    "link6_7.obj",
    "link6_8.obj",
    "link6_9.obj",
    "link6_10.obj",
    "link6_11.obj",
    "link6_12.obj",
    "link6_13.obj",
    "link6_14.obj",
    "link6_15.obj",
    "link6_16.obj",
    "link7_0.obj",
    "link7_1.obj",
    "link7_2.obj",
    "link7_3.obj",
    "link7_4.obj",
    "link7_5.obj",
    "link7_6.obj",
    "link7_7.obj",
    "hand_0.obj",
    "hand_1.obj",
    "hand_2.obj",
    "hand_3.obj",
    "hand_4.obj",
    "finger_0.obj",
    "finger_1.obj",
)


def default_cache_dir() -> Path:
    """Resolve the cache root, honoring ``ROBOSANDBOX_CACHE`` if set."""
    return _cache_root("franka_visuals")


def _fetch(url: str, dst: Path, *, force: bool) -> tuple[_FetchStatus, int]:
    """Download ``url`` to ``dst``.

    Returns ``(status, bytes_on_disk)``:
      - ``("cached", size)``  — ``dst`` already existed (and ``force`` is False).
      - ``("downloaded", size)`` — newly written.
      - ``("missing", 0)`` — source URL returned 404 (pinned menagerie
        layout drift — caller surfaces this, never silently swallows).
    """
    if dst.exists() and not force:
        return "cached", dst.stat().st_size
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = resp.read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return "missing", 0
        raise
    dst.write_bytes(data)
    return "downloaded", len(data)


def download_all(
    cache_dir: Path | None = None,
    *,
    force: bool = False,
    verbose: bool = True,
    max_workers: int = 8,
) -> dict[str, tuple[_FetchStatus, int]]:
    """Fetch every visual OBJ into ``cache_dir/assets/``.

    Downloads run in parallel (``max_workers`` threads) since the bottleneck
    is GitHub CDN RTT, not CPU. ``mkdir(..., exist_ok=True)`` makes the
    parent-dir creation safe under concurrent writers.

    Returns ``{filename: (status, bytes)}`` ordered like ``VISUAL_OBJS``.
    """
    cache = cache_dir or default_cache_dir()
    assets = cache / "assets"
    assets.mkdir(parents=True, exist_ok=True)

    def _job(name: str) -> tuple[str, _FetchStatus, int]:
        status, size = _fetch(f"{_BASE}/{name}", assets / name, force=force)
        return name, status, size

    out: dict[str, tuple[_FetchStatus, int]] = {}
    # max_workers=1 == serial, useful for deterministic test order.
    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as ex:
        for name, status, size in ex.map(_job, VISUAL_OBJS):
            out[name] = (status, size)
    if verbose:
        # Preserve VISUAL_OBJS order in the output.
        for name in VISUAL_OBJS:
            status, size = out[name]
            tag = {"cached": "·", "downloaded": "+", "missing": "!"}[status]
            kb = size // 1024 if status != "missing" else "—"
            print(f"  {tag} {name}  ({kb} KB)")
    return out


def cli(cache_dir: str | None = None, *, force: bool = False) -> int:
    """Entry point wired to ``robo-sandbox download-franka-visuals``."""
    import sys as _sys

    cache = Path(cache_dir) if cache_dir else default_cache_dir()
    print(f"Downloading Franka visuals to {cache}")
    print(f"Source: {_BASE}")
    results = download_all(cache, force=force)

    total_bytes = sum(size for _, size in results.values())
    missing = [n for n, (s, _) in results.items() if s == "missing"]
    downloaded = sum(1 for s, _ in results.values() if s == "downloaded")
    cached = sum(1 for s, _ in results.values() if s == "cached")
    print()
    print(f"  downloaded: {downloaded}, cached: {cached}, missing: {len(missing)}")
    print(f"  total on disk: {total_bytes / 1024 / 1024:.1f} MB")
    if missing:
        print(
            f"  WARN: {len(missing)} file(s) returned 404: {missing[:5]}...",
            file=_sys.stderr,
        )
        print(
            f"  The menagerie layout at ref={_MENAGERIE_REF} may have changed — open an issue.",
            file=_sys.stderr,
        )
        return 1
    return 0
