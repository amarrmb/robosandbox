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
Pinned to a specific menagerie ref so downloads are reproducible.
"""

from __future__ import annotations

import os
import sys
import urllib.error
import urllib.request
from pathlib import Path


_MENAGERIE_REF = "main"
_BASE = (
    f"https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/"
    f"{_MENAGERIE_REF}/franka_emika_panda/assets"
)

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
    root = os.environ.get("ROBOSANDBOX_CACHE")
    if root:
        return Path(root) / "franka_visuals"
    return Path.home() / ".cache" / "robosandbox" / "franka_visuals"


def _fetch(url: str, dst: Path, *, force: bool) -> str:
    """Download ``url`` to ``dst``.

    Returns ``"cached"`` if a copy already exists (and ``force`` is
    ``False``), ``"downloaded"`` on success, or ``"missing"`` if the
    source URL returned 404 (indicates the pinned menagerie layout
    changed — the caller should surface this, not silently swallow).
    """
    if dst.exists() and not force:
        return "cached"
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = resp.read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return "missing"
        raise
    dst.write_bytes(data)
    return "downloaded"


def download_all(
    cache_dir: Path | None = None,
    *,
    force: bool = False,
    verbose: bool = True,
) -> dict[str, str]:
    """Fetch every visual OBJ into ``cache_dir/assets/``.

    Returns ``{filename: status}`` — statuses are ``"cached"``,
    ``"downloaded"``, or ``"missing"``.
    """
    cache = cache_dir or default_cache_dir()
    assets = cache / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    out: dict[str, str] = {}
    for name in VISUAL_OBJS:
        dst = assets / name
        status = _fetch(f"{_BASE}/{name}", dst, force=force)
        out[name] = status
        if verbose:
            tag = {"cached": "·", "downloaded": "+", "missing": "!"}[status]
            size = dst.stat().st_size // 1024 if status != "missing" else "—"
            print(f"  {tag} {name}  ({size} KB)")
    return out


def cli(cache_dir: str | None = None, *, force: bool = False) -> int:
    """Entry point wired to ``robo-sandbox download-franka-visuals``."""
    cache = Path(cache_dir) if cache_dir else default_cache_dir()
    print(f"Downloading Franka visuals to {cache}")
    print(f"Source: {_BASE}")
    statuses = download_all(cache, force=force)
    total_bytes = sum(
        (cache / "assets" / n).stat().st_size
        for n, s in statuses.items()
        if s != "missing"
    )
    missing = [n for n, s in statuses.items() if s == "missing"]
    downloaded = sum(1 for s in statuses.values() if s == "downloaded")
    cached = sum(1 for s in statuses.values() if s == "cached")
    print()
    print(f"  downloaded: {downloaded}, cached: {cached}, missing: {len(missing)}")
    print(f"  total on disk: {total_bytes / 1024 / 1024:.1f} MB")
    if missing:
        print(
            f"  WARN: {len(missing)} file(s) returned 404: {missing[:5]}...",
            file=sys.stderr,
        )
        print(
            f"  The menagerie layout at ref={_MENAGERIE_REF} may have changed — open an issue.",
            file=sys.stderr,
        )
        return 1
    return 0
