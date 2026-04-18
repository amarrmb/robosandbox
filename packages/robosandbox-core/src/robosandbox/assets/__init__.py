"""Bundled assets + on-demand asset fetchers."""

from __future__ import annotations

from robosandbox.assets.franka_visuals import (
    VISUAL_OBJS as FRANKA_VISUAL_OBJS,
    default_cache_dir as franka_visuals_cache_dir,
    download_all as download_franka_visuals,
)

__all__ = [
    "FRANKA_VISUAL_OBJS",
    "franka_visuals_cache_dir",
    "download_franka_visuals",
]
