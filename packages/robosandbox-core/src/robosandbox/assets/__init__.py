"""Bundled assets + on-demand asset fetchers."""

from __future__ import annotations

from robosandbox.assets.franka_visuals import (
    VISUAL_OBJS as FRANKA_VISUAL_OBJS,
)
from robosandbox.assets.franka_visuals import (
    default_cache_dir as franka_visuals_cache_dir,
)
from robosandbox.assets.franka_visuals import (
    download_all as download_franka_visuals,
)

__all__ = [
    "FRANKA_VISUAL_OBJS",
    "download_franka_visuals",
    "franka_visuals_cache_dir",
]
