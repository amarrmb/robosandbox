"""VLMPointer: use an LLM that can reason about an image to locate
objects by natural-language name. Returns pixel coords + a projected
3D world pose (via the Observation's depth + intrinsics + extrinsics).

This is the default perception in the v0.1 agentic flow. It needs a
configured VLMClient. For offline / no-API-key flows, see
`GroundTruthPerception`.
"""

from __future__ import annotations

import math
import textwrap
from typing import Any

import numpy as np

from robosandbox.types import DetectedObject, Observation, Pose
from robosandbox.vlm.client import OpenAIVLMClient, rgb_to_data_url
from robosandbox.vlm.json_recovery import VLMOutputError, parse_json_loose

_SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a precise object locator for a robotics workspace.
    The user will give you a text query and a top-side view of a table.

    Respond ONLY with a JSON object of the form:
      {"objects": [{"label": "...", "bbox": [x1,y1,x2,y2], "point":[cx,cy], "confidence": 0.0-1.0}]}

    - Coordinates are pixels, origin top-left, x right, y down.
    - Return zero objects if no match is visible.
    - Prefer tight bounding boxes. `point` should be the visible centroid.
    - Do not wrap the JSON in prose or markdown fences.
    """
).strip()


def _pixel_to_world(
    u: int,
    v: int,
    obs: Observation,
) -> Pose | None:
    """Project a pixel to a 3D point in world frame using the Observation's
    depth + intrinsics + extrinsics."""
    if obs.depth is None or obs.camera_intrinsics is None or obs.camera_extrinsics is None:
        return None
    H, W = obs.depth.shape
    if not (0 <= u < W and 0 <= v < H):
        return None
    d = float(obs.depth[v, u])
    if not math.isfinite(d) or d <= 1e-4 or d > 100.0:
        return None

    intr = obs.camera_intrinsics
    # MuJoCo camera convention: +X right, +Y up, looking along -Z.
    # PNG image: origin top-left, x right, y down — flip y to get camera +Y.
    x_cam = (u - intr.cx) * d / intr.fx
    y_cam = -(v - intr.cy) * d / intr.fy
    z_cam = -d  # negative because camera looks in -Z

    # Rotate cam-frame point by extrinsic rotation, translate by origin.
    # Reconstruct rotation matrix from quaternion (x, y, z, w).
    qx, qy, qz, qw = obs.camera_extrinsics.quat_xyzw
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz
    R = np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ]
    )
    p_cam = np.array([x_cam, y_cam, z_cam])
    p_world = R @ p_cam + np.array(obs.camera_extrinsics.xyz)
    return Pose(xyz=tuple(p_world.tolist()), quat_xyzw=(0.0, 0.0, 0.0, 1.0))


class VLMPointer:
    name = "vlm_pointer"

    def __init__(self, vlm: OpenAIVLMClient | None = None, detail: str = "high") -> None:
        self._vlm = vlm
        self._detail = detail

    def locate(self, query: str, obs: Observation) -> list[DetectedObject]:
        if self._vlm is None:
            raise RuntimeError(
                "VLMPointer.locate called without a configured VLMClient; "
                "either pass one in __init__ or use GroundTruthPerception"
            )

        data_url = rgb_to_data_url(obs.rgb)
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Query: {query}"},
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url, "detail": self._detail},
                    },
                ],
            },
        ]
        resp = self._vlm.chat(messages)
        content = resp.get("content") or ""
        try:
            parsed: Any = parse_json_loose(content)
        except VLMOutputError:
            return []

        raw_objs = parsed.get("objects") if isinstance(parsed, dict) else None
        if not isinstance(raw_objs, list):
            return []

        results: list[DetectedObject] = []
        for obj in raw_objs:
            if not isinstance(obj, dict):
                continue
            label = str(obj.get("label", query))
            confidence = float(obj.get("confidence", 0.8))
            bbox = obj.get("bbox")
            point = obj.get("point")
            if isinstance(point, list) and len(point) == 2:
                u, v = int(point[0]), int(point[1])
            elif isinstance(bbox, list) and len(bbox) == 4:
                u = int((bbox[0] + bbox[2]) / 2)
                v = int((bbox[1] + bbox[3]) / 2)
            else:
                continue
            pose_3d = _pixel_to_world(u, v, obs)
            if pose_3d is None:
                continue
            results.append(
                DetectedObject(
                    label=label,
                    pixel_xy=(u, v),
                    bbox_2d=tuple(bbox) if isinstance(bbox, list) and len(bbox) == 4 else None,
                    pose_3d=pose_3d,
                    confidence=confidence,
                )
            )
        return results
