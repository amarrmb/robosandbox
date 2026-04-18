"""Analytic top-down grasp planner.

Assumptions (v0.1):
- Gripper approaches from above along world -Z.
- Gripper palm is aligned with world XY, gripper jaws open along world X.
- Object is on a table at or near the known table_height.

Works for cubes, short cylinders, cans, mugs. Fails for flat plates that
need a side grasp — that's a job for Contact-GraspNet in contrib.
"""

from __future__ import annotations

import numpy as np

from robosandbox.types import DetectedObject, Grasp, Observation, Pose

# Quaternion for: rotate by 180° around world X so gripper palm faces down.
# In (x, y, z, w): sin(90°) * X, cos(90°) -> (1, 0, 0, 0).
_PALM_DOWN = (1.0, 0.0, 0.0, 0.0)


class AnalyticTopDown:
    name = "analytic_topdown"

    def __init__(
        self,
        default_object_width: float = 0.04,
        grasp_height_offset: float = 0.013,
    ) -> None:
        """
        ``grasp_height_offset`` is the z-offset (meters) to add above the
        object's center when targeting the end-effector site. Tuned for the
        v0.1 built-in arm whose 4cm fingers extend 2cm below ee_site.
        """
        self._default_width = default_object_width
        self._grasp_height_offset = grasp_height_offset

    def plan(self, obs: Observation, target: DetectedObject) -> list[Grasp]:
        if target.pose_3d is None:
            return []
        x, y, z = target.pose_3d.xyz
        grasp_pose = Pose(
            xyz=(x, y, z + self._grasp_height_offset),
            quat_xyzw=_PALM_DOWN,
        )
        width = min(0.07, self._default_width + 0.01)
        return [
            Grasp(
                pose=grasp_pose,
                gripper_width=width,
                approach_offset=0.1,
                score=1.0,
            )
        ]

    # Future extension: return N grasps rotated about world Z so the VLM
    # can pick the orientation that best matches the object's principal axis.
    @staticmethod
    def rotate_about_z(g: Grasp, theta: float) -> Grasp:
        cz, sz = float(np.cos(theta / 2)), float(np.sin(theta / 2))
        # Compose with existing palm-down rotation.
        # For v0.1 we don't use this — left for v0.2.
        _ = cz, sz
        return g
