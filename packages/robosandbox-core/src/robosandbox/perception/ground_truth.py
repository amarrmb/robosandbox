"""Ground-truth perception: look up object pose in the Observation.

Useful for two things:
- Tier-3 skill integration tests (no VLM in the loop).
- Letting a user debug the skill/grasp/motion chain without paying for VLM
  calls while a scene is being built.

NOT used in the default VLM-driven agent flow — that's `vlm_pointer`.
"""

from __future__ import annotations

from robosandbox.types import DetectedObject, Observation, Pose


class GroundTruthPerception:
    """Match the text query against scene_objects by substring.

    Matching is dumb-on-purpose: the query "blue cube" matches an object
    with id "blue_cube". For realism, use VLMPointer instead.
    """

    name = "ground_truth"

    def locate(self, query: str, obs: Observation) -> list[DetectedObject]:
        if not obs.scene_objects:
            return []
        q = query.strip().lower().replace(" ", "_")
        matches: list[DetectedObject] = []
        for oid, pose in obs.scene_objects.items():
            oid_norm = oid.lower().replace(" ", "_")
            if q in oid_norm or oid_norm in q:
                matches.append(
                    DetectedObject(
                        label=oid,
                        pose_3d=Pose(xyz=pose.xyz, quat_xyzw=pose.quat_xyzw),
                        confidence=1.0,
                    )
                )
        return matches
