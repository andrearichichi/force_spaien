"""Validated production contact loading and initial-frame geometry."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml


def load_overrides(path: Path) -> dict[str, dict[str, Any]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Contact override file must contain a mapping: {path}")
    result = {}
    for key, value in data.items():
        if not isinstance(value, dict):
            raise ValueError(f"Override {key} must be a mapping")
        result[str(key)] = dict(value)
    return result


def normalize(vector: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(vector))
    if length <= 1e-12:
        raise ValueError("zero-length production contact vector")
    return vector / length


def initial_contact_geometry(
    model_dir: Path,
    joint_name: str,
    link_name: str,
    local_point: list[float] | np.ndarray,
    joint_axis_local: list[float] | np.ndarray,
) -> dict[str, list[float] | float]:
    import sapien

    scene = sapien.Scene([sapien.physx.PhysxCpuSystem()])
    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    articulation_builders, actor_builders, _ = loader.parse(str(model_dir / "mobility.urdf"))
    if len(articulation_builders) != 1 or actor_builders:
        raise ValueError(f"Expected one articulation in {model_dir}")
    builder = articulation_builders[0]
    for link_builder in builder.link_builders:
        link_builder.visual_records = []
    articulation = builder.build()
    articulation.set_qpos(np.zeros_like(articulation.get_qpos(), dtype=np.float32))
    joint = articulation.find_joint_by_name(joint_name)
    link = articulation.find_link_by_name(link_name)
    if joint is None or link is None:
        raise ValueError(f"Could not resolve {joint_name}/{link_name} in {model_dir}")
    link_matrix = link.get_entity_pose().to_transformation_matrix()
    joint_matrix = joint.get_global_pose().to_transformation_matrix()
    point_world = (link_matrix @ np.asarray([*local_point, 1.0], dtype=float))[:3]
    origin_world = np.asarray(joint_matrix[:3, 3], dtype=float)
    axis_world = normalize(
        np.asarray(joint_matrix[:3, :3], dtype=float)
        @ normalize(np.asarray(joint_axis_local, dtype=float))
    )
    radius = point_world - origin_world
    perpendicular = radius - axis_world * float(np.dot(radius, axis_world))
    return {
        "contact_point_world": point_world.tolist(),
        "joint_origin_world": origin_world.tolist(),
        "joint_axis_world": axis_world.tolist(),
        "lever_arm_perpendicular": float(np.linalg.norm(perpendicular)),
        "tangent_opening_world": normalize(np.cross(axis_world, perpendicular)).tolist(),
    }
