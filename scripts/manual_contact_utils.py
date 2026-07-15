#!/usr/bin/env python3
"""Geometry-only helpers for manual ForceSAPIEN contact selection."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np
import trimesh
import yaml


CANDIDATE_PREFIXES = {
    "10211": "laptop_screen_edge",
    "10905": "refrigerator_door_edge",
    "11100": "scissors_handle_or_blade",
    "101917": "oven_door_edge",
    "102255": "foldingchair_surface",
    "103111": "stapler_lid_front_top",
    "103706": "knife_prismatic_point",
    "103776": "washingmachine_door_rim",
}

CONTACT_STRATEGIES = {
    "10211": "laptop_screen_free_edge",
    "10905": "door_handle_or_free_edge",
    "11100": "scissors_handle_or_blade_candidate",
    "101917": "door_handle_or_free_edge",
    "102255": "farthest_from_joint",
    "103111": "stapler_lid_front_top",
    "103706": "prismatic_aligned_moving_link_point",
    "103776": "washingmachine_door_rim_or_handle",
}

CONTACT_VERDICTS = {
    "10211": "CONTACT_WARN",
    "10905": "CONTACT_PASS",
    "11100": "CONTACT_WARN",
    "101917": "CONTACT_WARN",
    "102255": "CONTACT_PASS",
    "103111": "CONTACT_PASS",
    "103706": "CONTACT_PASS",
    "103776": "CONTACT_WARN",
}

USB_SEMANTIC_CANDIDATES = {
    5: ("usb_cover_tip", "usb_cover_tip"),
    7: ("usb_cover_grip_edge", "usb_cover_grip_edge"),
    9: ("usb_cover_outer_edge", "usb_cover_outer_edge"),
    16: ("usb_far_from_hinge_edge", "usb_far_from_hinge_edge"),
}


def load_overrides(path: Path) -> dict[str, dict[str, Any]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Contact override file must contain a mapping: {path}")
    result: dict[str, dict[str, Any]] = {}
    for key, value in data.items():
        if not isinstance(value, dict):
            raise ValueError(f"Override {key} must be a mapping")
        result[str(key)] = dict(value)
    return result


def parse_xyz(text: str | None, default: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> np.ndarray:
    return np.asarray([float(v) for v in text.split()] if text else default, dtype=float)


def rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.asarray(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=float,
    )


def moving_link_vertices(model_dir: Path, link_name: str) -> np.ndarray:
    root = ET.parse(model_dir / "mobility.urdf").getroot()
    link = root.find(f".//link[@name='{link_name}']")
    if link is None:
        raise ValueError(f"Link {link_name!r} not found in {model_dir / 'mobility.urdf'}")
    clouds: list[np.ndarray] = []
    for visual in link.findall("visual"):
        mesh_node = visual.find("./geometry/mesh")
        if mesh_node is None or not mesh_node.get("filename"):
            continue
        mesh_path = model_dir / str(mesh_node.get("filename"))
        loaded = trimesh.load(mesh_path, force="mesh", process=False)
        vertices = np.asarray(loaded.vertices, dtype=float)
        scale = parse_xyz(mesh_node.get("scale"), (1.0, 1.0, 1.0))
        origin = visual.find("origin")
        xyz = parse_xyz(origin.get("xyz") if origin is not None else None)
        rpy = parse_xyz(origin.get("rpy") if origin is not None else None)
        clouds.append((vertices * scale) @ rpy_matrix(rpy).T + xyz)
    if not clouds:
        raise ValueError(f"No usable visual mesh vertices for {link_name} in {model_dir}")
    return np.concatenate(clouds, axis=0)


def link_inertial_properties(model_dir: Path, link_name: str) -> tuple[float, list[float]]:
    """Return URDF link mass and diagonal inertia in dataset/SAPIEN units."""
    root = ET.parse(model_dir / "mobility.urdf").getroot()
    link = root.find(f".//link[@name='{link_name}']")
    inertial = link.find("inertial") if link is not None else None
    mass_node = inertial.find("mass") if inertial is not None else None
    inertia = inertial.find("inertia") if inertial is not None else None
    mass = float(mass_node.get("value", "0")) if mass_node is not None else 0.0
    diag = [float(inertia.get(key, "0")) if inertia is not None else 0.0 for key in ("ixx", "iyy", "izz")]
    return mass, diag


def _unique_points(points: list[np.ndarray], tolerance: float = 1e-7) -> list[np.ndarray]:
    result: list[np.ndarray] = []
    for point in points:
        if not any(float(np.linalg.norm(point - existing)) <= tolerance for existing in result):
            result.append(np.asarray(point, dtype=float))
    return result


def candidate_local_points(vertices: np.ndarray, count: int = 20) -> list[np.ndarray]:
    """Return deterministic surface/extreme candidates in moving-link coordinates."""
    lo, hi = vertices.min(axis=0), vertices.max(axis=0)
    center = (lo + hi) * 0.5
    points: list[np.ndarray] = []
    # Corners are snapped to their nearest actual mesh vertices.
    for bits in range(8):
        corner = np.asarray([hi[i] if bits & (1 << i) else lo[i] for i in range(3)])
        points.append(vertices[int(np.argmin(np.linalg.norm(vertices - corner, axis=1)))])
    # Face/edge targets are also snapped to the mesh surface.
    targets = []
    for axis in range(3):
        for side in (lo[axis], hi[axis]):
            target = center.copy()
            target[axis] = side
            targets.append(target)
    for a, b in ((0, 1), (0, 2), (1, 2)):
        for sa in (lo[a], hi[a]):
            for sb in (lo[b], hi[b]):
                target = center.copy()
                target[a], target[b] = sa, sb
                targets.append(target)
    points.extend(vertices[int(np.argmin(np.linalg.norm(vertices - target, axis=1)))] for target in targets)
    unique = _unique_points(points)
    if len(unique) < count:
        distances = np.linalg.norm(vertices - center, axis=1)
        for index in np.argsort(distances)[::-1]:
            unique = _unique_points(unique + [vertices[int(index)]], tolerance=max(float(np.linalg.norm(hi - lo)) * 0.015, 1e-7))
            if len(unique) >= count:
                break
    return unique[: max(10, min(count, 30))]


def candidate_id(object_id: str, index: int) -> str:
    if object_id == "100109" and index in USB_SEMANTIC_CANDIDATES:
        return USB_SEMANTIC_CANDIDATES[index][0]
    return f"{CANDIDATE_PREFIXES.get(object_id, 'contact')}_{index + 1:02d}"


def candidate_strategy(object_id: str, index: int) -> str:
    if object_id == "100109" and index in USB_SEMANTIC_CANDIDATES:
        return USB_SEMANTIC_CANDIDATES[index][1]
    if index == 0:
        return CONTACT_STRATEGIES.get(object_id, "farthest_from_joint")
    cycle = ("free_edge_candidate", "bbox_edge_midpoint", "bbox_corner_candidate", "farthest_from_joint")
    return cycle[(index - 1) % len(cycle)]


def normalize(vector: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(vector))
    if length <= 1e-12:
        return np.zeros(3, dtype=float)
    return vector / length


def initial_contact_geometry(
    model_dir: Path,
    joint_name: str,
    link_name: str,
    local_point: list[float] | np.ndarray,
    joint_axis_local: list[float] | np.ndarray,
) -> dict[str, list[float] | float]:
    """Load an articulation without stepping dynamics and transform one contact."""
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
    axis_world = normalize(np.asarray(joint_matrix[:3, :3], dtype=float) @ normalize(np.asarray(joint_axis_local, dtype=float)))
    radius = point_world - origin_world
    perpendicular_radius = radius - axis_world * float(np.dot(radius, axis_world))
    lever = float(np.linalg.norm(perpendicular_radius))
    tangent = normalize(np.cross(axis_world, perpendicular_radius))
    return {
        "contact_point_world": point_world.tolist(),
        "joint_origin_world": origin_world.tolist(),
        "joint_axis_world": axis_world.tolist(),
        "lever_arm_perpendicular": lever,
        "tangent_opening_world": tangent.tolist(),
    }


def initial_contact_geometries(
    model_dir: Path,
    joint_name: str,
    link_name: str,
    local_points: list[np.ndarray],
    joint_axis_local: list[float] | np.ndarray,
) -> list[dict[str, list[float] | float]]:
    """Transform many candidates with one geometry-only articulation load."""
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
    origin_world = np.asarray(joint_matrix[:3, 3], dtype=float)
    axis_world = normalize(np.asarray(joint_matrix[:3, :3], dtype=float) @ normalize(np.asarray(joint_axis_local, dtype=float)))
    results = []
    for local_point in local_points:
        point_world = (link_matrix @ np.asarray([*local_point, 1.0], dtype=float))[:3]
        radius = point_world - origin_world
        perpendicular_radius = radius - axis_world * float(np.dot(radius, axis_world))
        lever = float(np.linalg.norm(perpendicular_radius))
        results.append(
            {
                "contact_point_world": point_world.tolist(),
                "joint_origin_world": origin_world.tolist(),
                "joint_axis_world": axis_world.tolist(),
                "lever_arm_perpendicular": lever,
                "tangent_opening_world": normalize(np.cross(axis_world, perpendicular_radius)).tolist(),
            }
        )
    return results


def initial_world_to_local(model_dir: Path, link_name: str, world_point: list[float]) -> list[float]:
    """Convert a world-space point to the selected link frame without stepping dynamics."""
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
    link = articulation.find_link_by_name(link_name)
    if link is None:
        raise ValueError(f"Could not resolve link {link_name} in {model_dir}")
    inverse = np.linalg.inv(link.get_entity_pose().to_transformation_matrix())
    return (inverse @ np.asarray([*world_point, 1.0], dtype=float))[:3].astype(float).tolist()


def initial_local_direction_to_world(model_dir: Path, link_name: str, local_direction: list[float]) -> list[float]:
    """Rotate and normalize a link-local direction without stepping dynamics."""
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
    link = articulation.find_link_by_name(link_name)
    if link is None:
        raise ValueError(f"Could not resolve link {link_name} in {model_dir}")
    rotation = link.get_entity_pose().to_transformation_matrix()[:3, :3]
    return normalize(rotation @ normalize(np.asarray(local_direction, dtype=float))).tolist()


def resolve_override(
    object_id: str,
    override: dict[str, Any],
    model_dir: Path,
    link_name: str,
    candidates: list[np.ndarray] | None = None,
) -> tuple[list[float], str]:
    mode = str(override.get("contact_mode", ""))
    if mode == "manual_local_point":
        point = override.get("local_point")
        if not isinstance(point, list) or len(point) != 3:
            raise ValueError(f"{object_id}: manual_local_point requires local_point: [x, y, z]")
        return [float(v) for v in point], "manual_override"
    if mode == "candidate_id":
        requested = str(override.get("candidate_id", ""))
        points = candidates or candidate_local_points(moving_link_vertices(model_dir, link_name))
        ids = [candidate_id(object_id, i) for i in range(len(points))]
        if requested not in ids:
            raise ValueError(f"{object_id}: unknown candidate_id {requested!r}; generate contact candidates first")
        return points[ids.index(requested)].astype(float).tolist(), "candidate_id"
    if mode == "manual_world_point":
        raise ValueError("manual_world_point must be converted using the initial link pose")
    raise ValueError(f"{object_id}: unsupported contact_mode {mode!r}")
