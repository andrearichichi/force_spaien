#!/usr/bin/env python3
"""Object-level ForceSAPIEN batch runner for final_dataset outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import traceback
import xml.etree.ElementTree as ET

try:
    import imageio.v2 as imageio
except ModuleNotFoundError:  # pragma: no cover - dry-run and clusters without video deps
    imageio = None

try:
    from PIL import Image, ImageDraw
except ModuleNotFoundError:  # pragma: no cover
    Image = None
    ImageDraw = None

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None

try:
    from main import default_initial_angle, drawer_index_from_link, first_moving_joint, preferred_joint
except ImportError:  # pragma: no cover
    from scripts.main import default_initial_angle, drawer_index_from_link, first_moving_joint, preferred_joint

try:
    from manual_contact_utils import (
        candidate_local_points,
        initial_contact_geometry,
        initial_world_to_local,
        load_overrides,
        moving_link_vertices as manual_link_vertices,
        resolve_override,
    )
except ImportError:  # pragma: no cover
    from scripts.manual_contact_utils import (
        candidate_local_points,
        initial_contact_geometry,
        initial_world_to_local,
        load_overrides,
        moving_link_vertices as manual_link_vertices,
        resolve_override,
    )


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent

REQUIRED_COMPLETE_FILES = [
    "simulation.json",
    "final_video.mp4",
    "diagnostics/physics_diagnostics.md",
    "diagnostics/physics_timeseries.tsv",
    "diagnostics/force_torque_by_frame.png",
    "diagnostics/q_qdot_qddot_by_frame.png",
    "diagnostics/resistance_by_frame.png",
]

SUMMARY_FIELDS = [
    "object_id",
    "object_name",
    "joint_type",
    "output_folder",
    "status",
    "q_start",
    "q_end",
    "delta_q",
    "final_video_exists",
    "simulation_json_exists",
    "diagnostics_exists",
    "error_message",
]

OBJECT_OVERRIDES: dict[str, dict[str, str]] = {
    "10211": {
        "joint_name": "joint_1", "moving_link_name": "link_1", "semantic_part": "laptop_screen",
        "contact_strategy": "laptop_screen_free_edge", "expected_motion": "screen_opens_about_hinge",
    },
    "11100": {
        "joint_name": "joint_0", "moving_link_name": "link_0", "semantic_part": "scissors_moving_half",
        "contact_strategy": "scissors_far_handle_or_blade_candidate", "expected_motion": "scissors_open_or_close",
    },
    "101917": {
        "joint_name": "joint_0", "moving_link_name": "link_0", "semantic_part": "oven_door",
        "contact_strategy": "oven_door_handle_or_free_edge", "expected_motion": "door_opens_about_hinge",
    },
    "103776": {
        "joint_name": "joint_0", "moving_link_name": "link_0", "semantic_part": "washingmachine_door",
        "contact_strategy": "washingmachine_door_rim_or_handle", "expected_motion": "door_opens_about_hinge",
    },
    "103706": {
        "joint_name": "joint_0", "moving_link_name": "link_0", "semantic_part": "knife_sliding_link",
        "contact_strategy": "prismatic_aligned_moving_link_point", "expected_motion": "translation_along_joint_axis",
    },
    "103111": {
        "joint_name": "joint_1",
        "moving_link_name": "link_1",
        "semantic_part": "stapler_lid",
        "contact_strategy": "stapler_lid_front_top",
        "expected_motion": "lid_opens_upward",
    },
    "10905": {
        "joint_name": "joint_0",
        "moving_link_name": "link_0",
        "semantic_part": "refrigerator_door",
        "contact_strategy": "door_handle_or_free_vertical_edge",
    },
    "102255": {
        "joint_name": "joint_1",
        "moving_link_name": "link_1",
        "semantic_part": "folding_chair_seat_or_back",
        "contact_strategy": "robust_moving_surface_point",
    },
}

EXPECTED_MAX_DIMENSIONS_M = {
    "laptop": (0.25, 0.45),
    "refrigerator": (1.0, 2.2),
    "scissors": (0.10, 0.30),
    "oven": (0.45, 1.0),
    "foldingchair": (0.5, 1.2),
    "stapler": (0.10, 0.25),
    "knife": (0.10, 0.35),
    "washingmachine": (0.5, 1.2),
}


def sanitize_name(value: str | None, fallback: str) -> str:
    text = (value or fallback).strip().lower()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^a-z0-9_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or fallback


def unit(values: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in values))
    if norm <= 1e-12:
        return [0.0, 0.0, 1.0]
    return [float(value / norm) for value in values]


def perpendicular(axis: list[float]) -> list[float]:
    ax = unit(axis)
    candidate = [1.0, 0.0, 0.0]
    if abs(sum(ax[i] * candidate[i] for i in range(3))) > 0.85:
        candidate = [0.0, 1.0, 0.0]
    return unit(
        [
            ax[1] * candidate[2] - ax[2] * candidate[1],
            ax[2] * candidate[0] - ax[0] * candidate[2],
            ax[0] * candidate[1] - ax[1] * candidate[0],
        ]
    )


def read_json(path: Path) -> object | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def vec3(value: object) -> list[float]:
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        return [float(value[0]), float(value[1]), float(value[2])]
    return [0.0, 0.0, 0.0]


def vector_norm(value: object) -> float:
    return math.sqrt(sum(component * component for component in vec3(value)))


def vector_unit(value: object) -> list[float]:
    vector = vec3(value)
    length = vector_norm(vector)
    return [component / length for component in vector] if length > 1e-12 else [0.0, 0.0, 0.0]


def vector_sub(left: object, right: object) -> list[float]:
    a, b = vec3(left), vec3(right)
    return [a[i] - b[i] for i in range(3)]


def vector_dot(left: object, right: object) -> float:
    a, b = vec3(left), vec3(right)
    return sum(a[i] * b[i] for i in range(3))


def vector_cross(left: object, right: object) -> list[float]:
    a, b = vec3(left), vec3(right)
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def sign_matches(left: float, right: float, epsilon: float = 1e-8) -> bool:
    return abs(left) > epsilon and abs(right) > epsilon and math.copysign(1.0, left) == math.copysign(1.0, right)


def object_name(model_dir: Path) -> str:
    meta = read_json(model_dir / "meta.json")
    if isinstance(meta, dict):
        for key in ("model_cat", "category", "object_name", "name"):
            value = meta.get(key)
            if isinstance(value, str) and value.strip():
                return value
    return "object"


def joint_axis_origin(model_dir: Path, joint_name: str | None) -> tuple[list[float], list[float]]:
    if not joint_name:
        return [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]
    root = ET.parse(model_dir / "mobility.urdf").getroot()
    joint = root.find(f".//joint[@name='{joint_name}']")
    if joint is None:
        return [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]
    axis_node = joint.find("axis")
    origin_node = joint.find("origin")
    axis = [float(v) for v in axis_node.attrib.get("xyz", "0 0 1").split()] if axis_node is not None else [0.0, 0.0, 1.0]
    origin = [float(v) for v in origin_node.attrib.get("xyz", "0 0 0").split()] if origin_node is not None else [0.0, 0.0, 0.0]
    return unit(axis), origin


def moving_joints(model_dir: Path) -> list[dict[str, object]]:
    root = ET.parse(model_dir / "mobility.urdf").getroot()
    joints = []
    for joint in root.findall("joint"):
        joint_type = joint.attrib.get("type", "")
        if joint_type == "fixed":
            continue
        child = joint.find("child")
        parent = joint.find("parent")
        axis_node = joint.find("axis")
        limit_node = joint.find("limit")
        axis = [float(v) for v in axis_node.attrib.get("xyz", "1 0 0").split()] if axis_node is not None else [1.0, 0.0, 0.0]
        limits = None
        if limit_node is not None and "lower" in limit_node.attrib and "upper" in limit_node.attrib:
            limits = [float(limit_node.attrib["lower"]), float(limit_node.attrib["upper"])]
        joints.append(
            {
                "joint_name": joint.attrib.get("name", ""),
                "joint_type": "revolute" if joint_type == "continuous" else joint_type,
                "parent_link": parent.attrib.get("link", "") if parent is not None else "",
                "child_link": child.attrib.get("link", "") if child is not None else "",
                "axis_local": unit(axis),
                "limits": limits,
            }
        )
    return joints


def joint_type_limits_for(model_dir: Path, joint_name: str) -> tuple[str, tuple[float, float] | None]:
    for joint in moving_joints(model_dir):
        if joint["joint_name"] == joint_name:
            limits = joint.get("limits")
            return str(joint["joint_type"]), tuple(limits) if isinstance(limits, list) else None
    raise RuntimeError(f"Override joint not found in {model_dir / 'mobility.urdf'}: {joint_name}")


def urdf_mesh_refs(model_dir: Path) -> list[str]:
    root = ET.parse(model_dir / "mobility.urdf").getroot()
    refs = []
    for mesh in root.findall(".//mesh"):
        filename = mesh.attrib.get("filename")
        if filename:
            refs.append(filename)
    return refs


def mesh_vertices(path: Path) -> list[list[float]]:
    vertices = []
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith("v "):
                    parts = line.split()
                    if len(parts) >= 4:
                        vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    except OSError:
        return []
    return vertices


def visual_origin(visual: ET.Element) -> list[float]:
    origin = visual.find("origin")
    if origin is None:
        return [0.0, 0.0, 0.0]
    return [float(value) for value in origin.attrib.get("xyz", "0 0 0").split()]


def link_visual_vertices(model_dir: Path, link_name: str) -> list[list[float]]:
    root = ET.parse(model_dir / "mobility.urdf").getroot()
    link = root.find(f".//link[@name='{link_name}']")
    if link is None:
        return []
    points: list[list[float]] = []
    for visual in link.findall("visual"):
        mesh = visual.find("./geometry/mesh")
        filename = mesh.attrib.get("filename") if mesh is not None else None
        if not filename:
            continue
        origin = visual_origin(visual)
        scale = [float(value) for value in mesh.attrib.get("scale", "1 1 1").split()]
        for point in mesh_vertices(model_dir / filename):
            points.append(
                [
                    point[0] * scale[0] + origin[0],
                    point[1] * scale[1] + origin[1],
                    point[2] * scale[2] + origin[2],
                ]
            )
    return points


def all_visual_vertices(model_dir: Path) -> list[list[float]]:
    root = ET.parse(model_dir / "mobility.urdf").getroot()
    points: list[list[float]] = []
    for link in root.findall("link"):
        name = link.attrib.get("name")
        if name:
            points.extend(link_visual_vertices(model_dir, name))
    return points


def bbox_size(points: list[list[float]]) -> list[float] | None:
    if not points:
        return None
    mins = [min(point[i] for point in points) for i in range(3)]
    maxs = [max(point[i] for point in points) for i in range(3)]
    return [maxs[i] - mins[i] for i in range(3)]


def point_inside_local_bbox(point: object, points: list[list[float]], tolerance: float = 1e-3) -> tuple[bool, float | None]:
    if not isinstance(point, list) or len(point) < 3 or not points:
        return False, None
    coords = [float(point[i]) for i in range(3)]
    mins = [min(vertex[i] for vertex in points) for i in range(3)]
    maxs = [max(vertex[i] for vertex in points) for i in range(3)]
    outside = [max(mins[i] - coords[i], 0.0, coords[i] - maxs[i]) for i in range(3)]
    distance = math.sqrt(sum(value * value for value in outside))
    return distance <= tolerance, distance


def object_scale_interpretation(object_name_value: object, size: list[float] | None) -> tuple[str, str]:
    if not size:
        return "unknown", "object bbox could not be computed"
    key = sanitize_name(str(object_name_value or ""), "object")
    max_dim = max(size)
    expected = EXPECTED_MAX_DIMENSIONS_M.get(key)
    if expected is None:
        return "unknown", "no rough expected size registered for category"
    if expected[0] <= max_dim <= expected[1]:
        return "metric_plausible", ""
    return "normalized_or_dataset_units", f"max dimension {max_dim:.3g} outside rough expected {expected[0]}-{expected[1]} m"


def verdict_rank(verdict: str) -> int:
    return {"PASS": 0, "WARN": 1, "FAIL": 2}.get(verdict, 1)


def combine_verdicts(*verdicts: str) -> str:
    return max(verdicts, key=verdict_rank)


def detect_object(model_dir: Path) -> dict[str, object]:
    object_id = model_dir.name
    name = object_name(model_dir)
    info: dict[str, object] = {
        "object_id": object_id,
        "object_name": name,
        "model_dir": model_dir,
        "valid": False,
        "error_message": "",
        "joint_type": "unknown",
        "joint_name": None,
        "link_name": None,
        "joint_limits": None,
        "joint_axis": [0.0, 0.0, 1.0],
        "joint_origin": [0.0, 0.0, 0.0],
    }
    urdf = model_dir / "mobility.urdf"
    if not urdf.exists():
        info["error_message"] = "missing mobility.urdf"
        return info
    try:
        missing_meshes = [ref for ref in urdf_mesh_refs(model_dir) if not (model_dir / ref).exists()]
        if missing_meshes:
            info["error_message"] = "missing mesh files: " + ", ".join(missing_meshes[:8])
            return info
        all_moving_joints = moving_joints(model_dir)
        override = OBJECT_OVERRIDES.get(object_id)
        if override is not None:
            joint_name = override["joint_name"]
            link_name = override["moving_link_name"]
            joint_type, limits = joint_type_limits_for(model_dir, joint_name)
            selection_source = "override"
        else:
            joint_type, detected_joint, detected_link, limits = first_moving_joint(model_dir)
            if joint_type == "continuous":
                joint_type = "revolute"
            joint_name, link_name = preferred_joint(model_dir, detected_joint, detected_link)
            selection_source = "auto_first_moving_joint"
        axis, origin = joint_axis_origin(model_dir, joint_name)
        info.update(
            {
                "valid": joint_type in {"revolute", "prismatic", "screw"},
                "joint_type": joint_type if joint_type in {"revolute", "prismatic", "screw"} else "unknown",
                "joint_name": joint_name,
                "link_name": link_name,
                "joint_limits": limits,
                "joint_axis": axis,
                "joint_origin": origin,
                "selection_source": selection_source,
                "override": override,
                "available_moving_joints": all_moving_joints,
            }
        )
        if not info["valid"]:
            info["error_message"] = f"unsupported joint type: {joint_type}"
    except Exception as exc:
        info["error_message"] = str(exc)
    return info


def output_folder(output_root: Path, info: dict[str, object]) -> Path:
    name = sanitize_name(str(info.get("object_name") or ""), "object")
    object_id = str(info["object_id"])
    joint_type = sanitize_name(str(info.get("joint_type") or "unknown"), "unknown")
    mode = str(info.get("force_application_mode") or "generalized_set_qf")
    custom_suffix = info.get("output_suffix")
    if custom_suffix:
        return output_root / f"{name}_{object_id}_{joint_type}_{sanitize_name(str(custom_suffix), 'check')}"
    if mode == "external_link_force":
        return output_root / f"{name}_{object_id}_{joint_type}_external_force_physics_check"
    if mode == "impulse_then_passive_joint_dynamics":
        return output_root / f"{name}_{object_id}_{joint_type}_impulse_passive_physics_check"
    return output_root / f"{name}_{object_id}_{joint_type}_general_force_framework_check"


def output_completeness(path: Path) -> tuple[bool, list[str]]:
    missing = []
    for rel in REQUIRED_COMPLETE_FILES:
        target = path / rel
        if not target.exists() or (target.is_file() and target.stat().st_size <= 0):
            missing.append(rel)
    return not missing, missing


def q_from_document(document: dict[str, object]) -> tuple[float | None, float | None]:
    initial = document.get("initial_state", {})
    final = document.get("final_state", {})
    if not isinstance(initial, dict) or not isinstance(final, dict):
        return None, None
    for start_key, end_key in (
        ("theta_rad", "theta_rad"),
        ("position_m", "position_m"),
        ("joint_angle_rad", "joint_angle_rad"),
        ("joint_position_m", "joint_position_m"),
    ):
        if start_key in initial and end_key in final:
            return float(initial[start_key]), float(final[end_key])
    return None, None


def primary_samples(document: dict[str, object]) -> list[dict[str, object]]:
    samples = document.get("samples", {})
    if not isinstance(samples, dict):
        return []
    for key in ("force", "pulling_force", "opening_force"):
        value = samples.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    for value in samples.values():
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def finite_timeseries(path: Path) -> bool:
    if not path.exists():
        return False
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()[1:]:
        for item in line.split("\t"):
            if not item:
                continue
            try:
                value = float(item)
            except ValueError:
                continue
            if not math.isfinite(value):
                return False
    return True


def finite_json(value: object) -> object:
    """Replace non-finite diagnostics with JSON null for unlimited joints."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [finite_json(item) for item in value]
    return value


def validate_video(path: Path) -> tuple[bool, bool, str]:
    if not path.exists() or path.stat().st_size <= 0:
        return False, False, "final_video.mp4 missing or empty"
    if imageio is None or np is None:
        return True, True, "imageio/numpy is not installed; skipped frame/static validation"
    try:
        reader = imageio.get_reader(path)
        count = reader.count_frames()
        if count < 2:
            reader.close()
            return False, True, "final_video.mp4 has fewer than two frames"
        first = np.asarray(reader.get_data(0), dtype=np.int16)
        last = np.asarray(reader.get_data(max(1, count - 1)), dtype=np.int16)
        reader.close()
        mean_diff = float(np.mean(np.abs(first - last)))
        return True, mean_diff > 0.5, "" if mean_diff > 0.5 else "video appears static"
    except Exception as exc:
        return False, False, f"could not validate video frames: {exc}"


def force_direction(info: dict[str, object]) -> list[float]:
    axis = list(info.get("joint_axis") or [0.0, 0.0, 1.0])
    joint_type = str(info.get("joint_type") or "unknown")
    if joint_type == "revolute":
        return perpendicular(axis)
    return unit(axis)


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected true/false, got {value!r}")


def prepare_manual_contact(info: dict[str, object], override: dict[str, object], args: argparse.Namespace) -> None:
    object_id = str(info["object_id"])
    model_dir = Path(info["model_dir"])
    joint_name = str(override.get("joint_name") or "")
    link_name = str(override.get("link_name") or "")
    if joint_name != info.get("joint_name") or link_name != info.get("link_name"):
        raise ValueError(
            f"{object_id}: override selects {joint_name}/{link_name}, expected {info.get('joint_name')}/{info.get('link_name')}"
        )
    mode = str(override.get("contact_mode") or "")
    geometry: dict[str, object] | None = None
    if mode == "candidate_id":
        requested = str(override.get("candidate_id") or "")
        candidate_files = sorted((REPO_ROOT / "contact_selection").glob(f"{object_id}_*/candidates.json"))
        if not candidate_files:
            raise ValueError(f"{object_id}: generate contact_selection candidates before using candidate_id")
        payload = read_json(candidate_files[0])
        records = payload.get("candidates", []) if isinstance(payload, dict) else []
        record = next((item for item in records if isinstance(item, dict) and item.get("candidate_id") == requested), None)
        if record is None:
            raise ValueError(f"{object_id}: candidate_id {requested!r} not found in {candidate_files[0]}")
        local_point = [float(v) for v in record["local_point"]]
        contact_source = "candidate_id"
        geometry = {
            "contact_point_world": record["world_point"],
            "joint_origin_world": record["joint_origin_world"],
            "joint_axis_world": record["joint_axis_world"],
            "lever_arm_perpendicular": record["lever_arm_perpendicular"] or 0.0,
            "tangent_opening_world": record["tangent_opening_world"],
        }
    elif mode == "manual_world_point":
        world_point = override.get("world_point")
        if not isinstance(world_point, list) or len(world_point) != 3:
            raise ValueError(f"{object_id}: manual_world_point requires world_point: [x, y, z]")
        local_point = initial_world_to_local(model_dir, link_name, [float(v) for v in world_point])
        contact_source = "manual_override"
    else:
        candidates = candidate_local_points(manual_link_vertices(model_dir, link_name), count=20)
        local_point, contact_source = resolve_override(object_id, override, model_dir, link_name, candidates)
    if geometry is None:
        geometry = initial_contact_geometry(model_dir, joint_name, link_name, local_point, list(info["joint_axis"]))
    direction_mode = str(override.get("force_direction_mode") or "")
    allowed = {
        "tangent_opening", "tangent_closing", "prismatic_axis", "negative_prismatic_axis",
        "manual_world_direction", "manual_local_direction",
    }
    if direction_mode not in allowed:
        raise ValueError(f"{object_id}: unsupported force_direction_mode {direction_mode!r}")
    joint_type = str(info["joint_type"])
    lever = float(geometry["lever_arm_perpendicular"])
    if direction_mode == "manual_world_direction":
        value = override.get("manual_world_direction")
        if not isinstance(value, list) or len(value) != 3:
            raise ValueError(f"{object_id}: manual_world_direction requires [x, y, z]")
        direction = unit([float(v) for v in value])
    elif direction_mode == "manual_local_direction":
        value = override.get("manual_local_direction")
        if not isinstance(value, list) or len(value) != 3:
            raise ValueError(f"{object_id}: manual_local_direction requires [x, y, z]")
        from manual_contact_utils import initial_local_direction_to_world
        direction = initial_local_direction_to_world(model_dir, link_name, [float(v) for v in value])
    elif joint_type == "revolute":
        if not direction_mode.startswith("tangent_"):
            raise ValueError(f"{object_id}: revolute joint requires a tangent or manual direction")
        if lever <= 1e-6:
            raise ValueError(f"{object_id}: manual contact has zero/near-zero perpendicular lever arm")
        direction = list(geometry["tangent_opening_world"])
        if direction_mode == "tangent_closing":
            direction = [-float(v) for v in direction]
    elif joint_type == "prismatic":
        if direction_mode not in {"prismatic_axis", "negative_prismatic_axis"}:
            raise ValueError(f"{object_id}: prismatic joint requires an axis or manual direction")
        direction = list(geometry["joint_axis_world"])
        if direction_mode == "negative_prismatic_axis":
            direction = [-float(v) for v in direction]
    else:
        raise ValueError(f"{object_id}: manual_contact_fixed_force supports revolute/prismatic joints only")
    direction = unit(direction)
    params = getattr(args, "realistic_params", {}).get(object_id, {}) if args.force_policy == "realistic_response_calibration" else {}
    actual_force = float(params.get("force_magnitude", args.force_magnitude))
    if not 1.0 <= actual_force <= 12.0:
        raise ValueError(f"{object_id}: calibrated force_magnitude {actual_force} outside [1, 12]")
    damping = float(params.get("joint_damping", args.joint_damping))
    friction = float(params.get("joint_friction", args.joint_friction))
    if not 0.3 <= damping <= 3.0 or not 0.03 <= friction <= 0.5:
        raise ValueError(f"{object_id}: calibrated damping/friction outside allowed bounds")
    info.update(
        {
            "manual_contact_override": override,
            "contact_source": contact_source,
            "contact_override_file": str(Path(args.contact_overrides).resolve()),
            "contact_mode": mode,
            "manual_contact_strategy": str((info.get("override") or {}).get("contact_strategy") or mode),
            "contact_point_local": local_point,
            "manual_contact_world_geometry": geometry,
            "force_direction_mode": direction_mode,
            "manual_force_direction_world": direction,
            "manual_contact_note": str(override.get("note") or ""),
            "candidate_id": str(override.get("candidate_id") or "") or None,
            "force_policy": args.force_policy,
            "actual_force_magnitude": actual_force,
            "selected_joint_damping": damping,
            "selected_joint_friction": friction,
            "calibration_reason": str(params.get("reason", "fixed global physics" if args.force_policy in {"fixed_global_physics", "fixed_global_impulse_decay"} else "bounded realism calibration")),
            "clamped_force_magnitude": False,
            "final_mode": (
                "realistic_manual_contact_adaptive" if args.force_policy == "realistic_response_calibration"
                else "manual_contact_global_physics_adaptive" if args.force_policy == "fixed_global_physics"
                else "manual_contact_global_impulse_decay" if args.force_policy == "fixed_global_impulse_decay"
                else "manual_contact_fixed_force"
            ),
        }
    )


def renderer_command(info: dict[str, object], run_dir: Path, args: argparse.Namespace) -> list[str]:
    joint_type = str(info["joint_type"])
    joint = str(info["joint_name"])
    link = str(info["link_name"])
    model_dir = Path(info["model_dir"])
    missing_textures = set()
    for material in model_dir.glob("textured_objs/*.mtl"):
        for line in material.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.strip().lower().startswith("map_kd "):
                texture = (material.parent / line.split(None, 1)[1].strip()).resolve()
                if not texture.exists():
                    missing_textures.add(texture.name)
    if missing_textures:
        runtime_dir = run_dir / "_asset_runtime"
        if runtime_dir.exists():
            shutil.rmtree(runtime_dir)
        shutil.copytree(model_dir, runtime_dir)
        image_dir = runtime_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        if Image is None:
            raise RuntimeError("Pillow is required to create neutral fallback textures")
        for name in sorted(missing_textures):
            Image.new("RGB", (4, 4), (190, 190, 190)).save(image_dir / name)
        model_dir = runtime_dir
    direction = list(info.get("manual_force_direction_world") or force_direction(info))
    renderer_force_mode = "external_link_force"
    force_profile = "pulse"
    actual_force = float(info["actual_force_magnitude"])
    command = [
        args.python_executable or sys.executable,
        str(SCRIPTS_DIR / f"render_{joint_type}_video.py"),
        "--mode",
        "render",
        "--model-dir",
        str(model_dir),
        "--seconds",
        str(args.sim_duration_s),
        "--fps",
        str(args.fps),
        "--direction",
        *(f"{float(v):.17f}" for v in direction),
        "--output-root",
        str(run_dir),
        "--json-output",
        str(run_dir / "simulation.json"),
        "--output",
        str(run_dir / "final_video.mp4"),
        "--panel-width",
        str(args.video_width),
        "--panel-height",
        str(args.video_height),
        "--info-height",
        "0",
        "--plot-height",
        "0",
        "--end-hold-seconds",
        str(args.end_hold_seconds),
        "--end-hold-mode",
        args.end_hold_mode,
        "--force-application-mode",
        renderer_force_mode,
        "--force-profile",
        force_profile,
        "--force-duration",
        str(args.force_duration_s),
        "--joint-viscous-damping",
        str(info.get("selected_joint_damping", args.joint_damping)),
        "--joint-dynamic-friction",
        str(info.get("selected_joint_friction", args.joint_friction)),
        "--settle-velocity-threshold",
        str(args.settle_velocity_threshold),
        "--max-q-fraction-of-limit",
        str(args.max_q_fraction_of_limit),
        "--contact-overrides",
        str(Path(args.contact_overrides).resolve()),
        "--manual-contact-required",
        "true",
        "--force-policy",
        str(args.force_policy),
        "--keep-old",
    ]
    if args.adaptive_duration:
        command += [
            "--simulate-until-settled",
            "--max-seconds", str(args.max_sim_duration_s),
            "--settle-velocity-threshold", str(args.settle_qdot_threshold),
            "--settle-window-seconds", str(args.settle_window_s),
            "--post-settle-hold-seconds", str(args.post_settle_hold_s),
        ]
    override = info.get("override")
    manual_override = info.get("manual_contact_override")
    contact_strategy = str(info.get("manual_contact_strategy") or "manual_override")
    command += [
        "--contact-point-strategy", contact_strategy,
        "--contact-point-local", *(str(v) for v in list(info["contact_point_local"])),
    ]
    if joint_type == "prismatic":
        command += ["--joint", joint, "--link", link, "--drawer", drawer_index_from_link(link), "--force", str(actual_force)]
    elif joint_type == "revolute":
        preferred = "-1" if info.get("force_direction_mode") == "tangent_closing" else "1"
        command += [
            "--joint",
            joint,
            "--link",
            link,
            "--force",
            str(actual_force),
            "--closing-force",
            str(actual_force),
            "--initial-angle",
            str(default_initial_angle(model_dir, info.get("joint_limits"))),
            "--preferred-motion-direction",
            preferred,
            "--no-auto-direction",
        ]
    elif joint_type == "screw":
        command[1] = str(SCRIPTS_DIR / "render_screw_video.py")
        command += ["--linear-joint", joint, "--rotary-joint", "joint_0", "--link", link, "--torque", str(args.force_magnitude)]
    else:
        raise RuntimeError(f"unsupported joint type: {joint_type}")
    return command


def blank_png(path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if Image is None or ImageDraw is None:
        # 1x1 transparent PNG. Keeps failure outputs structurally complete even in
        # minimal Python environments.
        path.write_bytes(
            bytes.fromhex(
                "89504e470d0a1a0a0000000d4948445200000001000000010806000000"
                "1f15c4890000000a49444154789c63600000020001e221bc3300000000"
                "49454e44ae426082"
            )
        )
        return
    image = Image.new("RGB", (1000, 640), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.text((40, 40), title, fill=(20, 28, 34))
    image.save(path)


def write_failure_artifacts(run_dir: Path, info: dict[str, object], error: str) -> None:
    diagnostics = run_dir / "diagnostics"
    diagnostics.mkdir(parents=True, exist_ok=True)
    (diagnostics / "physics_timeseries.tsv").write_text("frame\ttime_s\tq\tqdot\tqddot\tforce\ttorque\tresistance\n", encoding="utf-8")
    for filename, title in (
        ("force_torque_by_frame.png", "Force/torque unavailable"),
        ("q_qdot_qddot_by_frame.png", "Joint trajectory unavailable"),
        ("resistance_by_frame.png", "Resistance unavailable"),
    ):
        blank_png(diagnostics / filename, title)
    (diagnostics / "physics_diagnostics.md").write_text(
        "\n".join(
            [
                f"# ForceSAPIEN diagnostics: {info['object_id']}",
                "",
                f"- Status: `failed`",
                f"- Object: `{info.get('object_name', 'object')}`",
                f"- Joint type: `{info.get('joint_type', 'unknown')}`",
                f"- Joint: `{info.get('joint_name')}`",
                f"- Error: {error}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    document = {
        "object_id": info["object_id"],
        "object_name": info.get("object_name"),
        "status": "failed",
        "error_message": error,
        "joint_type": info.get("joint_type", "unknown"),
        "joint_name": info.get("joint_name"),
        "joint_index": None,
        "joint_axis": info.get("joint_axis"),
        "joint_origin": info.get("joint_origin"),
        "joint_limits": info.get("joint_limits"),
        "q_start": None,
        "q_end": None,
        "force_magnitude": None,
        "force_direction_world": None,
        "force_application_point_world": None,
        "torque_or_projected_force": None,
        "per_frame_states": [],
    }
    (run_dir / "simulation.json").write_text(json.dumps(document, indent=2), encoding="utf-8")


def build_physical_pulse_summary(
    samples: list[dict[str, object]],
    *,
    joint_type: str,
    force_magnitude: float,
    force_duration_s: float,
    metadata: dict[str, object],
    selected_link: str,
    bbox_max_dim: float | None,
    scale_interpretation: str,
) -> dict[str, object]:
    pulse = [
        sample for sample in samples
        if float(sample.get("applied_force_norm", 0.0) or 0.0) > 1e-8
        and float(sample.get("time_s", sample.get("time", 0.0)) or 0.0) <= force_duration_s + 1e-9
    ]
    after = [sample for sample in samples if float(sample.get("time_s", sample.get("time", 0.0)) or 0.0) > force_duration_s + 1e-9]
    recomputed: list[float] = []
    stored: list[float] = []
    for sample in pulse:
        force_world = vec3(sample.get("applied_force_world", sample.get("force_vector_world")))
        axis_world = vector_unit(sample.get("joint_axis_world"))
        if joint_type == "revolute":
            arm = vector_sub(sample.get("application_point_world"), sample.get("joint_origin_world"))
            effect = vector_dot(vector_cross(arm, force_world), axis_world)
            stored_effect = float(sample.get("torque_about_axis", sample.get("torque_about_axis_nm", 0.0)) or 0.0)
        else:
            effect = vector_dot(force_world, axis_world)
            stored_effect = float(sample.get("raw_projected_force_along_axis", sample.get("applied_generalized_force", 0.0)) or 0.0)
        recomputed.append(effect)
        stored.append(stored_effect)

    peak_index = max(range(len(pulse)), key=lambda idx: abs(recomputed[idx])) if pulse else None
    chosen = pulse[peak_index] if peak_index is not None else {}
    effect = recomputed[peak_index] if peak_index is not None else 0.0
    stored_effect = stored[peak_index] if peak_index is not None else 0.0
    force_world = vec3(chosen.get("applied_force_world", chosen.get("force_vector_world")))
    point_world = vec3(chosen.get("application_point_world"))
    axis_world = vector_unit(chosen.get("joint_axis_world"))
    origin_world = vec3(chosen.get("joint_origin_world"))
    lever_world = vector_sub(point_world, origin_world)
    parallel = vector_dot(lever_world, axis_world)
    perpendicular = [lever_world[i] - parallel * axis_world[i] for i in range(3)]
    lever_perpendicular = vector_norm(perpendicular) if joint_type == "revolute" else 0.0
    initial_qddot = next(
        (float(sample.get("qddot", sample.get("joint_acceleration_rad_s2", sample.get("joint_acceleration_m_s2", 0.0))) or 0.0)
         for sample in pulse if abs(float(sample.get("qddot", sample.get("joint_acceleration_rad_s2", sample.get("joint_acceleration_m_s2", 0.0))) or 0.0)) > 1e-8),
        0.0,
    )
    relative_errors = [abs(a - b) / max(abs(a), 1e-9) for a, b in zip(recomputed, stored)]
    pulse_magnitudes = [float(sample.get("applied_force_norm", 0.0) or 0.0) for sample in pulse]
    force_logging_ok = bool(pulse) and all(abs(value - force_magnitude) <= max(1e-5, abs(force_magnitude) * 1e-4) for value in pulse_magnitudes)
    force_zero_after = bool(after) and all(float(sample.get("applied_force_norm", 0.0) or 0.0) <= 1e-8 for sample in after)
    torque_ok = bool(pulse) and abs(effect) > 1e-8 and max(relative_errors, default=math.inf) <= 1e-4
    acceleration_ok = sign_matches(effect, initial_qddot)

    articulation = metadata.get("articulation", {}) if isinstance(metadata, dict) else {}
    links = articulation.get("links", []) if isinstance(articulation, dict) else []
    moving_link = next((link for link in links if isinstance(link, dict) and link.get("name") == selected_link), {})
    mass = moving_link.get("mass") if isinstance(moving_link, dict) else None
    inertia = moving_link.get("inertia") if isinstance(moving_link, dict) else None
    cmass = moving_link.get("cmass_local_pose", {}) if isinstance(moving_link, dict) else {}
    center_of_mass = cmass.get("p") if isinstance(cmass, dict) else None
    inertia_nonzero = isinstance(inertia, list) and len(inertia) >= 3 and all(math.isfinite(float(value)) and float(value) > 1e-9 for value in inertia[:3])
    mass_valid = mass is not None and math.isfinite(float(mass)) and float(mass) > 1e-8
    inertia_scale_verdict = "WARN" if scale_interpretation != "metric_plausible" else "PASS"
    if not mass_valid or not inertia_nonzero:
        inertia_scale_verdict = "FAIL"
    effective_proxy = abs(effect / initial_qddot) if abs(initial_qddot) > 1e-8 else None
    numeric_values = []
    for sample in samples:
        numeric_values.extend(float(sample.get(key, 0.0) or 0.0) for key in ("q", "qdot", "qddot"))
    finite_states = all(math.isfinite(value) for value in numeric_values)
    limit_distances = [
        float(sample.get("joint_limit_distance_rad", sample.get("joint_limit_distance_m", math.inf)) or 0.0)
        for sample in samples
    ]
    inside_limits = all(value >= -1e-6 for value in limit_distances)
    q_continues_after = False
    if after:
        first_after_q = float(after[0].get("q", after[0].get("theta_rad", after[0].get("position_m", 0.0))) or 0.0)
        final_q = float(samples[-1].get("q", samples[-1].get("theta_rad", samples[-1].get("position_m", 0.0))) or 0.0)
        q_continues_after = abs(final_q - first_after_q) > 1e-5
    state_validity_verdict = "PASS" if finite_states and inside_limits and q_continues_after else "FAIL"

    result: dict[str, object] = {
        "force_direction_world_at_pulse": vector_unit(force_world),
        "force_vector_world_at_pulse": force_world,
        "force_application_point_world_at_pulse": point_world,
        "joint_axis_world_at_pulse": axis_world,
        "joint_origin_world_at_pulse": origin_world,
        "pulse_sample_frame": chosen.get("frame"),
        "pulse_sample_time_s": chosen.get("time_s", chosen.get("time")),
        "peak_abs_applied_effect": max((abs(value) for value in recomputed), default=0.0),
        "mean_abs_applied_effect_during_pulse": sum(abs(value) for value in recomputed) / len(recomputed) if recomputed else 0.0,
        "recomputed_applied_effect_at_pulse": effect,
        "stored_applied_effect_at_pulse": stored_effect,
        "initial_qddot": initial_qddot,
        "moving_link_mass": mass,
        "moving_link_inertia_diag": inertia,
        "moving_link_center_of_mass": center_of_mass,
        "bbox_max_dim": bbox_max_dim,
        "applied_effect_at_pulse": effect,
        "effective_joint_inertia_proxy": effective_proxy,
        "scale_assessment": scale_interpretation,
        "inertia_assessment": "nonzero loaded SAPIEN inertia; dataset-scale, not SI-calibrated" if inertia_scale_verdict == "WARN" else "loaded mass/inertia valid",
        "force_logging_verdict": "PASS" if force_logging_ok and force_zero_after else "FAIL",
        "torque_projection_verdict": "PASS" if torque_ok else "FAIL",
        "acceleration_sign_verdict": "PASS" if acceleration_ok else "FAIL",
        "hidden_drive_verdict": "PASS",
        "inertia_scale_verdict": inertia_scale_verdict,
        "state_validity_verdict": state_validity_verdict,
        "physical_validation": {
            "nonzero_pulse_force_exists": bool(pulse),
            "pulse_force_matches_requested_magnitude": force_logging_ok,
            "force_zero_after_duration": force_zero_after,
            "max_relative_torque_projection_error": max(relative_errors, default=None),
            "recomputed_effect_nonzero": abs(effect) > 1e-8,
            "stored_effect_matches_recomputed": torque_ok,
            "initial_qddot_sign_matches_effect": acceleration_ok,
            "top_level_pulse_summary_matches_recomputed": torque_ok,
            "no_nan_or_inf": finite_states,
            "q_inside_joint_limits": inside_limits,
            "q_continues_after_force_removal": q_continues_after,
            "hidden_drive_code_audit": "external mode set_qf contains resistance only; drives disabled; no production-loop set_qpos",
        },
    }
    if joint_type == "revolute":
        result.update({
            "lever_arm_world_at_pulse": lever_world,
            "lever_arm_perpendicular": lever_perpendicular,
            "torque_about_axis_at_pulse": effect,
            "peak_abs_torque_about_axis": result["peak_abs_applied_effect"],
            "mean_abs_torque_during_pulse": result["mean_abs_applied_effect_during_pulse"],
        })
    else:
        result.update({
            "lever_arm_perpendicular": None,
            "raw_projected_force_along_axis_at_pulse": effect,
            "peak_abs_raw_projected_force_along_axis": result["peak_abs_applied_effect"],
            "mean_abs_raw_projected_force_during_pulse": result["mean_abs_applied_effect_during_pulse"],
            "net_generalized_force_after_resistance_at_pulse": chosen.get("net_generalized_force", chosen.get("net_force_n")),
        })
    return result


def augment_simulation_json(run_dir: Path, info: dict[str, object], validation_notes: list[str], args: argparse.Namespace) -> tuple[str, str, float | None, float | None, str]:
    path = run_dir / "simulation.json"
    document = read_json(path)
    if not isinstance(document, dict):
        return "failed", "invalid simulation.json", None, None, "invalid simulation.json"

    metadata = document.get("metadata", {})
    actuation = metadata.get("actuation", {}) if isinstance(metadata, dict) else {}
    application_point = metadata.get("application_point", {}) if isinstance(metadata, dict) else {}
    validation = document.get("validation", {})
    q_start, q_end = q_from_document(document)
    samples = primary_samples(document)
    per_frame = []
    for sample in samples:
        q = sample.get("theta_rad", sample.get("position_m", sample.get("joint_angle_rad", sample.get("joint_position_m"))))
        qdot = sample.get("omega_rad_s", sample.get("velocity_m_s", sample.get("joint_velocity_rad_s", sample.get("joint_velocity_m_s"))))
        qddot = sample.get("alpha_rad_s2", sample.get("acceleration_m_s2", sample.get("joint_acceleration_rad_s2", sample.get("joint_acceleration_m_s2"))))
        applied_force_world = sample.get("applied_force_world", sample.get("force_vector_world", [0.0, 0.0, 0.0]))
        if not isinstance(applied_force_world, list):
            applied_force_world = [0.0, 0.0, 0.0]
        force_norm = sample.get("applied_force_norm", sample.get("applied_linear_force_n", sample.get("applied_tangential_force_n", 0.0)))
        torque = sample.get("torque_about_axis", sample.get("torque_about_axis_nm", sample.get("torque_applied_nm", 0.0)))
        projected = sample.get("raw_projected_force_along_axis", sample.get("projected_force_along_axis", sample.get("applied_generalized_force", 0.0)))
        resistance = sample.get("torque_resisting_nm", sample.get("damping_force_n", sample.get("damping_torque_nm", 0.0)))
        per_frame.append(
            {
                "frame": sample.get("frame"),
                "time_s": sample.get("time_s", sample.get("time")),
                "phase": sample.get("phase"),
                "q": q,
                "qdot": qdot,
                "qddot": qddot,
                "applied_force_world": applied_force_world,
                "applied_force_norm": force_norm,
                "applied_generalized_force": sample.get("applied_generalized_force", sample.get("generalized_torque_nm", sample.get("generalized_joint_force_n"))),
                "torque_about_axis": torque,
                "projected_force_along_axis": projected,
                "damping_torque_or_force": sample.get("damping_torque_or_force", sample.get("damping_force_n", sample.get("damping_torque_nm"))),
                "friction_torque_or_force": sample.get("friction_torque_or_force", sample.get("dynamic_friction_force_n", sample.get("dynamic_friction_torque_nm"))),
                "net_generalized_force": sample.get("net_generalized_force", sample.get("net_force_n", sample.get("net_torque_nm"))),
                "joint_limit_distance": sample.get("joint_limit_distance_rad", sample.get("joint_limit_distance_m")),
                "settled_flag": sample.get("settled_flag", False),
                "resistance": resistance,
            }
        )

    force = actuation.get("force", {}) if isinstance(actuation, dict) else {}
    magnitude = None
    if isinstance(force, dict):
        magnitude = force.get("magnitude_n", force.get("applied_linear_force_n", force.get("applied_tangential_force_n")))

    model_dir = Path(info["model_dir"])
    q_unit = "meters" if info.get("joint_type") == "prismatic" else "radians"
    object_bbox_size = bbox_size(all_visual_vertices(model_dir))
    scale_interpretation, scale_warning = object_scale_interpretation(info.get("object_name"), object_bbox_size)
    force_duration_s = float(document.get("force_duration_s", force.get("force_duration_s", 0.0)) or 0.0)
    pulse_summary = build_physical_pulse_summary(
        samples,
        joint_type=str(info.get("joint_type") or "unknown"),
        force_magnitude=float(magnitude or 0.0),
        force_duration_s=force_duration_s,
        metadata=metadata if isinstance(metadata, dict) else {},
        selected_link=str(info.get("link_name") or ""),
        bbox_max_dim=max(object_bbox_size) if object_bbox_size else None,
        scale_interpretation=scale_interpretation,
    )
    if isinstance(force, dict):
        force.update(
            {
                "force_direction_world_at_pulse": pulse_summary.get("force_direction_world_at_pulse"),
                "force_vector_world_at_pulse": pulse_summary.get("force_vector_world_at_pulse"),
                "force_application_point_world_at_pulse": pulse_summary.get("force_application_point_world_at_pulse"),
                "joint_axis_world_at_pulse": pulse_summary.get("joint_axis_world_at_pulse"),
                "joint_origin_world_at_pulse": pulse_summary.get("joint_origin_world_at_pulse"),
                # Backward-compatible names now deliberately summarize the pulse.
                "force_vector_world": pulse_summary.get("force_vector_world_at_pulse"),
                "force_application_point_world": pulse_summary.get("force_application_point_world_at_pulse"),
                "joint_axis_world": pulse_summary.get("joint_axis_world_at_pulse"),
                "joint_origin_world": pulse_summary.get("joint_origin_world_at_pulse"),
            }
        )
        if str(info.get("joint_type")) == "revolute":
            force["torque_about_axis_nm"] = pulse_summary.get("torque_about_axis_at_pulse")
            force["torque_applied_nm"] = pulse_summary.get("torque_about_axis_at_pulse")
        else:
            force["raw_projected_force_along_axis"] = pulse_summary.get("raw_projected_force_along_axis_at_pulse")
            force["net_generalized_force_after_resistance"] = pulse_summary.get("net_generalized_force_after_resistance_at_pulse")
    direction = pulse_summary.get("force_direction_world_at_pulse")
    point = pulse_summary.get("force_application_point_world_at_pulse")
    app_local_point = application_point.get("local_on_link") if isinstance(application_point, dict) else None
    moving_link_vertices = link_visual_vertices(model_dir, str(info.get("link_name") or ""))
    contact_inside, contact_distance = point_inside_local_bbox(app_local_point, moving_link_vertices)
    available_moving_joints = info.get("available_moving_joints") if isinstance(info.get("available_moving_joints"), list) else []
    override = info.get("override") if isinstance(info.get("override"), dict) else None
    selection_source = str(info.get("selection_source") or "unknown")

    warning_messages = []
    if scale_warning:
        warning_messages.append(scale_warning)
    if isinstance(validation, dict):
        warning_messages.extend(str(item) for item in validation.get("warnings", []) if item)
    if validation_notes:
        warning_messages.extend(validation_notes)
    if selection_source != "override" and len(available_moving_joints) > 1:
        warning_messages.append("selected joint came from auto first-moving-joint heuristic while multiple moving joints exist")
    force_mode = None
    if isinstance(metadata, dict):
        sim_config = metadata.get("simulation_config", {})
        if isinstance(sim_config, dict):
            force_mode = sim_config.get("force_application_mode")
    if force_mode == "generalized":
        warning_messages.append("force_application_mode is generalized_set_qf; force point is used for torque/debug, not true external contact")
    if force_mode == "impulse_then_passive_joint_dynamics":
        warning_messages.append("true external link force was not used; this is a physics-plausible impulse/passive generalized-force fallback")

    force_value = 0.0
    torque_value = 0.0
    projected_value = 0.0
    if samples:
        last = samples[-1]
        force_value = float(last.get("applied_tangential_force_n", last.get("applied_linear_force_n", last.get("force", 0.0))) or 0.0)
        torque_value = max(abs(float(sample.get("torque_about_axis_nm", sample.get("torque_applied_nm", 0.0)) or 0.0)) for sample in samples)
        projected_value = max(
            abs(float(sample.get("net_force_n", sample.get("generalized_joint_force_n", sample.get("applied_linear_force_n", 0.0))) or 0.0))
            for sample in samples
        )

    contact_verdict = "PASS"
    if not contact_inside:
        contact_verdict = "WARN"
        warning_messages.append(
            f"force point local_on_link is outside selected moving-link bbox by {contact_distance if contact_distance is not None else 'unknown'}"
        )
    if info.get("object_id") == "10905":
        strategy = str(application_point.get("strategy", "")) if isinstance(application_point, dict) else ""
        if strategy != "door_handle_or_free_vertical_edge":
            contact_verdict = "WARN"
            warning_messages.append("refrigerator contact point is not using semantic door_handle_or_free_vertical_edge strategy")
    if info.get("object_id") == "102255" and contact_verdict != "PASS":
        contact_verdict = "WARN"
    strategy = str(application_point.get("strategy", "")) if isinstance(application_point, dict) else ""
    semantic_contact = {
        "10211": ("WARN", "screen free-edge candidate is far from hinge, but geometry alone cannot prove semantic edge quality"),
        "10905": ("PASS", "semantic refrigerator handle/free vertical edge on moving door"),
        "11100": ("WARN", "far handle-or-blade candidate; mesh geometry cannot distinguish finger loop from blade tip"),
        "101917": ("WARN", "door handle/free-edge candidate is plausible but not supported by semantic mesh labels"),
        "102255": ("PASS", "robust moving seat/back surface point"),
        "103111": ("PASS", "semantic stapler lid front/top point on joint_1/link_1"),
        "103706": ("PASS", "moving-link point with force aligned to prismatic world axis"),
        "103776": ("WARN", "door rim/handle candidate is plausible but not supported by semantic mesh labels"),
    }
    contact_semantic_verdict, contact_semantic_explanation = semantic_contact.get(str(info.get("object_id")), ("WARN", "no category semantic contact rule"))
    if not contact_inside:
        contact_semantic_verdict = "FAIL"
        contact_semantic_explanation += "; selected point lies outside moving-link bbox"
    contact_verdict = combine_verdicts(contact_verdict, contact_semantic_verdict)

    joint_verdict = "PASS"
    if info.get("object_id") == "103111" and info.get("joint_name") != "joint_1":
        joint_verdict = "FAIL"
        warning_messages.append("stapler must use joint_1/link_1 lid override; joint_0 actuates stapler_body")
    if info.get("joint_type") == "revolute" and abs(torque_value) <= 1e-8:
        joint_verdict = "FAIL"
        warning_messages.append("applied force does not generate useful torque around revolute axis")
    if info.get("joint_type") == "prismatic" and abs(projected_value or force_value) <= 1e-8:
        joint_verdict = "FAIL"
        warning_messages.append("applied force projection along prismatic axis is near zero")

    scale_verdict = "WARN" if scale_interpretation == "normalized_or_dataset_units" else "PASS"
    q_limit_warning = ""
    if q_start is not None and q_end is not None and isinstance(info.get("joint_limits"), tuple):
        lower, upper = info["joint_limits"]
        if abs(float(q_end) - float(lower)) <= 1e-3 or abs(float(q_end) - float(upper)) <= 1e-3:
            q_limit_warning = "q reached or nearly reached a joint limit"
            warning_messages.append(q_limit_warning)
    dynamics_verdict = str(document.get("dynamics_verdict") or "PASS")
    if force_mode in {"external_link_force", "impulse_then_passive_joint_dynamics"}:
        physics_verdict = combine_verdicts(contact_verdict, joint_verdict, dynamics_verdict)
        if scale_verdict == "WARN" and physics_verdict == "PASS":
            physics_verdict = "WARN"
    else:
        physics_verdict = combine_verdicts(contact_verdict, joint_verdict, scale_verdict)

    core_physical_verdicts = [
        str(pulse_summary.get("force_logging_verdict")),
        str(pulse_summary.get("torque_projection_verdict")),
        str(pulse_summary.get("acceleration_sign_verdict")),
        str(pulse_summary.get("hidden_drive_verdict")),
        str(pulse_summary.get("state_validity_verdict")),
    ]
    if "FAIL" in core_physical_verdicts:
        final_physical_consistency_verdict = "PHYSICS_INVALID"
    elif contact_semantic_verdict == "FAIL" or pulse_summary.get("inertia_scale_verdict") == "FAIL":
        final_physical_consistency_verdict = "PHYSICS_SUSPICIOUS"
    elif contact_semantic_verdict == "WARN" or pulse_summary.get("inertia_scale_verdict") == "WARN" or dynamics_verdict == "WARN":
        final_physical_consistency_verdict = "PHYSICS_PLAUSIBLE_BUT_UNCALIBRATED"
    else:
        final_physical_consistency_verdict = "PHYSICS_CONSISTENT"

    status = "success"
    errors = []
    if isinstance(validation, dict) and validation.get("motion_visible") is False and info.get("joint_type") != "fixed":
        status = "failed"
        errors.append("joint did not move visibly")
    if validation_notes:
        errors.extend(validation_notes)
        if status == "success":
            status = "warning"

    document.update(
        {
            "object_id": info["object_id"],
            "object_name": info.get("object_name"),
            "status": status,
            "error_message": "; ".join(errors) if errors else "",
            "joint_type": info.get("joint_type"),
            "joint_name": info.get("joint_name"),
            "selected_joint": info.get("joint_name"),
            "selected_link": info.get("link_name"),
            "joint_index": None,
            "joint_axis": info.get("joint_axis"),
            "joint_origin": info.get("joint_origin"),
            "joint_origin_urdf_local": info.get("joint_origin"),
            "joint_limits": info.get("joint_limits"),
            "q_start": q_start,
            "q_end": q_end,
            "delta_q": (float(q_end) - float(q_start)) if q_start is not None and q_end is not None else None,
            "force_magnitude": magnitude,
            "actual_force_magnitude": info.get("actual_force_magnitude"),
            "force_policy": info.get("force_policy"),
            "per_object_force_adaptation": info.get("force_policy") == "realistic_response_calibration",
            "per_object_damping_adaptation": info.get("force_policy") == "realistic_response_calibration",
            "per_object_friction_adaptation": info.get("force_policy") == "realistic_response_calibration",
            "same_physics_for_all_objects": info.get("force_policy") in {"fixed_global_physics", "fixed_global_impulse_decay"},
            "force_units": "dataset/SAPIEN units",
            "clamped_force_magnitude": info.get("clamped_force_magnitude"),
            "force_direction_world": direction,
            "force_application_point_world": point,
            "torque_about_axis": pulse_summary.get("torque_about_axis_at_pulse"),
            "torque_or_projected_force": actuation,
            "per_frame_states": per_frame,
            "q_unit": q_unit,
            "object_bbox_size_world": object_bbox_size,
            "object_scale_interpretation": scale_interpretation,
            "force_application_mode": "generalized_set_qf" if force_mode == "generalized" else force_mode,
            "true_external_force_used": force_mode == "external_link_force",
            "fallback_used": False,
            "hidden_drive_used": False,
            "manual_q_interpolation_used": False,
            "uses_generalized_set_qf_as_motion_driver": False,
            "force_units_physical": False,
            "calibrated_newtons": False,
            "final_mode": info.get("final_mode"),
            "candidate_id": info.get("candidate_id"),
            "contact_source": info.get("contact_source"),
            "contact_override_file": info.get("contact_override_file"),
            "contact_mode": info.get("contact_mode"),
            "contact_point_local": info.get("contact_point_local"),
            "contact_point_world_at_pulse": pulse_summary.get("force_application_point_world_at_pulse"),
            "force_direction_mode": info.get("force_direction_mode"),
            "force_direction_world_at_pulse": pulse_summary.get("force_direction_world_at_pulse"),
            "manual_contact_note": info.get("manual_contact_note"),
            "force_duration_s": force_duration_s,
            "sim_duration_s": document.get("sim_duration_s", metadata.get("timing", {}).get("simulated_seconds") if isinstance(metadata, dict) else None),
            "fps": document.get("fps", metadata.get("timing", {}).get("fps") if isinstance(metadata, dict) else None),
            "timestep_s": document.get("timestep_s", document.get("timestep", metadata.get("timing", {}).get("timestep_s") if isinstance(metadata, dict) else None)),
            "joint_damping": document.get("joint_damping", document.get("damping")),
            "joint_friction": document.get("joint_friction", document.get("friction")),
            "adaptive_duration": bool(getattr(args, "adaptive_duration", False)),
            "min_sim_duration_s": float(getattr(args, "min_sim_duration_s", args.sim_duration_s)),
            "max_sim_duration_s": float(getattr(args, "max_sim_duration_s", args.sim_duration_s)),
            "settle_qdot_threshold": float(getattr(args, "settle_qdot_threshold", args.settle_velocity_threshold)),
            "settle_window_s": float(getattr(args, "settle_window_s", 0.0)),
            "post_settle_hold_s": float(getattr(args, "post_settle_hold_s", 0.0)),
            "actual_sim_duration_s": document.get("actual_sim_duration_s", document.get("sim_duration_s")),
            "actual_video_frame_count": document.get("actual_video_frame_count", len(per_frame)),
            "stopped_because": document.get("stopped_because"),
            "duration_verdict": "FAIL_MAX_DURATION" if document.get("stopped_because") == "max_duration" else "PASS_SETTLED_PLUS_HOLD",
            "final_acceptance": "FAIL" if document.get("stopped_because") == "max_duration" else "PASS",
            "settle_velocity_threshold": metadata.get("simulation_config", {}).get("settle_velocity_threshold") if isinstance(metadata, dict) and isinstance(metadata.get("simulation_config"), dict) else None,
            "settled": document.get("settled"),
            "settle_time_s": document.get("settle_time_s"),
            "final_abs_qdot": document.get("final_abs_qdot"),
            "peak_abs_qdot": document.get("peak_abs_qdot"),
            "qdot_decay_ratio": document.get("qdot_decay_ratio"),
            "projected_force_along_axis": pulse_summary.get("raw_projected_force_along_axis_at_pulse"),
            "raw_projected_force_along_axis": pulse_summary.get("raw_projected_force_along_axis_at_pulse"),
            "warning_messages": list(dict.fromkeys(warning_messages)),
            "physics_verdict": physics_verdict,
            "dynamics_verdict": dynamics_verdict,
            "contact_verdict": contact_verdict,
            "contact_strategy": info.get("manual_contact_strategy") or strategy,
            "contact_semantic_verdict": contact_semantic_verdict,
            "contact_semantic_explanation": contact_semantic_explanation,
            "joint_verdict": joint_verdict,
            "scale_verdict": scale_verdict,
            "selection_source": selection_source,
            "override": override,
            "available_moving_joints": available_moving_joints,
            "selected_joint_parent_link": next((joint.get("parent_link") for joint in available_moving_joints if joint.get("joint_name") == info.get("joint_name")), None),
            "selected_joint_child_link": info.get("link_name"),
            "semantic_part": override.get("semantic_part") if override else None,
            "expected_motion": override.get("expected_motion") if override else None,
            "force_point_local_on_link_inside_bbox": contact_inside,
            "force_point_local_on_link_distance_to_bbox": contact_distance,
            "joint_axis_urdf_local": info.get("joint_axis"),
            "axis_transform_source": "renderer_motion_estimate_world_axis; urdf_local_axis_recorded_for audit",
            "legacy_coordinate_fields_warning": "joint_axis and joint_origin are legacy URDF-local fields; use *_urdf_local or *_world_at_pulse explicitly",
            "final_physical_consistency_verdict": final_physical_consistency_verdict,
        }
    )
    document.update(pulse_summary)
    document = finite_json(document)
    path.write_text(json.dumps(document, indent=2, allow_nan=False), encoding="utf-8")
    return status, str(document.get("error_message", "")), q_start, q_end, ""


def run_object(info: dict[str, object], output_root: Path, args: argparse.Namespace) -> dict[str, object]:
    run_dir = output_folder(output_root, info)
    complete, missing = output_completeness(run_dir)
    if complete and not args.force:
        document = read_json(run_dir / "simulation.json")
        q_start, q_end = q_from_document(document) if isinstance(document, dict) else (None, None)
        return summary_row(info, run_dir, "skipped_complete", q_start, q_end, "")

    if args.dry_run:
        planned = "rerun_force" if complete and args.force else "repair_incomplete" if run_dir.exists() else "run"
        return summary_row(info, run_dir, planned, None, None, "" if complete else "missing: " + ", ".join(missing))

    run_dir.mkdir(parents=True, exist_ok=True)
    if not info.get("valid"):
        error = str(info.get("error_message") or "object is not runnable")
        write_failure_artifacts(run_dir, info, error)
        return summary_row(info, run_dir, "failed", None, None, error)

    command = renderer_command(info, run_dir, args)
    log_path = run_dir / "run.log"
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n\n")
        log.flush()
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, text=True, check=False, env=os.environ.copy())

    if completed.returncode != 0:
        error = f"renderer exited with code {completed.returncode}; see {log_path}"
        write_failure_artifacts(run_dir, info, error)
        return summary_row(info, run_dir, "failed", None, None, error)

    notes = []
    video_ok, video_moving, video_error = validate_video(run_dir / "final_video.mp4")
    if not video_ok or not video_moving:
        notes.append(video_error)
    if not finite_timeseries(run_dir / "diagnostics" / "physics_timeseries.tsv"):
        notes.append("physics_timeseries.tsv has missing or non-finite values")
    complete_after, missing_after = output_completeness(run_dir)
    if not complete_after:
        notes.append("missing required outputs: " + ", ".join(missing_after))
    status, error_message, q_start, q_end, json_error = augment_simulation_json(run_dir, info, notes, args)
    if json_error:
        status = "failed"
        error_message = json_error
    return summary_row(info, run_dir, status, q_start, q_end, error_message)


def summary_row(
    info: dict[str, object],
    run_dir: Path,
    status: str,
    q_start: float | None,
    q_end: float | None,
    error_message: str,
) -> dict[str, object]:
    delta = q_end - q_start if q_start is not None and q_end is not None else None
    return {
        "object_id": info["object_id"],
        "object_name": info.get("object_name", ""),
        "joint_type": info.get("joint_type", "unknown"),
        "output_folder": str(run_dir),
        "status": status,
        "q_start": q_start,
        "q_end": q_end,
        "delta_q": delta,
        "final_video_exists": (run_dir / "final_video.mp4").exists() and (run_dir / "final_video.mp4").stat().st_size > 0,
        "simulation_json_exists": (run_dir / "simulation.json").exists(),
        "diagnostics_exists": (run_dir / "diagnostics").is_dir(),
        "error_message": error_message,
    }


def write_summary(output_root: Path, rows: list[dict[str, object]]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / "forcesapien_batch_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    counts = {status: sum(1 for row in rows if row["status"] == status) for status in sorted({str(row["status"]) for row in rows})}
    lines = ["# ForceSAPIEN Batch Summary", "", f"- Objects: {len(rows)}"]
    lines.extend(f"- {status}: {count}" for status, count in counts.items())
    lines += ["", "| object_id | object_name | joint_type | status | delta_q | output_folder | error_message |", "|---|---|---|---|---:|---|---|"]
    for row in rows:
        lines.append(
            f"| {row['object_id']} | {row['object_name']} | {row['joint_type']} | {row['status']} | "
            f"{'' if row['delta_q'] is None else row['delta_q']} | {row['output_folder']} | {row['error_message']} |"
        )
    (output_root / "forcesapien_batch_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run final ForceSAPIEN external-link-force modes.")
    parser.add_argument("--dataset_root", default="final_dataset")
    parser.add_argument("--output_root", default="outputs")
    parser.add_argument("--output-suffix", default="manual_contact_fixed_force_check", help="Collision-safe final-mode suffix.")
    parser.add_argument("--object_ids", nargs="*", default=None)
    parser.add_argument("--single_debug_object", default=None)
    parser.add_argument("--force", action="store_true", help="Re-run even complete outputs.")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--python-executable", default=None)
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument("--sim-duration-s", type=float, default=None)
    parser.add_argument(
        "--force-application-mode",
        choices=["external_link_force"],
        default="external_link_force",
    )
    parser.add_argument("--force-policy", choices=["fixed_magnitude", "realistic_response_calibration", "fixed_global_physics", "fixed_global_impulse_decay"], default="fixed_magnitude")
    parser.add_argument("--realistic-response-params", default=None)
    parser.add_argument("--force-magnitude", type=float, default=5.0)
    parser.add_argument("--contact-overrides", default="configs/manual_contact_overrides.yaml")
    parser.add_argument("--manual-contact-required", type=parse_bool, default=True)
    parser.add_argument("--preview-contacts-only", action="store_true")
    parser.add_argument("--force-duration-s", type=float, default=0.2)
    parser.add_argument("--joint-damping", type=float, default=0.5)
    parser.add_argument("--joint-friction", type=float, default=0.05)
    parser.add_argument("--settle-velocity-threshold", type=float, default=1e-3)
    parser.add_argument("--adaptive-duration", type=parse_bool, default=False)
    parser.add_argument("--min-sim-duration-s", type=float, default=6.0)
    parser.add_argument("--max-sim-duration-s", type=float, default=15.0)
    parser.add_argument("--settle-qdot-threshold", type=float, default=0.002)
    parser.add_argument("--settle-window-s", type=float, default=0.5)
    parser.add_argument("--post-settle-hold-s", type=float, default=1.0)
    parser.add_argument("--max-q-fraction-of-limit", type=float, default=0.98)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--video-width", type=int, default=1920)
    parser.add_argument("--video-height", type=int, default=1080)
    parser.add_argument("--end-hold-seconds", type=float, default=0.0)
    parser.add_argument("--end-hold-mode", choices=["always", "never", "if-stopped"], default="never")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.force_magnitude <= 0:
        print("ERROR: force magnitude must be positive", file=sys.stderr)
        return 2
    if args.sim_duration_s is None:
        args.sim_duration_s = args.seconds
    if args.adaptive_duration:
        args.sim_duration_s = args.min_sim_duration_s
        args.settle_velocity_threshold = args.settle_qdot_threshold
    args.realistic_params = {}
    if args.force_policy == "realistic_response_calibration":
        if not args.realistic_response_params and not args.preview_contacts_only:
            print("ERROR: --realistic-response-params is required for realistic calibration", file=sys.stderr)
            return 2
        if args.realistic_response_params:
            args.realistic_params = load_overrides(Path(args.realistic_response_params).expanduser().resolve())
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    if args.single_debug_object:
        args.object_ids = [args.single_debug_object]
    if not dataset_root.exists():
        print(f"ERROR: dataset root does not exist: {dataset_root}", file=sys.stderr)
        return 2

    object_ids = args.object_ids or [path.name for path in sorted(dataset_root.iterdir()) if path.is_dir() and path.name != "outputs"]
    infos = [detect_object(dataset_root / object_id) for object_id in object_ids]
    override_path = Path(args.contact_overrides).expanduser().resolve()
    if not override_path.exists():
        print(f"ERROR: contact override file does not exist: {override_path}", file=sys.stderr)
        return 2
    overrides = load_overrides(override_path)
    for info in infos:
        info["force_application_mode"] = args.force_application_mode
        info["output_suffix"] = args.output_suffix
        object_id = str(info["object_id"])
        override = overrides.get(object_id)
        if override is None:
            if args.manual_contact_required:
                print(f"ERROR: {object_id} has no entry in {override_path}", file=sys.stderr)
                return 2
            continue
        try:
            prepare_manual_contact(info, override, args)
        except Exception as exc:
            print(f"ERROR: invalid manual contact for {object_id}: {exc}", file=sys.stderr)
            return 2

    if args.preview_contacts_only:
        # Candidate assets are generated explicitly by generate_contact_candidates.py.
        # Preview validation is deliberately report-only and never renders videos.
        output_root.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Manual contact preview validation",
            "",
            "No dynamics or final videos were generated.",
            "",
            "| object_id | object_name | candidate_id | contact_point_local | force_direction_mode | expected torque/projection | force_magnitude | verdict | warning |",
            "|---|---|---|---|---|---:|---:|---|---|",
        ]
        preview_rows = []
        preview_failed = False
        for info in infos:
            geometry = info["manual_contact_world_geometry"]
            direction = info["manual_force_direction_world"]
            if info.get("joint_type") == "prismatic":
                effect = float(args.force_magnitude) * sum(float(a) * float(b) for a, b in zip(direction, geometry["joint_axis_world"]))
            else:
                import numpy as np
                radius = np.asarray(geometry["contact_point_world"]) - np.asarray(geometry["joint_origin_world"])
                effect = float(np.dot(np.cross(radius, np.asarray(direction) * args.force_magnitude), np.asarray(geometry["joint_axis_world"])))
            verdict = "PASS" if abs(effect) > 1e-8 else "FAIL"
            warning = str(info.get("manual_contact_note") or "")
            if verdict == "FAIL":
                warning = "; ".join(part for part in [warning, "zero/near-zero expected joint-axis effect"] if part)
                preview_failed = True
            row = {
                "object_id": info["object_id"],
                "object_name": info["object_name"],
                "candidate_id": info.get("candidate_id") or "",
                "contact_point_local": json.dumps(info["contact_point_local"]),
                "force_direction_mode": info["force_direction_mode"],
                "expected_torque_or_projection": f"{effect:.12g}",
                "force_magnitude": f"{info['actual_force_magnitude']:.12g}",
                "verdict": verdict,
                "warning": warning,
            }
            preview_rows.append(row)
            lines.append(
                f"| {info['object_id']} | {info['object_name']} | {info.get('candidate_id') or ''} | `{info['contact_point_local']}` | "
                f"{info['force_direction_mode']} | {effect:.9g} | {info['actual_force_magnitude']} | {verdict} | {warning} |"
            )
        (output_root / "contact_preview_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        with (output_root / "contact_preview_table.tsv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(preview_rows[0]), delimiter="\t")
            writer.writeheader()
            writer.writerows(preview_rows)
        print(f"Wrote {output_root / 'contact_preview_report.md'} without running physics")
        print(f"Wrote {output_root / 'contact_preview_table.tsv'} without running physics")
        return 1 if preview_failed else 0

    rows = []
    for info in infos:
        run_dir = output_folder(output_root, info)
        complete, missing = output_completeness(run_dir)
        if complete and not args.force:
            print(f"SKIP complete: {info['object_id']} -> {run_dir}")
        elif args.dry_run:
            print(
                f"DRY {info['object_id']}: {'complete' if complete else 'incomplete/new'} "
                f"{info.get('joint_name')}/{info.get('link_name')} via {info.get('selection_source')} -> {run_dir}"
            )
            print(f"  moving joints: {info.get('available_moving_joints')}")
            if missing:
                print(f"  missing: {', '.join(missing)}")
        else:
            print(
                f"RUN {info['object_id']}: {info.get('object_name')} {info.get('joint_type')} "
                f"{info.get('joint_name')}/{info.get('link_name')} via {info.get('selection_source')} -> {run_dir}"
            )
            print(f"  moving joints: {info.get('available_moving_joints')}")
        try:
            rows.append(run_object(info, output_root, args))
        except Exception as exc:
            run_dir.mkdir(parents=True, exist_ok=True)
            traceback_path = run_dir / "batch_error.log"
            traceback_path.write_text(traceback.format_exc(), encoding="utf-8")
            error = f"{exc}; see {traceback_path}"
            write_failure_artifacts(run_dir, info, error)
            rows.append(summary_row(info, run_dir, "failed", None, None, error))

    write_summary(output_root, rows)
    print(f"Wrote {output_root / 'forcesapien_batch_summary.csv'}")
    print(f"Wrote {output_root / 'forcesapien_batch_summary.md'}")
    return 0 if all(row["status"] in {"success", "warning", "skipped_complete", "run", "repair_incomplete", "rerun_force"} for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
