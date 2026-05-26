"""Helpers for consistent simulation.json metadata."""

from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import sapien


SCHEMA_VERSION = 3


def sample_time_from_frame(frame_index: int, steps_per_frame: int, timestep_s: float) -> float:
    return float((frame_index + 1) * steps_per_frame * timestep_s)


def sample_time_from_step(step_index: int, timestep_s: float) -> float:
    return float((step_index + 1) * timestep_s)


def urdf_joint_dynamics(model_dir: Path) -> dict[str, dict[str, str]]:
    tree = ET.parse(model_dir / "mobility.urdf")
    result = {}
    for joint in tree.findall("joint"):
        dynamics = joint.find("dynamics")
        if dynamics is not None:
            result[joint.attrib.get("name", "")] = dict(dynamics.attrib)
    return result


def pose_to_dict(pose: sapien.Pose) -> dict[str, list[float]]:
    return {"p": np.asarray(pose.p, dtype=float).tolist(), "q": np.asarray(pose.q, dtype=float).tolist()}


def optional_float(value_fn) -> float | None:
    try:
        return float(value_fn())
    except RuntimeError:
        return None


def optional_array(value_fn) -> list[float] | list[list[float]] | None:
    try:
        return np.asarray(value_fn(), dtype=float).tolist()
    except RuntimeError:
        return None


def optional_string(value_fn) -> str | None:
    try:
        return str(value_fn())
    except RuntimeError:
        return None


def link_dynamics_to_dict(link: sapien.physx.PhysxArticulationLinkComponent) -> dict[str, object]:
    return {
        "name": link.name,
        "mass": float(link.mass),
        "inertia": np.asarray(link.inertia, dtype=float).tolist(),
        "cmass_local_pose": pose_to_dict(link.cmass_local_pose),
        "linear_damping": float(link.linear_damping),
        "angular_damping": float(link.angular_damping),
        "disable_gravity": bool(link.disable_gravity),
    }


def joint_to_dict(joint: sapien.physx.PhysxArticulationJoint, limit_key: str) -> dict[str, object]:
    return {
        "name": joint.name,
        limit_key: optional_array(joint.get_limit),
        "friction": optional_float(lambda: joint.friction),
        "damping": optional_float(lambda: joint.damping),
        "drive_mode": optional_string(lambda: joint.drive_mode),
        "drive_target": optional_array(lambda: joint.drive_target),
        "drive_velocity_target": optional_array(lambda: joint.drive_velocity_target),
        "force_limit": optional_float(lambda: joint.force_limit),
    }


def articulation_to_dict(
    articulation: sapien.physx.PhysxArticulation,
    *,
    limit_key: str,
) -> dict[str, list[dict[str, object]]]:
    return {
        "links": [link_dynamics_to_dict(link) for link in articulation.get_links()],
        "joints": [joint_to_dict(joint, limit_key) for joint in articulation.get_joints() if joint.name],
    }


def physics_to_dict(model_dir: Path, linear_damping: float, angular_damping: float) -> dict[str, object]:
    urdf_dynamics = urdf_joint_dynamics(model_dir)
    return {
        "urdf_joint_dynamics": urdf_dynamics or None,
        "uses_separate_static_dynamic_friction": False,
        "uses_air_friction_model": False,
        "overrides": {
            "link_linear_damping": linear_damping,
            "link_angular_damping": angular_damping,
            "joint_drive": {
                "stiffness": 0.0,
                "damping": 0.0,
                "force_limit": 0.0,
            },
        },
    }


def series_summary(
    samples: list[dict[str, object]],
    *,
    position_key: str,
    velocity_key: str,
    secondary_position_key: str | None = None,
    initial_position_value: float | None = None,
    initial_secondary_position_value: float | None = None,
) -> dict[str, object]:
    if not samples:
        return {"sample_count": 0}

    initial_position = float(initial_position_value) if initial_position_value is not None else float(samples[0][position_key])
    final_position = float(samples[-1][position_key])
    max_velocity_index = max(range(len(samples)), key=lambda idx: abs(float(samples[idx][velocity_key])))
    max_abs_velocity = abs(float(samples[max_velocity_index][velocity_key]))
    summary: dict[str, object] = {
        "sample_count": len(samples),
        f"initial_{position_key}": initial_position,
        f"final_{position_key}": final_position,
        f"delta_{position_key}": final_position - initial_position,
        f"max_abs_{velocity_key}": max_abs_velocity,
        "time_of_max_abs_joint_velocity_s": float(samples[max_velocity_index]["time_s"]),
    }
    if secondary_position_key is not None:
        initial_secondary_position = (
            float(initial_secondary_position_value)
            if initial_secondary_position_value is not None
            else float(samples[0][secondary_position_key])
        )
        final_secondary_position = float(samples[-1][secondary_position_key])
        summary[f"initial_{secondary_position_key}"] = initial_secondary_position
        summary[f"final_{secondary_position_key}"] = final_secondary_position
        summary[f"delta_{secondary_position_key}"] = final_secondary_position - initial_secondary_position
    return summary


def build_summary(
    *,
    sample_series: dict[str, list[dict[str, object]]],
    physics_step_count: int,
    position_key: str,
    velocity_key: str,
    secondary_position_key: str | None = None,
    initial_position_value: float | None = None,
    initial_secondary_position_value: float | None = None,
) -> dict[str, object]:
    return {
        "physics_step_count": physics_step_count,
        "total_sample_count": sum(len(series) for series in sample_series.values()),
        "sample_series": {
            name: series_summary(
                samples,
                position_key=position_key,
                velocity_key=velocity_key,
                secondary_position_key=secondary_position_key,
                initial_position_value=initial_position_value,
                initial_secondary_position_value=initial_secondary_position_value,
            )
            for name, samples in sample_series.items()
        },
    }


def build_metadata(
    *,
    model_dir: Path,
    mode: str,
    joint_type: str,
    joint_name: str,
    link_name: str,
    json_output: Path,
    fps: int,
    requested_seconds: float,
    simulated_seconds: float,
    timestep_s: float,
    sample_interval_s: float,
    actuation: dict[str, object],
    application_point: dict[str, object],
    summary: dict[str, object],
    articulation: sapien.physx.PhysxArticulation,
    limit_key: str,
    linear_damping: float,
    angular_damping: float,
    video_output: Path | None = None,
    end_hold_seconds: float | None = None,
    drawer_index: int | None = None,
) -> dict[str, object]:
    output = {"json_output": str(json_output)}
    if video_output is not None:
        output["video_output"] = str(video_output)

    timing = {
        "fps": fps,
        "requested_seconds": requested_seconds,
        "simulated_seconds": simulated_seconds,
        "sample_interval_s": sample_interval_s,
        "timestep_s": timestep_s,
    }
    if end_hold_seconds is not None:
        timing["end_hold_seconds"] = end_hold_seconds
        timing["video_duration_seconds"] = simulated_seconds + end_hold_seconds

    simulated_object: dict[str, object] = {
        "model_dir": str(model_dir),
        "joint_type": joint_type,
        "joint": joint_name,
        "link": link_name,
    }
    if drawer_index is not None:
        simulated_object["drawer"] = drawer_index

    return {
        "schema_version": SCHEMA_VERSION,
        "pipeline": {"mode": mode},
        "object": simulated_object,
        "output": output,
        "timing": timing,
        "actuation": actuation,
        "application_point": application_point,
        "summary": summary,
        "physics": physics_to_dict(model_dir, linear_damping, angular_damping),
        "articulation": articulation_to_dict(articulation, limit_key=limit_key),
    }
