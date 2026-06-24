#!/usr/bin/env python3
"""Generic force-profile, generalized-force, and resistance helpers.

The functions in this module are intentionally independent of SAPIEN scenes and
dataset object IDs. Renderers pass world-space vectors and joint state in; the
helpers return scalar generalized forces/torques and diagnostic terms.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass
class ForceStep:
    scale: float
    applied_magnitude: float


@dataclass
class Resistance:
    static: float
    dynamic: float
    viscous: float
    total: float
    net: float
    static_engaged: bool


def unit(vector: np.ndarray) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(array))
    if norm <= 1e-12:
        return np.zeros_like(array, dtype=np.float32)
    return (array / norm).astype(np.float32)


def compute_force_profile_scale(
    profile: str,
    time_s: float,
    *,
    start_time_s: float = 0.0,
    duration_s: float = math.inf,
    ramp_time_s: float = 0.0,
) -> float:
    """Return force multiplier for constant, pulse, or ramp/hold/release profiles."""
    if time_s < start_time_s:
        return 0.0
    elapsed = time_s - start_time_s
    if profile == "constant":
        return 1.0
    if elapsed >= duration_s:
        return 0.0
    if profile == "pulse":
        return 1.0
    if profile != "ramp_hold_release":
        raise ValueError(f"Unsupported force profile: {profile}")
    ramp = max(0.0, float(ramp_time_s))
    if ramp <= 1e-9:
        return 1.0
    release_start = max(0.0, duration_s - ramp)
    if elapsed < ramp:
        return elapsed / ramp
    if elapsed > release_start:
        return max(0.0, (duration_s - elapsed) / ramp)
    return 1.0


def scaled_force_step(
    magnitude: float,
    profile: str,
    time_s: float,
    *,
    start_time_s: float = 0.0,
    duration_s: float = math.inf,
    ramp_time_s: float = 0.0,
) -> ForceStep:
    scale = compute_force_profile_scale(
        profile,
        time_s,
        start_time_s=start_time_s,
        duration_s=duration_s,
        ramp_time_s=ramp_time_s,
    )
    return ForceStep(scale=scale, applied_magnitude=float(magnitude) * scale)


def compute_resistance(
    applied_generalized: float,
    velocity: float,
    *,
    static_friction: float = 0.0,
    dynamic_friction: float = 0.0,
    viscous_damping: float = 0.0,
    static_velocity_threshold: float = 1e-4,
) -> Resistance:
    """Compute generalized resistance opposing joint motion.

    Units are whatever the joint uses: Nm for revolute, N for prismatic.
    Static friction cancels sub-threshold applied generalized force near rest;
    dynamic friction and viscous damping oppose velocity.
    """
    static = max(0.0, float(static_friction))
    dynamic = max(0.0, float(dynamic_friction))
    viscous = max(0.0, float(viscous_damping)) * float(velocity)
    near_rest = abs(float(velocity)) <= float(static_velocity_threshold)
    applied = float(applied_generalized)
    if near_rest and abs(applied) <= static:
        return Resistance(
            static=-applied,
            dynamic=0.0,
            viscous=0.0,
            total=-applied,
            net=0.0,
            static_engaged=True,
        )

    direction_basis = float(velocity) if not near_rest else applied
    direction = math.copysign(1.0, direction_basis) if abs(direction_basis) > 1e-12 else 0.0
    static_term = -direction * static if near_rest and static > 0.0 else 0.0
    dynamic_term = -direction * dynamic if direction else 0.0
    viscous_term = -viscous
    total = static_term + dynamic_term + viscous_term
    return Resistance(
        static=static_term,
        dynamic=dynamic_term,
        viscous=viscous_term,
        total=total,
        net=applied + total,
        static_engaged=False,
    )


def compute_revolute_generalized_torque(
    joint_axis_world: np.ndarray,
    joint_origin_world: np.ndarray,
    contact_point_world: np.ndarray,
    force_vector_world: np.ndarray,
) -> float:
    """Compute tau = axis . ((contact - origin) x force)."""
    axis = unit(joint_axis_world)
    radius = np.asarray(contact_point_world, dtype=np.float32) - np.asarray(joint_origin_world, dtype=np.float32)
    force = np.asarray(force_vector_world, dtype=np.float32)
    return float(np.dot(axis, np.cross(radius, force)))


def compute_prismatic_generalized_force(joint_axis_world: np.ndarray, force_vector_world: np.ndarray) -> float:
    """Compute generalized prismatic force as axis . force."""
    return float(np.dot(unit(joint_axis_world), np.asarray(force_vector_world, dtype=np.float32)))


def compute_screw_generalized_force(
    joint_axis_world: np.ndarray,
    force_vector_world: np.ndarray,
    *,
    pitch_m_per_rad: float = 0.0,
    torque_about_axis_nm: float = 0.0,
) -> float:
    """Approximate screw generalized effort.

    This combines axial force and torque through a simple pitch coupling. It is
    useful for diagnostics but is not a contact-resolved thread model.
    """
    axial_force = compute_prismatic_generalized_force(joint_axis_world, force_vector_world)
    return axial_force * float(pitch_m_per_rad) + float(torque_about_axis_nm)


def compute_generalized_force(
    joint_type: str,
    joint_axis_world: np.ndarray,
    force_vector_world: np.ndarray,
    *,
    joint_origin_world: np.ndarray | None = None,
    contact_point_world: np.ndarray | None = None,
    pitch_m_per_rad: float = 0.0,
    torque_about_axis_nm: float = 0.0,
) -> float:
    if joint_type == "revolute":
        if joint_origin_world is None or contact_point_world is None:
            raise ValueError("revolute generalized force requires joint_origin_world and contact_point_world")
        return compute_revolute_generalized_torque(
            joint_axis_world,
            joint_origin_world,
            contact_point_world,
            force_vector_world,
        )
    if joint_type == "prismatic":
        return compute_prismatic_generalized_force(joint_axis_world, force_vector_world)
    if joint_type == "screw":
        return compute_screw_generalized_force(
            joint_axis_world,
            force_vector_world,
            pitch_m_per_rad=pitch_m_per_rad,
            torque_about_axis_nm=torque_about_axis_nm,
        )
    raise ValueError(f"Unsupported joint type: {joint_type}")
