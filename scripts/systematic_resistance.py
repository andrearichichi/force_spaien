#!/usr/bin/env python3
"""Shared joint-space passive resistance calibration for ForceSAPIEN."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np


@dataclass(frozen=True)
class EffectiveCoordinateCalibration:
    joint_type: str
    joint_index: int
    diagnostic_generalized_force: float
    diagnostic_force_magnitude: float
    production_generalized_force_sign: int
    initial_joint_position: float
    joint_lower_limit: float
    joint_upper_limit: float
    available_positive_travel: float
    available_negative_travel: float
    selected_diagnostic_direction: int
    limit_aware_direction_fallback_activated: bool
    analytical_effective_inertia_or_mass: float
    measured_effective_inertia_or_mass: float
    analytical_measured_relative_error: float
    selected_effective_inertia_or_mass: float
    selected_effective_value_source: str
    diagnostic_initial_acceleration: float
    diagnostic_timestep_s: float
    validation_tolerance_fraction: float


@dataclass(frozen=True)
class SystematicResistance:
    decay_time_s: float
    friction_ratio: float
    effective_inertia_or_mass: float
    damping_coefficient: float
    friction_magnitude: float
    generalized_external_force_magnitude: float
    damping_clamp_activated: bool
    friction_clamp_activated: bool
    damping_bounds: tuple[float, float]
    friction_bounds: tuple[float, float]


def analytical_effective_coordinate_inertia(articulation: Any, joint_index: int) -> float:
    """Return 1/(M^-1)_ii, including all free-coordinate coupling."""
    qpos = np.asarray(articulation.get_qpos(), dtype=np.float64)
    model = articulation.create_pinocchio_model()
    mass_matrix = np.asarray(model.compute_generalized_mass_matrix(qpos), dtype=np.float64)
    inverse_mass = np.linalg.inv(mass_matrix)
    value = 1.0 / float(inverse_mass[joint_index, joint_index])
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"invalid analytical effective coordinate inertia/mass: {value}")
    return value


def calibrate_effective_coordinate_inertia(
    articulation: Any,
    scene: Any,
    joint_index: int,
    joint_type: str,
    generalized_external_force_magnitude: float,
    timestep_s: float,
    tolerance_fraction: float = 0.10,
    joint_limit_tolerance: float = 1e-6,
) -> EffectiveCoordinateCalibration:
    """Compare full-mass-matrix inertia with an isolated one-step force diagnostic."""
    analytical = analytical_effective_coordinate_inertia(articulation, joint_index)
    diagnostic_force_magnitude = min(1.0, max(1e-4, abs(generalized_external_force_magnitude) * 1e-3))
    production_sign = 1 if generalized_external_force_magnitude >= 0.0 else -1
    qpos = float(articulation.get_qpos()[joint_index])
    limits = np.asarray(articulation.get_qlimits(), dtype=np.float64)
    lower, upper = float(limits[joint_index, 0]), float(limits[joint_index, 1])
    available_positive = upper - qpos if math.isfinite(upper) else math.inf
    available_negative = qpos - lower if math.isfinite(lower) else math.inf
    production_travel = available_positive if production_sign > 0 else available_negative
    opposite_travel = available_negative if production_sign > 0 else available_positive
    fallback = production_travel <= joint_limit_tolerance and opposite_travel > joint_limit_tolerance
    selected_sign = -production_sign if fallback else production_sign
    if (
        (available_positive if selected_sign > 0 else available_negative)
        <= joint_limit_tolerance
    ):
        raise ValueError(
            "effective-inertia diagnostic has no available travel in either direction "
            f"at q={qpos}, limits=[{lower}, {upper}]"
        )
    diagnostic_force = selected_sign * diagnostic_force_magnitude
    qf = np.zeros_like(articulation.get_qf(), dtype=np.float32)
    qf[joint_index] = diagnostic_force
    articulation.set_qvel(np.zeros_like(articulation.get_qvel(), dtype=np.float32))
    articulation.set_qf(qf)
    scene.step()
    measured_acceleration = float(articulation.get_qvel()[joint_index]) / timestep_s
    articulation.set_qf(np.zeros_like(qf))
    if not math.isfinite(measured_acceleration) or abs(measured_acceleration) <= 1e-12:
        measured = math.inf
    else:
        measured = abs(diagnostic_force / measured_acceleration)
    relative_error = abs(measured - analytical) / analytical
    use_measured = math.isfinite(measured) and measured > 0.0 and relative_error > tolerance_fraction
    selected = measured if use_measured else analytical
    return EffectiveCoordinateCalibration(
        joint_type=joint_type,
        joint_index=joint_index,
        diagnostic_generalized_force=diagnostic_force,
        diagnostic_force_magnitude=diagnostic_force_magnitude,
        production_generalized_force_sign=production_sign,
        initial_joint_position=qpos,
        joint_lower_limit=lower,
        joint_upper_limit=upper,
        available_positive_travel=available_positive,
        available_negative_travel=available_negative,
        selected_diagnostic_direction=selected_sign,
        limit_aware_direction_fallback_activated=fallback,
        analytical_effective_inertia_or_mass=analytical,
        measured_effective_inertia_or_mass=measured,
        analytical_measured_relative_error=relative_error,
        selected_effective_inertia_or_mass=selected,
        selected_effective_value_source="measured_diagnostic" if use_measured else "analytical_full_mass_matrix",
        diagnostic_initial_acceleration=measured_acceleration,
        diagnostic_timestep_s=timestep_s,
        validation_tolerance_fraction=tolerance_fraction,
    )


def derive_systematic_resistance(
    calibration: EffectiveCoordinateCalibration,
    generalized_external_force_magnitude: float,
    decay_time_s: float,
    friction_ratio: float,
    damping_bounds: tuple[float, float] = (1e-8, 1e6),
    friction_bounds: tuple[float, float] = (0.0, 1e6),
) -> SystematicResistance:
    if decay_time_s <= 0.0:
        raise ValueError("decay_time_s must be positive")
    if friction_ratio < 0.0:
        raise ValueError("friction_ratio must be nonnegative")
    raw_damping = calibration.selected_effective_inertia_or_mass / decay_time_s
    raw_friction = friction_ratio * abs(generalized_external_force_magnitude)
    damping = min(max(raw_damping, damping_bounds[0]), damping_bounds[1])
    friction = min(max(raw_friction, friction_bounds[0]), friction_bounds[1])
    return SystematicResistance(
        decay_time_s=decay_time_s,
        friction_ratio=friction_ratio,
        effective_inertia_or_mass=calibration.selected_effective_inertia_or_mass,
        damping_coefficient=damping,
        friction_magnitude=friction,
        generalized_external_force_magnitude=abs(generalized_external_force_magnitude),
        damping_clamp_activated=not math.isclose(damping, raw_damping),
        friction_clamp_activated=not math.isclose(friction, raw_friction),
        damping_bounds=damping_bounds,
        friction_bounds=friction_bounds,
    )


def calibration_metadata(
    calibration: EffectiveCoordinateCalibration,
    resistance: SystematicResistance,
) -> dict[str, object]:
    return {
        "effective_coordinate_calibration": asdict(calibration),
        "systematic_resistance": asdict(resistance),
        "systematic_resistance_equations": {
            "effective_coordinate_value": "1 / inv(M)[i,i], validated by diagnostic_qf / diagnostic_qacc",
            "viscous_coefficient": "effective_inertia_or_mass / T_decay",
            "viscous_generalized_force": "-viscous_coefficient * qdot",
            "coulomb_friction": "alpha * abs(generalized_external_force)",
        },
    }
