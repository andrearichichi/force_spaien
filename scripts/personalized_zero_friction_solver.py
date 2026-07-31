"""Pure-viscous travel solver and deterministic bisection utilities."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable


@dataclass(frozen=True)
class PureViscousEstimate:
    target_displacement: float
    pulse_duration_s: float
    decay_time_s: float
    effective_inertia_or_mass: float
    cartesian_efficiency: float
    analytical_generalized_force: float
    analytical_cartesian_force: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def analytical_estimate(
    target_displacement: float,
    effective_inertia_or_mass: float,
    cartesian_efficiency: float,
    pulse_duration_s: float = 2.0,
    decay_time_s: float = 2.0,
) -> PureViscousEstimate:
    if min(target_displacement, effective_inertia_or_mass, cartesian_efficiency) <= 0:
        raise ValueError("target, effective property, and efficiency must be positive")
    # For M*qddot + (M/Td)*qdot = Q during Tp, then passive decay,
    # total displacement is exactly (Q/M)*Tp*Td.
    generalized = effective_inertia_or_mass * target_displacement / (
        pulse_duration_s * decay_time_s
    )
    pure_force = generalized / cartesian_efficiency

    return PureViscousEstimate(
        target_displacement=target_displacement,
        pulse_duration_s=pulse_duration_s,
        decay_time_s=decay_time_s,
        effective_inertia_or_mass=effective_inertia_or_mass,
        cartesian_efficiency=cartesian_efficiency,
        analytical_generalized_force=generalized,
        analytical_cartesian_force=pure_force,
    )


def bisect_force(
    target: float,
    initial_force: float,
    evaluate: Callable[[float], float],
    *,
    minimum_force: float = 1e-7,
    maximum_force: float = 150.0,
    relative_tolerance: float = 0.01,
    maximum_iterations: int = 18,
) -> dict[str, object]:
    low = minimum_force
    high = min(maximum_force, max(initial_force * 2.0, minimum_force * 2.0))
    low_value = evaluate(low)
    high_value = evaluate(high)
    while high_value < target and high < maximum_force:
        low, low_value = high, high_value
        high = min(maximum_force, high * 2.0)
        high_value = evaluate(high)
    history: list[dict[str, float]] = []
    chosen_force, chosen_value = high, high_value
    for iteration in range(1, maximum_iterations + 1):
        force = 0.5 * (low + high)
        value = evaluate(force)
        error = abs(value - target) / target
        history.append({"iteration": iteration, "force_n": force,
                        "displacement": value, "relative_error": error})
        chosen_force, chosen_value = force, value
        if error <= relative_tolerance:
            break
        if value < target:
            low = force
        else:
            high = force
    return {
        "method": "deterministic_no_rendering_sapien_bisection",
        "force_n": chosen_force,
        "predicted_displacement": chosen_value,
        "relative_error": abs(chosen_value - target) / target,
        "converged": abs(chosen_value - target) / target <= relative_tolerance,
        "iterations": len(history),
        "history": history,
        "minimum_clamp_activated": chosen_force <= minimum_force,
        "maximum_clamp_activated": chosen_force >= maximum_force,
    }
