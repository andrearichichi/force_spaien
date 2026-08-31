#!/usr/bin/env python3
"""Audit recorded ForceSAPIEN trajectories against the viscous joint model."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean


CSV_FIELDS = [
    "object", "object_id", "joint_type", "effective_inertia_or_mass", "c",
    "theoretical_decay_time_s", "fitted_decay_time_s", "relative_decay_time_error",
    "peak_abs_qdot", "joint_limit_hit", "damping_logging_valid",
    "physics_residual_during_force_rms", "physics_residual_during_force_relative",
    "physics_residual_after_release_rms", "physics_residual_after_release_relative",
    "friction_zero", "gravity_off", "decay_fit_samples", "status", "reasons",
]


def rms(values: list[float]) -> float:
    return math.sqrt(mean(value * value for value in values)) if values else math.inf


def fit_decay(samples: list[dict], release: float, threshold: float,
              limits: tuple[float, float]) -> tuple[float, int, bool]:
    lower, upper = limits
    margin = 1e-5
    post_release = [sample for sample in samples if sample["time_s"] > release]
    limit_during_fit = any(
        float(sample["q"]) <= lower + margin or float(sample["q"]) >= upper - margin
        for sample in post_release if abs(float(sample["qdot"])) > threshold
    )
    selected = [
        sample for sample in post_release
        if abs(float(sample["qdot"])) > threshold
        and lower + margin < float(sample["q"]) < upper - margin
    ]
    if len(selected) < 3:
        return math.nan, len(selected), limit_during_fit
    x = [float(sample["time_s"]) for sample in selected]
    y = [math.log(abs(float(sample["qdot"]))) for sample in selected]
    x_mean, y_mean = mean(x), mean(y)
    denominator = sum((value - x_mean) ** 2 for value in x)
    slope = sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y)) / denominator
    return (-1.0 / slope if slope < 0.0 else math.nan), len(selected), limit_during_fit


def audit(path: Path) -> dict[str, object]:
    document = json.loads(path.read_text())
    samples = document.get("stage_a_physics_samples", [])
    resistance = document.get("systematic_resistance", {})
    inertia = float(resistance["effective_inertia_or_mass"])
    damping = float(resistance["damping_coefficient"])
    theoretical = inertia / damping
    timestep = float(document["timestep_s"])
    release = float(document["actual_force_stop_time_s"])
    threshold = max(float(document.get("settle_qdot_threshold", 0.002)), 1e-8)
    raw_limits = document.get("joint_limits")
    limits = tuple(map(float, raw_limits)) if raw_limits is not None else (-math.inf, math.inf)
    fitted, fit_count, limit_during_fit = fit_decay(samples, release, threshold, limits)

    during_residuals: list[float] = []
    after_residuals: list[float] = []
    maximum_external = 0.0
    maximum_post_release_damping = 0.0
    damping_law_valid = True
    net_logging_valid = True
    for sample in samples:
        external = float(sample["external_generalized_force"])
        viscous = float(sample["viscous_generalized_resistance"])
        qdot_before = float(sample["qdot_before_step"])
        qddot = (float(sample["qdot"]) - qdot_before) / timestep
        net = float(sample["net_generalized_force"])
        residual = inertia * qddot - net
        damping_law_valid &= math.isclose(viscous, -damping * qdot_before, rel_tol=1e-7, abs_tol=1e-12)
        net_logging_valid &= math.isclose(net, external + viscous, rel_tol=1e-7, abs_tol=1e-12)
        maximum_external = max(maximum_external, abs(external))
        if abs(external) > 1e-12:
            during_residuals.append(residual)
        else:
            after_residuals.append(residual)
            maximum_post_release_damping = max(maximum_post_release_damping, abs(viscous))

    frame_samples = document.get("samples", {}).get("force", [])
    moving_frames = [sample for sample in frame_samples if abs(float(sample["qdot"])) > 1e-6]
    frame_logging_valid = bool(moving_frames) and all(
        abs(float(sample.get("applied_viscous_generalized_resistance", 0.0))) > 1e-12
        and math.isclose(
            float(sample.get("damping_torque_or_force", math.nan)),
            float(sample["applied_viscous_generalized_resistance"]),
            rel_tol=1e-7,
            abs_tol=1e-12,
        )
        and float(sample["qdot"]) * float(sample["applied_viscous_generalized_resistance"]) <= 1e-12
        for sample in moving_frames
    )
    logging_valid = damping_law_valid and net_logging_valid and frame_logging_valid

    during_rms = rms(during_residuals)
    after_rms = rms(after_residuals)
    during_relative = during_rms / max(maximum_external, 1e-12)
    after_relative = after_rms / max(maximum_post_release_damping, maximum_external, 1e-12)
    decay_error = abs(fitted - theoretical) / theoretical if math.isfinite(fitted) else math.inf
    friction_zero = (
        abs(float(document.get("joint_friction_installed", math.inf))) <= 1e-12
        and abs(float(resistance.get("friction_magnitude", math.inf))) <= 1e-12
    )
    gravity_off = document.get("gravity_enabled") is False
    joint_limit_hit = bool(document.get("joint_limit_reached", False) or limit_during_fit)

    failures = []
    if decay_error > 0.02:
        failures.append(f"decay error {decay_error:.3%} exceeds 2%")
    if during_relative > 0.01:
        failures.append(f"during-force residual {during_relative:.3%} exceeds 1%")
    if after_relative > 0.01:
        failures.append(f"post-release residual {after_relative:.3%} exceeds 1%")
    if not logging_valid:
        failures.append("damping/net logging does not match applied qf")
    if not friction_zero:
        failures.append("friction is nonzero")
    if not gravity_off:
        failures.append("gravity is enabled")
    if joint_limit_hit:
        failures.append("joint limit reached during the evaluated trajectory")
    if fit_count < 20:
        failures.append("insufficient post-release samples for decay fit")

    return {
        "object": document["object_name"],
        "object_id": str(document["object_id"]),
        "joint_type": document["joint_type"],
        "effective_inertia_or_mass": inertia,
        "c": damping,
        "theoretical_decay_time_s": theoretical,
        "fitted_decay_time_s": fitted,
        "relative_decay_time_error": decay_error,
        "peak_abs_qdot": max(abs(float(sample["qdot"])) for sample in samples),
        "joint_limit_hit": joint_limit_hit,
        "damping_logging_valid": logging_valid,
        "physics_residual_during_force_rms": during_rms,
        "physics_residual_during_force_relative": during_relative,
        "physics_residual_after_release_rms": after_rms,
        "physics_residual_after_release_relative": after_relative,
        "friction_zero": friction_zero,
        "gravity_off": gravity_off,
        "decay_fit_samples": fit_count,
        "status": "FAIL" if failures else "PASS",
        "reasons": "; ".join(failures),
        "simulation_json": str(path.resolve()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output = (args.output_root or root).resolve()
    paths = sorted(root.glob("objects/*/simulation.json")) or sorted(root.glob("*/simulation.json"))
    if not paths:
        raise SystemExit(f"no simulation.json files found under {root}")
    rows = [audit(path) for path in paths]
    output.mkdir(parents=True, exist_ok=True)
    (output / "physics_validation.json").write_text(json.dumps(rows, indent=2) + "\n")
    with (output / "physics_validation.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=CSV_FIELDS, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(f"{row['object']}: {row['status']} (tau={row['fitted_decay_time_s']:.6g}s)")
    return 1 if any(row["status"] == "FAIL" for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
