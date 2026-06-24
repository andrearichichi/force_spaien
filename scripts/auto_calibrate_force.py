#!/usr/bin/env python3
"""Run a small per-object physical-force calibration grid."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_grid(text: str) -> list[float]:
    values = [float(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("grid must contain at least one numeric value")
    return values


def render_script_for(motion_type: str) -> Path:
    if motion_type == "revolute":
        return REPO_ROOT / "scripts" / "render_revolute_video.py"
    if motion_type == "prismatic":
        return REPO_ROOT / "scripts" / "render_prismatic_video.py"
    raise ValueError(motion_type)


def sample_series(document: dict[str, object]) -> list[dict[str, object]]:
    samples = document.get("samples", {})
    if isinstance(samples, dict):
        if "force" in samples and isinstance(samples["force"], list):
            return samples["force"]
        if "opening_force" in samples and isinstance(samples["opening_force"], list):
            return samples["opening_force"]
        if "pulling_force" in samples and isinstance(samples["pulling_force"], list):
            return samples["pulling_force"]
    raise RuntimeError("Could not find primary force sample series in simulation.json")


def metric_keys(motion_type: str) -> tuple[str, str]:
    if motion_type == "prismatic":
        return "joint_position_m", "joint_velocity_m_s"
    return "joint_angle_rad", "joint_velocity_rad_s"


def run_variant(args: argparse.Namespace, force: float, friction: float, damping: float, output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(render_script_for(args.motion_type)),
        "--model-dir",
        str(args.model_dir),
        "--joint",
        args.joint,
        "--link",
        args.link,
        "--force",
        str(force),
        "--seconds",
        str(args.seconds),
        "--fps",
        str(args.fps),
        "--panel-width",
        str(args.panel_width),
        "--panel-height",
        str(args.panel_height),
        "--force-profile",
        args.force_profile,
        "--force-start-time",
        str(args.force_start_time),
        "--force-duration",
        str(args.force_duration),
        "--force-ramp-time",
        str(args.force_ramp_time),
        "--joint-dynamic-friction",
        str(friction),
        "--joint-viscous-damping",
        str(damping),
        "--simulate-until-settled",
        "--settle-velocity-threshold",
        str(args.settle_velocity_threshold),
        "--max-seconds",
        str(args.max_seconds),
        "--end-hold-seconds",
        "0",
        "--end-hold-mode",
        "never",
        "--output",
        str(output_dir / "final_video.mp4"),
        "--json-output",
        str(output_dir / "simulation.json"),
    ]
    if args.motion_type == "revolute":
        command += ["--initial-angle", str(args.initial_angle)]
    if args.contact_point_local is not None:
        command += ["--contact-point-local", *(str(value) for value in args.contact_point_local)]
    if args.contact_point_strategy is not None:
        command += ["--contact-point-strategy", args.contact_point_strategy]

    result = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        return {
            "status": "failed",
            "force": force,
            "joint_dynamic_friction": friction,
            "joint_viscous_damping": damping,
            "output_dir": str(output_dir),
            "stderr": result.stderr.strip(),
        }

    document = json.loads((output_dir / "simulation.json").read_text(encoding="utf-8"))
    samples = sample_series(document)
    position_key, velocity_key = metric_keys(args.motion_type)
    initial_position = float(samples[0][position_key])
    final_position = float(samples[-1][position_key])
    final_velocity = float(samples[-1][velocity_key])
    displacement = abs(final_position - initial_position)
    max_velocity = max(abs(float(sample[velocity_key])) for sample in samples)
    timing = document["metadata"]["timing"]
    validation = document.get("validation", {})
    settled = bool(validation.get("motion_completed")) and not bool(document["metadata"]["simulation_config"].get("still_moving_at_max_seconds"))
    clamped = bool(validation.get("clamped_at_limit"))
    plausible = (
        displacement >= args.target_min_displacement
        and displacement <= args.target_max_displacement
        and settled
        and not clamped
        and float(timing.get("end_hold_seconds", 0.0)) == 0.0
    )
    target_mid = 0.5 * (args.target_min_displacement + args.target_max_displacement)
    score = abs(displacement - target_mid)
    if not settled:
        score += 10.0
    if clamped:
        score += 5.0
    if displacement < args.target_min_displacement:
        score += args.target_min_displacement - displacement
    if displacement > args.target_max_displacement:
        score += displacement - args.target_max_displacement
    return {
        "status": "ok",
        "force": force,
        "joint_dynamic_friction": friction,
        "joint_viscous_damping": damping,
        "output_dir": str(output_dir),
        "final_q": final_position,
        "displacement": displacement,
        "final_qdot": final_velocity,
        "max_abs_qdot": max_velocity,
        "duration_s": float(timing["video_duration_seconds"]),
        "end_hold_seconds": float(timing.get("end_hold_seconds", 0.0)),
        "settled": settled,
        "clamped_at_limit": clamped,
        "plausible": plausible,
        "score": score,
    }


def write_outputs(output_root: Path, rows: list[dict[str, object]], best: dict[str, object] | None) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    columns = [
        "status",
        "force",
        "joint_dynamic_friction",
        "joint_viscous_damping",
        "final_q",
        "displacement",
        "final_qdot",
        "max_abs_qdot",
        "duration_s",
        "end_hold_seconds",
        "settled",
        "clamped_at_limit",
        "plausible",
        "score",
        "output_dir",
    ]
    with (output_root / "calibration_results.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    if best is not None:
        (output_root / "best_config.json").write_text(json.dumps(best, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Physical Force Calibration Summary",
        "",
        f"- Variants: {len(rows)}",
        f"- Successful variants: {sum(1 for row in rows if row.get('status') == 'ok')}",
        f"- Plausible variants: {sum(1 for row in rows if row.get('plausible'))}",
        "",
    ]
    if best is None:
        lines.append("No plausible completed variant was found.")
    else:
        lines += [
            "## Best Configuration",
            "",
            f"- Force: `{best['force']}`",
            f"- Dynamic friction: `{best['joint_dynamic_friction']}`",
            f"- Viscous damping: `{best['joint_viscous_damping']}`",
            f"- Final q: `{best['final_q']}`",
            f"- Final qdot: `{best['final_qdot']}`",
            f"- Duration: `{best['duration_s']}`",
            f"- Output: `{best['output_dir']}`",
        ]
    (output_root / "calibration_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate force/friction/damping for one object joint.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--joint", required=True)
    parser.add_argument("--link", required=True)
    parser.add_argument("--motion-type", choices=["revolute", "prismatic"], required=True)
    parser.add_argument("--target-min-displacement", type=float, default=0.15)
    parser.add_argument("--target-max-displacement", type=float, default=0.60)
    parser.add_argument("--settle-velocity-threshold", type=float, default=1e-3)
    parser.add_argument("--max-seconds", type=float, default=25.0)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--force-grid", type=parse_grid, default=parse_grid("2.5,5.0"))
    parser.add_argument("--dynamic-friction-grid", type=parse_grid, default=parse_grid("0.05,0.15,0.25,0.35"))
    parser.add_argument("--viscous-damping-grid", type=parse_grid, default=parse_grid("0.2,0.5,1.0"))
    parser.add_argument("--force-profile", choices=["constant", "pulse", "ramp_hold_release"], default="ramp_hold_release")
    parser.add_argument("--force-start-time", type=float, default=0.0)
    parser.add_argument("--force-duration", type=float, default=1.2)
    parser.add_argument("--force-ramp-time", type=float, default=0.2)
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--panel-width", type=int, default=640)
    parser.add_argument("--panel-height", type=int, default=480)
    parser.add_argument("--initial-angle", type=float, default=0.0)
    parser.add_argument("--contact-point-local", nargs=3, type=float, default=None)
    parser.add_argument("--contact-point-strategy", default=None)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    rows: list[dict[str, object]] = []
    for force in args.force_grid:
        for friction in args.dynamic_friction_grid:
            for damping in args.viscous_damping_grid:
                name = f"force_{force:g}_fric_{friction:g}_damp_{damping:g}".replace(".", "p")
                print(f"Running {name}")
                rows.append(run_variant(args, force, friction, damping, output_root / name))
                write_outputs(output_root, rows, None)

    candidates = [row for row in rows if row.get("status") == "ok" and row.get("plausible")]
    best = min(candidates, key=lambda row: float(row["score"])) if candidates else None
    write_outputs(output_root, rows, best)
    return 0 if best is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
