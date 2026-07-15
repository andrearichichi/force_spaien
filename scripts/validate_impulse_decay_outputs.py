#!/usr/bin/env python3
"""Validate and summarize the fixed global impulse-decay ForceSAPIEN run."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


SUFFIX = "manual_contact_global_impulse_decay_check"
EXPECTED_IDS = {"10211", "10905", "11100", "45135", "100109", "101917", "102255", "103111", "103706", "103776"}
VISUAL = {
    "10211": ("VISUAL_WARN_SUBTLE", "Short laptop-screen impulse is subtle but visible and stops naturally."),
    "10905": ("VISUAL_WARN_SUBTLE", "Short refrigerator-door impulse is subtle but coherent and stops naturally."),
    "11100": ("VISUAL_WARN_SUBTLE", "Short scissor impulse is subtle but visible at the selected handle contact."),
    "45135": ("VISUAL_WARN_SUBTLE", "Drawer displacement is very subtle but consistent with the short impulse and clean decay."),
    "100109": ("VISUAL_WARN_SUBTLE", "USB cover displacement is very subtle at the correct cover-tip contact and cleanly decays."),
    "101917": ("VISUAL_WARN_SUBTLE", "Oven-door impulse is subtle but follows the selected hinge and stops naturally."),
    "102255": ("VISUAL_WARN_SUBTLE", "Folding-chair articulation is subtle but physically coherent and cleanly settles."),
    "103111": ("VISUAL_WARN_SUBTLE", "Stapler-top impulse is subtle but visible and passively settles."),
    "103706": ("VISUAL_PASS", "Knife slider shows a clear quick displacement followed by passive settling."),
    "103776": ("VISUAL_PASS", "Washing-machine door shows a clear quick rotation followed by passive settling."),
}


def finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(finite(item) for item in value)
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    return True


def validate(folder: Path) -> dict[str, Any]:
    sim_path = folder / "simulation.json"
    video_path = folder / "final_video.mp4"
    data = json.loads(sim_path.read_text(encoding="utf-8"), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    object_id = str(data["object_id"])
    samples = data.get("samples", {}).get("force", [])
    pulse = [sample for sample in samples if float(sample.get("time_s", 0.0)) <= 0.100000001]
    passive = [sample for sample in samples if float(sample.get("time_s", 0.0)) > 0.100000001]
    moving_passive = [sample for sample in passive if abs(float(sample.get("qdot", 0.0))) > 0.002]
    opposing = [sample for sample in moving_passive if float(sample.get("qdot", 0.0)) * float(sample.get("qddot", 0.0)) <= 0.0]
    checks = {
        "finite": finite(data),
        "video_exists": video_path.is_file(),
        "final_mode": data.get("final_mode") == "manual_contact_global_impulse_decay",
        "force_policy": data.get("force_policy") == "fixed_global_impulse_decay",
        "same_physics": data.get("same_physics_for_all_objects") is True,
        "gravity_disabled": data.get("metadata", {}).get("actuation", {}).get("force", {}).get("gravity_enabled") is False,
        "global_values": all(math.isclose(float(data.get(key, -1)), value) for key, value in {
            "force_magnitude": 5.0, "actual_force_magnitude": 5.0, "joint_damping": 2.0,
            "joint_friction": 0.30, "force_duration_s": 0.10,
        }.items()),
        "no_adaptation": all(data.get(key) is False for key in (
            "per_object_force_adaptation", "per_object_damping_adaptation", "per_object_friction_adaptation")),
        "true_force": data.get("true_external_force_used") is True and data.get("force_application_mode") == "external_link_force",
        "no_fake_driver": all(data.get(key) is False for key in (
            "hidden_drive_used", "manual_q_interpolation_used", "uses_generalized_set_qf_as_motion_driver")),
        "contact_recorded": bool(data.get("contact_point_world_at_pulse")) and bool(data.get("force_direction_world_at_pulse")),
        "adaptive_fields": data.get("adaptive_duration") is True and all(key in data for key in (
            "min_sim_duration_s", "max_sim_duration_s", "settle_qdot_threshold", "settle_window_s",
            "post_settle_hold_s", "settled", "stopped_because", "duration_verdict", "final_acceptance")),
        "adaptive_values": all(math.isclose(float(data.get(key, -1)), value) for key, value in {
            "min_sim_duration_s": 1.0, "max_sim_duration_s": 10.0, "settle_qdot_threshold": 0.002,
            "settle_window_s": 0.5, "post_settle_hold_s": 1.0,
        }.items()),
        "pulse": bool(pulse) and all(float(sample.get("applied_force_norm", 0.0)) > 0.0
            and float(sample.get("force_profile_scale", 0.0)) > 0.0 and sample.get("phase") == "force_applied" for sample in pulse),
        "passive_zero_force": bool(passive) and all(float(sample.get("applied_force_norm", -1.0)) == 0.0
            and float(sample.get("force_profile_scale", -1.0)) == 0.0 and sample.get("phase") == "passive_motion" for sample in passive),
        "passive_decay": bool(moving_passive) and len(opposing) / len(moving_passive) >= 0.9
            and abs(float(passive[-1].get("qdot", 1.0))) < abs(float(passive[0].get("qdot", 0.0))),
        "joint_limits": data.get("physical_validation", {}).get("q_inside_joint_limits") is True,
        "not_forced_to_six_seconds": float(data.get("actual_sim_duration_s", 6.0)) < 6.0,
        "duration_policy": (data.get("stopped_because") != "max_duration") or (
            data.get("settled") is False and data.get("duration_verdict") == "FAIL_MAX_DURATION"
            and data.get("final_acceptance") == "FAIL"),
    }
    failed = [name for name, passed in checks.items() if not passed]
    visual_verdict, visual_note = VISUAL[object_id]
    return {
        "object_id": object_id, "object_name": data.get("object_name"), "joint_type": data.get("joint_type"),
        "selected_joint": data.get("selected_joint"), "selected_link": data.get("selected_link"),
        "candidate_id": data.get("candidate_id"), "force_direction_mode": data.get("force_direction_mode"),
        "q_start": data.get("q_start"), "q_end": data.get("q_end"), "delta_q": data.get("delta_q"),
        "q_unit": data.get("q_unit"), "force_magnitude": data.get("force_magnitude"),
        "joint_damping": data.get("joint_damping"), "joint_friction": data.get("joint_friction"),
        "force_duration_s": data.get("force_duration_s"), "actual_sim_duration_s": data.get("actual_sim_duration_s"),
        "actual_video_frame_count": data.get("actual_video_frame_count"), "peak_abs_qdot": data.get("peak_abs_qdot"),
        "settle_time_s": data.get("settle_time_s"),
        "final_abs_qdot": data.get("final_abs_qdot"), "settled": data.get("settled"),
        "stopped_because": data.get("stopped_because"), "duration_verdict": data.get("duration_verdict"),
        "visual_verdict": visual_verdict, "visual_note": visual_note,
        "final_acceptance": data.get("final_acceptance") if not failed else "FAIL_VALIDATION",
        "validation": "PASS" if not failed else "FAIL: " + ", ".join(failed),
        "folder": folder.as_posix(), "video": video_path.as_posix(),
    }


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    folders = sorted(path.parent for path in args.output_root.glob(f"*_{SUFFIX}/simulation.json"))
    rows = sorted((validate(folder) for folder in folders), key=lambda row: int(row["object_id"]))
    ids = {row["object_id"] for row in rows}
    if ids != EXPECTED_IDS:
        raise SystemExit(f"expected IDs {sorted(EXPECTED_IDS)}, got {sorted(ids)}")
    failures = [row for row in rows if row["validation"] != "PASS" or row["final_acceptance"] != "PASS"]

    final_fields = ["object_id", "object_name", "joint_type", "selected_joint", "selected_link", "candidate_id",
                    "force_direction_mode", "q_start", "q_end", "delta_q", "q_unit", "force_magnitude",
                    "joint_damping", "joint_friction", "force_duration_s", "actual_sim_duration_s",
                    "actual_video_frame_count", "settle_time_s", "peak_abs_qdot", "final_abs_qdot", "settled", "stopped_because",
                    "duration_verdict", "visual_verdict", "final_acceptance", "validation", "video"]
    write_tsv(args.output_root / "final_manual_contact_global_impulse_decay_table.tsv", rows, final_fields)
    write_tsv(args.output_root / "physical_consistency_manual_contact_global_impulse_decay.tsv", rows,
              ["object_id", "object_name", "joint_type", "delta_q", "q_unit", "peak_abs_qdot", "final_abs_qdot",
               "settled", "stopped_because", "duration_verdict", "final_acceptance", "validation"])
    write_tsv(args.output_root / "final_visual_review_manual_contact_global_impulse_decay.tsv", rows,
              ["object_id", "object_name", "visual_verdict", "visual_note", "settled", "stopped_because", "final_acceptance", "video"])

    parameters = "Force 5.0 dataset/SAPIEN units; pulse 0.10 s; joint damping 2.0; joint friction 0.30; gravity disabled; 30 fps."
    lines = ["# Final manual-contact global impulse-decay run", "", parameters, "",
             "All objects use true `external_link_force` with identical dynamics. Only the manually selected semantic contact point and opening direction vary. There is no per-object calibration, target torque, hidden drive, manual q interpolation, or generalized-force motion driver.", "",
             "| ID | Object | Δq | Duration (s) | Stop reason | Visual | Final |", "|---:|---|---:|---:|---|---|---|"]
    for row in rows:
        lines.append(f"| {row['object_id']} | {row['object_name']} | {row['delta_q']:.6f} {row['q_unit']} | {row['actual_sim_duration_s']:.2f} | {row['stopped_because']} | {row['visual_verdict']} | {row['final_acceptance']} |")
    lines += ["", f"Validation: {'PASS' if not failures else 'FAIL'}. Max-duration failures: {sum(row['stopped_because'] == 'max_duration' for row in rows)}.", ""]
    (args.output_root / "final_manual_contact_global_impulse_decay_report.md").write_text("\n".join(lines), encoding="utf-8")

    physics = ["# Physical consistency — global impulse decay", "", parameters, "",
               "A maximum-duration stop is a hard failure; visual review cannot override it.", "",
               "| ID | Object | Peak |qdot| | Final |qdot| | Settled | Duration verdict | Acceptance |", "|---:|---|---:|---:|---|---|---|"]
    for row in rows:
        physics.append(f"| {row['object_id']} | {row['object_name']} | {row['peak_abs_qdot']:.6g} | {row['final_abs_qdot']:.6g} | {row['settled']} | {row['duration_verdict']} | {row['final_acceptance']} |")
    physics += ["", f"All {len(rows)} JSON files parse, contain finite numeric data, retain the fixed global parameters, record true pulse-only external forces, and link an MP4.", ""]
    (args.output_root / "physical_consistency_manual_contact_global_impulse_decay.md").write_text("\n".join(physics), encoding="utf-8")

    visual = ["# Visual review — global impulse decay", "", "Visual review used six-frame contact sheets from each final MP4. Subtle motion is warned explicitly; a visual verdict never overrides a duration failure.", "",
              "| ID | Object | Verdict | Note |", "|---:|---|---|---|"]
    for row in rows:
        visual.append(f"| {row['object_id']} | {row['object_name']} | {row['visual_verdict']} | {row['visual_note']} |")
    visual.append("")
    (args.output_root / "final_visual_review_manual_contact_global_impulse_decay.md").write_text("\n".join(visual), encoding="utf-8")
    if failures:
        raise SystemExit("validation failures: " + "; ".join(f"{row['object_id']} {row['validation']} {row['final_acceptance']}" for row in failures))
    print(f"validated {len(rows)} global impulse-decay outputs; all accepted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
