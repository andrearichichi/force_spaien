#!/usr/bin/env python3
"""Validate canonical physics, metadata, portability, and media."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import subprocess

import imageio_ffmpeg

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/production_pipeline.json"


def close(left: float, right: float, tolerance: float = 1e-10) -> bool:
    return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)


def vector_close(left: list[float], right: list[float]) -> bool:
    return len(left) == len(right) and all(close(a, b) for a, b in zip(left, right))


def validate_document(document: dict, expected: dict, physics: dict) -> list[str]:
    failures = []
    articulation_links = {
        link["name"]: link
        for link in document.get("metadata", {}).get("articulation", {}).get("links", [])
    }
    calibrated_links = {link["name"]: link for link in expected["links"]}
    link_calibration_matches = articulation_links.keys() == calibrated_links.keys()
    if link_calibration_matches:
        for name, calibrated in calibrated_links.items():
            actual = articulation_links[name]
            link_calibration_matches = (
                close(actual["mass"], calibrated["mass_kg"])
                and vector_close(actual["inertia"], calibrated["local_inertia_kg_m2"])
                and vector_close(actual["cmass_local_pose"]["p"], calibrated["local_com_m"])
            )
            if not link_calibration_matches:
                break
    resistance = document.get("systematic_resistance", {})
    checks = {
        "object_id": document.get("object_id") == expected["object_id"],
        "contact": document.get("candidate_id") == expected["contact_label"],
        "joint": document.get("joint_name") == expected["joint_name"],
        "link": document.get("selected_link") == expected["moving_link"],
        "force": close(document.get("force_magnitude"), expected["personalized_force_n"]),
        "force_direction": vector_close(document.get("force_direction_world", []),
                                          expected["force_direction_world"]),
        "contact_point": vector_close(
            document.get("metadata", {}).get("application_point", {}).get("local_on_link", []),
            expected["contact_local_scaled_m"],
        ),
        "target": close(document["joint_travel_personalized_actuation"]["target_displacement"],
                        expected["target_displacement"]),
        "link_mass_com_inertia": link_calibration_matches,
        "total_mass": close(sum(link["mass"] for link in articulation_links.values()),
                            expected["actual_total_mass_kg"]),
        "friction": close(document.get("joint_friction_installed"), 0.0),
        "no_friction_fallback": document.get("joint_friction_fallback_used") is False,
        "no_friction_compensation": document.get("friction_compensation_used") is False,
        "decay": close(document.get("T_decay"), 2.0),
        "viscous_only": document.get("resistance_model") == "viscous_damping_only",
        "damping_rule": close(resistance.get("damping_coefficient"),
                               expected["effective_inertia_or_mass"] / 2.0),
        "zero_generalized_friction": (
            close(resistance.get("friction_magnitude"), 0.0)
            and document.get("zero_generalized_friction_force_verified") is True
        ),
        "gravity": document.get("gravity_enabled") is False,
        "timestep": close(document.get("timestep_s"), 1.0 / 240.0),
        "pulse_steps": int(document.get("active_force_physics_steps")) == 480,
        "native_force": document.get("native_sapien_add_force_at_point_used") is True,
        "trajectory": document.get("application_point_world_trajectory") ==
                      document.get("contact_point_world_per_frame"),
        "trajectory_complete": len(document.get("application_point_world_trajectory", [])) ==
                               len(document.get("q", [])),
        "settling": document.get("stopped_because") != "maximum_duration_cap",
        "finite": document.get("non_finite_state_detected") is False,
        "force_release": all(
            not row.get("force_active", False)
            for row in document.get("samples", {}).get("force", [])
            if float(row.get("time_s", 0.0)) >= 2.0
        ),
    }
    for name, passed in checks.items():
        if not passed:
            failures.append(name)
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    package = args.package.resolve()
    config = json.loads(CONFIG.read_text())
    failures = []
    canonical_object_files = {
        "contact_sheet.png", "final_video.mp4", "run.log", "simulation.json"
    }
    object_directories = sorted(path for path in (package / "objects").iterdir()
                                if path.is_dir())
    if len(object_directories) != 10:
        failures.append("object_directory_cardinality")
    for directory in object_directories:
        relative_files = {
            path.relative_to(directory).as_posix()
            for path in directory.rglob("*") if path.is_file()
        }
        if relative_files != canonical_object_files:
            failures.append(f"object_structure:{directory.name}")
    videos = list(package.glob("objects/*/final_video.mp4"))
    simulations = list(package.glob("objects/*/simulation.json"))
    if len(videos) != 10 or len(simulations) != 10:
        failures.append("package_cardinality")
    for path in simulations:
        document = json.loads(path.read_text())
        object_id = document["object_id"]
        expected = {"object_id": object_id, **config["objects"][object_id]}
        failures.extend(f"{object_id}:{name}" for name in
                        validate_document(document, expected, config["physics"]))
    index = (package / "index.html").read_text()
    if index.count("<article>") != 10:
        failures.append("index_cards")
    references = re.findall(r'(?:href|src)="([^"]+)"', index)
    failures.extend(f"missing_reference:{ref}" for ref in references
                    if not (package / ref).is_file())
    if any(path.is_symlink() for path in package.rglob("*")):
        failures.append("symlink")
    if not args.metadata_only:
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        for video in videos:
            result = subprocess.run([ffmpeg, "-v", "error", "-i", str(video),
                                     "-f", "null", "-"], check=False)
            if result.returncode:
                failures.append(f"decode:{video}")
        for sheet in package.glob("objects/*/contact_sheet.png"):
            result = subprocess.run([ffmpeg, "-v", "error", "-i", str(sheet),
                                     "-f", "null", "-"], check=False)
            if result.returncode:
                failures.append(f"decode:{sheet}")
        result = subprocess.run(["sha256sum", "-c", "SHA256SUMS"], cwd=package,
                                stdout=subprocess.DEVNULL, check=False)
        if result.returncode:
            failures.append("checksums")
    if failures:
        print(json.dumps({"verdict": "FAIL", "failures": failures}, indent=2))
        return 1
    print(json.dumps({"verdict": "PASS", "objects": 10,
                      "metadata_only": args.metadata_only}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
