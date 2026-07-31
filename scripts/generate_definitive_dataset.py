#!/usr/bin/env python3
"""Single production entry point for the definitive ForceSAPIEN dataset."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.personalized_zero_friction_solver import analytical_estimate

CONFIG = ROOT / "configs/production_pipeline.json"
CONTACTS = ROOT / "configs/manual_contact_overrides.yaml"
INTERNAL_RUNNER = ROOT / "scripts/run_forcesapien_batch_final_dataset.py"
PYTHON = Path("/leonardo_work/IscrC_EditGS/andrea/FORCEARTGS/.venv/bin/python")


def scale_vector(text: str, scale: float) -> str:
    return " ".join(f"{float(value) * scale:.17g}" for value in text.split())


def prepare_scaled_object(source: Path, destination: Path, scale: float) -> None:
    shutil.copytree(source, destination, symlinks=False)
    tree = ET.parse(destination / "mobility.urdf")
    root = tree.getroot()
    for link in root.findall("link"):
        for inertial in list(link.findall("inertial")):
            link.remove(inertial)
    for origin in root.iter("origin"):
        if "xyz" in origin.attrib:
            origin.attrib["xyz"] = scale_vector(origin.attrib["xyz"], scale)
    for mesh in root.iter("mesh"):
        mesh.attrib["scale"] = scale_vector(mesh.attrib.get("scale", "1 1 1"), scale)
    for joint in root.findall("joint"):
        if joint.attrib.get("type") == "prismatic":
            limit = joint.find("limit")
            if limit is not None:
                for key in ("lower", "upper"):
                    if key in limit.attrib:
                        limit.attrib[key] = f"{float(limit.attrib[key]) * scale:.17g}"
    tree.write(destination / "mobility.urdf", encoding="utf-8", xml_declaration=True)


def write_scaled_contacts(config: dict, destination: Path) -> None:
    source = yaml.safe_load(CONTACTS.read_text())
    resolved = {}
    for object_id in config["object_ids"]:
        row = dict(source.get(int(object_id), source.get(object_id)))
        row["contact_mode"] = "manual_local_point"
        row["local_point"] = config["objects"][object_id]["contact_local_scaled_m"]
        row["candidate_id"] = config["objects"][object_id]["contact_label"]
        resolved[int(object_id)] = row
    destination.write_text(yaml.safe_dump(resolved, sort_keys=False))


def validate_resolved_force(row: dict, physics: dict) -> None:
    estimate = analytical_estimate(
        float(row["target_displacement"]),
        float(row["effective_inertia_or_mass"]),
        float(row["cartesian_efficiency"]),
        float(physics["force_duration_s"]),
        float(physics["T_decay_s"]),
    ).analytical_cartesian_force
    force = float(row["personalized_force_n"])
    if not physics["minimum_force_n"] <= force <= physics["maximum_force_n"]:
        raise RuntimeError(f"{row['name']}: personalized force violates safety bounds")
    if abs(force - estimate) / force > 0.08:
        raise RuntimeError(f"{row['name']}: resolved force is inconsistent with pure-viscous estimate")


def dry_run(config: dict, dataset_root: Path) -> int:
    print("object_id\tcontact\ttarget\tpersonalized_force_n\tfriction\tT_decay")
    for object_id in config["object_ids"]:
        row = config["objects"][object_id]
        if not (dataset_root / object_id / "mobility.urdf").is_file():
            raise RuntimeError(f"missing source object {object_id}")
        validate_resolved_force(row, config["physics"])
        print(f"{object_id}\t{row['contact_label']}\t{row['target_displacement']:.12g}\t"
              f"{row['personalized_force_n']:.12g}\t0\t2.0")
    subprocess.run([
        sys.executable, str(ROOT / "scripts/validate_definitive_dataset.py"),
        "--package", str(ROOT / "final_scaled_70pct_zero_friction_definitive"),
        "--metadata-only",
    ], cwd=ROOT, check=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    dataset_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve()
    canonical = (ROOT / "final_scaled_70pct_zero_friction_definitive").resolve()
    if output_root == canonical:
        raise SystemExit("production generation must use a new output path, not the canonical package")
    config = json.loads(CONFIG.read_text())
    if args.dry_run:
        if output_root.exists():
            raise SystemExit(f"dry-run output path must not exist: {output_root}")
        return dry_run(config, dataset_root)
    if output_root.exists():
        raise SystemExit(f"refusing to overwrite existing output: {output_root}")

    work = output_root.parent / f".{output_root.name}.production_work"
    if work.exists():
        raise SystemExit(f"refusing to overwrite existing work path: {work}")
    runtime = work / "runtime_dataset"
    raw = work / "raw"
    runtime.mkdir(parents=True)
    raw.mkdir()
    for object_id in config["object_ids"]:
        prepare_scaled_object(dataset_root / object_id, runtime / object_id,
                              float(config["objects"][object_id]["scale"]))
    contacts = work / "manual_contact_overrides_scaled.yaml"
    write_scaled_contacts(config, contacts)
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT / 'scripts'}:{ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}"
    for object_id in config["object_ids"]:
        row = config["objects"][object_id]
        validate_resolved_force(row, config["physics"])
        object_env = env.copy()
        object_env["FORCESAPIEN_CALIBRATION_DENSITY"] = str(row["equivalent_density_kg_m3"])
        object_env["FORCESAPIEN_CAMERA_TARGET_DISPLACEMENT"] = str(
            float(row["target_displacement"]) * float(row["intended_sign"])
        )
        command = [
            str(PYTHON), str(INTERNAL_RUNNER), "--dataset_root", str(runtime),
            "--output_root", str(raw), "--object_ids", object_id, "--force",
            "--output-suffix", "production_70pct_zero_friction",
            "--force-policy", "native_sapien_force",
            "--force-application-mode", "native_sapien_add_force_at_point",
            "--force-magnitude", str(row["personalized_force_n"]),
            "--force-duration-s", "2.0", "--sim-duration-s", "6.0",
            "--contact-overrides", str(contacts), "--joint-damping", "0",
            "--joint-friction", "0", "--resistance-model", "systematic_joint_space",
            "--systematic-decay-time-s", "2.0", "--systematic-friction-ratio", "0",
            "--require-zero-joint-friction", "--reverse-force-direction-object-ids", "10211",
            "--adaptive-duration", "true", "--min-sim-duration-s", "6.0",
            "--max-sim-duration-s", "20.0", "--settle-qdot-threshold", "0.002",
            "--settle-window-s", "0.5", "--post-settle-hold-s", "1.0",
            "--end-hold-seconds", "0", "--end-hold-mode", "never",
            "--python-executable", str(PYTHON),
        ]
        subprocess.run(command, cwd=ROOT, env=object_env, check=True)
    subprocess.run([
        str(PYTHON), str(ROOT / "scripts/package_definitive_dataset.py"),
        "--raw-root", str(raw), "--output-root", str(output_root),
    ], cwd=ROOT, env=env, check=True)
    subprocess.run([
        str(PYTHON), str(ROOT / "scripts/validate_definitive_dataset.py"),
        "--package", str(output_root),
    ], cwd=ROOT, env=env, check=True)
    shutil.rmtree(work)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
