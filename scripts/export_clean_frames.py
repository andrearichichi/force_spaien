#!/usr/bin/env python3
"""Create a non-destructive dual export from recorded ForceSAPIEN trajectories.

The physical simulation is never stepped.  Each clean RGB/mask pair is rendered
by assigning the recorded generalized coordinate from simulation.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.generate_definitive_dataset import prepare_scaled_object
from scripts.replay_definitive_video import ensure_neutral_missing_textures, set_coordinate
from scripts import render_prismatic_video as prismatic
from scripts import render_revolute_video as revolute


CONFIG = ROOT / "configs/production_pipeline.json"
SOURCE_PACKAGE = ROOT / "output"
OUTPUT_ROOT = ROOT / "outputs"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def foreground_mask(camera, shape: tuple[int, int]) -> np.ndarray:
    segmentation = np.asarray(camera.get_picture("Segmentation"))
    mask = segmentation[..., 0] != 0
    if mask.shape != shape:
        mask = cv2.resize(mask.astype(np.uint8), (shape[1], shape[0]),
                          interpolation=cv2.INTER_NEAREST).astype(bool)
    return mask.astype(np.uint8) * 255


def make_sim(document: dict, model_dir: Path):
    local = np.append(np.asarray(document["metadata"]["application_point"]["local_on_link"],
                                 dtype=np.float32), np.float32(1.0))
    target = (float(document["joint_travel_personalized_actuation"]["target_displacement"])
              * float(document["intended_joint_direction_sign"]))
    os.environ["FORCESAPIEN_CAMERA_TARGET_DISPLACEMENT"] = str(target)
    if document["joint_type"] == "prismatic":
        direction = np.asarray(document["requested_force_direction_world"], dtype=np.float32)
        sim = prismatic.setup_sim(
            model_dir, document["joint_name"], document["selected_link"], 1920, 1080,
            direction, override_point=local,
            override_strategy=document.get("contact_strategy"),
        )
        return sim, sim.cabinet, sim.joint_index, prismatic.render_panel
    sim = revolute.setup_sim(
        model_dir, document["joint_name"], document["selected_link"], 1920, 1080,
        float(document["q_start"]), override_point=local,
        override_strategy=document.get("contact_strategy"), asset_scale=1.0,
    )
    return sim, sim.laptop, sim.joint_index, revolute.render_panel


def contact_sheet(rgb_paths: list[Path], destination: Path) -> None:
    indices = np.linspace(0, len(rgb_paths) - 1, 7, dtype=int)
    frames = [cv2.imread(str(rgb_paths[index]), cv2.IMREAD_COLOR) for index in indices]
    frames.append(np.full_like(frames[0], 255))
    cv2.imwrite(str(destination), np.vstack((np.hstack(frames[:4]), np.hstack(frames[4:]))))


def export_object(source: Path, destination: Path, model_dir: Path) -> dict:
    destination.mkdir(parents=True)
    for name in ("final_video.mp4", "simulation.json"):
        shutil.copy2(source / name, destination / name)
    document = json.loads((source / "simulation.json").read_text(encoding="utf-8"))
    samples = document["samples"]["force"]
    rgb_dir = destination / "frames" / "rgb"
    mask_dir = destination / "frames" / "masks"
    rgb_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)

    ensure_neutral_missing_textures(model_dir)
    sim, articulation, joint_index, render = make_sim(document, model_dir)
    rgb_paths: list[Path] = []
    for index, sample in enumerate(samples):
        set_coordinate(articulation, joint_index, float(sample["q"]))
        rgb = render(sim)  # take_picture occurs here; no overlay drawing is called
        mask = foreground_mask(sim.camera, rgb.shape[:2])
        name = f"frame_{index:04d}.png"
        rgb_path, mask_path = rgb_dir / name, mask_dir / name
        if not cv2.imwrite(str(rgb_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
            raise RuntimeError(f"could not write {rgb_path}")
        if not cv2.imwrite(str(mask_path), mask):
            raise RuntimeError(f"could not write {mask_path}")
        rgb_paths.append(rgb_path)

    contact_sheet(rgb_paths, destination / "contact_sheet.png")
    original_video_sha = sha256(source / "final_video.mp4")
    copied_video_sha = sha256(destination / "final_video.mp4")
    log = (source / "run.log").read_text(encoding="utf-8", errors="replace")
    log += (
        "\n[clean_frame_export]\n"
        "status=SUCCESS\nphysics_reexecuted=false\ntrajectory_source=simulation.json:samples.force.q\n"
        "final_video_action=copied_byte_for_byte\n"
        f"final_video_sha256={copied_video_sha}\n"
        f"clean_rgb_frames={len(rgb_paths)}\nclean_masks={len(rgb_paths)}\n"
        "mask_encoding=uint8_background_0_object_255\n"
        "overlays_in_clean_rgb=false\ncontact_sheet_source=clean_rgb\n"
    )
    (destination / "run.log").write_text(log, encoding="utf-8")
    if copied_video_sha != original_video_sha:
        raise RuntimeError("copied final_video.mp4 hash mismatch")
    return {"frames": len(rgb_paths), "video_sha256": copied_video_sha}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-package", type=Path, default=SOURCE_PACKAGE)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "final_dataset")
    args = parser.parse_args()
    source, dataset, output = (args.source_package.resolve(), args.dataset_root.resolve(),
                               OUTPUT_ROOT.resolve())
    if output.exists():
        raise SystemExit(f"refusing to overwrite output: {output}")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    output.mkdir(parents=True)

    results, failures = [], []
    with tempfile.TemporaryDirectory(prefix="forcesapien_clean_models_") as temporary:
        scaled_root = Path(temporary)
        for row in manifest["objects"]:
            object_id, slug = str(row["object_id"]), row["directory"]
            category = str(row["object_name"])
            model_dir = scaled_root / object_id
            prepare_scaled_object(dataset / object_id, model_dir,
                                  float(config["objects"][object_id]["scale"]))
            try:
                result = export_object(source / "objects" / slug,
                                       output / category, model_dir)
                results.append({"object_id": object_id, "directory": category, **result})
                print(f"SUCCESS {category}: {result['frames']} RGB/mask pairs", flush=True)
            except Exception as exc:
                failures.append({"object_id": object_id, "directory": category, "error": str(exc)})
                print(f"FAILED {category}: {exc}", flush=True)

    print(f"SUMMARY successful={len(results)} failed={len(failures)} output={output}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
