#!/usr/bin/env python3
"""Measure object-only occupancy from SAPIEN segmentation on a render node."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts import render_prismatic_video as prismatic
from scripts import render_revolute_video as revolute
from scripts.replay_definitive_video import ensure_neutral_missing_textures, set_coordinate


def bbox_metrics(mask: np.ndarray) -> tuple[float, bool]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise RuntimeError("empty object segmentation")
    height, width = mask.shape
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    occupancy = max((x1 - x0 + 1) / width, (y1 - y0 + 1) / height)
    clipping = x0 == 0 or y0 == 0 or x1 == width - 1 or y1 == height - 1
    return float(occupancy), clipping


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    rows = []
    for object_id in config["object_ids"]:
        row = config["objects"][object_id]
        directory = next(args.package.glob(f"objects/*_{object_id}"))
        document = json.loads((directory / "simulation.json").read_text())
        model_dir = args.runtime_root / object_id
        ensure_neutral_missing_textures(model_dir)
        local = np.append(np.asarray(document["metadata"]["application_point"]["local_on_link"], dtype=np.float32), 1.0)
        target = float(document["joint_travel_personalized_actuation"]["target_displacement"]) * float(document["intended_joint_direction_sign"])
        import os
        os.environ["FORCESAPIEN_CAMERA_TARGET_DISPLACEMENT"] = str(target)
        if document["joint_type"] == "prismatic":
            sim = prismatic.setup_sim(
                model_dir, document["joint_name"], document["selected_link"], 1920, 1080,
                np.asarray(document["requested_force_direction_world"], dtype=np.float32),
                override_point=local, override_strategy=document.get("contact_strategy"),
            )
            project = prismatic.project
        else:
            sim = revolute.setup_sim(
                model_dir, document["joint_name"], document["selected_link"], 1920, 1080,
                float(document["q_start"]), override_point=local,
                override_strategy=document.get("contact_strategy"), asset_scale=1.0,
            )
            project = revolute.project
        samples = document["samples"]["force"]
        indices = sorted(set([0, min(len(samples)-1, 30), min(len(samples)-1, 57),
                              min(len(samples)-1, 60), min(len(samples)-1, 63),
                              len(samples)//2, len(samples)-1]))
        occupancies = []
        clipping = False
        axis_lengths = []
        articulation = sim.cabinet if document["joint_type"] == "prismatic" else sim.laptop
        for index in indices:
            sample = samples[index]
            set_coordinate(articulation, sim.joint_index, float(sample["q"]))
            sim.scene.update_render()
            sim.camera.take_picture()
            segmentation = sim.camera.get_picture("Segmentation")
            occupancy, clipped = bbox_metrics(segmentation[..., 0] != 0)
            occupancies.append(occupancy)
            clipping = clipping or clipped
            origin = np.asarray(sample["joint_origin_world"], dtype=np.float32)
            axis = np.asarray(sample["joint_axis_world"], dtype=np.float32)
            axis /= np.linalg.norm(axis) or 1.0
            half_length = 0.30 * float(sim.camera_object_dominant_size)
            a, b = project(sim.camera, origin - axis * half_length), project(sim.camera, origin + axis * half_length)
            if a is not None and b is not None:
                axis_lengths.append(float(np.linalg.norm(np.asarray(a) - np.asarray(b))))
        minimum, maximum = min(occupancies), max(occupancies)
        rows.append({
            "object_id": object_id, "object": row["name"],
            "minimum_occupancy": minimum, "maximum_occupancy": maximum,
            "camera_distance_m": float(sim.camera_distance),
            "red_axis_projected_length_px_min": min(axis_lengths),
            "red_axis_projected_length_px_max": max(axis_lengths),
            "clipping": clipping,
            "verdict": "PASS" if not clipping and maximum >= 0.50 and maximum <= 0.85 else "FAIL",
            "note": "occupancy is measured from object-only SAPIEN segmentation; overlays are excluded",
        })
    args.output.write_text(json.dumps({"schema_version": 1, "resolution": [1920, 1080], "rows": rows}, indent=2) + "\n")
    print(json.dumps(rows, indent=2))
    return 0 if all(row["verdict"] == "PASS" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
