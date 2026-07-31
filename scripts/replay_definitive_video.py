#!/usr/bin/env python3
"""Replay an immutable definitive trajectory with object-only camera framing."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import imageio.v2 as imageio
import numpy as np
from PIL import Image

from scripts import render_prismatic_video as prismatic
from scripts import render_revolute_video as revolute


def ensure_neutral_missing_textures(model_dir: Path) -> None:
    for material in model_dir.glob("textured_objs/*.mtl"):
        for line in material.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not line.strip().lower().startswith("map_kd "):
                continue
            texture = (material.parent / line.split(None, 1)[1].strip()).resolve()
            if not texture.exists():
                texture.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (4, 4), (190, 190, 190)).save(texture)


def set_coordinate(articulation, joint_index: int, value: float) -> None:
    qpos = articulation.get_qpos().copy()
    qpos[joint_index] = value
    articulation.set_qpos(qpos)


def replay_revolute(document: dict, model_dir: Path, output: Path) -> None:
    samples = document["samples"]["force"]
    local_point = np.append(np.asarray(document["metadata"]["application_point"]["local_on_link"], dtype=np.float32), 1.0)
    os.environ["FORCESAPIEN_CAMERA_TARGET_DISPLACEMENT"] = str(
        document["joint_travel_personalized_actuation"]["target_displacement"]
        * document["intended_joint_direction_sign"]
    )
    sim = revolute.setup_sim(
        model_dir, document["joint_name"], document["selected_link"], 1920, 1080,
        float(document["q_start"]), override_point=local_point,
        override_strategy=document.get("contact_strategy"), asset_scale=1.0,
    )
    history: list[np.ndarray] = []
    with imageio.get_writer(output, fps=int(document["fps"]), codec="libx264",
                            pixelformat="yuv420p", quality=8, macro_block_size=1) as writer:
        for sample in samples:
            set_coordinate(sim.laptop, sim.joint_index, float(sample["q"]))
            point = np.asarray(sample["application_point_world"], dtype=np.float32)
            history.append(point)
            force_detail = SimpleNamespace(
                joint_origin_world=np.asarray(sample["joint_origin_world"], dtype=np.float32),
                joint_axis_world=np.asarray(sample["joint_axis_world"], dtype=np.float32),
                radius_perpendicular_world=np.asarray(sample["radius_perpendicular_world"], dtype=np.float32),
                force_application_point_world=point,
                tangential_direction_world=np.asarray(sample["tangential_direction_world"], dtype=np.float32),
            )
            frame = revolute.render_panel(sim)
            revolute.draw_revolute_geometry_overlay(frame, sim, force_detail, revolute.COLOR_ACCENT)
            active = float(sample["time_s"]) < 2.0
            revolute.draw_force_annotation(
                frame, sim, force_detail.tangential_direction_world,
                float(document["force_magnitude"]), 1920, 1080,
                revolute.COLOR_ACCENT, history, active,
            )
            writer.append_data(frame)


def replay_prismatic(document: dict, model_dir: Path, output: Path) -> None:
    samples = document["samples"]["force"]
    local_point = np.append(np.asarray(document["metadata"]["application_point"]["local_on_link"], dtype=np.float32), 1.0)
    direction = np.asarray(document["requested_force_direction_world"], dtype=np.float32)
    os.environ["FORCESAPIEN_CAMERA_TARGET_DISPLACEMENT"] = str(
        document["joint_travel_personalized_actuation"]["target_displacement"]
        * document["intended_joint_direction_sign"]
    )
    sim = prismatic.setup_sim(
        model_dir, document["joint_name"], document["selected_link"], 1920, 1080,
        direction, override_point=local_point, override_strategy=document.get("contact_strategy"),
    )
    history: list[np.ndarray] = []
    with imageio.get_writer(output, fps=int(document["fps"]), codec="libx264",
                            pixelformat="yuv420p", quality=8, macro_block_size=1) as writer:
        for sample in samples:
            set_coordinate(sim.cabinet, sim.joint_index, float(sample["q"]))
            point = np.asarray(sample["application_point_world"], dtype=np.float32)
            history.append(point)
            force_direction = np.asarray(sample["applied_force_world"], dtype=np.float32)
            force_direction /= np.linalg.norm(force_direction) or 1.0
            frame = prismatic.render_panel(sim)
            prismatic.draw_prismatic_axis(frame, sim)
            active = float(sample["time_s"]) < 2.0
            prismatic.draw_force_annotation(
                frame, sim, force_direction, float(document["force_magnitude"]),
                prismatic.COLOR_ACCENT, history, active,
            )
            writer.append_data(frame)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulation", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    document = json.loads(args.simulation.read_text())
    ensure_neutral_missing_textures(args.model_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if document["joint_type"] == "prismatic":
        replay_prismatic(document, args.model_dir, args.output)
    else:
        replay_revolute(document, args.model_dir, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
