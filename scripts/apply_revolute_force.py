#!/usr/bin/env python3
"""Apply external force to a revolute joint link and save simulation samples."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import sapien

try:
    from paths import resolve_model_dir
    from simulation_json import build_metadata, sample_time_from_step
except ModuleNotFoundError:
    from scripts.paths import resolve_model_dir
    from scripts.simulation_json import build_metadata, sample_time_from_step


TIMESTEP = 1.0 / 240.0
LINEAR_DAMPING = 0.0
ANGULAR_DAMPING = 0.02


def output_paths(model_dir: Path, output_root: Path, json_output: str | None) -> Path:
    object_dir = output_root / f"{model_dir.name}_output"
    object_dir.mkdir(parents=True, exist_ok=True)
    return Path(json_output).resolve() if json_output else object_dir / "simulation.json"


def clear_object_output(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    for old_output in path.parent.iterdir():
        if old_output.is_file() and old_output.name in {"simulation.json", "final_video.mp4"}:
            old_output.unlink()


def load_application_point_override(object_dir: Path, link_name: str) -> tuple[np.ndarray | None, str | None]:
    model_name = object_dir.name.removesuffix("_output")
    override_path = object_dir.parent / "application_point_overrides" / f"{model_name}.json"
    if not override_path.exists():
        return None, None
    data = json.loads(override_path.read_text())
    if data.get("link") != link_name:
        return None, None
    point = np.append(np.asarray(data["local_point"], dtype=np.float32), np.float32(1.0))
    return point, f"manual candidate {data.get('candidate_id')} from application_point_override.json"


def mesh_vertices(mesh_path: Path) -> np.ndarray:
    vertices: list[list[float]] = []
    with mesh_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if not vertices:
        raise RuntimeError(f"No vertices found in {mesh_path}")
    return np.asarray(vertices, dtype=np.float32)


def visual_origin(visual: ET.Element) -> np.ndarray:
    origin = visual.find("origin")
    if origin is None:
        return np.zeros(3, dtype=np.float32)
    return np.asarray([float(value) for value in origin.attrib.get("xyz", "0 0 0").split()], dtype=np.float32)


def pick_handle_point_local(model_dir: Path, link_name: str) -> np.ndarray | None:
    tree = ET.parse(model_dir / "mobility.urdf")
    link = tree.find(f".//link[@name='{link_name}']")
    if link is None:
        return None

    handle_vertices: list[np.ndarray] = []
    for visual in link.findall("visual"):
        if "handle" not in visual.attrib.get("name", "").lower():
            continue
        mesh = visual.find("./geometry/mesh")
        if mesh is None or "filename" not in mesh.attrib:
            continue
        handle_vertices.append(mesh_vertices(model_dir / mesh.attrib["filename"]) + visual_origin(visual))

    if not handle_vertices:
        return None

    vertices = np.concatenate(handle_vertices, axis=0)
    center = 0.5 * (vertices.min(axis=0) + vertices.max(axis=0))
    return np.append(center.astype(np.float32), np.float32(1.0))


def pick_link_edge_point(link: sapien.physx.PhysxArticulationLinkComponent) -> np.ndarray:
    aabb = link.compute_global_aabb_tight()
    return np.array([aabb[0, 0], 0.5 * (aabb[0, 1] + aabb[1, 1]), aabb[1, 2], 1.0], dtype=np.float32)


def application_point_world(link: sapien.physx.PhysxArticulationLinkComponent, local_point: np.ndarray) -> np.ndarray:
    return (link.get_entity_pose().to_transformation_matrix() @ local_point)[:3].astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="11691")
    parser.add_argument("--joint", default="joint_1")
    parser.add_argument("--link", default="link_1")
    parser.add_argument("--force", type=float, default=0.5)
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--initial-angle", type=float, default=-1.5)
    parser.add_argument("--direction", nargs=3, type=float, default=[0.0, 0.0, 1.0])
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--json-output", default=None)
    parser.add_argument("--keep-old", action="store_true")
    args = parser.parse_args()

    model_dir = resolve_model_dir(args.model_dir)
    if not (model_dir / "mobility.urdf").exists():
        raise FileNotFoundError(model_dir / "mobility.urdf")

    json_output = output_paths(model_dir, Path(args.output_root).resolve(), args.json_output)
    if not args.keep_old:
        clear_object_output(json_output)

    scene = sapien.Scene()
    scene.set_timestep(TIMESTEP)
    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    articulation = loader.load(str(model_dir / "mobility.urdf"))
    target_joint = articulation.find_joint_by_name(args.joint)
    target_link = articulation.find_link_by_name(args.link)
    if target_joint is None or target_link is None:
        raise RuntimeError(f"Could not find {args.joint}/{args.link}.")

    active_joints = list(articulation.get_active_joints())
    joint_index = active_joints.index(target_joint)
    qpos = np.zeros_like(articulation.get_qpos(), dtype=np.float32)
    qpos[joint_index] = args.initial_angle
    articulation.set_qpos(qpos)

    for joint in articulation.get_joints():
        joint.set_drive_property(0.0, 0.0, 0.0)
    for link in articulation.get_links():
        link.disable_gravity = True
        link.linear_damping = LINEAR_DAMPING
        link.angular_damping = ANGULAR_DAMPING

    direction = np.asarray(args.direction, dtype=np.float32)
    direction /= np.linalg.norm(direction) or 1.0
    force_world = direction * args.force
    local_application_point = pick_handle_point_local(model_dir, args.link)
    application_point_strategy = "center of handle mesh on selected link"
    if local_application_point is None:
        override_point, override_strategy = load_application_point_override(json_output.parent, args.link)
        if override_point is not None:
            local_application_point = override_point
            application_point_strategy = override_strategy or "manual application point override"
        else:
            local_application_point = np.linalg.inv(target_link.get_entity_pose().to_transformation_matrix()) @ pick_link_edge_point(target_link)
            application_point_strategy = "free edge of selected link from tight AABB at initial pose"

    samples = []
    steps = max(1, int(args.seconds / TIMESTEP))
    sample_interval = max(1, round(1.0 / (TIMESTEP * args.fps)))
    for step in range(steps):
        point = application_point_world(target_link, local_application_point)
        target_link.add_force_at_point(force_world, point, "force")
        scene.step()

        if step % sample_interval == 0 or step == steps - 1:
            angle = float(articulation.get_qpos()[joint_index])
            samples.append(
                {
                    "time_s": sample_time_from_step(step, TIMESTEP),
                    "joint_angle_rad": angle,
                    "joint_angle_deg": math.degrees(angle),
                    "joint_velocity_rad_s": float(articulation.get_qvel()[joint_index]),
                    "application_point_world": application_point_world(target_link, local_application_point).astype(float).tolist(),
                    "applied_force_world": force_world.astype(float).tolist(),
                }
            )

    metadata = build_metadata(
        model_dir=model_dir,
        mode="apply",
        joint_type="revolute",
        joint_name=args.joint,
        link_name=args.link,
        json_output=json_output,
        fps=args.fps,
        requested_seconds=args.seconds,
        simulated_seconds=steps * TIMESTEP,
        timestep_s=TIMESTEP,
        sample_interval_s=sample_interval * TIMESTEP,
        actuation={
            "initial_joint_position": {
                "rad": args.initial_angle,
                "deg": math.degrees(args.initial_angle),
            },
            "force": {
                "magnitude_n": args.force,
                "direction_world": direction.astype(float).tolist(),
            },
            "joint_limits_rad": target_joint.get_limit().tolist(),
        },
        application_point={
            "strategy": application_point_strategy,
            "local_on_link": local_application_point[:3].astype(float).tolist(),
        },
        articulation=articulation,
        limit_key="limits_rad",
        linear_damping=LINEAR_DAMPING,
        angular_damping=ANGULAR_DAMPING,
    )

    with json_output.open("w", encoding="utf-8") as f:
        json.dump({"metadata": metadata, "samples": {"force": samples}}, f, indent=2)

    print(f"Wrote {json_output}")
    print(f"Final angle: {samples[-1]['joint_angle_deg']:.2f} deg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
