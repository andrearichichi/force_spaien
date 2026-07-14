#!/usr/bin/env python3
"""Render a SAPIEN video for a prismatic joint force simulation."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import sapien

try:
    from physics_force_model import (
        ForceStep,
        Resistance,
        compute_force_profile_scale,
        compute_prismatic_generalized_force,
        compute_resistance,
        scaled_force_step as model_scaled_force_step,
    )
    from simulation_json import (
        build_metadata,
        build_summary,
        motion_document,
        sample_time_from_frame,
        sample_time_from_step,
        validation_for_motion,
    )
except ModuleNotFoundError:
    from scripts.physics_force_model import (
        ForceStep,
        Resistance,
        compute_force_profile_scale,
        compute_prismatic_generalized_force,
        compute_resistance,
        scaled_force_step as model_scaled_force_step,
    )
    from scripts.simulation_json import (
        build_metadata,
        build_summary,
        motion_document,
        sample_time_from_frame,
        sample_time_from_step,
        validation_for_motion,
    )


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = REPO_ROOT / "dataset"


def resolve_model_dir(model_dir_arg: str | Path) -> Path:
    model_dir = Path(model_dir_arg).expanduser()
    if model_dir.is_absolute():
        return model_dir.resolve()

    direct = (Path.cwd() / model_dir).resolve()
    if (direct / "mobility.urdf").exists():
        return direct

    dataset_model = (DATASET_DIR / model_dir).resolve()
    if (dataset_model / "mobility.urdf").exists():
        return dataset_model

    return direct


FONT_REGULAR = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
FONT_BOLD = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
FONT_SMALL = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
TIMESTEP = 1.0 / 240.0
LINEAR_DAMPING = 0.0
ANGULAR_DAMPING = 0.02
END_HOLD_MOTION_THRESHOLD = 1e-3
CAMERA_ZOOM_OUT = 1.35
DEFAULT_VIDEO_WIDTH = 1920
DEFAULT_PANEL_HEIGHT = 1080
DEFAULT_INFO_HEIGHT = 0
COLOR_BG = (242, 244, 246)
COLOR_PANEL = (255, 255, 255)
COLOR_PANEL_2 = (248, 249, 250)
COLOR_BORDER = (210, 216, 224)
COLOR_GRID = (226, 230, 235)
COLOR_TEXT = (24, 28, 34)
COLOR_MUTED = (101, 112, 126)
COLOR_ACCENT = (0, 122, 255)
COLOR_ALT = (255, 59, 48)


def should_append_end_hold(mode: str, hold_frames: int, final_motion: float) -> bool:
    """Use a small velocity threshold to avoid freezing active diagnostic motion."""
    if hold_frames <= 0 or mode == "never":
        return False
    if mode == "if-stopped":
        return final_motion <= END_HOLD_MOTION_THRESHOLD
    return True


def warn_if_holding_active_motion(final_motion: float) -> None:
    if final_motion <= END_HOLD_MOTION_THRESHOLD:
        return
    print(
        "WARNING: appending end-hold frames while final motion is still active.\n"
        "This may look like an abrupt stop in the video.\n"
        "Use --end-hold-seconds 0 or --end-hold-mode never for diagnostic videos."
    )


def force_profile_scale(args: argparse.Namespace, time_s: float) -> float:
    return compute_force_profile_scale(
        args.force_profile,
        time_s,
        start_time_s=args.force_start_time,
        duration_s=args.force_duration,
        ramp_time_s=args.force_ramp_time,
    )


def scaled_force_step(args: argparse.Namespace, time_s: float) -> ForceStep:
    return model_scaled_force_step(
        args.force,
        args.force_profile,
        time_s,
        start_time_s=args.force_start_time,
        duration_s=args.force_duration,
        ramp_time_s=args.force_ramp_time,
    )


def resistance_for_motion(applied: float, velocity: float, args: argparse.Namespace) -> Resistance:
    return compute_resistance(
        applied,
        velocity,
        static_friction=args.joint_static_friction,
        dynamic_friction=args.joint_dynamic_friction,
        viscous_damping=args.joint_viscous_damping,
        static_velocity_threshold=args.static_friction_velocity_threshold,
    )


def finite_range(values: list[float]) -> tuple[float, float]:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return 0.0, 1.0
    lo, hi = min(finite), max(finite)
    if abs(hi - lo) < 1e-12:
        pad = max(1.0, abs(hi)) * 0.05
        return lo - pad, hi + pad
    pad = (hi - lo) * 0.05
    return lo - pad, hi + pad


def draw_plot(path: Path, title: str, series: list[tuple[str, list[float], tuple[int, int, int]]]) -> None:
    width, height = 1000, 640
    margin_l, margin_r, margin_t, margin_b = 80, 30, 55, 70
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    cv2.putText(image, title, (margin_l, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_TEXT, 2, cv2.LINE_AA)
    max_len = max((len(values) for _, values, _ in series), default=0)
    if max_len <= 1:
        cv2.imwrite(str(path), image)
        return
    all_values = [value for _, values, _ in series for value in values]
    y_min, y_max = finite_range(all_values)
    x0, x1 = margin_l, width - margin_r
    y0, y1 = height - margin_b, margin_t
    cv2.rectangle(image, (x0, y1), (x1, y0), (220, 220, 220), 1)
    for tick in range(6):
        y = int(y0 + (y1 - y0) * tick / 5)
        cv2.line(image, (x0, y), (x1, y), (235, 235, 235), 1)
        value = y_min + (y_max - y_min) * tick / 5
        cv2.putText(image, f"{value:.3g}", (8, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_MUTED, 1, cv2.LINE_AA)
    for label, values, color in series:
        points = []
        for idx, value in enumerate(values):
            if not math.isfinite(value):
                continue
            x = int(x0 + (x1 - x0) * idx / (max_len - 1))
            y = int(y0 - (y0 - y1) * (value - y_min) / (y_max - y_min))
            points.append((x, y))
        if len(points) >= 2:
            cv2.polylines(image, [np.asarray(points, dtype=np.int32)], False, color, 2, cv2.LINE_AA)
    legend_y = height - 25
    legend_x = margin_l
    for label, _, color in series:
        cv2.line(image, (legend_x, legend_y), (legend_x + 30, legend_y), color, 3)
        cv2.putText(image, label, (legend_x + 38, legend_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXT, 1, cv2.LINE_AA)
        legend_x += 190
    cv2.imwrite(str(path), image)


def write_physics_diagnostics(output_dir: Path, samples: list[dict[str, object]], metadata: dict[str, object], validation: dict[str, object]) -> None:
    diagnostics_dir = output_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    columns = [
        "frame",
        "time_s",
        "phase",
        "q",
        "qdot",
        "qddot",
        "joint_position_m",
        "joint_velocity_m_s",
        "joint_acceleration_m_s2",
        "applied_linear_force_n",
        "applied_force_norm",
        "applied_generalized_force",
        "damping_torque_or_force",
        "friction_torque_or_force",
        "static_friction_force_n",
        "dynamic_friction_force_n",
        "damping_force_n",
        "net_force_n",
        "net_generalized_force",
        "mechanical_work_j",
        "joint_limit_distance_m",
        "joint_limit_distance",
        "settled_flag",
    ]
    with (diagnostics_dir / "physics_timeseries.tsv").open("w", encoding="utf-8") as f:
        f.write("\t".join(columns) + "\n")
        for sample in samples:
            f.write("\t".join(str(sample.get(column, "")) for column in columns) + "\n")

    def values(key: str) -> list[float]:
        return [float(sample.get(key, 0.0)) for sample in samples]

    draw_plot(
        diagnostics_dir / "force_torque_by_frame.png",
        "Applied and net generalized force",
        [
            ("applied N", values("applied_linear_force_n"), (0, 122, 255)),
            ("net N", values("net_force_n"), (255, 59, 48)),
        ],
    )
    draw_plot(
        diagnostics_dir / "q_qdot_qddot_by_frame.png",
        "q, qdot, qddot",
        [
            ("q m", values("joint_position_m"), (0, 122, 255)),
            ("qdot", values("joint_velocity_m_s"), (255, 59, 48)),
            ("qddot", values("joint_acceleration_m_s2"), (88, 86, 214)),
        ],
    )
    draw_plot(
        diagnostics_dir / "resistance_by_frame.png",
        "Resistance terms",
        [
            ("static", values("static_friction_force_n"), (255, 149, 0)),
            ("dynamic", values("dynamic_friction_force_n"), (255, 59, 48)),
            ("damping", values("damping_force_n"), (88, 86, 214)),
        ],
    )

    final = samples[-1] if samples else {}
    lines = [
        "# Physics Diagnostics",
        "",
        "## Model",
        "",
        "- Motion source: physical force through SAPIEN integration",
        f"- Force application mode: `{metadata.get('simulation_config', {}).get('force_application_mode', 'unknown')}`",
        "- Prismatic generalized force is applied along the selected joint axis.",
        "- No prescribed q(t) is used in force-driven mode.",
        "- End hold is presentation-only and is not part of settling physics.",
        "",
        "## Current Code Behavior",
        "",
        "- Main render loop uses SAPIEN `scene.step()` for state evolution.",
        "- In `generalized` mode, the net joint force is applied with `set_qf`.",
        "- In `external_link_force` mode, SAPIEN `add_force_at_point` is used and only resistance is applied through joint generalized force.",
        "- Joint drives are set to zero stiffness/damping/force limit.",
        f"- Gravity enabled: `{metadata.get('actuation', {}).get('force', {}).get('gravity_enabled', False)}`",
        "- URDF joint dynamics are logged under `metadata.physics.urdf_joint_dynamics`; the render path exposes explicit friction/damping overrides.",
        "",
        "## Result",
        "",
        f"- Sample count: {len(samples)}",
        f"- Final q: {final.get('joint_position_m')}",
        f"- Final qdot: {final.get('joint_velocity_m_s')}",
        f"- Final qddot: {final.get('joint_acceleration_m_s2')}",
        f"- Final net force: {final.get('net_force_n')}",
        f"- Final work proxy: {final.get('mechanical_work_j')}",
        f"- Motion completed: {validation.get('motion_completed')}",
        f"- Warnings: {validation.get('warnings', [])}",
        "",
        "## Artifacts",
        "",
        "- `physics_timeseries.tsv`",
        "- `force_torque_by_frame.png`",
        "- `q_qdot_qddot_by_frame.png`",
        "- `resistance_by_frame.png`",
    ]
    (diagnostics_dir / "physics_diagnostics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def look_at_pose(eye: np.ndarray, target: np.ndarray) -> sapien.Pose:
    forward = target - eye
    forward = forward / (np.linalg.norm(forward) or 1.0)
    left = np.cross(np.array([0.0, 0.0, 1.0], dtype=np.float32), forward)
    left = left / (np.linalg.norm(left) or 1.0)
    up = np.cross(forward, left)
    mat = np.eye(4, dtype=np.float32)
    mat[:3, 0] = forward
    mat[:3, 1] = left
    mat[:3, 2] = up
    mat[:3, 3] = eye
    return sapien.Pose(mat)


def zoomed_eye(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    return target + (eye - target) * CAMERA_ZOOM_OUT


def border_connected_mask(mask: np.ndarray) -> np.ndarray:
    count, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    if count <= 1:
        return mask
    border_labels = np.unique(
        np.concatenate(
            [
                labels[0, :],
                labels[-1, :],
                labels[:, 0],
                labels[:, -1],
            ]
        )
    )
    border_labels = border_labels[border_labels != 0]
    if len(border_labels) == 0:
        return np.zeros_like(mask, dtype=bool)
    return np.isin(labels, border_labels)


def background_mask_from_render(camera: sapien.render.RenderCameraComponent, image: np.ndarray) -> np.ndarray:
    try:
        segmentation = camera.get_picture("Segmentation")
    except Exception:
        channel_range = image.max(axis=2) - image.min(axis=2)
        luminance = image.mean(axis=2)
        return border_connected_mask((channel_range < 70) & (luminance > 128))

    mask = segmentation[..., 0] == 0
    if mask.shape != image.shape[:2]:
        mask = cv2.resize(mask.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
    return mask


def output_paths(model_dir: Path, output_root: Path, output: str | None, json_output: str | None) -> tuple[Path, Path]:
    object_dir = output_root / f"{model_dir.name}_output"
    object_dir.mkdir(parents=True, exist_ok=True)
    video_path = Path(output).resolve() if output else object_dir / "final_video.mp4"
    json_path = Path(json_output).resolve() if json_output else object_dir / "simulation.json"
    return video_path, json_path


def clear_object_output(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    for old_output in path.parent.iterdir():
        if old_output.is_file() and old_output.name in {"simulation.json", "final_video.mp4"}:
            old_output.unlink()


def load_physics_only_articulation(scene: sapien.Scene, model_dir: Path) -> sapien.physx.PhysxArticulation:
    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    articulation_builders, actor_builders, _cameras = loader.parse(str(model_dir / "mobility.urdf"))
    if len(articulation_builders) != 1 or actor_builders:
        raise RuntimeError("Expected one articulation and no standalone actors in URDF.")
    builder = articulation_builders[0]
    for link_builder in builder.link_builders:
        link_builder.visual_records = []
    return builder.build()


@dataclass
class DrawerSim:
    scene: sapien.Scene
    cabinet: sapien.physx.PhysxArticulation
    drawer: sapien.physx.PhysxArticulationLinkComponent
    joint_index: int
    camera: sapien.render.RenderCameraComponent
    local_application_point: np.ndarray
    positive_pull_dir_world: np.ndarray
    application_point_strategy: str


def _mesh_vertices(mesh_path: Path) -> np.ndarray:
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


def _visual_origin(visual: ET.Element) -> np.ndarray:
    origin = visual.find("origin")
    if origin is None:
        return np.zeros(3, dtype=np.float32)
    xyz = origin.attrib.get("xyz", "0 0 0")
    return np.asarray([float(v) for v in xyz.split()], dtype=np.float32)


def pick_handle_pull_point_local(model_dir: Path, drawer_index: int) -> np.ndarray:
    """Pick the center of the drawer handle mesh in drawer-link coordinates."""
    tree = ET.parse(model_dir / "mobility.urdf")
    link = tree.find(f".//link[@name='link_{drawer_index}']")
    if link is None:
        raise RuntimeError(f"Could not find link_{drawer_index} in mobility.urdf")

    handle_vertices: list[np.ndarray] = []
    for visual in link.findall("visual"):
        name = visual.attrib.get("name", "")
        if not name.startswith("handle-"):
            continue

        mesh = visual.find("./geometry/mesh")
        if mesh is None or "filename" not in mesh.attrib:
            continue

        vertices = _mesh_vertices(model_dir / mesh.attrib["filename"])
        handle_vertices.append(vertices + _visual_origin(visual))

    if not handle_vertices:
        raise RuntimeError(f"Could not find a handle visual on link_{drawer_index}")

    vertices = np.concatenate(handle_vertices, axis=0)
    return (0.5 * (vertices.min(axis=0) + vertices.max(axis=0))).astype(np.float32)


def pick_drawer_pull_point(drawer: sapien.physx.PhysxArticulationLinkComponent, direction: np.ndarray) -> np.ndarray:
    aabb = drawer.compute_global_aabb_tight()
    point = 0.5 * (aabb[0] + aabb[1])
    axis = int(np.argmax(np.abs(direction)))
    point[axis] = aabb[1, axis] if direction[axis] >= 0 else aabb[0, axis]
    return np.append(point.astype(np.float32), np.float32(1.0))


def pick_drawer_center_point(drawer: sapien.physx.PhysxArticulationLinkComponent) -> np.ndarray:
    aabb = drawer.compute_global_aabb_tight()
    return np.append((0.5 * (aabb[0] + aabb[1])).astype(np.float32), np.float32(1.0))


def pick_drawer_bbox_extreme_point(drawer: sapien.physx.PhysxArticulationLinkComponent, direction: np.ndarray) -> np.ndarray:
    return pick_drawer_pull_point(drawer, direction)


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
        handle_vertices.append(_mesh_vertices(model_dir / mesh.attrib["filename"]) + _visual_origin(visual))

    if not handle_vertices:
        return None

    vertices = np.concatenate(handle_vertices, axis=0)
    center = 0.5 * (vertices.min(axis=0) + vertices.max(axis=0))
    return np.append(center.astype(np.float32), np.float32(1.0))


def application_point_world_on_link(link: sapien.physx.PhysxArticulationLinkComponent, local_point: np.ndarray) -> np.ndarray:
    return (link.get_entity_pose().to_transformation_matrix() @ local_point)[:3].astype(np.float32)


def pick_link_face_point(link: sapien.physx.PhysxArticulationLinkComponent, direction: np.ndarray) -> np.ndarray:
    aabb = link.compute_global_aabb_tight()
    point = 0.5 * (aabb[0] + aabb[1])
    axis = int(np.argmax(np.abs(direction)))
    point[axis] = aabb[1, axis] if direction[axis] >= 0 else aabb[0, axis]
    return np.append(point.astype(np.float32), np.float32(1.0))


def explicit_contact_point(args: argparse.Namespace) -> tuple[np.ndarray | None, str | None]:
    if args.contact_point_local is None:
        return None, args.contact_point_strategy
    point = np.append(np.asarray(args.contact_point_local, dtype=np.float32), np.float32(1.0))
    return point, args.contact_point_strategy


def setup_sim(
    model_dir: Path,
    joint_name: str,
    link_name: str,
    width: int,
    height: int,
    force_dir: np.ndarray,
    override_point: np.ndarray | None = None,
    override_strategy: str | None = None,
    link_linear_damping: float = LINEAR_DAMPING,
    link_angular_damping: float = ANGULAR_DAMPING,
    disable_gravity: bool = True,
) -> DrawerSim:
    scene = sapien.Scene()
    scene.set_timestep(TIMESTEP)
    scene.set_ambient_light([0.78, 0.80, 0.84])
    scene.add_directional_light([0.2, -0.45, -1.0], [1.0, 1.0, 0.96], shadow=False)
    scene.add_directional_light([-0.7, 0.25, -1.0], [0.42, 0.48, 0.56], shadow=False)

    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    cabinet = loader.load(str(model_dir / "mobility.urdf"))
    cabinet.set_qpos(np.zeros_like(cabinet.get_qpos(), dtype=np.float32))

    for joint in cabinet.get_joints():
        joint.set_drive_property(0.0, 0.0, 0.0)
    for link in cabinet.get_links():
        link.disable_gravity = disable_gravity
        link.linear_damping = link_linear_damping
        link.angular_damping = link_angular_damping

    joint = cabinet.find_joint_by_name(joint_name)
    drawer = cabinet.find_link_by_name(link_name)
    if joint is None or drawer is None:
        raise RuntimeError(f"Could not find {joint_name}/{link_name}.")

    strategy = override_strategy
    if override_point is not None:
        local_application_point = override_point
        application_point_strategy = strategy or "user_given"
    else:
        if strategy in {"moving_link_bbox_extreme", "mesh_surface_farthest_point"}:
            world_point = pick_drawer_bbox_extreme_point(drawer, force_dir)
            local_application_point = np.linalg.inv(drawer.get_entity_pose().to_transformation_matrix()) @ world_point
            application_point_strategy = strategy
        elif strategy == "farthest_from_joint_axis":
            world_point = pick_drawer_center_point(drawer)
            local_application_point = np.linalg.inv(drawer.get_entity_pose().to_transformation_matrix()) @ world_point
            application_point_strategy = "moving_link_center_for_prismatic_axis_force"
        else:
            handle_point_local = pick_handle_point_local(model_dir, link_name)
            if handle_point_local is not None:
                local_application_point = handle_point_local
                application_point_strategy = "center of handle mesh on selected link"
            else:
                local_application_point = np.linalg.inv(drawer.get_entity_pose().to_transformation_matrix()) @ pick_drawer_pull_point(drawer, force_dir)
                application_point_strategy = "center of selected link face along force direction"
            if strategy:
                application_point_strategy = strategy
    joint_index = list(cabinet.get_active_joints()).index(joint)

    base_qpos = cabinet.get_qpos()
    base_point = (drawer.get_entity_pose().to_transformation_matrix() @ local_application_point)[:3]
    probe_qpos = base_qpos.copy()
    probe_qpos[joint_index] += 0.01
    cabinet.set_qpos(probe_qpos)
    probe_point = (drawer.get_entity_pose().to_transformation_matrix() @ local_application_point)[:3]
    cabinet.set_qpos(base_qpos)
    positive_pull_dir_world = (probe_point - base_point).astype(np.float32)
    positive_pull_dir_world /= np.linalg.norm(positive_pull_dir_world) or 1.0

    camera = scene.add_camera("camera", width, height, math.radians(44), 0.01, 20.0)
    camera_eye = np.array([-1.45, -1.55, 0.86], dtype=np.float32)
    camera_target = np.array([0.0, -0.04, 0.06], dtype=np.float32)
    camera.set_entity_pose(
        look_at_pose(
            zoomed_eye(camera_eye, camera_target),
            camera_target,
        )
    )

    return DrawerSim(scene, cabinet, drawer, joint_index, camera, local_application_point, positive_pull_dir_world, application_point_strategy)


def application_point_world(sim: DrawerSim) -> np.ndarray:
    point = sim.drawer.get_entity_pose().to_transformation_matrix() @ sim.local_application_point
    return point[:3].astype(np.float32)


def project(camera: sapien.render.RenderCameraComponent, point: np.ndarray) -> tuple[int, int] | None:
    camera_point = camera.get_extrinsic_matrix() @ np.array([point[0], point[1], point[2], 1.0], dtype=np.float32)
    if camera_point[2] <= 0:
        return None
    uvw = camera.get_intrinsic_matrix() @ camera_point
    return int(round(uvw[0] / uvw[2])), int(round(uvw[1] / uvw[2]))


def render_panel(sim: DrawerSim) -> np.ndarray:
    sim.scene.update_render()
    sim.camera.take_picture()
    image = (sim.camera.get_picture("Color")[..., :3].clip(0, 1) * 255).astype(np.uint8)
    image = cv2.convertScaleAbs(image, alpha=1.0, beta=0)
    background_mask = background_mask_from_render(sim.camera, image)
    top = np.array([246, 248, 251], dtype=np.float32)
    bottom = np.array([229, 236, 243], dtype=np.float32)
    t = np.linspace(0.0, 1.0, image.shape[0], dtype=np.float32)[:, None]
    gradient = (top * (1.0 - t) + bottom * t).astype(np.uint8)
    gradient = np.repeat(gradient[:, None, :], image.shape[1], axis=1)
    image[background_mask] = gradient[background_mask]
    return image


def fit_panel(image: np.ndarray, width: int, height: int) -> np.ndarray:
    if image.shape[1] == width and image.shape[0] == height:
        return image
    interpolation = cv2.INTER_AREA if image.shape[1] > width or image.shape[0] > height else cv2.INTER_CUBIC
    return cv2.resize(image, (width, height), interpolation=interpolation)


def draw_text(img: np.ndarray, text: str, xy: tuple[int, int], color: tuple[int, int, int], font=FONT_REGULAR) -> None:
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    draw.text(xy, text, fill=color, font=font)
    img[:] = np.asarray(pil)


def draw_label(img: np.ndarray, text: str, xy: tuple[int, int], color: tuple[int, int, int], scale: float = 0.75) -> None:
    font = FONT_SMALL if scale < 0.6 else FONT_REGULAR
    x, y = xy
    x = min(max(10, x), img.shape[1] - 10)
    y = min(max(24, y), img.shape[0] - 12)
    draw_text(img, text, (x + 2, y + 2), (255, 255, 255), font)
    draw_text(img, text, (x, y), color, font)


def draw_panel_frame(canvas: np.ndarray, x: int, y: int, w: int, h: int, color: tuple[int, int, int]) -> None:
    cv2.rectangle(canvas, (x, y), (x + w - 1, y + h - 1), COLOR_BORDER, 1, cv2.LINE_AA)


def draw_force_annotation(
    img: np.ndarray,
    sim: DrawerSim,
    force_dir: np.ndarray,
    force: float,
    color: tuple[int, int, int],
    point_history: list[np.ndarray],
) -> None:
    projected_history = [uv for p in point_history if (uv := project(sim.camera, p)) is not None]
    if len(projected_history) > 1:
        cv2.polylines(img, [np.array(projected_history, dtype=np.int32)], False, color, 3, cv2.LINE_AA)
        for history_uv in projected_history[:: max(1, len(projected_history) // 12)]:
            cv2.circle(img, history_uv, 3, color, -1, cv2.LINE_AA)

    point = application_point_world(sim)
    uv = project(sim.camera, point)
    if uv is None:
        return

    px, py = uv
    cv2.circle(img, (px, py), 17, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(img, (px, py), 14, color, 4, cv2.LINE_AA)
    cv2.circle(img, (px, py), 4, color, -1, cv2.LINE_AA)

    if force > 0:
        projected_dir = project(sim.camera, point + force_dir * 0.18)
        arrow_len = int(120 + 12 * math.log10(max(1.0, force)))
        if projected_dir is None:
            direction_px = np.array([-1.0, 0.0], dtype=np.float32)
        else:
            direction_px = np.array([projected_dir[0] - px, projected_dir[1] - py], dtype=np.float32)
            norm = float(np.linalg.norm(direction_px))
            if norm < 1.0:
                direction_px = np.array([-1.0, 0.0], dtype=np.float32)
            else:
                direction_px /= norm

        ex = int(np.clip(px + direction_px[0] * arrow_len, 32, img.shape[1] - 32))
        ey = int(np.clip(py + direction_px[1] * arrow_len, 32, img.shape[0] - 32))
        cv2.arrowedLine(img, (px, py), (ex, ey), color, 10, cv2.LINE_AA, tipLength=0.2)
        cv2.arrowedLine(img, (px, py), (ex, ey), (255, 255, 255), 4, cv2.LINE_AA, tipLength=0.2)
        label_offset = (direction_px * 20 + np.array([-42.0, -10.0], dtype=np.float32)).astype(int)
        draw_label(img, f"{force:g} N", (ex + int(label_offset[0]), ey + int(label_offset[1])), color, 0.58)

def draw_info_card(canvas: np.ndarray, x: int, y: int, w: int, h: int, title: str, force_text: str, disp: float, color: tuple[int, int, int]) -> None:
    x0, y0 = x + 18, y + 12
    x1, y1 = x + w - 18, y + h - 10
    cv2.rectangle(canvas, (x0 + 2, y0 + 3), (x1 + 2, y1 + 3), (226, 232, 240), -1, cv2.LINE_AA)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), COLOR_PANEL, -1, cv2.LINE_AA)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), COLOR_BORDER, 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (x0, y0), (x0 + 5, y1), color, -1, cv2.LINE_AA)
    cv2.circle(canvas, (x + 44, y + 43), 17, (224, 238, 255), -1, cv2.LINE_AA)
    cv2.circle(canvas, (x + 44, y + 43), 17, color, 1, cv2.LINE_AA)
    cv2.circle(canvas, (x + 44, y + 43), 6, color, -1, cv2.LINE_AA)
    draw_text(canvas, title.upper(), (x + 70, y + 29), COLOR_TEXT, FONT_BOLD)
    draw_text(canvas, force_text, (x + 34, y + 72), color, FONT_REGULAR)
    draw_text(canvas, f"spostamento {disp:5.3f} m", (x + w - 310, y + 72), COLOR_TEXT, FONT_REGULAR)


def draw_displacement_plot(
    canvas: np.ndarray,
    no_force: list[float],
    pulling: list[float],
    x: int,
    y: int,
    w: int,
    h: int,
    *,
    comparison: bool = True,
) -> None:
    cv2.rectangle(canvas, (x + 2, y + 3), (x + w + 2, y + h + 3), (226, 232, 240), -1)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), COLOR_PANEL_2, -1)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), COLOR_BORDER, 1)
    for tick in range(1, 5):
        gx = x + int(w * tick / 5)
        gy = y + int(h * tick / 5)
        cv2.line(canvas, (gx, y + 42), (gx, y + h - 18), COLOR_GRID, 1, cv2.LINE_AA)
        cv2.line(canvas, (x + 24, gy), (x + w - 24, gy), COLOR_GRID, 1, cv2.LINE_AA)
    draw_label(canvas, "spostamento nel tempo", (x + 14, y + 28), COLOR_TEXT, 0.55)

    values = [0.0, *no_force, *pulling]
    vmin = min(values) - 0.02
    vmax = max(values) + 0.02
    if abs(vmax - vmin) < 1e-4:
        vmax += 0.1

    def to_px(i: int, value: float, n: int) -> tuple[int, int]:
        px = x + 42 + int((w - 60) * (i / max(1, n - 1)))
        py = y + h - 24 - int((h - 58) * ((value - vmin) / (vmax - vmin)))
        return px, py

    for values_line, color in ((no_force, COLOR_MUTED), (pulling, COLOR_ACCENT)):
        pts = [to_px(i, value, len(values_line)) for i, value in enumerate(values_line)]
        if len(pts) > 1:
            cv2.polylines(canvas, [np.array(pts, dtype=np.int32)], False, (210, 228, 250), 5, cv2.LINE_AA)
            cv2.polylines(canvas, [np.array(pts, dtype=np.int32)], False, color, 3, cv2.LINE_AA)

    if comparison:
        draw_label(canvas, "senza forza", (x + w - 168, y + 30), COLOR_MUTED, 0.48)
        draw_label(canvas, "tiro", (x + w - 168, y + 54), COLOR_ACCENT, 0.48)
    else:
        draw_label(canvas, "movimento", (x + w - 168, y + 30), COLOR_ACCENT, 0.48)


def sample_to_dict(
    time_s: float,
    sim: DrawerSim,
    applied_force: np.ndarray,
    generalized_force: float,
    *,
    frame: int | None = None,
    force_step: ForceStep | None = None,
    resistance: Resistance | None = None,
    previous_velocity: float | None = None,
    previous_time_s: float | None = None,
    previous_position: float | None = None,
    cumulative_work: float | None = None,
    force_application_mode: str = "generalized",
    phase: str | None = None,
    settled_flag: bool = False,
) -> dict[str, object]:
    position = float(sim.cabinet.get_qpos()[sim.joint_index])
    velocity = float(sim.cabinet.get_qvel()[sim.joint_index])
    sample_dt = time_s - previous_time_s if previous_time_s is not None else 0.0
    acceleration = 0.0 if previous_velocity is None or sample_dt <= 0.0 else (velocity - previous_velocity) / sample_dt
    delta_q = 0.0 if previous_position is None else position - previous_position
    resistance = resistance or Resistance(0.0, 0.0, 0.0, 0.0, generalized_force, False)
    limits = sim.cabinet.get_active_joints()[sim.joint_index].get_limit().tolist()
    joint_origin_world = sim.cabinet.get_active_joints()[sim.joint_index].get_global_pose().p.astype(float)
    lower, upper = float(limits[0][0]), float(limits[0][1])
    clamped = (math.isfinite(lower) and abs(position - lower) <= 1e-3) or (math.isfinite(upper) and abs(position - upper) <= 1e-3)
    if math.isfinite(lower) and math.isfinite(upper):
        joint_limit_distance = min(position - lower, upper - position)
    elif math.isfinite(lower):
        joint_limit_distance = position - lower
    elif math.isfinite(upper):
        joint_limit_distance = upper - position
    else:
        joint_limit_distance = math.inf
    return {
        "frame": int(frame) if frame is not None else None,
        "time": float(time_s),
        "time_s": float(time_s),
        "phase": phase or "force_applied",
        "q": position,
        "qdot": velocity,
        "qddot": acceleration,
        "position_m": position,
        "velocity_m_s": velocity,
        "acceleration_m_s2": acceleration,
        "joint_position_m": position,
        "joint_velocity_m_s": velocity,
        "joint_acceleration_m_s2": acceleration,
        "delta_q_m": delta_q,
        "application_point_world": application_point_world(sim).astype(float).tolist(),
        "applied_force_world": applied_force.astype(float).tolist(),
        "force_model": "linear_force",
        "force_application_mode": force_application_mode,
        "force_profile_scale": float(force_step.scale) if force_step is not None else 1.0,
        "applied_linear_force_n": float(abs(generalized_force)),
        "applied_force_norm": float(np.linalg.norm(applied_force)),
        "applied_generalized_force": float(generalized_force),
        "static_friction_force_n": float(resistance.static),
        "dynamic_friction_force_n": float(resistance.dynamic),
        "damping_force_n": float(resistance.viscous),
        "damping_torque_or_force": float(resistance.viscous),
        "friction_torque_or_force": float(resistance.static + resistance.dynamic),
        "opposing_linear_friction_n": float(abs(resistance.dynamic)),
        "net_force_n": float(resistance.net),
        "net_generalized_force": float(resistance.net),
        "generalized_joint_force_n": float(resistance.net),
        "generalized_force_n": float(resistance.net),
        "static_friction_engaged": bool(resistance.static_engaged),
        "mechanical_work_j": float(cumulative_work) if cumulative_work is not None else 0.0,
        "damping_n_s_m": 0.0,
        "mass_or_effective_mass": None,
        "joint_axis_world": sim.positive_pull_dir_world.astype(float).tolist(),
        "joint_origin_world": joint_origin_world.tolist(),
        "raw_projected_force_along_axis": float(generalized_force),
        "joint_lower_limit_m": lower,
        "joint_upper_limit_m": upper,
        "joint_limit_distance_m": float(joint_limit_distance),
        "joint_limit_distance": float(joint_limit_distance),
        "clamped_at_limit": bool(clamped),
        "settled_flag": bool(settled_flag),
    }


def physics_mode_results(samples: list[dict[str, object]], args: argparse.Namespace) -> tuple[dict[str, object], list[str]]:
    q = [float(sample["q"]) for sample in samples]
    qdot = [float(sample["qdot"]) for sample in samples]
    force = [float(sample["applied_force_norm"]) for sample in samples]
    times = [float(sample["time_s"]) for sample in samples]
    force_end = float(args.force_start_time + args.force_duration)
    during = [i for i, time in enumerate(times) if args.force_start_time <= time <= force_end + 1e-9]
    after = [i for i, time in enumerate(times) if time > force_end + 1e-9]
    peak = max((abs(value) for value in qdot), default=0.0)
    final = abs(qdot[-1]) if qdot else 0.0
    ratio = final / peak if peak > 1e-12 else math.inf
    checks = {
        "force_nonzero_during_force_window": any(force[i] > 1e-8 for i in during),
        "force_zero_after_force_window": bool(after) and all(force[i] <= 1e-8 for i in after),
        "qdot_changes_during_force": any(abs(qdot[i]) > 1e-5 for i in during),
        "q_continues_after_force_removed": bool(after) and abs(q[-1] - q[after[0]]) > 1e-5,
        "qdot_decays_after_force_removed": ratio <= 0.5 or final <= args.settle_velocity_threshold,
        "no_nan_or_inf": all(math.isfinite(value) for value in q + qdot),
        "q_inside_joint_limits": all(float(sample["joint_limit_distance"]) >= -1e-6 for sample in samples),
    }
    messages = {
        "force_nonzero_during_force_window": "force was never non-zero during the force window",
        "force_zero_after_force_window": "force did not become zero after force_duration_s",
        "qdot_changes_during_force": "qdot did not change during force application",
        "q_continues_after_force_removed": "q did not continue changing after force removal",
        "no_nan_or_inf": "q/qdot contains NaN or infinity",
        "q_inside_joint_limits": "q left the joint limits",
    }
    warnings = [messages[key] for key in messages if not checks[key]]
    verdict = "FAIL" if warnings else "PASS"
    if not checks["qdot_decays_after_force_removed"]:
        warnings.append("qdot did not decay enough after force removal")
        if verdict == "PASS":
            verdict = "WARN"
    settled = final <= args.settle_velocity_threshold
    if not settled and verdict == "PASS":
        verdict = "WARN"
        warnings.append("motion did not fully settle by the final frame")
    checks["warnings"] = warnings
    return {
        "q_start": q[0], "q_end": q[-1], "delta_q": q[-1] - q[0],
        "peak_abs_qdot": peak, "final_abs_qdot": final, "qdot_decay_ratio": ratio,
        "settled": settled,
        "settle_time_s": next((float(sample["time_s"]) for sample in samples if sample.get("settled_flag")), None),
        "dynamics_verdict": verdict, "physics_mode_validation": checks,
    }, warnings


def run_apply(args: argparse.Namespace) -> int:
    model_dir = resolve_model_dir(args.model_dir)
    if not (model_dir / "mobility.urdf").exists():
        raise FileNotFoundError(model_dir / "mobility.urdf")

    json_output = output_paths(model_dir, Path(args.output_root).resolve(), None, args.json_output)[1]
    if not args.keep_old:
        clear_object_output(json_output)

    scene = sapien.Scene([sapien.physx.PhysxCpuSystem()])
    scene.set_timestep(TIMESTEP)
    articulation = load_physics_only_articulation(scene, model_dir)
    articulation.set_qpos(np.zeros_like(articulation.get_qpos(), dtype=np.float32))

    for joint in articulation.get_joints():
        joint.set_drive_property(0.0, 0.0, 0.0)
    for link in articulation.get_links():
        link.disable_gravity = True
        link.linear_damping = LINEAR_DAMPING
        link.angular_damping = ANGULAR_DAMPING

    target_joint = articulation.find_joint_by_name(args.joint)
    target_link = articulation.find_link_by_name(args.link)
    if target_joint is None or target_link is None:
        raise RuntimeError(f"Could not find {args.joint}/{args.link}.")

    active_joints = list(articulation.get_active_joints())
    joint_index = active_joints.index(target_joint)
    direction = np.asarray(args.direction, dtype=np.float32)
    direction /= np.linalg.norm(direction) or 1.0
    axis = int(np.argmax(np.abs(direction)))
    generalized_force = args.force if direction[axis] >= 0 else -args.force
    explicit_point, explicit_strategy = explicit_contact_point(args)
    if explicit_point is not None:
        local_application_point = explicit_point
        application_point_strategy = explicit_strategy or "manual application point from picker"
    else:
        local_application_point = pick_handle_point_local(model_dir, args.link)
        application_point_strategy = "center of handle mesh on selected link"
        if local_application_point is None:
            local_application_point = np.linalg.inv(target_link.get_entity_pose().to_transformation_matrix()) @ pick_link_face_point(target_link, direction)
            application_point_strategy = "center of selected link face along force direction"
    force_world = direction * args.force
    temp_sim = DrawerSim(
        scene,
        articulation,
        target_link,
        joint_index,
        None,
        local_application_point,
        direction,
        application_point_strategy,
    )

    samples = []
    steps = max(1, int(args.seconds / TIMESTEP))
    sample_interval = max(1, round(1.0 / (TIMESTEP * args.fps)))
    for step in range(steps):
        qf = np.zeros_like(articulation.get_qf(), dtype=np.float32)
        qf[joint_index] = generalized_force
        articulation.set_qf(qf)
        scene.step()

        if step % sample_interval == 0 or step == steps - 1:
            samples.append(sample_to_dict(sample_time_from_step(step, TIMESTEP), temp_sim, force_world, generalized_force, frame=len(samples)))

    summary = build_summary(
        sample_series={"force": samples},
        physics_step_count=steps,
        position_key="joint_position_m",
        velocity_key="joint_velocity_m_s",
        initial_position_value=0.0,
    )
    metadata = build_metadata(
        model_dir=model_dir,
        mode="apply",
        joint_type="prismatic",
        joint_name=args.joint,
        link_name=args.link,
        json_output=json_output,
        fps=args.fps,
        requested_seconds=args.seconds,
        simulated_seconds=steps * TIMESTEP,
        timestep_s=TIMESTEP,
        sample_interval_s=sample_interval * TIMESTEP,
        actuation={
            "force": {
                "magnitude_n": args.force,
                "direction_world": direction.astype(float).tolist(),
                "force_model": "linear_force",
                "applied_linear_force_n": float(abs(generalized_force)),
                "opposing_linear_friction_n": 0.0,
                "net_force_n": float(generalized_force),
                "damping_n_s_m": LINEAR_DAMPING,
                "mass_or_effective_mass": None,
                "joint_axis_world": direction.astype(float).tolist(),
                "generalized_joint_force_n": float(generalized_force),
            },
            "joint_limits_m": target_joint.get_limit().tolist(),
        },
        application_point={
            "strategy": application_point_strategy,
            "local_on_link": local_application_point[:3].astype(float).tolist(),
        },
        summary=summary,
        articulation=articulation,
        limit_key="limits_m",
        linear_damping=LINEAR_DAMPING,
        angular_damping=ANGULAR_DAMPING,
    )
    validation = validation_for_motion(
        initial_position=0.0,
        final_position=float(samples[-1]["position_m"]),
        final_velocity=float(samples[-1]["velocity_m_s"]),
        limits=target_joint.get_limit().tolist(),
        actuation_sign=float(generalized_force),
    )
    document = motion_document(
        motion_type="prismatic",
        metadata=metadata,
        sample_series={"force": samples},
        initial_state={"position_m": 0.0},
        final_state={
            "position_m": float(samples[-1]["position_m"]),
            "velocity_m_s": float(samples[-1]["velocity_m_s"]),
        },
        validation=validation,
    )
    with json_output.open("w", encoding="utf-8") as f:
        json.dump(document, f, indent=2)

    print(f"Wrote {json_output}")
    print(f"Final displacement: {samples[-1]['joint_position_m']:.4f} m")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["render", "apply"], default="render")
    parser.add_argument("--model-dir", default="44817")
    parser.add_argument("--output", default=None)
    parser.add_argument("--json-output", default=None)
    parser.add_argument("--drawer", type=int, default=1)
    parser.add_argument("--joint", default="joint_1")
    parser.add_argument("--link", default="link_1")
    parser.add_argument("--force", "--force-magnitude", dest="force", type=float, default=0.5)
    parser.add_argument("--seconds", "--sim-duration-s", dest="seconds", type=float, default=4.0)
    parser.add_argument("--motion-source", choices=["physical_force"], default="physical_force")
    parser.add_argument(
        "--force-application-mode",
        choices=["generalized", "generalized_set_qf", "external_link_force", "impulse_then_passive_joint_dynamics"],
        default="generalized",
    )
    parser.add_argument("--joint-static-friction", type=float, default=0.0, help="Static friction threshold in N for prismatic joints.")
    parser.add_argument("--joint-dynamic-friction", "--joint-friction", dest="joint_dynamic_friction", type=float, default=0.0, help="Coulomb friction magnitude in N for prismatic joints.")
    parser.add_argument("--joint-viscous-damping", "--joint-damping", dest="joint_viscous_damping", type=float, default=0.02, help="Viscous joint damping in N*s/m.")
    parser.add_argument("--static-friction-velocity-threshold", type=float, default=1e-4)
    parser.add_argument("--link-linear-damping", type=float, default=LINEAR_DAMPING)
    parser.add_argument("--link-angular-damping", type=float, default=ANGULAR_DAMPING)
    parser.add_argument("--enable-gravity", action="store_true", help="Leave gravity enabled on articulation links.")
    parser.add_argument("--force-profile", choices=["constant", "pulse", "ramp_hold_release"], default="constant")
    parser.add_argument("--force-start-time", type=float, default=0.0)
    parser.add_argument("--force-duration", "--force-duration-s", dest="force_duration", type=float, default=4.0)
    parser.add_argument("--force-ramp-time", type=float, default=0.25)
    parser.add_argument("--simulate-until-settled", action="store_true")
    parser.add_argument("--settle-velocity-threshold", type=float, default=1e-3)
    parser.add_argument("--max-seconds", type=float, default=10.0)
    parser.add_argument("--end-hold-seconds", type=float, default=2.0, help="Freeze the last frame for this many seconds")
    parser.add_argument(
        "--end-hold-mode",
        choices=["always", "never", "if-stopped"],
        default="always",
        help=f"Control final-frame hold behavior; if-stopped uses |velocity| <= {END_HOLD_MOTION_THRESHOLD:g}.",
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--panel-width", type=int, default=DEFAULT_VIDEO_WIDTH)
    parser.add_argument("--panel-height", type=int, default=DEFAULT_PANEL_HEIGHT)
    parser.add_argument("--info-height", type=int, default=DEFAULT_INFO_HEIGHT)
    parser.add_argument("--plot-height", type=int, default=0)
    parser.add_argument("--direction", nargs=3, type=float, default=[0.0, 0.0, 1.0])
    parser.add_argument("--movement", choices=["single", "comparison"], default="single")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--contact-point-local", nargs=3, type=float, default=None)
    parser.add_argument("--contact-point-strategy", default=None)
    parser.add_argument("--max-q-fraction-of-limit", type=float, default=0.98)
    parser.add_argument("--keep-old", action="store_true", help="Do not delete old files in the output directory")
    args = parser.parse_args()
    if args.force_application_mode in {"external_link_force", "impulse_then_passive_joint_dynamics"} and args.force_profile == "constant":
        args.force_profile = "pulse"
    if args.force_application_mode == "generalized_set_qf":
        args.force_application_mode = "generalized"

    if args.mode == "apply":
        return run_apply(args)

    model_dir = resolve_model_dir(args.model_dir)
    selected_joint = args.joint
    selected_link = args.link
    if (
        args.drawer != parser.get_default("drawer")
        and args.joint == parser.get_default("joint")
        and args.link == parser.get_default("link")
    ):
        selected_joint = f"joint_{args.drawer}"
        selected_link = f"link_{args.drawer}"
    output, json_output = output_paths(model_dir, Path(args.output_root).resolve(), args.output, args.json_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not args.keep_old:
        clear_object_output(output)

    force_dir = np.array(args.direction, dtype=np.float32)
    force_dir /= np.linalg.norm(force_dir) or 1.0
    slider_axis = int(np.argmax(np.abs(force_dir)))
    generalized_force = compute_prismatic_generalized_force(force_dir, force_dir * args.force)

    explicit_point, explicit_strategy = explicit_contact_point(args)
    still_sim = (
        setup_sim(
            model_dir,
            selected_joint,
            selected_link,
            args.panel_width,
            args.panel_height,
            force_dir,
            explicit_point,
            explicit_strategy,
            args.link_linear_damping,
            args.link_angular_damping,
            not args.enable_gravity,
        )
        if args.movement == "comparison"
        else None
    )
    pulling_sim = setup_sim(
        model_dir,
        selected_joint,
        selected_link,
        args.panel_width,
        args.panel_height,
        force_dir,
        explicit_point,
        explicit_strategy,
        args.link_linear_damping,
        args.link_angular_damping,
        not args.enable_gravity,
    )
    pull_dir_world = pulling_sim.positive_pull_dir_world if generalized_force >= 0 else -pulling_sim.positive_pull_dir_world
    force = pull_dir_world * args.force

    steps_per_frame = max(1, round(240 / args.fps))
    frame_count = max(1, int(args.seconds * args.fps))
    still_displacements: list[float] = []
    pulling_displacements: list[float] = []
    samples = {"no_force": [], "pulling_force": []} if args.movement == "comparison" else {"force": []}
    point_histories = {"no_force": [], "pulling_force": []} if args.movement == "comparison" else {"force": []}

    out_w = args.panel_width * 2 if args.movement == "comparison" else args.panel_width
    out_h = args.panel_height + args.info_height + args.plot_height

    final_frame = None
    actual_end_hold_seconds = 0.0
    frame_index = 0
    physics_step_count = 0
    previous_sample_velocity: float | None = None
    previous_sample_time: float | None = None
    previous_sample_position: float | None = None
    cumulative_work = 0.0
    last_force_step = ForceStep(0.0, 0.0)
    last_resistance = Resistance(0.0, 0.0, 0.0, 0.0, 0.0, False)
    still_moving_at_max_seconds = False
    with imageio.get_writer(output, fps=args.fps, codec="libx264", quality=8, macro_block_size=1) as writer:
        while True:
            if frame_index >= frame_count:
                if not args.simulate_until_settled:
                    break
                if abs(float(pulling_sim.cabinet.get_qvel()[pulling_sim.joint_index])) <= args.settle_velocity_threshold:
                    break
                if physics_step_count * TIMESTEP >= args.max_seconds:
                    still_moving_at_max_seconds = True
                    break
            for _ in range(steps_per_frame):
                step_time_s = physics_step_count * TIMESTEP
                force_step = scaled_force_step(args, step_time_s)
                signed_generalized_force = compute_prismatic_generalized_force(
                    pulling_sim.positive_pull_dir_world,
                    force * force_step.scale,
                )
                resistance = resistance_for_motion(signed_generalized_force, float(pulling_sim.cabinet.get_qvel()[pulling_sim.joint_index]), args)
                q_before = float(pulling_sim.cabinet.get_qpos()[pulling_sim.joint_index])
                qf = np.zeros_like(pulling_sim.cabinet.get_qf(), dtype=np.float32)
                if args.force_application_mode == "external_link_force":
                    pulling_sim.drawer.add_force_at_point(force * force_step.scale, application_point_world(pulling_sim), "force")
                    qf[pulling_sim.joint_index] = resistance.total
                else:
                    qf[pulling_sim.joint_index] = resistance.net
                pulling_sim.cabinet.set_qf(qf)
                if still_sim is not None:
                    still_sim.scene.step()
                pulling_sim.scene.step()
                q_after = float(pulling_sim.cabinet.get_qpos()[pulling_sim.joint_index])
                cumulative_work += resistance.net * (q_after - q_before)
                last_force_step = force_step
                last_resistance = resistance
                physics_step_count += 1

            time_s = physics_step_count * TIMESTEP
            current_abs_qdot = abs(float(pulling_sim.cabinet.get_qvel()[pulling_sim.joint_index]))
            sample_phase = (
                "force_applied"
                if last_force_step.applied_magnitude > 1e-9
                else "settled"
                if current_abs_qdot <= args.settle_velocity_threshold
                else "passive_motion"
            )
            if args.movement == "comparison":
                samples["no_force"].append(sample_to_dict(time_s, still_sim, np.zeros(3, dtype=np.float32), 0.0, frame=len(pulling_displacements)))
                samples["pulling_force"].append(
                    sample_to_dict(
                        time_s,
                        pulling_sim,
                        force * last_force_step.scale,
                        generalized_force * last_force_step.scale,
                        frame=frame_index,
                        force_step=last_force_step,
                        resistance=last_resistance,
                        previous_velocity=previous_sample_velocity,
                        previous_time_s=previous_sample_time,
                        previous_position=previous_sample_position,
                        cumulative_work=cumulative_work,
                        force_application_mode=args.force_application_mode,
                        phase=sample_phase,
                        settled_flag=sample_phase == "settled",
                    )
                )
                point_histories["no_force"].append(application_point_world(still_sim))
                point_histories["pulling_force"].append(application_point_world(pulling_sim))
            else:
                samples["force"].append(
                    sample_to_dict(
                        time_s,
                        pulling_sim,
                        force * last_force_step.scale,
                        generalized_force * last_force_step.scale,
                        frame=frame_index,
                        force_step=last_force_step,
                        resistance=last_resistance,
                        previous_velocity=previous_sample_velocity,
                        previous_time_s=previous_sample_time,
                        previous_position=previous_sample_position,
                        cumulative_work=cumulative_work,
                        force_application_mode=args.force_application_mode,
                        phase=sample_phase,
                        settled_flag=sample_phase == "settled",
                    )
                )
                point_histories["force"].append(application_point_world(pulling_sim))
            previous_sample_velocity = float(pulling_sim.cabinet.get_qvel()[pulling_sim.joint_index])
            previous_sample_time = time_s
            previous_sample_position = float(pulling_sim.cabinet.get_qpos()[pulling_sim.joint_index])

            right = fit_panel(render_panel(pulling_sim), args.panel_width, args.panel_height)
            canvas = np.full((out_h, out_w, 3), COLOR_BG, dtype=np.uint8)

            draw_force_annotation(
                right,
                pulling_sim,
                pull_dir_world,
                args.force,
                COLOR_ACCENT,
                point_histories["pulling_force"] if args.movement == "comparison" else point_histories["force"],
            )
            canvas[: args.panel_height, : args.panel_width] = right
            draw_panel_frame(canvas, 0, 0, args.panel_width, args.panel_height, COLOR_ACCENT)
            pulling_disp = float(pulling_sim.cabinet.get_qpos()[pulling_sim.joint_index])
            pulling_displacements.append(pulling_disp)

            if args.info_height > 0:
                draw_info_card(canvas, 0, args.panel_height, args.panel_width, args.info_height, "movimento", f"F = {args.force:g} N", pulling_disp, COLOR_ACCENT)

            if still_sim is not None:
                left = fit_panel(render_panel(still_sim), args.panel_width, args.panel_height)
                draw_force_annotation(left, still_sim, pull_dir_world, 0.0, COLOR_MUTED, point_histories["no_force"])
                canvas[: args.panel_height, : args.panel_width] = left
                canvas[: args.panel_height, args.panel_width :] = right
                draw_panel_frame(canvas, 0, 0, args.panel_width, args.panel_height, COLOR_MUTED)
                draw_panel_frame(canvas, args.panel_width, 0, args.panel_width, args.panel_height, COLOR_ACCENT)
                cv2.line(canvas, (args.panel_width, 0), (args.panel_width, args.panel_height + args.info_height), COLOR_BORDER, 1)
                still_disp = float(still_sim.cabinet.get_qpos()[still_sim.joint_index])
                still_displacements.append(still_disp)
                if args.info_height > 0:
                    draw_info_card(canvas, 0, args.panel_height, args.panel_width, args.info_height, "senza forza", "F = 0 N", still_disp, COLOR_MUTED)
                    draw_info_card(canvas, args.panel_width, args.panel_height, args.panel_width, args.info_height, "trazione cassetto", f"F = {args.force:g} N", pulling_disp, COLOR_ACCENT)
            if args.plot_height > 0:
                draw_displacement_plot(
                    canvas,
                    still_displacements,
                    pulling_displacements,
                    22,
                    args.panel_height + args.info_height + 16,
                    out_w - 44,
                    args.plot_height - 32,
                    comparison=args.movement == "comparison",
                )
            writer.append_data(canvas)
            final_frame = canvas.copy()
            frame_index += 1

        hold_frames = int(round(args.end_hold_seconds * args.fps))
        final_motion = abs(float(pulling_sim.cabinet.get_qvel()[pulling_sim.joint_index]))
        append_hold = should_append_end_hold(args.end_hold_mode, hold_frames, final_motion)
        if args.end_hold_mode == "always" and append_hold:
            warn_if_holding_active_motion(final_motion)
        if final_frame is not None and append_hold:
            actual_end_hold_seconds = hold_frames / args.fps
            for _ in range(hold_frames):
                writer.append_data(final_frame)

    simulated_seconds = physics_step_count * TIMESTEP
    summary = build_summary(
        sample_series=samples,
        physics_step_count=physics_step_count,
        position_key="joint_position_m",
        velocity_key="joint_velocity_m_s",
        initial_position_value=0.0,
    )
    primary_samples = samples["force" if args.movement != "comparison" else "pulling_force"]
    metadata = build_metadata(
        model_dir=model_dir,
        mode="render",
        joint_type="prismatic",
        joint_name=selected_joint,
        link_name=selected_link,
        json_output=json_output,
        video_output=output,
        fps=args.fps,
        requested_seconds=args.seconds,
        simulated_seconds=simulated_seconds,
        timestep_s=TIMESTEP,
        sample_interval_s=steps_per_frame * TIMESTEP,
        end_hold_seconds=actual_end_hold_seconds,
        actuation={
            "force": {
                "magnitude_n": args.force,
                "direction_world": pull_dir_world.astype(float).tolist(),
                "force_model": "linear_force",
                "motion_source": args.motion_source,
                "force_application_mode": args.force_application_mode,
                "force_profile": args.force_profile,
                "force_start_time_s": args.force_start_time,
                "force_duration_s": args.force_duration,
                "force_ramp_time_s": args.force_ramp_time,
                "applied_linear_force_n": float(abs(generalized_force)),
                "joint_static_friction_n": args.joint_static_friction,
                "joint_dynamic_friction_n": args.joint_dynamic_friction,
                "joint_viscous_damping_n_s_m": args.joint_viscous_damping,
                "link_linear_damping": args.link_linear_damping,
                "link_angular_damping": args.link_angular_damping,
                "gravity_enabled": bool(args.enable_gravity),
                "opposing_linear_friction_n": float(abs(primary_samples[-1]["dynamic_friction_force_n"])),
                "net_force_n": float(primary_samples[-1]["net_force_n"]),
                "damping_n_s_m": args.joint_viscous_damping,
                "mass_or_effective_mass": None,
                "joint_axis_world": pull_dir_world.astype(float).tolist(),
                "generalized_joint_force_n": float(generalized_force),
            },
            "joint_limits_m": pulling_sim.cabinet.get_active_joints()[pulling_sim.joint_index].get_limit().tolist(),
        },
        application_point={
            "strategy": pulling_sim.application_point_strategy,
            "local_on_link": pulling_sim.local_application_point[:3].astype(float).tolist(),
        },
        summary=summary,
        articulation=pulling_sim.cabinet,
        limit_key="limits_m",
        linear_damping=args.link_linear_damping,
        angular_damping=args.link_angular_damping,
        drawer_index=args.drawer,
    )
    metadata["simulation_config"] = {
        "motion_source": args.motion_source,
        "force_application_mode": args.force_application_mode,
        "force_profile": args.force_profile,
        "simulate_until_settled": bool(args.simulate_until_settled),
        "settle_velocity_threshold": args.settle_velocity_threshold,
        "max_seconds": args.max_seconds,
        "stopped_by_settle_threshold": bool(args.simulate_until_settled and not still_moving_at_max_seconds),
        "still_moving_at_max_seconds": bool(still_moving_at_max_seconds),
    }
    metadata["physics"]["uses_separate_static_dynamic_friction"] = True
    metadata["physics"]["overrides"]["joint_resistance"] = {
        "static_friction": args.joint_static_friction,
        "dynamic_friction": args.joint_dynamic_friction,
        "viscous_damping": args.joint_viscous_damping,
    }
    validation = validation_for_motion(
        initial_position=0.0,
        final_position=float(primary_samples[-1]["position_m"]),
        final_velocity=float(primary_samples[-1]["velocity_m_s"]),
        limits=pulling_sim.cabinet.get_active_joints()[pulling_sim.joint_index].get_limit().tolist(),
        actuation_sign=float(generalized_force),
        completion_velocity_threshold=1e-3,
    )
    if still_moving_at_max_seconds:
        validation["warnings"].append("Simulate-until-settled reached max seconds while motion was still active.")
    document = motion_document(
        motion_type="prismatic",
        metadata=metadata,
        sample_series=samples,
        initial_state={"position_m": 0.0},
        final_state={
            "position_m": float(primary_samples[-1]["position_m"]),
            "velocity_m_s": float(primary_samples[-1]["velocity_m_s"]),
        },
        validation=validation,
    )
    physics_results, dynamics_warnings = physics_mode_results(primary_samples, args)
    validation.setdefault("warnings", []).extend(dynamics_warnings)
    document.update(physics_results)
    document.update(
        {
            "force_application_mode": args.force_application_mode,
            "true_external_force_used": args.force_application_mode == "external_link_force",
            "fallback_used": args.force_application_mode == "impulse_then_passive_joint_dynamics",
            "force_units_physical": args.force_application_mode == "external_link_force",
            "force_duration_s": float(args.force_duration),
            "sim_duration_s": float(args.seconds),
            "timestep_s": float(TIMESTEP),
            "fps": int(args.fps),
            "joint_damping": float(args.joint_viscous_damping),
            "joint_friction": float(args.joint_dynamic_friction),
            "warning_messages": dynamics_warnings,
        }
    )
    with json_output.open("w", encoding="utf-8") as f:
        json.dump(document, f, indent=2)
    write_physics_diagnostics(output.parent, primary_samples, metadata, validation)

    print(f"Wrote {output}")
    print(f"Wrote {json_output}")
    if still_displacements:
        print(f"No-force final displacement: {still_displacements[-1]:.4f} m")
    print(f"Final displacement: {pulling_displacements[-1]:.4f} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
