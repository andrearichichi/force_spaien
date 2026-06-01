#!/usr/bin/env python3
"""Render a SAPIEN video for a revolute joint force simulation."""

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
    from simulation_json import (
        build_metadata,
        build_summary,
        motion_document,
        sample_time_from_frame,
        sample_time_from_step,
        validation_for_motion,
    )
except ModuleNotFoundError:
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


@dataclass
class LaptopSim:
    scene: sapien.Scene
    laptop: sapien.physx.PhysxArticulation
    screen: sapien.physx.PhysxArticulationLinkComponent
    joint: sapien.physx.PhysxArticulationJoint
    joint_index: int
    marker: sapien.Entity
    camera: sapien.render.RenderCameraComponent
    local_application_point: np.ndarray
    application_point_strategy: str
    joint_axis_local: np.ndarray
    joint_axis_world_reference: np.ndarray


@dataclass
class RevoluteForce:
    force_vector_world: np.ndarray
    tangential_direction_world: np.ndarray
    joint_axis_world: np.ndarray
    joint_origin_world: np.ndarray
    force_application_point_world: np.ndarray
    radius_vector_world: np.ndarray
    radius_perpendicular_world: np.ndarray
    tangential_force_radius_m: float
    force_perpendicular_to_axis_error: float
    torque_about_axis_nm: float
    generalized_torque_nm: float
    torque_direction: str
    direction_auto_flipped: bool
    warnings: list[str]


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


def joint_axis_from_urdf(model_dir: Path, joint_name: str) -> np.ndarray:
    root = ET.parse(model_dir / "mobility.urdf").getroot()
    joint = root.find(f".//joint[@name='{joint_name}']")
    axis = joint.find("axis") if joint is not None else None
    if axis is None:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    values = np.asarray([float(value) for value in axis.attrib.get("xyz", "1 0 0").split()], dtype=np.float32)
    return values / (np.linalg.norm(values) or 1.0)


def estimate_revolute_axis_from_motion(
    articulation: sapien.physx.PhysxArticulation,
    link: sapien.physx.PhysxArticulationLinkComponent,
    joint_index: int,
    *,
    delta: float = 1e-3,
) -> np.ndarray:
    """Estimate the real world rotation axis from a tiny link motion.

    Some assets have joint frames whose URDF axis is not visually intuitive in
    world coordinates. The trajectory of the moving link is the source of truth
    for drawing the circle and constructing the tangential force.
    """
    qpos = articulation.get_qpos().copy()
    qvel = articulation.get_qvel().copy()
    before = link.get_entity_pose().to_transformation_matrix()
    qpos_delta = qpos.copy()
    qpos_delta[joint_index] += delta
    articulation.set_qpos(qpos_delta)
    after = link.get_entity_pose().to_transformation_matrix()
    articulation.set_qpos(qpos)
    articulation.set_qvel(qvel)

    rotation = after[:3, :3] @ before[:3, :3].T
    cos_angle = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    angle = math.acos(cos_angle)
    if abs(math.sin(angle)) < 1e-9:
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)
    axis = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ],
        dtype=np.float32,
    ) / (2.0 * math.sin(angle))
    axis = axis / (np.linalg.norm(axis) or 1.0)
    return axis.astype(np.float32)


def pick_screen_edge_point(screen: sapien.physx.PhysxArticulationLinkComponent) -> np.ndarray:
    """Pick a visible point on the free upper screen edge in world coordinates.

    The old version used a hand-written world point. Here we derive the point
    from link_1's current tight AABB after setting the initial hinge angle.
    For this laptop, the visible free screen border is the minimum X / maximum Z
    side of link_1. Y is kept at the edge center to avoid selecting a corner.
    """
    aabb = screen.compute_global_aabb_tight()
    return np.array(
        [
            aabb[0, 0],
            0.5 * (aabb[0, 1] + aabb[1, 1]),
            aabb[1, 2],
            1.0,
        ],
        dtype=np.float32,
    )


def create_marker(scene: sapien.Scene) -> sapien.Entity:
    return scene.create_actor_builder().build_kinematic(name="force_application_point")


def setup_sim(
    model_dir: Path,
    joint_name: str,
    link_name: str,
    width: int,
    height: int,
    initial_angle: float,
    override_point: np.ndarray | None = None,
    override_strategy: str | None = None,
) -> LaptopSim:
    scene = sapien.Scene()
    scene.set_timestep(TIMESTEP)
    scene.set_ambient_light([0.78, 0.80, 0.84])
    scene.add_directional_light([0.25, 0.45, -1.0], [1.0, 1.0, 0.96], shadow=False)
    scene.add_directional_light([-0.6, -0.2, -1.0], [0.42, 0.48, 0.56], shadow=False)

    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    laptop = loader.load(str(model_dir / "mobility.urdf"))
    joint = laptop.find_joint_by_name(joint_name)
    screen = laptop.find_link_by_name(link_name)
    if joint is None or screen is None:
        raise RuntimeError(f"Could not find {joint_name}/{link_name}.")

    active_joints = list(laptop.get_active_joints())
    joint_index = active_joints.index(joint)
    qpos = np.zeros_like(laptop.get_qpos(), dtype=np.float32)
    qpos[joint_index] = initial_angle
    laptop.set_qpos(qpos)

    for current_joint in laptop.get_joints():
        current_joint.set_drive_property(0.0, 0.0, 0.0)
    for link in laptop.get_links():
        link.disable_gravity = True
        link.linear_damping = LINEAR_DAMPING
        link.angular_damping = ANGULAR_DAMPING

    if override_point is not None:
        local_application_point = override_point
        application_point_strategy = override_strategy or "manual application point override"
        initial_world_point = screen.get_entity_pose().to_transformation_matrix() @ local_application_point
    else:
        local_application_point = pick_handle_point_local(model_dir, link_name)
        application_point_strategy = "center of handle mesh on selected link"
        if local_application_point is None:
            initial_world_point = pick_screen_edge_point(screen)
            local_application_point = np.linalg.inv(screen.get_entity_pose().to_transformation_matrix()) @ initial_world_point
            application_point_strategy = f"upper free border of {link_name} from tight AABB at initial pose"
        else:
            initial_world_point = screen.get_entity_pose().to_transformation_matrix() @ local_application_point

    marker = create_marker(scene)
    marker.set_pose(sapien.Pose(initial_world_point[:3]))

    camera = scene.add_camera("camera", width, height, math.radians(48), 0.01, 20.0)
    camera_eye = np.array([-1.18, -1.46, 0.86], dtype=np.float32)
    camera_target = np.array([-0.08, 0.10, 0.05], dtype=np.float32)
    camera.set_entity_pose(
        look_at_pose(
            zoomed_eye(camera_eye, camera_target),
            camera_target,
        )
    )

    return LaptopSim(
        scene,
        laptop,
        screen,
        joint,
        joint_index,
        marker,
        camera,
        local_application_point,
        application_point_strategy,
        joint_axis_from_urdf(model_dir, joint_name),
        estimate_revolute_axis_from_motion(laptop, screen, joint_index),
    )


def application_point_world(sim: LaptopSim) -> np.ndarray:
    point = sim.screen.get_entity_pose().to_transformation_matrix() @ sim.local_application_point
    return point[:3].astype(np.float32)


def joint_axis_world(sim: LaptopSim) -> np.ndarray:
    return sim.joint_axis_world_reference.astype(np.float32) / (np.linalg.norm(sim.joint_axis_world_reference) or 1.0)


def joint_origin_world(sim: LaptopSim) -> np.ndarray:
    return sim.joint.get_global_pose().p.astype(np.float32)


def tangential_force_world(
    sim: LaptopSim,
    magnitude: float,
    preferred_motion_direction: float = 1.0,
    *,
    auto_direction: bool = True,
) -> RevoluteForce:
    return geometric_tangential_force_for_joint(
        sim.joint,
        sim.joint_axis_local,
        application_point_world(sim),
        magnitude,
        preferred_motion_direction,
        float(sim.laptop.get_qpos()[sim.joint_index]),
        sim.joint.get_limit().tolist(),
        auto_direction=auto_direction,
        axis_world_override=sim.joint_axis_world_reference,
    )


def project(camera: sapien.render.RenderCameraComponent, point: np.ndarray) -> tuple[int, int] | None:
    camera_point = camera.get_extrinsic_matrix() @ np.array([point[0], point[1], point[2], 1.0], dtype=np.float32)
    if camera_point[2] <= 0:
        return None
    uvw = camera.get_intrinsic_matrix() @ camera_point
    return int(round(uvw[0] / uvw[2])), int(round(uvw[1] / uvw[2]))


def render_panel(sim: LaptopSim) -> np.ndarray:
    point = application_point_world(sim)
    sim.marker.set_pose(sapien.Pose(point))
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


def draw_text(
    img: np.ndarray,
    text: str,
    xy: tuple[int, int],
    color: tuple[int, int, int],
    font: ImageFont.FreeTypeFont = FONT_REGULAR,
    anchor: str = "la",
) -> None:
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    draw.text(xy, text, fill=color, font=font, anchor=anchor)
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


def draw_info_card(
    canvas: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    title: str,
    force_text: str,
    direction_text: str,
    angle_deg: float,
    color: tuple[int, int, int],
) -> None:
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
    draw_text(canvas, direction_text, (x + 34, y + 100), COLOR_MUTED, FONT_SMALL)
    draw_text(canvas, f"angolo {angle_deg:6.1f} deg", (x + w - 255, y + 72), COLOR_TEXT, FONT_REGULAR)


def draw_force_annotation(
    img: np.ndarray,
    sim: LaptopSim,
    force_dir: np.ndarray,
    force: float,
    width: int,
    height: int,
    color: tuple[int, int, int],
    point_history: list[np.ndarray],
) -> None:
    projected_history = []
    for history_point in point_history:
        history_uv = project(sim.camera, history_point)
        if history_uv is not None:
            projected_history.append(history_uv)
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
    cv2.line(img, (px - 18, py), (px + 18, py), (255, 255, 255), 2, cv2.LINE_AA)
    cv2.line(img, (px, py - 18), (px, py + 18), (255, 255, 255), 2, cv2.LINE_AA)
    cv2.line(img, (px - 15, py), (px + 15, py), color, 1, cv2.LINE_AA)
    cv2.line(img, (px, py - 15), (px, py + 15), color, 1, cv2.LINE_AA)

    arrow_length = 0.24 + 0.05 * math.log10(max(1.0, force))
    endpoint = point + force_dir * arrow_length
    uv_end = project(sim.camera, endpoint)
    if uv_end is not None and force > 0:
        ex, ey = uv_end
        cv2.arrowedLine(img, (px, py), (ex, ey), color, 9, cv2.LINE_AA, tipLength=0.22)
        cv2.arrowedLine(img, (px, py), (ex, ey), (255, 255, 255), 3, cv2.LINE_AA, tipLength=0.22)
        draw_label(img, f"{force:g} N", (ex + 14, ey - 12), color, 0.58)


def rotate_about_axis(vector: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / (np.linalg.norm(axis) or 1.0)
    return (
        vector * math.cos(angle)
        + np.cross(axis, vector) * math.sin(angle)
        + axis * float(np.dot(axis, vector)) * (1.0 - math.cos(angle))
    )


def draw_revolute_geometry_overlay(
    img: np.ndarray,
    sim: LaptopSim,
    force_detail: RevoluteForce,
    color: tuple[int, int, int],
) -> None:
    origin = force_detail.joint_origin_world
    axis = force_detail.joint_axis_world
    radius = force_detail.radius_perpendicular_world
    radius_norm = float(np.linalg.norm(radius))
    if radius_norm < 1e-8:
        return

    axis_len = max(0.35, min(1.2, radius_norm * 1.2))
    axis_a = origin - axis * axis_len
    axis_b = origin + axis * axis_len
    uv_a = project(sim.camera, axis_a)
    uv_b = project(sim.camera, axis_b)
    if uv_a is not None and uv_b is not None:
        cv2.line(img, uv_a, uv_b, (255, 59, 48), 4, cv2.LINE_AA)
        cv2.line(img, uv_a, uv_b, (255, 255, 255), 1, cv2.LINE_AA)

    uv_o = project(sim.camera, origin)
    uv_p = project(sim.camera, force_detail.force_application_point_world)
    if uv_o is not None:
        cv2.circle(img, uv_o, 7, (255, 59, 48), -1, cv2.LINE_AA)
    if uv_o is not None and uv_p is not None:
        cv2.line(img, uv_o, uv_p, (120, 120, 120), 2, cv2.LINE_AA)

    arc_points = []
    for angle in np.linspace(-math.pi, math.pi, 97):
        point = origin + rotate_about_axis(radius, axis, float(angle))
        uv = project(sim.camera, point)
        if uv is not None:
            arc_points.append(uv)
    if len(arc_points) > 2:
        cv2.polylines(img, [np.asarray(arc_points, dtype=np.int32)], True, (0, 122, 255), 3, cv2.LINE_AA)

    tangent_end = force_detail.force_application_point_world + force_detail.tangential_direction_world * min(0.32, max(0.18, radius_norm * 0.45))
    uv_tangent = project(sim.camera, tangent_end)
    if uv_p is not None and uv_tangent is not None:
        cv2.arrowedLine(img, uv_p, uv_tangent, color, 5, cv2.LINE_AA, tipLength=0.22)
        cv2.arrowedLine(img, uv_p, uv_tangent, (255, 255, 255), 2, cv2.LINE_AA, tipLength=0.22)

def draw_angle_plot(
    canvas: np.ndarray,
    opening_angles: list[float],
    closing_angles: list[float],
    initial_angle: float,
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
    draw_label(canvas, "risposta nel tempo", (x + 14, y + 28), COLOR_TEXT, 0.55)

    all_angles = [math.degrees(initial_angle), *opening_angles, *closing_angles]
    amin = min(all_angles) - 4.0
    amax = max(all_angles) + 4.0
    if abs(amax - amin) < 1e-3:
        amax += 1.0
        amin -= 1.0

    def to_px(i: int, value: float, n: int) -> tuple[int, int]:
        px = x + 42 + int((w - 60) * (i / max(1, n - 1)))
        py = y + h - 24 - int((h - 58) * ((value - amin) / (amax - amin)))
        return px, py

    for angles, color in (
        (opening_angles, COLOR_ACCENT),
        (closing_angles, COLOR_ALT),
    ):
        pts = [to_px(i, a, len(angles)) for i, a in enumerate(angles)]
        if len(pts) > 1:
            cv2.polylines(canvas, [np.array(pts, dtype=np.int32)], False, (210, 228, 250), 5, cv2.LINE_AA)
            cv2.polylines(canvas, [np.array(pts, dtype=np.int32)], False, color, 3, cv2.LINE_AA)

    if comparison:
        draw_label(canvas, "apre", (x + w - 130, y + 30), COLOR_ACCENT, 0.48)
        draw_label(canvas, "chiude", (x + w - 130, y + 54), COLOR_ALT, 0.48)
    else:
        draw_label(canvas, "movimento", (x + w - 150, y + 30), COLOR_ACCENT, 0.48)


def torque_about_joint(sim: LaptopSim, applied_force: np.ndarray) -> float:
    axis = joint_axis_world(sim)
    radius_vec = application_point_world(sim) - joint_origin_world(sim)
    radius_perp = radius_vec - axis * float(np.dot(radius_vec, axis))
    return float(np.dot(axis, np.cross(radius_perp, applied_force)))


def direction_pushes_outside_limit(sim: LaptopSim, applied_force: np.ndarray, threshold: float = 1e-3) -> bool:
    limits = sim.joint.get_limit().tolist()
    lower, upper = limits[0]
    angle = float(sim.laptop.get_qpos()[sim.joint_index])
    torque = torque_about_joint(sim, applied_force)
    lower_finite = math.isfinite(lower)
    upper_finite = math.isfinite(upper)
    return (lower_finite and abs(angle - lower) <= threshold and torque < 0.0) or (
        upper_finite and abs(angle - upper) <= threshold and torque > 0.0
    )


def reset_revolute_state(sim: LaptopSim, angle: float) -> None:
    qpos = np.zeros_like(sim.laptop.get_qpos(), dtype=np.float32)
    qvel = np.zeros_like(sim.laptop.get_qvel(), dtype=np.float32)
    qpos[sim.joint_index] = angle
    sim.laptop.set_qpos(qpos)
    sim.laptop.set_qvel(qvel)


def probe_direction_near_limit(
    sim: LaptopSim,
    force_magnitude: float,
    preferred_motion_direction: float,
    *,
    steps: int = 48,
    threshold: float = 1e-8,
) -> tuple[float, bool]:
    limits = sim.joint.get_limit().tolist()
    lower, upper = float(limits[0][0]), float(limits[0][1])
    initial_angle = float(sim.laptop.get_qpos()[sim.joint_index])
    near_lower = math.isfinite(lower) and abs(initial_angle - lower) <= 1e-3
    near_upper = math.isfinite(upper) and abs(initial_angle - upper) <= 1e-3
    if not near_lower and not near_upper:
        return preferred_motion_direction, False

    def probe(sign: float) -> float:
        reset_revolute_state(sim, initial_angle)
        for _ in range(steps):
            point = application_point_world(sim)
            detail = geometric_tangential_force_for_joint(
                sim.joint,
                sim.joint_axis_local,
                point,
                force_magnitude,
                sign,
                float(sim.laptop.get_qpos()[sim.joint_index]),
                limits,
                auto_direction=False,
                axis_world_override=sim.joint_axis_world_reference,
            )
            sim.screen.add_force_at_point(detail.force_vector_world, point, "force")
            sim.scene.step()
        displacement = float(sim.laptop.get_qpos()[sim.joint_index]) - initial_angle
        reset_revolute_state(sim, initial_angle)
        return displacement

    preferred = 1.0 if preferred_motion_direction >= 0.0 else -1.0
    opposite = -preferred
    preferred_delta = probe(preferred)
    opposite_delta = probe(opposite)
    if near_lower:
        preferred_ok = preferred_delta > threshold
        opposite_ok = opposite_delta > threshold
    else:
        preferred_ok = preferred_delta < -threshold
        opposite_ok = opposite_delta < -threshold
    if preferred_ok or not opposite_ok:
        return preferred, False
    return opposite, True


def generalized_torque_for_joint(
    joint: sapien.physx.PhysxArticulationJoint,
    current_angle: float,
    torque_magnitude: float,
    preferred_motion_direction: float,
    *,
    near_limit_threshold: float = 1e-3,
) -> float:
    limits = joint.get_limit().tolist()
    lower, upper = float(limits[0][0]), float(limits[0][1])
    if math.isfinite(lower) and abs(current_angle - lower) <= near_limit_threshold:
        return abs(torque_magnitude)
    if math.isfinite(upper) and abs(current_angle - upper) <= near_limit_threshold:
        return -abs(torque_magnitude)
    sign = 1.0 if preferred_motion_direction >= 0.0 else -1.0
    return sign * abs(torque_magnitude)


def initial_generalized_motion_direction(joint: sapien.physx.PhysxArticulationJoint, initial_angle: float, fallback: float) -> float:
    limits = joint.get_limit().tolist()
    lower, upper = float(limits[0][0]), float(limits[0][1])
    if math.isfinite(lower) and abs(initial_angle - lower) <= 1e-3:
        return 1.0
    if math.isfinite(upper) and abs(initial_angle - upper) <= 1e-3:
        return -1.0
    return 1.0 if fallback >= 0.0 else -1.0


def sample_to_dict(time_s: float, sim: LaptopSim, force: RevoluteForce, frame: int | None = None) -> dict[str, object]:
    angle = float(sim.laptop.get_qpos()[sim.joint_index])
    omega = float(sim.laptop.get_qvel()[sim.joint_index])
    force_magnitude = float(np.linalg.norm(force.force_vector_world))
    limits = sim.joint.get_limit().tolist()
    lower, upper = float(limits[0][0]), float(limits[0][1])
    clamped = (math.isfinite(lower) and abs(angle - lower) <= 1e-3) or (math.isfinite(upper) and abs(angle - upper) <= 1e-3)
    return {
        "frame": int(frame) if frame is not None else None,
        "time": float(time_s),
        "time_s": float(time_s),
        "theta_rad": angle,
        "theta_deg": math.degrees(angle),
        "omega_rad_s": omega,
        "joint_angle_rad": angle,
        "joint_angle_deg": math.degrees(angle),
        "joint_velocity_rad_s": omega,
        "application_point_world": force.force_application_point_world.astype(float).tolist(),
        "applied_force_world": force.force_vector_world.astype(float).tolist(),
        "force_vector_world": force.force_vector_world.astype(float).tolist(),
        "force_model": "geometric_tangential_force",
        "force_tangent_world": force.tangential_direction_world.astype(float).tolist(),
        "tangential_direction_world": force.tangential_direction_world.astype(float).tolist(),
        "joint_axis_world": force.joint_axis_world.astype(float).tolist(),
        "joint_origin_world": force.joint_origin_world.astype(float).tolist(),
        "radius_vector_world": force.radius_vector_world.astype(float).tolist(),
        "radius_perpendicular_world": force.radius_perpendicular_world.astype(float).tolist(),
        "applied_tangential_force_n": force_magnitude,
        "opposing_tangential_friction_n": 0.0,
        "tangential_force_radius_m": float(force.tangential_force_radius_m),
        "force_perpendicular_to_axis_error": float(force.force_perpendicular_to_axis_error),
        "torque_about_axis_nm": float(force.torque_about_axis_nm),
        "generalized_torque_nm": float(force.generalized_torque_nm),
        "torque_direction": force.torque_direction,
        "torque_applied_nm": float(force.torque_about_axis_nm),
        "torque_resisting_nm": 0.0,
        "net_torque_nm": float(force.generalized_torque_nm),
        "joint_lower_limit_rad": lower,
        "joint_upper_limit_rad": upper,
        "clamped_at_limit": bool(clamped),
    }


def application_point_world_on_link(link: sapien.physx.PhysxArticulationLinkComponent, local_point: np.ndarray) -> np.ndarray:
    return (link.get_entity_pose().to_transformation_matrix() @ local_point)[:3].astype(np.float32)


def geometric_tangential_force_for_joint(
    joint: sapien.physx.PhysxArticulationJoint,
    joint_axis_local: np.ndarray,
    point_world: np.ndarray,
    magnitude: float,
    preferred_motion_direction: float,
    current_angle: float,
    limits: list[list[float]],
    *,
    auto_direction: bool,
    axis_world_override: np.ndarray | None = None,
    near_limit_threshold: float = 1e-3,
) -> RevoluteForce:
    warnings: list[str] = []
    mat = joint.get_global_pose().to_transformation_matrix()
    if axis_world_override is None:
        axis = mat[:3, :3] @ joint_axis_local
        axis = axis.astype(np.float32) / (np.linalg.norm(axis) or 1.0)
    else:
        axis = np.asarray(axis_world_override, dtype=np.float32)
        axis = axis / (np.linalg.norm(axis) or 1.0)
    origin = joint.get_global_pose().p.astype(np.float32)
    radius_vec = point_world - origin
    radius_perp = radius_vec - axis * float(np.dot(radius_vec, axis))
    radius = float(np.linalg.norm(radius_perp))
    preferred_sign = 1.0 if preferred_motion_direction >= 0.0 else -1.0
    desired_sign = preferred_sign
    lower, upper = float(limits[0][0]), float(limits[0][1])
    if auto_direction:
        if math.isfinite(lower) and abs(current_angle - lower) <= near_limit_threshold:
            desired_sign = 1.0
        elif math.isfinite(upper) and abs(current_angle - upper) <= near_limit_threshold:
            desired_sign = -1.0
    direction_auto_flipped = desired_sign != preferred_sign

    if radius < 1e-8:
        warnings.append("Invalid force application point: radius around revolute axis is too small.")
        tangent = np.zeros(3, dtype=np.float32)
        force = np.zeros(3, dtype=np.float32)
        torque = 0.0
        error = 0.0
    else:
        r_hat = radius_perp / radius
        tangent_base = np.cross(axis, r_hat)
        tangent_base = tangent_base / (np.linalg.norm(tangent_base) or 1.0)
        candidate_force = magnitude * tangent_base
        candidate_torque = float(np.dot(axis, np.cross(radius_perp, candidate_force)))
        if candidate_torque == 0.0:
            sign = desired_sign
        else:
            candidate_sign = math.copysign(1.0, candidate_torque)
            sign = desired_sign * candidate_sign
        tangent = (sign * tangent_base).astype(np.float32)
        force = (float(magnitude) * tangent).astype(np.float32)
        torque = float(np.dot(axis, np.cross(radius_perp, force)))
        error = abs(float(np.dot(force / (np.linalg.norm(force) or 1.0), axis)))
        if abs(torque) < 1e-8:
            warnings.append("Applied force does not generate useful torque around the revolute axis.")
        if not auto_direction:
            if math.isfinite(lower) and abs(current_angle - lower) <= near_limit_threshold and torque < 0.0:
                warnings.append("Actuation pushes joint against lower limit; motion may be clamped.")
            if math.isfinite(upper) and abs(current_angle - upper) <= near_limit_threshold and torque > 0.0:
                warnings.append("Actuation pushes joint against upper limit; motion may be clamped.")

    torque_direction = "positive" if torque > 1e-12 else "negative" if torque < -1e-12 else "zero"
    generalized_sign = desired_sign
    generalized_torque = generalized_sign * abs(torque)
    return RevoluteForce(
        force_vector_world=force.astype(np.float32),
        tangential_direction_world=tangent.astype(np.float32),
        joint_axis_world=axis.astype(np.float32),
        joint_origin_world=origin.astype(np.float32),
        force_application_point_world=point_world.astype(np.float32),
        radius_vector_world=radius_vec.astype(np.float32),
        radius_perpendicular_world=radius_perp.astype(np.float32),
        tangential_force_radius_m=radius,
        force_perpendicular_to_axis_error=error,
        torque_about_axis_nm=torque,
        generalized_torque_nm=generalized_torque,
        torque_direction=torque_direction,
        direction_auto_flipped=direction_auto_flipped,
        warnings=warnings,
    )


def tangential_force_world_for_joint(
    joint: sapien.physx.PhysxArticulationJoint,
    joint_axis_local: np.ndarray,
    point_world: np.ndarray,
    magnitude: float,
    preferred_direction: np.ndarray,
) -> tuple[np.ndarray, float, np.ndarray]:
    detail = geometric_tangential_force_for_joint(
        joint,
        joint_axis_local,
        point_world,
        magnitude,
        1.0,
        0.0,
        [[-math.inf, math.inf]],
        auto_direction=False,
    )
    return detail.force_vector_world, detail.tangential_force_radius_m, detail.joint_axis_world


def explicit_contact_point(args: argparse.Namespace) -> tuple[np.ndarray | None, str | None]:
    if args.contact_point_local is None:
        return None, None
    point = np.append(np.asarray(args.contact_point_local, dtype=np.float32), np.float32(1.0))
    return point, args.contact_point_strategy


def run_apply(args: argparse.Namespace) -> int:
    model_dir = resolve_model_dir(args.model_dir)
    if not (model_dir / "mobility.urdf").exists():
        raise FileNotFoundError(model_dir / "mobility.urdf")

    json_output = output_paths(model_dir, Path(args.output_root).resolve(), None, args.json_output)[1]
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

    joint_axis_local = joint_axis_from_urdf(model_dir, args.joint)
    explicit_point, explicit_strategy = explicit_contact_point(args)
    if explicit_point is not None:
        local_application_point = explicit_point
        application_point_strategy = explicit_strategy or "manual application point from picker"
    else:
        local_application_point = pick_handle_point_local(model_dir, args.link)
        application_point_strategy = "center of handle mesh on selected link"
        if local_application_point is None:
            local_application_point = np.linalg.inv(target_link.get_entity_pose().to_transformation_matrix()) @ pick_link_edge_point(target_link)
            application_point_strategy = "free edge of selected link from tight AABB at initial pose"

    temp_sim = LaptopSim(
        scene,
        articulation,
        target_link,
        target_joint,
        joint_index,
        create_marker(scene),
        None,
        local_application_point,
        application_point_strategy,
        joint_axis_local,
        estimate_revolute_axis_from_motion(articulation, target_link, joint_index),
    )
    preferred_motion_direction = 1.0 if args.preferred_motion_direction >= 0.0 else -1.0
    force_auto_direction = args.auto_direction
    direction_auto_flipped = False
    if args.auto_direction:
        preferred_motion_direction, direction_auto_flipped = probe_direction_near_limit(temp_sim, args.force, preferred_motion_direction)
        force_auto_direction = not direction_auto_flipped
        if direction_auto_flipped:
            print("WARNING: Revolute direction probe found the opposite tangential direction moves inside the joint limits.")
    generalized_motion_direction = initial_generalized_motion_direction(target_joint, args.initial_angle, args.preferred_motion_direction)

    samples = []
    steps = max(1, int(args.seconds / TIMESTEP))
    sample_interval = max(1, round(1.0 / (TIMESTEP * args.fps)))
    for step in range(steps):
        point = application_point_world_on_link(target_link, local_application_point)
        force_detail = geometric_tangential_force_for_joint(
            target_joint,
            joint_axis_local,
            point,
            args.force,
            preferred_motion_direction,
            float(articulation.get_qpos()[joint_index]),
            target_joint.get_limit().tolist(),
            auto_direction=force_auto_direction,
            axis_world_override=temp_sim.joint_axis_world_reference,
        )
        force_detail.generalized_torque_nm = generalized_torque_for_joint(
            target_joint,
            float(articulation.get_qpos()[joint_index]),
            force_detail.torque_about_axis_nm,
            generalized_motion_direction,
        )
        direction_auto_flipped = direction_auto_flipped or force_detail.direction_auto_flipped
        qf = np.zeros_like(articulation.get_qf(), dtype=np.float32)
        qf[joint_index] = force_detail.generalized_torque_nm
        articulation.set_qf(qf)
        scene.step()

        if step % sample_interval == 0 or step == steps - 1:
            samples.append(sample_to_dict(sample_time_from_step(step, TIMESTEP), temp_sim, force_detail, frame=len(samples)))

    summary = build_summary(
        sample_series={"force": samples},
        physics_step_count=steps,
        position_key="joint_angle_rad",
        velocity_key="joint_velocity_rad_s",
        secondary_position_key="joint_angle_deg",
        initial_position_value=args.initial_angle,
        initial_secondary_position_value=math.degrees(args.initial_angle),
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
                "preferred_motion_direction": float(preferred_motion_direction),
                "generalized_motion_direction": float(generalized_motion_direction),
                "force_model": "geometric_tangential_force",
                "joint_axis_world": samples[-1]["joint_axis_world"],
                "joint_origin_world": samples[-1]["joint_origin_world"],
                "force_application_point_world": samples[-1]["application_point_world"],
                "radius_vector_world": samples[-1]["radius_vector_world"],
                "radius_perpendicular_world": samples[-1]["radius_perpendicular_world"],
                "tangential_direction_world": samples[-1]["tangential_direction_world"],
                "applied_tangential_force_n": float(args.force),
                "force_vector_world": samples[-1]["force_vector_world"],
                "force_perpendicular_to_axis_error": float(samples[-1]["force_perpendicular_to_axis_error"]),
                "opposing_tangential_friction_n": 0.0,
                "tangential_force_radius_m": float(samples[-1]["tangential_force_radius_m"]),
                "torque_about_axis_nm": float(samples[-1]["torque_about_axis_nm"]),
                "generalized_torque_nm": float(samples[-1]["generalized_torque_nm"]),
                "torque_direction": samples[-1]["torque_direction"],
                "torque_applied_nm": float(samples[-1]["torque_applied_nm"]),
                "torque_resisting_nm": 0.0,
                "net_torque_nm": float(samples[-1]["net_torque_nm"]),
                "damping_nm_s_rad": ANGULAR_DAMPING,
                "inertia": None,
            },
            "joint_limits_rad": target_joint.get_limit().tolist(),
        },
        application_point={
            "strategy": application_point_strategy,
            "local_on_link": local_application_point[:3].astype(float).tolist(),
        },
        summary=summary,
        articulation=articulation,
        limit_key="limits_rad",
        linear_damping=LINEAR_DAMPING,
        angular_damping=ANGULAR_DAMPING,
    )
    validation = validation_for_motion(
        initial_position=args.initial_angle,
        final_position=float(samples[-1]["theta_rad"]),
        final_velocity=float(samples[-1]["omega_rad_s"]),
        limits=target_joint.get_limit().tolist(),
        actuation_sign=float(samples[0]["net_torque_nm"]),
        direction_auto_flipped=direction_auto_flipped,
    )
    if float(primary_samples[-1]["tangential_force_radius_m"]) <= 1e-8:
        validation["warnings"].append("Invalid force application point: radius around revolute axis is too small.")
        validation["motion_visible"] = False
    if abs(float(primary_samples[-1]["torque_about_axis_nm"])) <= 1e-8:
        validation["warnings"].append("Applied force does not generate useful torque around the revolute axis.")
        validation["motion_visible"] = False
    if not validation["motion_visible"] and abs(float(primary_samples[-1]["torque_about_axis_nm"])) > 1e-8:
        validation["warnings"].append(
            "A non-zero geometric tangential torque was applied but the joint did not move visibly; check torque direction, joint limits, or physical constraints."
        )
    if float(samples[-1]["tangential_force_radius_m"]) <= 1e-8:
        validation["warnings"].append("Invalid force application point: radius around revolute axis is too small.")
        validation["motion_visible"] = False
    if abs(float(samples[-1]["torque_about_axis_nm"])) <= 1e-8:
        validation["warnings"].append("Applied force does not generate useful torque around the revolute axis.")
        validation["motion_visible"] = False
    if not validation["motion_visible"] and abs(float(samples[-1]["torque_about_axis_nm"])) > 1e-8:
        validation["warnings"].append(
            "A non-zero geometric tangential torque was applied but the joint did not move visibly; check torque direction, joint limits, or physical constraints."
        )
    document = motion_document(
        motion_type="revolute",
        metadata=metadata,
        sample_series={"force": samples},
        initial_state={"theta_rad": float(args.initial_angle), "theta_deg": math.degrees(args.initial_angle)},
        final_state={
            "theta_rad": float(samples[-1]["theta_rad"]),
            "theta_deg": float(samples[-1]["theta_deg"]),
            "omega_rad_s": float(samples[-1]["omega_rad_s"]),
        },
        validation=validation,
    )

    with json_output.open("w", encoding="utf-8") as f:
        json.dump(document, f, indent=2)

    print(f"Wrote {json_output}")
    print(f"Final angle: {samples[-1]['joint_angle_deg']:.2f} deg")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["render", "apply"], default="render")
    parser.add_argument("--model-dir", default="11691")
    parser.add_argument("--joint", default="joint_1")
    parser.add_argument("--link", default="link_1")
    parser.add_argument("--output", default=None)
    parser.add_argument("--force", type=float, default=0.5)
    parser.add_argument("--closing-force", type=float, default=0.5)
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--end-hold-seconds", type=float, default=2.0, help="Freeze the last frame for this many seconds")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--panel-width", type=int, default=DEFAULT_VIDEO_WIDTH)
    parser.add_argument("--panel-height", type=int, default=DEFAULT_PANEL_HEIGHT)
    parser.add_argument("--info-height", type=int, default=DEFAULT_INFO_HEIGHT)
    parser.add_argument("--plot-height", type=int, default=0)
    parser.add_argument("--initial-angle", type=float, default=-1.5)
    parser.add_argument("--direction", nargs=3, type=float, default=[0.0, 0.0, 1.0])
    parser.add_argument("--preferred-motion-direction", type=float, default=1.0)
    parser.add_argument("--movement", choices=["single", "comparison"], default="single")
    parser.add_argument("--json-output", default=None)
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--contact-point-local", nargs=3, type=float, default=None)
    parser.add_argument("--contact-point-strategy", default=None)
    parser.add_argument("--auto-direction", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--keep-old", action="store_true", help="Do not delete old files in the output directory")
    args = parser.parse_args()

    if args.mode == "apply":
        return run_apply(args)

    model_dir = resolve_model_dir(args.model_dir)
    output, json_output = output_paths(model_dir, Path(args.output_root).resolve(), args.output, args.json_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not args.keep_old:
        clear_object_output(output)

    opening_motion_direction = 1.0 if args.preferred_motion_direction >= 0.0 else -1.0
    closing_motion_direction = -opening_motion_direction

    explicit_point, explicit_strategy = explicit_contact_point(args)
    opening_sim = setup_sim(model_dir, args.joint, args.link, args.panel_width, args.panel_height, args.initial_angle, explicit_point, explicit_strategy)
    direction_auto_flipped = False
    force_auto_direction = args.auto_direction
    if args.auto_direction:
        opening_motion_direction, direction_auto_flipped = probe_direction_near_limit(
            opening_sim,
            args.force,
            opening_motion_direction,
        )
        closing_motion_direction = -opening_motion_direction
        force_auto_direction = not direction_auto_flipped
        if direction_auto_flipped:
            print("WARNING: Revolute direction probe found the opposite tangential direction moves inside the joint limits.")
    opening_generalized_direction = initial_generalized_motion_direction(opening_sim.joint, args.initial_angle, args.preferred_motion_direction)
    closing_generalized_direction = -opening_generalized_direction
    closing_sim = (
        setup_sim(model_dir, args.joint, args.link, args.panel_width, args.panel_height, args.initial_angle, explicit_point, explicit_strategy)
        if args.movement == "comparison"
        else None
    )

    steps_per_frame = max(1, round(240 / args.fps))
    frame_count = max(1, int(args.seconds * args.fps))
    opening_angles: list[float] = []
    closing_angles: list[float] = []
    samples = {"opening_force": [], "closing_force": []} if args.movement == "comparison" else {"force": []}
    point_histories = {"opening_force": [], "closing_force": []} if args.movement == "comparison" else {"force": []}

    out_w = args.panel_width * 2 if args.movement == "comparison" else args.panel_width
    out_h = args.panel_height + args.info_height + args.plot_height

    final_frame = None
    with imageio.get_writer(output, fps=args.fps, codec="libx264", quality=8, macro_block_size=1) as writer:
        for _ in range(frame_count):
            for _ in range(steps_per_frame):
                opening_point = application_point_world(opening_sim)
                opening_force = tangential_force_world(
                    opening_sim,
                    args.force,
                    opening_motion_direction,
                    auto_direction=force_auto_direction,
                )
                opening_force.generalized_torque_nm = generalized_torque_for_joint(
                    opening_sim.joint,
                    float(opening_sim.laptop.get_qpos()[opening_sim.joint_index]),
                    opening_force.torque_about_axis_nm,
                    opening_generalized_direction,
                )
                direction_auto_flipped = direction_auto_flipped or opening_force.direction_auto_flipped
                qf = np.zeros_like(opening_sim.laptop.get_qf(), dtype=np.float32)
                qf[opening_sim.joint_index] = opening_force.generalized_torque_nm
                opening_sim.laptop.set_qf(qf)
                opening_sim.scene.step()
                if closing_sim is not None:
                    closing_point = application_point_world(closing_sim)
                    closing_force = tangential_force_world(
                        closing_sim,
                        args.closing_force,
                        closing_motion_direction,
                        auto_direction=force_auto_direction,
                    )
                    closing_force.generalized_torque_nm = generalized_torque_for_joint(
                        closing_sim.joint,
                        float(closing_sim.laptop.get_qpos()[closing_sim.joint_index]),
                        closing_force.torque_about_axis_nm,
                        closing_generalized_direction,
                    )
                    qf = np.zeros_like(closing_sim.laptop.get_qf(), dtype=np.float32)
                    qf[closing_sim.joint_index] = closing_force.generalized_torque_nm
                    closing_sim.laptop.set_qf(qf)
                    closing_sim.scene.step()

            time_s = sample_time_from_frame(len(opening_angles), steps_per_frame, TIMESTEP)
            opening_force = tangential_force_world(
                opening_sim,
                args.force,
                opening_motion_direction,
                auto_direction=force_auto_direction,
            )
            opening_force.generalized_torque_nm = generalized_torque_for_joint(
                opening_sim.joint,
                float(opening_sim.laptop.get_qpos()[opening_sim.joint_index]),
                opening_force.torque_about_axis_nm,
                opening_generalized_direction,
            )
            direction_auto_flipped = direction_auto_flipped or opening_force.direction_auto_flipped
            if args.movement == "comparison":
                closing_force = tangential_force_world(
                    closing_sim,
                    args.closing_force,
                    closing_motion_direction,
                    auto_direction=force_auto_direction,
                )
                closing_force.generalized_torque_nm = generalized_torque_for_joint(
                    closing_sim.joint,
                    float(closing_sim.laptop.get_qpos()[closing_sim.joint_index]),
                    closing_force.torque_about_axis_nm,
                    closing_generalized_direction,
                )
                samples["opening_force"].append(sample_to_dict(time_s, opening_sim, opening_force, frame=len(opening_angles)))
                samples["closing_force"].append(sample_to_dict(time_s, closing_sim, closing_force, frame=len(opening_angles)))
                point_histories["opening_force"].append(application_point_world(opening_sim))
                point_histories["closing_force"].append(application_point_world(closing_sim))
            else:
                samples["force"].append(sample_to_dict(time_s, opening_sim, opening_force, frame=len(opening_angles)))
                point_histories["force"].append(application_point_world(opening_sim))

            left = fit_panel(render_panel(opening_sim), args.panel_width, args.panel_height)
            canvas = np.full((out_h, out_w, 3), COLOR_BG, dtype=np.uint8)
            draw_revolute_geometry_overlay(left, opening_sim, opening_force, COLOR_ACCENT)
            draw_force_annotation(
                left,
                opening_sim,
                opening_force.tangential_direction_world,
                args.force,
                args.panel_width,
                args.panel_height,
                COLOR_ACCENT,
                point_histories["opening_force"] if args.movement == "comparison" else point_histories["force"],
            )
            canvas[: args.panel_height, : args.panel_width] = left
            draw_panel_frame(canvas, 0, 0, args.panel_width, args.panel_height, COLOR_ACCENT)

            open_angle = math.degrees(float(opening_sim.laptop.get_qpos()[opening_sim.joint_index]))
            opening_angles.append(open_angle)

            if args.info_height > 0:
                draw_info_card(
                    canvas,
                    0,
                    args.panel_height,
                    args.panel_width,
                    args.info_height,
                    "apertura" if args.movement == "comparison" else "movimento",
                    f"F = {args.force:g} N",
                    f"torque {opening_force.torque_direction}",
                    open_angle,
                    COLOR_ACCENT,
                )

            if closing_sim is not None:
                right = fit_panel(render_panel(closing_sim), args.panel_width, args.panel_height)
                draw_revolute_geometry_overlay(right, closing_sim, closing_force, COLOR_ALT)
                draw_force_annotation(
                    right,
                    closing_sim,
                    closing_force.tangential_direction_world,
                    args.closing_force,
                    args.panel_width,
                    args.panel_height,
                    COLOR_ALT,
                    point_histories["closing_force"],
                )
                canvas[: args.panel_height, args.panel_width :] = right
                draw_panel_frame(canvas, args.panel_width, 0, args.panel_width, args.panel_height, COLOR_ALT)
                cv2.line(canvas, (args.panel_width, 0), (args.panel_width, args.panel_height + args.info_height), COLOR_BORDER, 1)
                close_angle = math.degrees(float(closing_sim.laptop.get_qpos()[closing_sim.joint_index]))
                closing_angles.append(close_angle)
                if args.info_height > 0:
                    draw_info_card(
                        canvas,
                        args.panel_width,
                        args.panel_height,
                        args.panel_width,
                        args.info_height,
                        "chiusura",
                        f"F = {args.closing_force:g} N",
                        f"torque {closing_force.torque_direction}",
                        close_angle,
                        COLOR_ALT,
                    )
            if args.plot_height > 0:
                draw_angle_plot(
                    canvas,
                    opening_angles,
                    closing_angles,
                    args.initial_angle,
                    22,
                    args.panel_height + args.info_height + 16,
                    out_w - 44,
                    args.plot_height - 32,
                    comparison=args.movement == "comparison",
                )
            writer.append_data(canvas)
            final_frame = canvas.copy()

        hold_frames = int(round(args.end_hold_seconds * args.fps))
        if final_frame is not None:
            for _ in range(hold_frames):
                writer.append_data(final_frame)

    simulated_seconds = frame_count * steps_per_frame * TIMESTEP
    summary = build_summary(
        sample_series=samples,
        physics_step_count=frame_count * steps_per_frame,
        position_key="joint_angle_rad",
        velocity_key="joint_velocity_rad_s",
        secondary_position_key="joint_angle_deg",
        initial_position_value=args.initial_angle,
        initial_secondary_position_value=math.degrees(args.initial_angle),
    )
    actuation = {
        "initial_joint_position": {
            "rad": args.initial_angle,
            "deg": math.degrees(args.initial_angle),
        },
        "force": {
            "magnitude_n": args.force,
            "preferred_motion_direction": float(opening_motion_direction),
            "generalized_motion_direction": float(opening_generalized_direction),
            "force_model": "geometric_tangential_force",
            "applied_tangential_force_n": float(args.force),
            "opposing_tangential_friction_n": 0.0,
            "tangential_force_radius_m": float(samples["force" if args.movement != "comparison" else "opening_force"][-1]["tangential_force_radius_m"]),
            "joint_axis_world": samples["force" if args.movement != "comparison" else "opening_force"][-1]["joint_axis_world"],
            "joint_origin_world": samples["force" if args.movement != "comparison" else "opening_force"][-1]["joint_origin_world"],
            "force_application_point_world": samples["force" if args.movement != "comparison" else "opening_force"][-1]["application_point_world"],
            "radius_vector_world": samples["force" if args.movement != "comparison" else "opening_force"][-1]["radius_vector_world"],
            "radius_perpendicular_world": samples["force" if args.movement != "comparison" else "opening_force"][-1]["radius_perpendicular_world"],
            "tangential_direction_world": samples["force" if args.movement != "comparison" else "opening_force"][-1]["tangential_direction_world"],
            "force_vector_world": samples["force" if args.movement != "comparison" else "opening_force"][-1]["force_vector_world"],
            "force_perpendicular_to_axis_error": float(samples["force" if args.movement != "comparison" else "opening_force"][-1]["force_perpendicular_to_axis_error"]),
            "torque_about_axis_nm": float(samples["force" if args.movement != "comparison" else "opening_force"][-1]["torque_about_axis_nm"]),
            "generalized_torque_nm": float(samples["force" if args.movement != "comparison" else "opening_force"][-1]["generalized_torque_nm"]),
            "torque_direction": samples["force" if args.movement != "comparison" else "opening_force"][-1]["torque_direction"],
            "torque_applied_nm": float(samples["force" if args.movement != "comparison" else "opening_force"][-1]["torque_applied_nm"]),
            "torque_resisting_nm": 0.0,
            "net_torque_nm": float(samples["force" if args.movement != "comparison" else "opening_force"][-1]["net_torque_nm"]),
            "damping_nm_s_rad": ANGULAR_DAMPING,
            "inertia": None,
        },
        "joint_limits_rad": opening_sim.laptop.get_active_joints()[opening_sim.joint_index].get_limit().tolist(),
    }
    if args.movement == "comparison":
        actuation = {
            "initial_joint_position": actuation["initial_joint_position"],
            "opening_force": actuation["force"],
            "closing_force": {
                "magnitude_n": args.closing_force,
                "force_model": "geometric_tangential_force",
                "preferred_motion_direction": float(closing_motion_direction),
            },
            "joint_limits_rad": actuation["joint_limits_rad"],
        }

    metadata = build_metadata(
        model_dir=model_dir,
        mode="render",
        joint_type="revolute",
        joint_name=args.joint,
        link_name=args.link,
        json_output=json_output,
        video_output=output,
        fps=args.fps,
        requested_seconds=args.seconds,
        simulated_seconds=simulated_seconds,
        timestep_s=TIMESTEP,
        sample_interval_s=steps_per_frame * TIMESTEP,
        end_hold_seconds=args.end_hold_seconds,
        actuation=actuation,
        application_point={
            "strategy": opening_sim.application_point_strategy,
            "local_on_link": opening_sim.local_application_point[:3].astype(float).tolist(),
        },
        summary=summary,
        articulation=opening_sim.laptop,
        limit_key="limits_rad",
        linear_damping=LINEAR_DAMPING,
        angular_damping=ANGULAR_DAMPING,
    )

    print(f"Wrote {output}")
    primary_samples = samples["force" if args.movement != "comparison" else "opening_force"]
    validation = validation_for_motion(
        initial_position=args.initial_angle,
        final_position=float(primary_samples[-1]["theta_rad"]),
        final_velocity=float(primary_samples[-1]["omega_rad_s"]),
        limits=opening_sim.laptop.get_active_joints()[opening_sim.joint_index].get_limit().tolist(),
        actuation_sign=float(primary_samples[0]["net_torque_nm"]),
        direction_auto_flipped=direction_auto_flipped,
    )
    if float(primary_samples[-1]["tangential_force_radius_m"]) <= 1e-8:
        validation["warnings"].append("Invalid force application point: radius around revolute axis is too small.")
        validation["motion_visible"] = False
    if abs(float(primary_samples[-1]["torque_about_axis_nm"])) <= 1e-8:
        validation["warnings"].append("Applied force does not generate useful torque around the revolute axis.")
        validation["motion_visible"] = False
    if not validation["motion_visible"] and abs(float(primary_samples[-1]["torque_about_axis_nm"])) > 1e-8:
        validation["warnings"].append(
            "A non-zero geometric tangential torque was applied but the joint did not move visibly; check torque direction, joint limits, or physical constraints."
        )
    document = motion_document(
        motion_type="revolute",
        metadata=metadata,
        sample_series=samples,
        initial_state={"theta_rad": float(args.initial_angle), "theta_deg": math.degrees(args.initial_angle)},
        final_state={
            "theta_rad": float(primary_samples[-1]["theta_rad"]),
            "theta_deg": float(primary_samples[-1]["theta_deg"]),
            "omega_rad_s": float(primary_samples[-1]["omega_rad_s"]),
        },
        validation=validation,
    )
    with json_output.open("w", encoding="utf-8") as f:
        json.dump(document, f, indent=2)
    print(f"Wrote {json_output}")
    print(f"Final angle: {opening_angles[-1]:.2f} deg")
    if closing_angles:
        print(f"Closing-force final angle: {closing_angles[-1]:.2f} deg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
