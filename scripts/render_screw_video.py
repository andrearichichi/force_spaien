#!/usr/bin/env python3
"""Render a SAPIEN video for a screw-like coupled translation/rotation."""

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

FONT_REGULAR = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
FONT_SMALL = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
TIMESTEP = 1.0 / 240.0
LINEAR_DAMPING = 0.0
ANGULAR_DAMPING = 0.02
END_HOLD_MOTION_THRESHOLD = 1e-3
CAMERA_ZOOM_OUT = 1.55
DEFAULT_VIDEO_WIDTH = 1920
DEFAULT_PANEL_HEIGHT = 1080
DEFAULT_INFO_HEIGHT = 0
COLOR_BG = (242, 244, 246)
COLOR_BORDER = (210, 216, 224)
COLOR_ACCENT = (0, 122, 255)
COLOR_ROTATION_MARKER = (255, 45, 85)


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


@dataclass
class ScrewSim:
    scene: sapien.Scene
    articulation: sapien.physx.PhysxArticulation
    link: sapien.physx.PhysxArticulationLinkComponent
    linear_joint: sapien.physx.PhysxArticulationJoint
    rotary_joint: sapien.physx.PhysxArticulationJoint
    linear_index: int
    rotary_index: int
    camera: sapien.render.RenderCameraComponent
    local_application_point: np.ndarray
    application_point_strategy: str
    local_link_center: np.ndarray
    local_marker_start: np.ndarray
    local_marker_end: np.ndarray
    linear_start: float
    linear_end: float
    rotation_start: float
    rotation_end: float


@dataclass
class ScrewState:
    theta: float = 0.0
    omega: float = 0.0
    translation: float = 0.0


class ScrewMotionController:
    """Virtual helical kinematic constraint with one independent coordinate.

    SAPIEN is not simulating geometric thread contact here. The URDF exposes a
    revolute joint and a prismatic joint, but this controller treats theta as
    the only evolved DOF and imposes the screw coupling kinematically.
    """

    def __init__(self, sim: ScrewSim, args: argparse.Namespace):
        self.sim = sim
        self.theta_min = 0.0
        self.theta_max = max(0.0, sim.rotation_end - sim.rotation_start)
        inferred_pitch = 0.0
        if abs(self.theta_max) > 1e-9:
            inferred_pitch = (sim.linear_end - sim.linear_start) * (2.0 * math.pi) / self.theta_max
        self.pitch = inferred_pitch if args.pitch is None else float(args.pitch)
        self.z0 = sim.linear_start if args.z0 is None else float(args.z0)
        self.inertia = max(float(args.rotational_inertia), 1e-9)
        self.damping = max(0.0, float(args.rotary_damping))
        self.force_radius = max(float(args.force_radius), 0.0)
        legacy_friction_force = float(args.friction_torque) / self.force_radius if self.force_radius > 1e-9 else 0.0
        self.applied_force_tangential = (
            float(args.applied_force_tangential)
            if args.applied_force_tangential is not None
            else float(args.torque) / self.force_radius
            if self.force_radius > 1e-9
            else 0.0
        )
        self.opposing_tangential_friction = (
            max(0.0, float(args.opposing_tangential_friction))
            if args.opposing_tangential_friction is not None
            else max(0.0, legacy_friction_force)
        )
        self.torque_applied = self.applied_force_tangential * self.force_radius
        self.torque_resisting = self.opposing_tangential_friction * self.force_radius
        force_after_resistance = max(abs(self.applied_force_tangential) - self.opposing_tangential_friction, 0.0)
        force_sign = math.copysign(1.0, self.applied_force_tangential) if self.applied_force_tangential else 0.0
        self.net_torque = force_sign * force_after_resistance * self.force_radius

    def translation_from_theta(self, theta: float) -> float:
        return self.z0 + self.pitch * theta / (2.0 * math.pi)

    def progress(self, theta: float) -> float:
        if self.theta_max <= 1e-9:
            return 0.0
        return float(np.clip(theta / self.theta_max, 0.0, 1.0))

    def linear_velocity(self, omega: float) -> float:
        return self.pitch * omega / (2.0 * math.pi)

    def apply(self, state: ScrewState) -> float:
        theta = float(np.clip(state.theta, self.theta_min, self.theta_max))
        translation = self.translation_from_theta(theta)
        qpos = self.sim.articulation.get_qpos()
        qpos[self.sim.rotary_index] = self.sim.rotation_start + theta
        qpos[self.sim.linear_index] = translation
        self.sim.articulation.set_qpos(qpos)
        state.theta = theta
        state.translation = translation
        return self.progress(theta)

    def step(self, state: ScrewState, dt: float) -> ScrewState:
        omega_dot = (self.net_torque - self.damping * state.omega) / self.inertia
        omega = state.omega + omega_dot * dt
        theta = state.theta + omega * dt

        if theta <= self.theta_min:
            theta = self.theta_min
            omega = max(0.0, omega)
        elif theta >= self.theta_max:
            theta = self.theta_max
            omega = min(0.0, omega)

        return ScrewState(
            theta=theta,
            omega=omega,
            translation=self.translation_from_theta(theta),
        )

    def metadata(self) -> dict[str, object]:
        return {
            "motion_type": "screw",
            "virtual_physics": True,
            "real_thread_contact": False,
            "master_joint": self.sim.rotary_joint.name,
            "coupled_joint": self.sim.linear_joint.name,
            "pitch": float(self.pitch),
            "pitch_m_per_revolution": float(self.pitch),
            "z0": float(self.z0),
            "force_model": "tangential_force_on_cap",
            "applied_tangential_force_n": float(self.applied_force_tangential),
            "applied_force_tangential_n": float(self.applied_force_tangential),
            "opposing_tangential_friction_n": float(self.opposing_tangential_friction),
            "tangential_force_radius_m": float(self.force_radius),
            "force_radius_m": float(self.force_radius),
            "torque_applied_nm": float(self.torque_applied),
            "torque_resisting_nm": float(self.torque_resisting),
            "net_torque_nm": float(self.net_torque),
            "damping_nm_s_rad": float(self.damping),
            "damping": float(self.damping),
            "inertia": float(self.inertia),
            "constraint_equation": "translation = z0 + pitch * theta / (2*pi)",
        }


def screw_constraint_stats(samples: list[dict[str, object]]) -> tuple[float, float]:
    errors = [abs(float(sample["constraint_error_m"])) for sample in samples]
    if not errors:
        return 0.0, 0.0
    return max(errors), float(np.mean(errors))


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


def camera_eye_target(view: str) -> tuple[np.ndarray, np.ndarray]:
    target = np.array([0.0, -0.06, 0.0], dtype=np.float32)
    if view == "top_side":
        return np.array([1.15, -2.15, 1.75], dtype=np.float32), target
    return np.array([-0.72, -1.85, 0.72], dtype=np.float32), target


def mesh_vertices(mesh_path: Path) -> np.ndarray:
    vertices = []
    with mesh_path.open("r", encoding="utf-8", errors="ignore") as mesh_file:
        for line in mesh_file:
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


def link_visual_aabb(model_dir: Path, link_name: str) -> tuple[np.ndarray, np.ndarray]:
    root = ET.parse(model_dir / "mobility.urdf").getroot()
    link = root.find(f".//link[@name='{link_name}']")
    if link is None:
        raise RuntimeError(f"Could not find {link_name} in {model_dir / 'mobility.urdf'}")
    vertices = []
    for visual in link.findall("visual"):
        mesh = visual.find("./geometry/mesh")
        filename = mesh.attrib.get("filename") if mesh is not None else None
        if filename:
            vertices.append(mesh_vertices(model_dir / filename) + visual_origin(visual))
    if not vertices:
        raise RuntimeError(f"No visual mesh found for {link_name}")
    all_vertices = np.concatenate(vertices, axis=0)
    return all_vertices.min(axis=0), all_vertices.max(axis=0)


def default_application_point(model_dir: Path, link_name: str, direction: np.ndarray) -> np.ndarray:
    mins, maxs = link_visual_aabb(model_dir, link_name)
    point = 0.5 * (mins + maxs)
    screw_axis = int(np.argmax(np.abs(direction)))
    plane_axes = [axis for axis in range(3) if axis != screw_axis]
    radial_axis = plane_axes[int((maxs[plane_axes[0]] - mins[plane_axes[0]]) < (maxs[plane_axes[1]] - mins[plane_axes[1]]))]
    point[radial_axis] = maxs[radial_axis]
    return np.append(point.astype(np.float32), np.float32(1.0))


def default_link_center(model_dir: Path, link_name: str) -> np.ndarray:
    mins, maxs = link_visual_aabb(model_dir, link_name)
    return np.append((0.5 * (mins + maxs)).astype(np.float32), np.float32(1.0))


def default_rotation_marker(model_dir: Path, link_name: str, direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mins, maxs = link_visual_aabb(model_dir, link_name)
    center = 0.5 * (mins + maxs)
    face_axis = int(np.argmax(np.abs(direction)))
    plane_axes = [axis for axis in range(3) if axis != face_axis]
    radial_axis = plane_axes[int((maxs[plane_axes[0]] - mins[plane_axes[0]]) < (maxs[plane_axes[1]] - mins[plane_axes[1]]))]
    face_value = maxs[face_axis] if direction[face_axis] < 0 else mins[face_axis]
    radius = maxs[radial_axis] - center[radial_axis]
    if abs(radius) < 1e-6:
        radius = (maxs[radial_axis] - mins[radial_axis]) * 0.5
    start = center.copy()
    end = center.copy()
    start[face_axis] = face_value
    end[face_axis] = face_value
    start[radial_axis] = center[radial_axis] + 0.18 * radius
    end[radial_axis] = center[radial_axis] + 0.78 * radius
    return np.append(start.astype(np.float32), np.float32(1.0)), np.append(end.astype(np.float32), np.float32(1.0))


def explicit_contact_point(args: argparse.Namespace) -> tuple[np.ndarray | None, str | None]:
    if args.contact_point_local is None:
        return None, None
    point = np.append(np.asarray(args.contact_point_local, dtype=np.float32), np.float32(1.0))
    return point, args.contact_point_strategy


def background_mask_from_render(camera: sapien.render.RenderCameraComponent, image: np.ndarray) -> np.ndarray:
    try:
        segmentation = camera.get_picture("Segmentation")
        return segmentation[..., 0] == 0
    except Exception:
        luminance = image.mean(axis=2)
        return luminance > 230


def application_point_world(sim: ScrewSim) -> np.ndarray:
    point = sim.link.get_entity_pose().to_transformation_matrix() @ sim.local_application_point
    return point[:3].astype(np.float32)


def local_point_world(sim: ScrewSim, local_point: np.ndarray) -> np.ndarray:
    point = sim.link.get_entity_pose().to_transformation_matrix() @ local_point
    return point[:3].astype(np.float32)


def append_if_moved(history: list[np.ndarray], point: np.ndarray, threshold: float = 1e-5) -> None:
    if not history or float(np.linalg.norm(point - history[-1])) > threshold:
        history.append(point)


def project(camera: sapien.render.RenderCameraComponent, point: np.ndarray) -> tuple[int, int] | None:
    camera_point = camera.get_extrinsic_matrix() @ np.array([point[0], point[1], point[2], 1.0], dtype=np.float32)
    if camera_point[2] <= 0:
        return None
    uvw = camera.get_intrinsic_matrix() @ camera_point
    return int(round(uvw[0] / uvw[2])), int(round(uvw[1] / uvw[2]))


def render_panel(sim: ScrewSim) -> np.ndarray:
    sim.scene.update_render()
    sim.camera.take_picture()
    image = (sim.camera.get_picture("Color")[..., :3].clip(0, 1) * 255).astype(np.uint8)
    mask = background_mask_from_render(sim.camera, image)
    top = np.array([246, 248, 251], dtype=np.float32)
    bottom = np.array([229, 236, 243], dtype=np.float32)
    t = np.linspace(0.0, 1.0, image.shape[0], dtype=np.float32)[:, None]
    gradient = (top * (1.0 - t) + bottom * t).astype(np.uint8)
    gradient = np.repeat(gradient[:, None, :], image.shape[1], axis=1)
    image[mask] = gradient[mask]
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


def draw_label(img: np.ndarray, text: str, xy: tuple[int, int], color: tuple[int, int, int]) -> None:
    x, y = xy
    x = min(max(10, x), img.shape[1] - 10)
    y = min(max(24, y), img.shape[0] - 12)
    draw_text(img, text, (x + 2, y + 2), (255, 255, 255), FONT_SMALL)
    draw_text(img, text, (x, y), color, FONT_SMALL)


def draw_torque_annotation(
    img: np.ndarray,
    sim: ScrewSim,
    torque_axis: np.ndarray,
    torque: float,
    point_history: list[np.ndarray],
) -> None:
    projected_history = [uv for p in point_history if (uv := project(sim.camera, p)) is not None]
    if len(projected_history) > 1:
        cv2.polylines(img, [np.array(projected_history[-80:], dtype=np.int32)], False, (170, 210, 255), 5, cv2.LINE_AA)
        cv2.polylines(img, [np.array(projected_history[-80:], dtype=np.int32)], False, COLOR_ACCENT, 2, cv2.LINE_AA)

    point = application_point_world(sim)
    uv = project(sim.camera, point)
    if uv is None:
        return
    px, py = uv
    cv2.circle(img, (px, py), 16, (235, 246, 255), -1, cv2.LINE_AA)
    cv2.circle(img, (px, py), 16, COLOR_ACCENT, 4, cv2.LINE_AA)

    axis_end = project(sim.camera, point + torque_axis * 0.18)
    clockwise = True
    if axis_end is not None:
        axis_px = np.array(axis_end, dtype=np.float32) - np.array([px, py], dtype=np.float32)
        clockwise = axis_px[1] >= 0

    radius = 62
    start_angle = 35 if clockwise else 325
    end_angle = 320 if clockwise else 40
    cv2.ellipse(img, (px, py), (radius, radius), 0, start_angle, end_angle, (235, 246, 255), 10, cv2.LINE_AA)
    cv2.ellipse(img, (px, py), (radius, radius), 0, start_angle, end_angle, COLOR_ACCENT, 5, cv2.LINE_AA)
    angle = math.radians(end_angle if clockwise else start_angle)
    tip = np.array([px + radius * math.cos(angle), py + radius * math.sin(angle)], dtype=np.float32)
    tangent = np.array([-math.sin(angle), math.cos(angle)], dtype=np.float32)
    if not clockwise:
        tangent *= -1.0
    tail = (tip - tangent * 28).astype(int)
    tip_i = tip.astype(int)
    cv2.arrowedLine(img, tuple(tail), tuple(tip_i), (235, 246, 255), 10, cv2.LINE_AA, tipLength=0.7)
    cv2.arrowedLine(img, tuple(tail), tuple(tip_i), COLOR_ACCENT, 5, cv2.LINE_AA, tipLength=0.7)
    draw_label(img, f"{torque:g} Nm", (px + radius + 16, py), COLOR_ACCENT)


def draw_tangential_force_annotation(
    img: np.ndarray,
    sim: ScrewSim,
    screw_axis_world: np.ndarray,
    controller: ScrewMotionController,
    point_history: list[np.ndarray],
) -> None:
    projected_history = [uv for p in point_history if (uv := project(sim.camera, p)) is not None]
    if len(projected_history) > 1:
        cv2.polylines(img, [np.array(projected_history[-80:], dtype=np.int32)], False, (170, 210, 255), 5, cv2.LINE_AA)
        cv2.polylines(img, [np.array(projected_history[-80:], dtype=np.int32)], False, COLOR_ACCENT, 2, cv2.LINE_AA)

    point = application_point_world(sim)
    start_uv = project(sim.camera, point)
    center_uv = project(sim.camera, local_point_world(sim, sim.local_link_center))
    if start_uv is None:
        return
    if center_uv is None:
        return

    radius_px = np.asarray(start_uv, dtype=np.float32) - np.asarray(center_uv, dtype=np.float32)
    if np.linalg.norm(radius_px) < 2.0:
        return
    radius_px = radius_px / np.linalg.norm(radius_px)
    direction_px = np.asarray([-radius_px[1], radius_px[0]], dtype=np.float32)
    force_sign = math.copysign(1.0, controller.applied_force_tangential) if controller.applied_force_tangential else 1.0
    direction_px *= -force_sign

    arrow_len = 150
    end_uv = (
        int(round(start_uv[0] + direction_px[0] * arrow_len)),
        int(round(start_uv[1] + direction_px[1] * arrow_len)),
    )

    cv2.circle(img, start_uv, 15, (235, 246, 255), -1, cv2.LINE_AA)
    cv2.circle(img, start_uv, 15, COLOR_ACCENT, 4, cv2.LINE_AA)
    cv2.arrowedLine(img, start_uv, end_uv, (235, 246, 255), 13, cv2.LINE_AA, tipLength=0.22)
    cv2.arrowedLine(img, start_uv, end_uv, COLOR_ACCENT, 7, cv2.LINE_AA, tipLength=0.22)
    draw_label(img, f"{controller.applied_force_tangential:g} N", (end_uv[0] + 14, end_uv[1]), COLOR_ACCENT)


def draw_rotation_marker(img: np.ndarray, sim: ScrewSim) -> None:
    start = project(sim.camera, local_point_world(sim, sim.local_marker_start))
    end = project(sim.camera, local_point_world(sim, sim.local_marker_end))
    if start is None or end is None:
        return
    cv2.line(img, start, end, (255, 240, 245), 18, cv2.LINE_AA)
    cv2.line(img, start, end, COLOR_ROTATION_MARKER, 10, cv2.LINE_AA)
    cv2.circle(img, end, 13, (255, 240, 245), -1, cv2.LINE_AA)
    cv2.circle(img, end, 8, COLOR_ROTATION_MARKER, -1, cv2.LINE_AA)


def draw_panel_frame(canvas: np.ndarray, width: int, height: int) -> None:
    cv2.rectangle(canvas, (0, 0), (width - 1, height - 1), COLOR_BORDER, 1, cv2.LINE_AA)


def setup_sim(args: argparse.Namespace, model_dir: Path) -> ScrewSim:
    scene = sapien.Scene()
    scene.set_timestep(TIMESTEP)
    scene.set_ambient_light([0.78, 0.80, 0.84])
    scene.add_directional_light([0.2, -0.45, -1.0], [1.0, 1.0, 0.96], shadow=False)
    scene.add_directional_light([-0.7, 0.25, -1.0], [0.42, 0.48, 0.56], shadow=False)

    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    articulation = loader.load(str(model_dir / "mobility.urdf"))
    linear_joint = articulation.find_joint_by_name(args.linear_joint)
    rotary_joint = articulation.find_joint_by_name(args.rotary_joint)
    link = articulation.find_link_by_name(args.link)
    if linear_joint is None or rotary_joint is None or link is None:
        raise RuntimeError(f"Could not find {args.linear_joint}/{args.rotary_joint}/{args.link}.")

    active_joints = list(articulation.get_active_joints())
    linear_index = active_joints.index(linear_joint)
    rotary_index = active_joints.index(rotary_joint)
    linear_limits = linear_joint.get_limit()
    if linear_limits.size:
        low, high = [float(v) for v in linear_limits[0]]
    else:
        low, high = -0.1, 0.0

    direction = np.asarray(args.direction, dtype=np.float32)
    direction /= np.linalg.norm(direction) or 1.0
    linear_start = high if args.translation_start is None else float(args.translation_start)
    linear_end = low if args.translation_end is None else float(args.translation_end)
    rotation_start = math.radians(args.rotation_start_degrees)
    rotation_end = math.radians(args.rotation_end_degrees)

    qpos = np.zeros_like(articulation.get_qpos(), dtype=np.float32)
    qpos[linear_index] = linear_start
    qpos[rotary_index] = rotation_start
    articulation.set_qpos(qpos)

    for joint in articulation.get_joints():
        joint.set_drive_property(0.0, 0.0, 0.0)
    for body_link in articulation.get_links():
        body_link.disable_gravity = True
        body_link.linear_damping = LINEAR_DAMPING
        body_link.angular_damping = ANGULAR_DAMPING

    explicit_point, explicit_strategy = explicit_contact_point(args)
    if explicit_point is not None:
        local_point = explicit_point
        strategy = explicit_strategy or "manual application point override"
    else:
        local_point = default_application_point(model_dir, args.link, direction)
        strategy = "side/rim point for tangential cap force from visual AABB"
    local_center = default_link_center(model_dir, args.link)
    marker_start, marker_end = default_rotation_marker(model_dir, args.link, direction)

    camera = scene.add_camera("camera", args.panel_width, args.panel_height, math.radians(44), 0.01, 20.0)
    camera_eye, camera_target = camera_eye_target(args.camera_view)
    camera.set_entity_pose(look_at_pose(zoomed_eye(camera_eye, camera_target), camera_target))

    return ScrewSim(
        scene,
        articulation,
        link,
        linear_joint,
        rotary_joint,
        linear_index,
        rotary_index,
        camera,
        local_point,
        strategy,
        local_center,
        marker_start,
        marker_end,
        linear_start,
        linear_end,
        rotation_start,
        rotation_end,
    )


def sample_to_dict(
    frame: int,
    time_s: float,
    sim: ScrewSim,
    torque_axis: np.ndarray,
    applied_axial_force: np.ndarray,
    controller: ScrewMotionController,
    state: ScrewState,
    progress: float,
) -> dict[str, object]:
    qpos = sim.articulation.get_qpos()
    theta = float(state.theta)
    omega = float(state.omega)
    angle = float(sim.rotation_start + theta)
    translation = float(qpos[sim.linear_index])
    expected_translation = float(controller.translation_from_theta(theta))
    constraint_error = translation - expected_translation
    master_limits = sim.rotary_joint.get_limit().tolist()
    coupled_limits = sim.linear_joint.get_limit().tolist()
    master_lower, master_upper = float(master_limits[0][0]), float(master_limits[0][1])
    coupled_lower, coupled_upper = float(coupled_limits[0][0]), float(coupled_limits[0][1])
    clamped = abs(theta - controller.theta_min) <= 1e-3 or abs(theta - controller.theta_max) <= 1e-3
    return {
        "frame": int(frame),
        "time": float(time_s),
        "time_s": float(time_s),
        "motion_type": "screw",
        "force_model": "tangential_force_on_cap",
        "screw_progress": float(progress),
        "theta_rad": theta,
        "theta_deg": math.degrees(theta),
        "omega_rad_s": omega,
        "translation_m": translation,
        "expected_translation_m": expected_translation,
        "constraint_error_m": float(constraint_error),
        "theta": theta,
        "omega": omega,
        "translation": translation,
        "pitch": float(controller.pitch),
        "z0": float(controller.z0),
        "joint_position_m": float(qpos[sim.linear_index]),
        "joint_velocity_m_s": float(controller.linear_velocity(omega)),
        "joint_angle_rad": angle,
        "joint_angle_deg": math.degrees(angle),
        "joint_velocity_rad_s": omega,
        "applied_force_tangential_n": float(controller.applied_force_tangential),
        "applied_tangential_force_n": float(controller.applied_force_tangential),
        "opposing_tangential_friction_n": float(controller.opposing_tangential_friction),
        "force_radius_m": float(controller.force_radius),
        "tangential_force_radius_m": float(controller.force_radius),
        "torque_applied_nm": float(controller.torque_applied),
        "torque_resisting_nm": float(controller.torque_resisting),
        "net_torque_nm": float(controller.net_torque),
        "pitch_m_per_revolution": float(controller.pitch),
        "master_joint_lower_limit_rad": master_lower,
        "master_joint_upper_limit_rad": master_upper,
        "coupled_joint_lower_limit_m": coupled_lower,
        "coupled_joint_upper_limit_m": coupled_upper,
        "clamped_at_limit": bool(clamped),
        "application_point_world": application_point_world(sim).astype(float).tolist(),
        "torque_visual_anchor_world": application_point_world(sim).astype(float).tolist(),
        "applied_torque_world": (controller.net_torque * torque_axis).astype(float).tolist(),
        "applied_axial_force_world": applied_axial_force.astype(float).tolist(),
    }


def run(args: argparse.Namespace) -> int:
    model_dir = resolve_model_dir(args.model_dir)
    output, json_output = output_paths(model_dir, Path(args.output_root).resolve(), args.output, args.json_output)
    if not args.keep_old:
        clear_object_output(output)

    sim = setup_sim(args, model_dir)
    torque_axis = np.asarray(args.direction, dtype=np.float32)
    torque_axis /= np.linalg.norm(torque_axis) or 1.0
    axial_force_dir = np.asarray(args.axial_force_direction, dtype=np.float32)
    axial_force_dir /= np.linalg.norm(axial_force_dir) or 1.0
    axial_force = axial_force_dir * args.axial_force

    steps_per_frame = max(1, round(240 / args.fps))
    frame_count = max(1, int(args.seconds * args.fps))
    render_simulated_seconds = frame_count * steps_per_frame * TIMESTEP
    samples: list[dict[str, object]] = []
    point_history: list[np.ndarray] = []
    final_frame = None
    actual_end_hold_seconds = 0.0
    controller = ScrewMotionController(sim, args)
    state = ScrewState(translation=controller.z0)
    controller.apply(state)

    if args.mode == "render":
        with imageio.get_writer(output, fps=args.fps, codec="libx264", quality=8, macro_block_size=1) as writer:
            for frame_index in range(frame_count):
                progress = controller.apply(state)
                time_s = sample_time_from_frame(frame_index, steps_per_frame, TIMESTEP)
                samples.append(
                    sample_to_dict(
                        frame_index,
                        time_s,
                        sim,
                        torque_axis,
                        axial_force,
                        controller,
                        state,
                        progress,
                    )
                )
                append_if_moved(point_history, application_point_world(sim))

                panel = fit_panel(render_panel(sim), args.panel_width, args.panel_height)
                canvas = np.full((args.panel_height + args.info_height + args.plot_height, args.panel_width, 3), COLOR_BG, dtype=np.uint8)
                draw_rotation_marker(panel, sim)
                draw_tangential_force_annotation(panel, sim, torque_axis, controller, point_history)
                canvas[: args.panel_height, : args.panel_width] = panel
                draw_panel_frame(canvas, args.panel_width, args.panel_height)
                writer.append_data(canvas)
                final_frame = canvas.copy()
                for _ in range(steps_per_frame):
                    state = controller.step(state, TIMESTEP)
            hold_frames = int(round(args.end_hold_seconds * args.fps))
            final_motion = abs(float(state.omega))
            append_hold = should_append_end_hold(args.end_hold_mode, hold_frames, final_motion)
            if args.end_hold_mode == "always" and append_hold:
                warn_if_holding_active_motion(final_motion)
            if final_frame is not None and append_hold:
                actual_end_hold_seconds = hold_frames / args.fps
                for _ in range(hold_frames):
                    writer.append_data(final_frame)
        simulated_seconds = render_simulated_seconds
        video_output = output
    else:
        steps = max(1, int(args.seconds / TIMESTEP))
        sample_interval = max(1, round(1.0 / (TIMESTEP * args.fps)))
        for step in range(steps):
            state = controller.step(state, TIMESTEP)
            progress = controller.apply(state)
            if step % sample_interval == 0 or step == steps - 1:
                samples.append(
                    sample_to_dict(
                        len(samples),
                        sample_time_from_step(step, TIMESTEP),
                        sim,
                        torque_axis,
                        axial_force,
                        controller,
                        state,
                        progress,
                    )
                )
        simulated_seconds = steps * TIMESTEP
        video_output = None

    sample_series = {"screw_motion": samples}
    summary = build_summary(
        sample_series=sample_series,
        physics_step_count=frame_count * steps_per_frame if args.mode == "render" else max(1, int(args.seconds / TIMESTEP)),
        position_key="joint_position_m",
        velocity_key="joint_velocity_m_s",
        secondary_position_key="joint_angle_deg",
        initial_position_value=sim.linear_start,
        initial_secondary_position_value=math.degrees(sim.rotation_start),
    )
    rotation_delta = controller.theta_max
    linear_delta = sim.linear_end - sim.linear_start
    final_sample = samples[-1]
    constraint_error_max, constraint_error_mean = screw_constraint_stats(samples)
    pitch_axis = "positive" if controller.pitch > 0.0 else "negative" if controller.pitch < 0.0 else "zero"
    screw_metadata = controller.metadata()
    screw_metadata.update(
        {
            "theta": final_sample["theta_rad"],
            "omega": final_sample["omega_rad_s"],
            "translation": final_sample["translation_m"],
            "theta_limit_rad": controller.theta_max,
            "translation_start_m": float(sim.linear_start),
            "translation_end_m": float(sim.linear_end),
            "constraint_error_max": constraint_error_max,
            "constraint_error_mean": constraint_error_mean,
            "positive_theta_translation_axis": pitch_axis,
        }
    )
    metadata = build_metadata(
        model_dir=model_dir,
        mode=args.mode,
        joint_type="screw",
        joint_name=f"{args.linear_joint}+{args.rotary_joint}",
        link_name=args.link,
        json_output=json_output,
        video_output=video_output,
        fps=args.fps,
        requested_seconds=args.seconds,
        simulated_seconds=simulated_seconds,
        timestep_s=TIMESTEP,
        sample_interval_s=steps_per_frame * TIMESTEP if args.mode == "render" else max(1, round(1.0 / (TIMESTEP * args.fps))) * TIMESTEP,
        end_hold_seconds=actual_end_hold_seconds if args.mode == "render" else None,
        actuation={
            "force_model": "tangential_force_on_cap",
            "applied_tangential_force_n": float(controller.applied_force_tangential),
            "applied_force_tangential_n": float(controller.applied_force_tangential),
            "opposing_tangential_friction_n": float(controller.opposing_tangential_friction),
            "tangential_force_radius_m": float(controller.force_radius),
            "force_radius_m": float(controller.force_radius),
            "torque_applied_nm": float(controller.torque_applied),
            "torque_resisting_nm": float(controller.torque_resisting),
            "net_torque_nm": float(controller.net_torque),
            "damping_nm_s_rad": float(controller.damping),
            "inertia": float(controller.inertia),
            "torque": {
                "magnitude_nm": float(controller.net_torque),
                "axis_world": torque_axis.astype(float).tolist(),
                "application": "derived_from_tangential_force_on_cap",
                "visual_anchor": "application_point",
                "deprecated_input_torque_nm": args.torque,
            },
            "axial_force": {
                "magnitude_n": args.axial_force,
                "direction_world": axial_force_dir.astype(float).tolist(),
            },
            "linear_joint": args.linear_joint,
            "rotary_joint": args.rotary_joint,
            "linear_limits_m": sim.linear_joint.get_limit().tolist(),
            "rotary_limits_rad": sim.rotary_joint.get_limit().tolist(),
            "legacy_aliases": {
                "note": "Kept for backward compatibility; canonical screw coupling lives in metadata.kinematic_constraint.",
                "screw_dynamics": screw_metadata,
                "screw": screw_metadata,
            },
            "screw_dynamics": {
                "legacy_alias": True,
                **screw_metadata,
            },
            "screw": {
                "legacy_alias": True,
                **screw_metadata,
            },
        },
        application_point={
            "strategy": sim.application_point_strategy,
            "local_on_link": sim.local_application_point[:3].astype(float).tolist(),
            "role": "visual_anchor_for_tangential_force_cue",
            "note": "The screw model uses a simplified tangential force on the cap; this side/rim point anchors the visual cue on the moving link.",
        },
        summary=summary,
        articulation=sim.articulation,
        limit_key="limits",
        linear_damping=LINEAR_DAMPING,
        angular_damping=ANGULAR_DAMPING,
    )
    metadata["joint_limits"] = {
        "master_joint_limits_rad": sim.rotary_joint.get_limit().tolist(),
        "coupled_joint_limits_m": sim.linear_joint.get_limit().tolist(),
    }
    metadata["kinematic_constraint"] = {
        "type": "virtual_helical",
        "virtual_physics": True,
        "real_thread_contact": False,
        "master_joint": args.rotary_joint,
        "coupled_joint": args.linear_joint,
        "pitch_m_per_revolution": float(controller.pitch),
        "z0": float(controller.z0),
        "theta_limit_rad": float(controller.theta_max),
        "translation_start_m": float(sim.linear_start),
        "translation_end_m": float(sim.linear_end),
        "constraint_equation": "translation = z0 + pitch * theta / (2*pi)",
        "constraint_error_max": constraint_error_max,
        "constraint_error_mean": constraint_error_mean,
        "positive_theta_translation_axis": pitch_axis,
    }
    metadata["simulation_config"] = {
        "timestep_s": TIMESTEP,
        "fps": args.fps,
        "requested_seconds": args.seconds,
        "end_hold_seconds": actual_end_hold_seconds if args.mode == "render" else None,
    }
    metadata["motion_bounds"] = {
        "translation_start_m": float(sim.linear_start),
        "translation_end_m": float(sim.linear_end),
        "theta_start_rad": 0.0,
        "theta_end_rad": float(controller.theta_max),
        "rotation_start_rad": float(sim.rotation_start),
        "rotation_end_rad": float(sim.rotation_start + controller.theta_max),
        "rotation_start_deg": math.degrees(sim.rotation_start),
        "rotation_end_deg": math.degrees(sim.rotation_start + controller.theta_max),
    }
    metadata["screw_coupling"] = {
        "legacy_alias": True,
        "linear_delta_m": float(linear_delta),
        "rotation_delta_rad": float(rotation_delta),
        "pitch_m_per_revolution": float(controller.pitch),
        "pitch_m_per_rad": float(controller.pitch / (2.0 * math.pi)),
        "z0": float(controller.z0),
        "constraint": "translation = z0 + pitch * theta / (2*pi)",
    }

    validation = validation_for_motion(
        initial_position=0.0,
        final_position=float(final_sample["theta_rad"]),
        final_velocity=float(final_sample["omega_rad_s"]),
        limits=[[0.0, float(controller.theta_max)]],
        actuation_sign=float(controller.net_torque),
    )
    if constraint_error_max > args.constraint_tolerance:
        validation["warnings"].append(
            f"Screw constraint error exceeds tolerance: {constraint_error_max:.6g} m > {args.constraint_tolerance:.6g} m."
        )
    validation["constraint_error_max"] = float(constraint_error_max)
    validation["constraint_error_mean"] = float(constraint_error_mean)
    document = motion_document(
        motion_type="screw",
        metadata=metadata,
        sample_series=sample_series,
        initial_state={
            "theta_rad": 0.0,
            "theta_deg": 0.0,
            "translation_m": float(sim.linear_start),
        },
        final_state={
            "theta_rad": float(final_sample["theta_rad"]),
            "theta_deg": float(final_sample["theta_deg"]),
            "omega_rad_s": float(final_sample["omega_rad_s"]),
            "translation_m": float(final_sample["translation_m"]),
        },
        validation=validation,
    )

    if not args.no_json_output:
        with json_output.open("w", encoding="utf-8") as f:
            json.dump(document, f, indent=2)

    if args.mode == "render":
        print(f"Wrote {output}")
    if not args.no_json_output:
        print(f"Wrote {json_output}")
    if constraint_error_max > args.constraint_tolerance:
        print(
            "WARNING: Screw constraint check failed: "
            f"max error = {constraint_error_max:.6g} m > tolerance = {args.constraint_tolerance:.6g} m"
        )
    else:
        print(f"Screw constraint check passed: max error = {constraint_error_max:.6g} m")
    print(f"Final screw theta: {final_sample['theta_deg']:.2f} deg")
    print(f"Final screw translation: {final_sample['translation_m']:.6f} m")
    print(f"Pitch used: {controller.pitch:.6f} m/revolution")
    print(f"Applied tangential force: {controller.applied_force_tangential:.6f} N")
    print(f"Opposing tangential friction: {controller.opposing_tangential_friction:.6f} N")
    print(f"Force radius: {controller.force_radius:.6f} m")
    print(f"Net torque: {controller.net_torque:.6f} Nm")
    print(f"Number of revolutions: {float(final_sample['theta_rad']) / (2.0 * math.pi):.6f}")
    print(f"Constraint max error: {constraint_error_max:.6g} m")
    print(f"Positive theta with this pitch produces translation along {pitch_axis} prismatic axis")
    print(f"Final screw: {samples[-1]['joint_position_m']:.4f} m, {samples[-1]['joint_angle_deg']:.2f} deg")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["render", "apply"], default="render")
    parser.add_argument("--model-dir", default="3763")
    parser.add_argument("--output", default=None)
    parser.add_argument("--json-output", default=None)
    parser.add_argument("--no-json-output", action="store_true")
    parser.add_argument("--linear-joint", default="joint_2")
    parser.add_argument("--rotary-joint", default="joint_0")
    parser.add_argument("--link", default="link_0")
    parser.add_argument("--torque", type=float, default=0.05, help="Deprecated screw input; used only if tangential force is omitted.")
    parser.add_argument("--applied-force-tangential", type=float, default=None)
    parser.add_argument("--opposing-tangential-friction", type=float, default=None)
    parser.add_argument("--force-radius", type=float, default=0.05)
    parser.add_argument("--rotational-inertia", type=float, default=0.002)
    parser.add_argument("--friction-torque", type=float, default=0.0, help="Deprecated; converted to opposing tangential friction if needed.")
    parser.add_argument("--rotary-damping", type=float, default=0.02)
    parser.add_argument("--pitch", type=float, default=None, help="Screw pitch in meters per full revolution.")
    parser.add_argument("--z0", type=float, default=None, help="Prismatic displacement at theta=0.")
    parser.add_argument("--constraint-tolerance", type=float, default=1e-6)
    parser.add_argument("--axial-force", type=float, default=0.0)
    parser.add_argument("--axial-force-direction", nargs=3, type=float, default=[0.0, 0.0, -1.0])
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--end-hold-seconds", type=float, default=2.0)
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
    parser.add_argument("--camera-view", choices=["main", "top_side"], default="main")
    parser.add_argument("--direction", nargs=3, type=float, default=[0.0, -1.0, 0.0], help="Torque axis in world coordinates")
    parser.add_argument("--translation-start", type=float, default=None)
    parser.add_argument("--translation-end", type=float, default=None)
    parser.add_argument("--rotation-start-degrees", type=float, default=0.0)
    parser.add_argument("--rotation-end-degrees", type=float, default=180.0)
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--contact-point-local", nargs=3, type=float, default=None)
    parser.add_argument("--contact-point-strategy", default=None)
    parser.add_argument("--keep-old", action="store_true")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
