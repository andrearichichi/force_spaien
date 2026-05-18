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


FONT_REGULAR = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
FONT_BOLD = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
FONT_SMALL = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)


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


def urdf_joint_dynamics(model_dir: Path) -> dict[str, dict[str, str]]:
    tree = ET.parse(model_dir / "mobility.urdf")
    result = {}
    for joint in tree.findall("joint"):
        dynamics = joint.find("dynamics")
        if dynamics is not None:
            result[joint.attrib.get("name", "")] = dict(dynamics.attrib)
    return result


@dataclass
class LaptopSim:
    scene: sapien.Scene
    laptop: sapien.physx.PhysxArticulation
    screen: sapien.physx.PhysxArticulationLinkComponent
    joint_index: int
    marker: sapien.Entity
    camera: sapien.render.RenderCameraComponent
    local_application_point: np.ndarray
    application_point_strategy: str


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
    scene.set_timestep(1.0 / 240.0)
    scene.set_ambient_light([0.72, 0.72, 0.72])
    scene.add_directional_light([0.25, 0.45, -1.0], [1.0, 1.0, 1.0], shadow=False)
    scene.add_directional_light([-0.6, -0.2, -1.0], [0.35, 0.35, 0.35], shadow=False)

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
        link.linear_damping = 0.0
        link.angular_damping = 0.02

    local_application_point = pick_handle_point_local(model_dir, link_name)
    application_point_strategy = "center of handle mesh on selected link"
    if local_application_point is None:
        if override_point is not None:
            local_application_point = override_point
            application_point_strategy = override_strategy or "manual application point override"
            initial_world_point = screen.get_entity_pose().to_transformation_matrix() @ local_application_point
        else:
            initial_world_point = pick_screen_edge_point(screen)
            local_application_point = np.linalg.inv(screen.get_entity_pose().to_transformation_matrix()) @ initial_world_point
            application_point_strategy = f"upper free border of {link_name} from tight AABB at initial pose"
    else:
        initial_world_point = screen.get_entity_pose().to_transformation_matrix() @ local_application_point

    marker = create_marker(scene)
    marker.set_pose(sapien.Pose(initial_world_point[:3]))

    camera = scene.add_camera("camera", width, height, math.radians(48), 0.01, 20.0)
    camera.set_entity_pose(
        look_at_pose(
            np.array([-1.18, -1.46, 0.86], dtype=np.float32),
            np.array([-0.08, 0.10, 0.05], dtype=np.float32),
        )
    )

    return LaptopSim(scene, laptop, screen, joint_index, marker, camera, local_application_point, application_point_strategy)


def application_point_world(sim: LaptopSim) -> np.ndarray:
    point = sim.screen.get_entity_pose().to_transformation_matrix() @ sim.local_application_point
    return point[:3].astype(np.float32)


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
    return (sim.camera.get_picture("Color")[..., :3].clip(0, 1) * 255).astype(np.uint8)


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
    draw.text(xy, text, fill=color[::-1], font=font, anchor=anchor)
    img[:] = np.asarray(pil)


def draw_label(img: np.ndarray, text: str, xy: tuple[int, int], color: tuple[int, int, int], scale: float = 0.75) -> None:
    font = FONT_SMALL if scale < 0.6 else FONT_REGULAR
    x, y = xy
    x = min(max(10, x), img.shape[1] - 10)
    y = min(max(24, y), img.shape[0] - 12)
    draw_text(img, text, (x + 2, y + 2), (255, 255, 255), font)
    draw_text(img, text, (x, y), color, font)


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
    cv2.rectangle(canvas, (x + 18, y + 12), (x + w - 18, y + h - 10), (248, 248, 248), -1, cv2.LINE_AA)
    cv2.rectangle(canvas, (x + 18, y + 12), (x + w - 18, y + h - 10), (205, 205, 205), 1, cv2.LINE_AA)
    cv2.circle(canvas, (x + 42, y + 42), 10, color, -1, cv2.LINE_AA)
    draw_text(canvas, title, (x + 62, y + 29), (24, 24, 24), FONT_BOLD)
    draw_text(canvas, force_text, (x + 34, y + 70), color, FONT_REGULAR)
    draw_text(canvas, direction_text, (x + 34, y + 98), (80, 80, 80), FONT_SMALL)
    draw_text(canvas, f"angolo: {angle_deg:6.1f} deg", (x + w - 245, y + 70), (24, 24, 24), FONT_REGULAR)


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

    inset_size = 150
    half = 54
    x0 = max(0, min(img.shape[1] - 2 * half, px - half))
    y0 = max(0, min(img.shape[0] - 2 * half, py - half))
    crop = img[y0 : y0 + 2 * half, x0 : x0 + 2 * half]
    if crop.size:
        zoom = cv2.resize(crop, (inset_size, inset_size), interpolation=cv2.INTER_CUBIC)
        inset_x = img.shape[1] - inset_size - 22
        inset_y = 22
        cv2.rectangle(img, (inset_x - 4, inset_y - 4), (inset_x + inset_size + 4, inset_y + inset_size + 4), (255, 255, 255), -1)
        cv2.rectangle(img, (inset_x - 4, inset_y - 4), (inset_x + inset_size + 4, inset_y + inset_size + 4), color, 3)
        img[inset_y : inset_y + inset_size, inset_x : inset_x + inset_size] = zoom


def draw_angle_plot(
    canvas: np.ndarray,
    opening_angles: list[float],
    closing_angles: list[float],
    initial_angle: float,
    x: int,
    y: int,
    w: int,
    h: int,
) -> None:
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (245, 245, 245), -1)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (180, 180, 180), 1)
    draw_label(canvas, "risposta nel tempo", (x + 14, y + 28), (20, 20, 20), 0.55)

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
        (opening_angles, (210, 55, 45)),
        (closing_angles, (25, 115, 210)),
    ):
        pts = [to_px(i, a, len(angles)) for i, a in enumerate(angles)]
        if len(pts) > 1:
            cv2.polylines(canvas, [np.array(pts, dtype=np.int32)], False, color, 2, cv2.LINE_AA)

    draw_label(canvas, "apre", (x + w - 130, y + 30), (210, 55, 45), 0.48)
    draw_label(canvas, "chiude", (x + w - 130, y + 54), (25, 115, 210), 0.48)


def pose_to_dict(pose: sapien.Pose) -> dict[str, list[float]]:
    return {"p": np.asarray(pose.p, dtype=float).tolist(), "q": np.asarray(pose.q, dtype=float).tolist()}


def link_dynamics_to_dict(link: sapien.physx.PhysxArticulationLinkComponent) -> dict[str, object]:
    return {
        "name": link.name,
        "mass": float(link.mass),
        "inertia": np.asarray(link.inertia, dtype=float).tolist(),
        "cmass_local_pose": pose_to_dict(link.cmass_local_pose),
        "linear_damping": float(link.linear_damping),
        "angular_damping": float(link.angular_damping),
        "disable_gravity": bool(link.disable_gravity),
    }


def optional_float(value_fn) -> float | None:
    try:
        return float(value_fn())
    except RuntimeError:
        return None


def optional_array(value_fn) -> list[float] | list[list[float]] | None:
    try:
        return np.asarray(value_fn(), dtype=float).tolist()
    except RuntimeError:
        return None


def optional_string(value_fn) -> str | None:
    try:
        return str(value_fn())
    except RuntimeError:
        return None


def joint_to_dict(joint: sapien.physx.PhysxArticulationJoint) -> dict[str, object]:
    return {
        "name": joint.name,
        "limits_rad": optional_array(joint.get_limit),
        "friction": optional_float(lambda: joint.friction),
        "damping": optional_float(lambda: joint.damping),
        "drive_mode": optional_string(lambda: joint.drive_mode),
        "drive_target": optional_array(lambda: joint.drive_target),
        "drive_velocity_target": optional_array(lambda: joint.drive_velocity_target),
        "force_limit": optional_float(lambda: joint.force_limit),
    }


def sample_to_dict(time_s: float, sim: LaptopSim, applied_force: np.ndarray) -> dict[str, object]:
    angle = float(sim.laptop.get_qpos()[sim.joint_index])
    return {
        "time_s": float(time_s),
        "hinge_angle_rad": angle,
        "hinge_angle_deg": math.degrees(angle),
        "hinge_velocity_rad_s": float(sim.laptop.get_qvel()[sim.joint_index]),
        "application_point_world": application_point_world(sim).astype(float).tolist(),
        "applied_force_world": applied_force.astype(float).tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="11691")
    parser.add_argument("--joint", default="joint_1")
    parser.add_argument("--link", default="link_1")
    parser.add_argument("--output", default=None)
    parser.add_argument("--force", type=float, default=0.5)
    parser.add_argument("--closing-force", type=float, default=0.5)
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--end-hold-seconds", type=float, default=2.0, help="Freeze the last frame for this many seconds")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--panel-width", type=int, default=720)
    parser.add_argument("--panel-height", type=int, default=448)
    parser.add_argument("--info-height", type=int, default=132)
    parser.add_argument("--plot-height", type=int, default=176)
    parser.add_argument("--initial-angle", type=float, default=-1.5)
    parser.add_argument("--direction", nargs=3, type=float, default=[0.0, 0.0, 1.0])
    parser.add_argument("--json-output", default=None)
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--keep-old", action="store_true", help="Do not delete old files in the output directory")
    args = parser.parse_args()

    model_dir = Path(args.model_dir).resolve()
    output, json_output = output_paths(model_dir, Path(args.output_root).resolve(), args.output, args.json_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not args.keep_old:
        clear_object_output(output)

    opening_dir = np.array(args.direction, dtype=np.float32)
    opening_dir /= np.linalg.norm(opening_dir) or 1.0
    closing_dir = -opening_dir
    opening_force = opening_dir * args.force
    closing_force = closing_dir * args.closing_force

    override_point, override_strategy = load_application_point_override(output.parent, args.link)
    opening_sim = setup_sim(model_dir, args.joint, args.link, args.panel_width, args.panel_height, args.initial_angle, override_point, override_strategy)
    closing_sim = setup_sim(model_dir, args.joint, args.link, args.panel_width, args.panel_height, args.initial_angle, override_point, override_strategy)
    urdf_dynamics = urdf_joint_dynamics(model_dir)
    metadata = {
        "model_dir": str(model_dir),
        "video_output": str(output),
        "fps": args.fps,
        "seconds": args.seconds,
        "end_hold_seconds": args.end_hold_seconds,
        "video_duration_seconds": args.seconds + args.end_hold_seconds,
        "timestep_s": 1.0 / 240.0,
        "initial_angle_rad": args.initial_angle,
        "initial_angle_deg": math.degrees(args.initial_angle),
        "joint": args.joint,
        "link": args.link,
        "force_magnitude_n": args.force,
        "closing_force_magnitude_n": args.closing_force,
        "opening_direction_world": opening_dir.astype(float).tolist(),
        "closing_direction_world": closing_dir.astype(float).tolist(),
        "application_point_strategy": opening_sim.application_point_strategy,
        "application_point_local_on_screen": opening_sim.local_application_point[:3].astype(float).tolist(),
        "joint_limits_rad": opening_sim.laptop.get_active_joints()[opening_sim.joint_index].get_limit().tolist(),
        "physics": {
            "urdf_joint_dynamics": urdf_dynamics,
            "urdf_joint_dynamics_present": bool(urdf_dynamics),
            "separate_static_dynamic_friction_present": False,
            "air_friction_model_present": False,
            "link_linear_damping_set_to": 0.0,
            "link_angular_damping_set_to": 0.02,
            "joint_drive_stiffness_damping_force_limit_set_to": [0.0, 0.0, 0.0],
        },
        "links": [link_dynamics_to_dict(link) for link in opening_sim.laptop.get_links()],
        "joints": [joint_to_dict(joint) for joint in opening_sim.laptop.get_joints()],
    }

    steps_per_frame = max(1, round(240 / args.fps))
    frame_count = max(1, int(args.seconds * args.fps))
    opening_angles: list[float] = []
    closing_angles: list[float] = []
    samples = {"opening_force": [], "closing_force": []}
    point_histories = {"opening_force": [], "closing_force": []}

    out_w = args.panel_width * 2
    out_h = args.panel_height + args.info_height + args.plot_height

    final_frame = None
    with imageio.get_writer(output, fps=args.fps, codec="libx264", quality=8) as writer:
        for _ in range(frame_count):
            for _ in range(steps_per_frame):
                opening_point = application_point_world(opening_sim)
                closing_point = application_point_world(closing_sim)
                opening_sim.screen.add_force_at_point(opening_force, opening_point, "force")
                closing_sim.screen.add_force_at_point(closing_force, closing_point, "force")
                opening_sim.scene.step()
                closing_sim.scene.step()

            time_s = len(opening_angles) / args.fps
            samples["opening_force"].append(sample_to_dict(time_s, opening_sim, opening_force))
            samples["closing_force"].append(sample_to_dict(time_s, closing_sim, closing_force))
            point_histories["opening_force"].append(application_point_world(opening_sim))
            point_histories["closing_force"].append(application_point_world(closing_sim))

            left = render_panel(opening_sim)
            right = render_panel(closing_sim)
            canvas = np.full((out_h, out_w, 3), 232, dtype=np.uint8)
            draw_force_annotation(
                left,
                opening_sim,
                opening_dir,
                args.force,
                args.panel_width,
                args.panel_height,
                (210, 55, 45),
                point_histories["opening_force"],
            )
            draw_force_annotation(
                right,
                closing_sim,
                closing_dir,
                args.closing_force,
                args.panel_width,
                args.panel_height,
                (25, 115, 210),
                point_histories["closing_force"],
            )
            canvas[: args.panel_height, : args.panel_width] = left
            canvas[: args.panel_height, args.panel_width :] = right
            cv2.line(canvas, (args.panel_width, 0), (args.panel_width, args.panel_height + args.info_height), (205, 205, 205), 2)

            open_angle = math.degrees(float(opening_sim.laptop.get_qpos()[opening_sim.joint_index]))
            close_angle = math.degrees(float(closing_sim.laptop.get_qpos()[closing_sim.joint_index]))
            opening_angles.append(open_angle)
            closing_angles.append(close_angle)

            draw_info_card(
                canvas,
                0,
                args.panel_height,
                args.panel_width,
                args.info_height,
                "apertura",
                f"F = {args.force:g} N",
                f"direzione world [{opening_dir[0]:.0f}, {opening_dir[1]:.0f}, {opening_dir[2]:.0f}]",
                open_angle,
                (210, 55, 45),
            )
            draw_info_card(
                canvas,
                args.panel_width,
                args.panel_height,
                args.panel_width,
                args.info_height,
                "chiusura",
                f"F = {args.closing_force:g} N",
                f"direzione world [{closing_dir[0]:.0f}, {closing_dir[1]:.0f}, {closing_dir[2]:.0f}]",
                close_angle,
                (25, 115, 210),
            )
            draw_angle_plot(
                canvas,
                opening_angles,
                closing_angles,
                args.initial_angle,
                22,
                args.panel_height + args.info_height + 16,
                out_w - 44,
                args.plot_height - 32,
            )
            writer.append_data(canvas)
            final_frame = canvas.copy()

        hold_frames = int(round(args.end_hold_seconds * args.fps))
        if final_frame is not None:
            for _ in range(hold_frames):
                writer.append_data(final_frame)

    print(f"Wrote {output}")
    with json_output.open("w", encoding="utf-8") as f:
        json.dump({"metadata": metadata, "samples": samples}, f, indent=2)
    print(f"Wrote {json_output}")
    print(f"Opening-force final angle: {opening_angles[-1]:.2f} deg")
    print(f"Closing-force final angle: {closing_angles[-1]:.2f} deg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
