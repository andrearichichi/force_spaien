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


def setup_sim(model_dir: Path, drawer_index: int, width: int, height: int, force_dir: np.ndarray, override_point: np.ndarray | None = None, override_strategy: str | None = None) -> DrawerSim:
    scene = sapien.Scene()
    scene.set_timestep(1.0 / 240.0)
    scene.set_ambient_light([0.72, 0.72, 0.72])
    scene.add_directional_light([0.2, -0.45, -1.0], [1.0, 1.0, 1.0], shadow=False)
    scene.add_directional_light([-0.7, 0.25, -1.0], [0.38, 0.38, 0.38], shadow=False)

    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    cabinet = loader.load(str(model_dir / "mobility.urdf"))
    cabinet.set_qpos(np.zeros_like(cabinet.get_qpos(), dtype=np.float32))

    for joint in cabinet.get_joints():
        joint.set_drive_property(0.0, 0.0, 0.0)
    for link in cabinet.get_links():
        link.disable_gravity = True
        link.linear_damping = 0.0
        link.angular_damping = 0.02

    joint = cabinet.find_joint_by_name(f"joint_{drawer_index}")
    drawer = cabinet.find_link_by_name(f"link_{drawer_index}")
    if joint is None or drawer is None:
        raise RuntimeError(f"Could not find joint_{drawer_index}/link_{drawer_index}. Try --drawer 0, 1, 2, or 3.")

    try:
        handle_point_local = pick_handle_pull_point_local(model_dir, drawer_index)
        local_application_point = np.append(handle_point_local, np.float32(1.0))
        application_point_strategy = "center of handle mesh on selected link"
    except RuntimeError:
        if override_point is not None:
            local_application_point = override_point
            application_point_strategy = override_strategy or "manual application point override"
        else:
            local_application_point = np.linalg.inv(drawer.get_entity_pose().to_transformation_matrix()) @ pick_drawer_pull_point(drawer, force_dir)
            application_point_strategy = "center of selected link face along force direction"
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
    camera.set_entity_pose(
        look_at_pose(
            np.array([-1.45, -1.55, 0.86], dtype=np.float32),
            np.array([0.0, -0.04, 0.06], dtype=np.float32),
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
    return (sim.camera.get_picture("Color")[..., :3].clip(0, 1) * 255).astype(np.uint8)


def draw_text(img: np.ndarray, text: str, xy: tuple[int, int], color: tuple[int, int, int], font=FONT_REGULAR) -> None:
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    draw.text(xy, text, fill=color[::-1], font=font)
    img[:] = np.asarray(pil)


def draw_label(img: np.ndarray, text: str, xy: tuple[int, int], color: tuple[int, int, int], scale: float = 0.75) -> None:
    font = FONT_SMALL if scale < 0.6 else FONT_REGULAR
    x, y = xy
    x = min(max(10, x), img.shape[1] - 10)
    y = min(max(24, y), img.shape[0] - 12)
    draw_text(img, text, (x + 2, y + 2), (255, 255, 255), font)
    draw_text(img, text, (x, y), color, font)


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


def draw_info_card(canvas: np.ndarray, x: int, y: int, w: int, h: int, title: str, force_text: str, disp: float, color: tuple[int, int, int]) -> None:
    cv2.rectangle(canvas, (x + 18, y + 12), (x + w - 18, y + h - 10), (248, 248, 248), -1, cv2.LINE_AA)
    cv2.rectangle(canvas, (x + 18, y + 12), (x + w - 18, y + h - 10), (205, 205, 205), 1, cv2.LINE_AA)
    cv2.circle(canvas, (x + 42, y + 42), 10, color, -1, cv2.LINE_AA)
    draw_text(canvas, title, (x + 62, y + 29), (24, 24, 24), FONT_BOLD)
    draw_text(canvas, force_text, (x + 34, y + 70), color, FONT_REGULAR)
    draw_text(canvas, f"spostamento: {disp:5.3f} m", (x + w - 286, y + 70), (24, 24, 24), FONT_REGULAR)


def draw_displacement_plot(canvas: np.ndarray, no_force: list[float], pulling: list[float], x: int, y: int, w: int, h: int) -> None:
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (245, 245, 245), -1)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (180, 180, 180), 1)
    draw_label(canvas, "spostamento nel tempo", (x + 14, y + 28), (20, 20, 20), 0.55)

    values = [0.0, *no_force, *pulling]
    vmin = min(values) - 0.02
    vmax = max(values) + 0.02
    if abs(vmax - vmin) < 1e-4:
        vmax += 0.1

    def to_px(i: int, value: float, n: int) -> tuple[int, int]:
        px = x + 42 + int((w - 60) * (i / max(1, n - 1)))
        py = y + h - 24 - int((h - 58) * ((value - vmin) / (vmax - vmin)))
        return px, py

    for values_line, color in ((no_force, (95, 95, 95)), (pulling, (210, 55, 45))):
        pts = [to_px(i, value, len(values_line)) for i, value in enumerate(values_line)]
        if len(pts) > 1:
            cv2.polylines(canvas, [np.array(pts, dtype=np.int32)], False, color, 2, cv2.LINE_AA)

    draw_label(canvas, "senza forza", (x + w - 168, y + 30), (95, 95, 95), 0.48)
    draw_label(canvas, "tiro", (x + w - 168, y + 54), (210, 55, 45), 0.48)


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
        "limits_m": optional_array(joint.get_limit),
        "friction": optional_float(lambda: joint.friction),
        "damping": optional_float(lambda: joint.damping),
        "drive_mode": optional_string(lambda: joint.drive_mode),
        "drive_target": optional_array(lambda: joint.drive_target),
        "drive_velocity_target": optional_array(lambda: joint.drive_velocity_target),
        "force_limit": optional_float(lambda: joint.force_limit),
    }


def sample_to_dict(time_s: float, sim: DrawerSim, applied_force: np.ndarray) -> dict[str, object]:
    return {
        "time_s": float(time_s),
        "drawer_displacement_m": float(sim.cabinet.get_qpos()[sim.joint_index]),
        "drawer_velocity_m_s": float(sim.cabinet.get_qvel()[sim.joint_index]),
        "application_point_world": application_point_world(sim).astype(float).tolist(),
        "applied_force_world": applied_force.astype(float).tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="44817")
    parser.add_argument("--output", default=None)
    parser.add_argument("--json-output", default=None)
    parser.add_argument("--drawer", type=int, default=1)
    parser.add_argument("--force", type=float, default=0.5)
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--end-hold-seconds", type=float, default=2.0, help="Freeze the last frame for this many seconds")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--panel-width", type=int, default=720)
    parser.add_argument("--panel-height", type=int, default=448)
    parser.add_argument("--info-height", type=int, default=132)
    parser.add_argument("--plot-height", type=int, default=176)
    parser.add_argument("--direction", nargs=3, type=float, default=[0.0, 0.0, 1.0])
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--keep-old", action="store_true", help="Do not delete old files in the output directory")
    args = parser.parse_args()

    model_dir = Path(args.model_dir).resolve()
    output, json_output = output_paths(model_dir, Path(args.output_root).resolve(), args.output, args.json_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not args.keep_old:
        clear_object_output(output)

    force_dir = np.array(args.direction, dtype=np.float32)
    force_dir /= np.linalg.norm(force_dir) or 1.0
    slider_axis = int(np.argmax(np.abs(force_dir)))
    generalized_force = args.force if force_dir[slider_axis] >= 0 else -args.force

    override_point, override_strategy = load_application_point_override(output.parent, f"link_{args.drawer}")
    still_sim = setup_sim(model_dir, args.drawer, args.panel_width, args.panel_height, force_dir, override_point, override_strategy)
    pulling_sim = setup_sim(model_dir, args.drawer, args.panel_width, args.panel_height, force_dir, override_point, override_strategy)
    pull_dir_world = pulling_sim.positive_pull_dir_world if generalized_force >= 0 else -pulling_sim.positive_pull_dir_world
    force = pull_dir_world * args.force

    steps_per_frame = max(1, round(240 / args.fps))
    frame_count = max(1, int(args.seconds * args.fps))
    still_displacements: list[float] = []
    pulling_displacements: list[float] = []
    samples = {"no_force": [], "pulling_force": []}
    point_histories = {"no_force": [], "pulling_force": []}

    out_w = args.panel_width * 2
    out_h = args.panel_height + args.info_height + args.plot_height

    final_frame = None
    with imageio.get_writer(output, fps=args.fps, codec="libx264", quality=8) as writer:
        for _ in range(frame_count):
            for _ in range(steps_per_frame):
                qf = np.zeros_like(pulling_sim.cabinet.get_qf(), dtype=np.float32)
                qf[pulling_sim.joint_index] = generalized_force
                pulling_sim.cabinet.set_qf(qf)
                still_sim.scene.step()
                pulling_sim.scene.step()

            time_s = len(pulling_displacements) / args.fps
            samples["no_force"].append(sample_to_dict(time_s, still_sim, np.zeros(3, dtype=np.float32)))
            samples["pulling_force"].append(sample_to_dict(time_s, pulling_sim, force))
            point_histories["no_force"].append(application_point_world(still_sim))
            point_histories["pulling_force"].append(application_point_world(pulling_sim))

            left = render_panel(still_sim)
            right = render_panel(pulling_sim)
            canvas = np.full((out_h, out_w, 3), 232, dtype=np.uint8)

            draw_force_annotation(left, still_sim, pull_dir_world, 0.0, (95, 95, 95), point_histories["no_force"])
            draw_force_annotation(right, pulling_sim, pull_dir_world, args.force, (210, 55, 45), point_histories["pulling_force"])

            canvas[: args.panel_height, : args.panel_width] = left
            canvas[: args.panel_height, args.panel_width :] = right
            cv2.line(canvas, (args.panel_width, 0), (args.panel_width, args.panel_height + args.info_height), (205, 205, 205), 2)

            still_disp = float(still_sim.cabinet.get_qpos()[still_sim.joint_index])
            pulling_disp = float(pulling_sim.cabinet.get_qpos()[pulling_sim.joint_index])
            still_displacements.append(still_disp)
            pulling_displacements.append(pulling_disp)

            draw_info_card(canvas, 0, args.panel_height, args.panel_width, args.info_height, "senza forza", "F = 0 N", still_disp, (95, 95, 95))
            draw_info_card(canvas, args.panel_width, args.panel_height, args.panel_width, args.info_height, "trazione cassetto", f"F = {args.force:g} N", pulling_disp, (210, 55, 45))
            draw_displacement_plot(canvas, still_displacements, pulling_displacements, 22, args.panel_height + args.info_height + 16, out_w - 44, args.plot_height - 32)
            writer.append_data(canvas)
            final_frame = canvas.copy()

        hold_frames = int(round(args.end_hold_seconds * args.fps))
        if final_frame is not None:
            for _ in range(hold_frames):
                writer.append_data(final_frame)

    urdf_dynamics = urdf_joint_dynamics(model_dir)
    metadata = {
        "model_dir": str(model_dir),
        "video_output": str(output),
        "fps": args.fps,
        "seconds": args.seconds,
        "end_hold_seconds": args.end_hold_seconds,
        "video_duration_seconds": args.seconds + args.end_hold_seconds,
        "drawer": args.drawer,
        "joint": f"joint_{args.drawer}",
        "link": f"link_{args.drawer}",
        "force_magnitude_n": args.force,
        "direction_world": pull_dir_world.astype(float).tolist(),
        "generalized_drawer_force_n": float(generalized_force),
        "application_point_strategy": pulling_sim.application_point_strategy,
        "application_point_local_on_drawer": pulling_sim.local_application_point[:3].astype(float).tolist(),
        "joint_limits_m": pulling_sim.cabinet.get_active_joints()[pulling_sim.joint_index].get_limit().tolist(),
        "timestep_s": 1.0 / 240.0,
        "physics": {
            "urdf_joint_dynamics": urdf_dynamics,
            "urdf_joint_dynamics_present": bool(urdf_dynamics),
            "separate_static_dynamic_friction_present": False,
            "air_friction_model_present": False,
            "link_linear_damping_set_to": 0.0,
            "link_angular_damping_set_to": 0.02,
            "joint_drive_stiffness_damping_force_limit_set_to": [0.0, 0.0, 0.0],
        },
        "links": [link_dynamics_to_dict(link) for link in pulling_sim.cabinet.get_links()],
        "joints": [joint_to_dict(joint) for joint in pulling_sim.cabinet.get_joints()],
    }
    with json_output.open("w", encoding="utf-8") as f:
        json.dump({"metadata": metadata, "samples": samples}, f, indent=2)

    print(f"Wrote {output}")
    print(f"Wrote {json_output}")
    print(f"No-force final displacement: {still_displacements[-1]:.4f} m")
    print(f"Pulling-force final displacement: {pulling_displacements[-1]:.4f} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
