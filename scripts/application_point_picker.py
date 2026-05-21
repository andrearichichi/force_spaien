#!/usr/bin/env python3
"""Create/select manual application-point candidates for an articulated object."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    import sapien
except ModuleNotFoundError:
    sapien = None

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


def load_font(size: int) -> ImageFont.ImageFont:
    for font_path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
    ):
        try:
            return ImageFont.truetype(font_path, size)
        except OSError:
            continue
    return ImageFont.load_default()


FONT = load_font(24)
SELECTION_PREFIX = "CONTACT_POINT_SELECTION_JSON="


def require_sapien() -> None:
    if sapien is None:
        raise ModuleNotFoundError(
            "sapien is required to generate a new preview. Reuse an existing preview or install sapien."
        )


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


def first_moving_joint(model_dir: Path) -> tuple[str, str, str, tuple[float, float] | None]:
    root = ET.parse(model_dir / "mobility.urdf").getroot()
    for joint in root.findall("joint"):
        joint_type = joint.attrib.get("type", "")
        if joint_type == "fixed":
            continue
        child = joint.find("child")
        limit = joint.find("limit")
        if child is None:
            continue
        limits = None
        if limit is not None and "lower" in limit.attrib and "upper" in limit.attrib:
            limits = (float(limit.attrib["lower"]), float(limit.attrib["upper"]))
        return joint_type, joint.attrib.get("name", ""), child.attrib.get("link", ""), limits
    raise RuntimeError(f"No moving joint found in {model_dir / 'mobility.urdf'}")


def preferred_joint(model_dir: Path, detected_joint: str, detected_link: str) -> tuple[str, str]:
    if model_dir.name == "44817":
        return "joint_1", "link_1"
    return detected_joint, detected_link


def default_initial_angle(model_dir: Path, limits: tuple[float, float] | None) -> float:
    if model_dir.name == "11691":
        return -1.5
    if limits is None:
        return 0.0
    lower, upper = limits
    return 0.0 if lower <= 0.0 <= upper else lower


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


def link_visual_vertices(model_dir: Path, link_name: str) -> np.ndarray:
    root = ET.parse(model_dir / "mobility.urdf").getroot()
    link = root.find(f".//link[@name='{link_name}']")
    if link is None:
        raise RuntimeError(f"Could not find {link_name}.")

    chunks: list[np.ndarray] = []
    for visual in link.findall("visual"):
        mesh = visual.find("./geometry/mesh")
        if mesh is not None and "filename" in mesh.attrib:
            chunks.append(mesh_vertices(model_dir / mesh.attrib["filename"]) + visual_origin(visual))
    if not chunks:
        raise RuntimeError(f"No visual mesh found for {link_name}.")
    return np.concatenate(chunks, axis=0)


def candidate_points(vertices: np.ndarray) -> list[dict[str, object]]:
    vmin = vertices.min(axis=0)
    vmax = vertices.max(axis=0)
    center = 0.5 * (vmin + vmax)
    raw = [
        ("center", center),
        ("x_min_face", np.array([vmin[0], center[1], center[2]], dtype=np.float32)),
        ("x_max_face", np.array([vmax[0], center[1], center[2]], dtype=np.float32)),
        ("y_min_face", np.array([center[0], vmin[1], center[2]], dtype=np.float32)),
        ("y_max_face", np.array([center[0], vmax[1], center[2]], dtype=np.float32)),
        ("z_min_face", np.array([center[0], center[1], vmin[2]], dtype=np.float32)),
        ("z_max_face", np.array([center[0], center[1], vmax[2]], dtype=np.float32)),
    ]
    for x in (vmin[0], vmax[0]):
        for y in (vmin[1], vmax[1]):
            for z in (vmin[2], vmax[2]):
                raw.append(("corner", np.array([x, y, z], dtype=np.float32)))

    raw.extend(
        [
            ("xy_min_min_edge", np.array([vmin[0], vmin[1], center[2]], dtype=np.float32)),
            ("xy_min_max_edge", np.array([vmin[0], vmax[1], center[2]], dtype=np.float32)),
            ("xy_max_min_edge", np.array([vmax[0], vmin[1], center[2]], dtype=np.float32)),
            ("xy_max_max_edge", np.array([vmax[0], vmax[1], center[2]], dtype=np.float32)),
            ("xz_min_min_edge", np.array([vmin[0], center[1], vmin[2]], dtype=np.float32)),
            ("xz_min_max_edge", np.array([vmin[0], center[1], vmax[2]], dtype=np.float32)),
            ("xz_max_min_edge", np.array([vmax[0], center[1], vmin[2]], dtype=np.float32)),
            ("xz_max_max_edge", np.array([vmax[0], center[1], vmax[2]], dtype=np.float32)),
            ("yz_min_min_edge", np.array([center[0], vmin[1], vmin[2]], dtype=np.float32)),
            ("yz_min_max_edge", np.array([center[0], vmin[1], vmax[2]], dtype=np.float32)),
            ("yz_max_min_edge", np.array([center[0], vmax[1], vmin[2]], dtype=np.float32)),
            ("yz_max_max_edge", np.array([center[0], vmax[1], vmax[2]], dtype=np.float32)),
        ]
    )

    return [{"id": index, "name": name, "local_point": point.astype(float).tolist()} for index, (name, point) in enumerate(raw)]


def project(camera: sapien.render.RenderCameraComponent, point: np.ndarray) -> tuple[int, int] | None:
    camera_point = camera.get_extrinsic_matrix() @ np.array([point[0], point[1], point[2], 1.0], dtype=np.float32)
    if camera_point[2] <= 0:
        return None
    uvw = camera.get_intrinsic_matrix() @ camera_point
    return int(round(uvw[0] / uvw[2])), int(round(uvw[1] / uvw[2]))


def draw_candidates(image: np.ndarray, projected: list[dict[str, object]]) -> np.ndarray:
    out = image.copy()
    pil = Image.fromarray(out)
    draw = ImageDraw.Draw(pil)
    for item in projected:
        uv = item.get("pixel")
        if uv is None:
            continue
        x, y = uv
        color = (30, 60, 220)
        cv2.circle(out, (x, y), 16, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(out, (x, y), 14, color, 3, cv2.LINE_AA)
        pil = Image.fromarray(out)
        draw = ImageDraw.Draw(pil)
        draw.text((x + 18, y - 17), str(item["id"]), fill=(255, 255, 255), font=FONT, stroke_width=3, stroke_fill=(20, 20, 20))
        out = np.asarray(pil).copy()
    return out


def object_output_dir(model_dir: Path, output_root: Path) -> Path:
    path = output_root / f"{model_dir.name}_output"
    path.mkdir(parents=True, exist_ok=True)
    return path


def fallback_preview_path(model_dir: Path, output_root: Path) -> Path:
    return object_output_dir(model_dir, output_root) / "contact_point_preview.png"


def save_preview_image(model_dir: Path, output_root: Path, preview: np.ndarray) -> Path:
    path = fallback_preview_path(model_dir, output_root)
    Image.fromarray(preview).save(path)
    return path


def can_open_window() -> bool:
    if sys.platform.startswith("linux"):
        return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return True


def build_preview_data(args: argparse.Namespace) -> tuple[dict[str, object], np.ndarray]:
    require_sapien()
    model_dir = resolve_model_dir(args.model_dir)
    detected_type, detected_joint, detected_link, limits = first_moving_joint(model_dir)
    joint_name = args.joint or preferred_joint(model_dir, detected_joint, detected_link)[0]
    link_name = args.link or preferred_joint(model_dir, detected_joint, detected_link)[1]

    scene = sapien.Scene()
    scene.set_timestep(1.0 / 240.0)
    scene.set_ambient_light([0.72, 0.72, 0.72])
    scene.add_directional_light([0.2, -0.45, -1.0], [1.0, 1.0, 1.0], shadow=False)
    scene.add_directional_light([-0.7, 0.25, -1.0], [0.38, 0.38, 0.38], shadow=False)
    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    articulation = loader.load(str(model_dir / "mobility.urdf"))

    joint = articulation.find_joint_by_name(joint_name)
    link = articulation.find_link_by_name(link_name)
    if joint is None or link is None:
        raise RuntimeError(f"Could not find {joint_name}/{link_name}.")

    active_joints = list(articulation.get_active_joints())
    if joint in active_joints:
        qpos = np.zeros_like(articulation.get_qpos(), dtype=np.float32)
        qpos[active_joints.index(joint)] = args.initial_angle if args.initial_angle is not None else default_initial_angle(model_dir, limits)
        articulation.set_qpos(qpos)

    camera = scene.add_camera("camera", args.width, args.height, math.radians(44), 0.01, 20.0)
    camera.set_entity_pose(
        look_at_pose(
            np.array(args.camera_eye, dtype=np.float32),
            np.array(args.camera_target, dtype=np.float32),
        )
    )

    scene.update_render()
    camera.take_picture()
    image = (camera.get_picture("Color")[..., :3].clip(0, 1) * 255).astype(np.uint8)

    local_candidates = candidate_points(link_visual_vertices(model_dir, link_name))
    transform = link.get_entity_pose().to_transformation_matrix()
    projected = []
    for candidate in local_candidates:
        local = np.array([*candidate["local_point"], 1.0], dtype=np.float32)
        world = (transform @ local)[:3]
        item = dict(candidate)
        item["world_point"] = world.astype(float).tolist()
        item["pixel"] = project(camera, world)
        projected.append(item)

    preview = draw_candidates(image, projected)
    return (
        {
            "model_dir": str(model_dir),
            "joint_type": detected_type,
            "joint": joint_name,
            "link": link_name,
            "candidates": projected,
        },
        preview,
    )


def print_candidate_summary(data: dict[str, object]) -> None:
    visible_candidates = [item for item in data["candidates"] if item.get("pixel") is not None]
    print(f"Visible candidates for {data['link']}:")
    for item in visible_candidates:
        x, y = item["pixel"]
        print(f"  {item['id']:>2}  {item['name']:<18} pixel=({x}, {y})")


def emit_selected_candidate(data: dict[str, object], candidate: dict[str, object]) -> None:
    payload = {
        "joint": data["joint"],
        "link": data["link"],
        "candidate_id": candidate["id"],
        "candidate_name": candidate["name"],
        "local_point": candidate["local_point"],
    }
    print(f"{SELECTION_PREFIX}{json.dumps(payload)}")


def prompt_candidate_selection(data: dict[str, object], preview_path: Path) -> dict[str, object] | None:
    if not sys.stdin.isatty():
        raise RuntimeError(
            "Picker fallback requires an interactive terminal. "
            f"Open {preview_path} and rerun with --select-point ID if needed."
        )

    visible_candidates = [item for item in data["candidates"] if item.get("pixel") is not None]
    candidate_by_id = {int(item["id"]): item for item in visible_candidates}
    if not candidate_by_id:
        raise RuntimeError("No visible candidates available for terminal selection.")

    print_candidate_summary(data)
    print(f"Preview saved to {preview_path}")
    print("Open the image, inspect the marker ids, then type the candidate number.")

    while True:
        response = input("Candidate id (or 'cancel'): ").strip().lower()
        if response in {"cancel", "c", "q", "quit", "exit"}:
            print("Selection cancelled.")
            return None
        try:
            candidate_id = int(response)
        except ValueError:
            print("Enter a numeric candidate id or 'cancel'.")
            continue
        candidate = candidate_by_id.get(candidate_id)
        if candidate is None:
            print(f"Candidate {candidate_id} is not visible in the saved preview.")
            continue
        return candidate


def create_preview(args: argparse.Namespace) -> int:
    data, preview = build_preview_data(args)
    print_candidate_summary(data)

    preview_bgr = cv2.cvtColor(preview, cv2.COLOR_RGB2BGR)
    window = "Application point candidates: press any key to close"
    if not can_open_window():
        preview_path = save_preview_image(resolve_model_dir(args.model_dir), Path(args.output_root).resolve(), preview)
        print(f"Could not open a GUI window here. Preview saved to {preview_path}")
        return 0

    try:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.imshow(window, preview_bgr)
        cv2.waitKey(0)
        cv2.destroyWindow(window)
    except cv2.error:
        preview_path = save_preview_image(resolve_model_dir(args.model_dir), Path(args.output_root).resolve(), preview)
        print(f"Could not open a GUI window here. Preview saved to {preview_path}")

    return 0


def select_candidate(args: argparse.Namespace) -> int:
    data, _preview = build_preview_data(args)
    candidate = next((item for item in data["candidates"] if int(item["id"]) == args.select_point), None)
    if candidate is None:
        raise RuntimeError(f"Candidate id {args.select_point} not found for {data['link']}")

    emit_selected_candidate(data, candidate)
    print(f"Selected candidate {candidate['id']}: {candidate['name']}")
    return 0


def pick_interactively(args: argparse.Namespace) -> int:
    model_dir = resolve_model_dir(args.model_dir)
    output_root = Path(args.output_root).resolve()
    data, preview = build_preview_data(args)
    image = cv2.cvtColor(preview, cv2.COLOR_RGB2BGR)

    visible_candidates = [item for item in data["candidates"] if item.get("pixel") is not None]
    if not visible_candidates:
        raise RuntimeError("No visible candidates available for interactive picking.")

    if not can_open_window():
        preview_path = save_preview_image(model_dir, output_root, preview)
        selected = prompt_candidate_selection(data, preview_path)
        if selected is None:
            return 1
        emit_selected_candidate(data, selected)
        print(f"Selected candidate {selected['id']}: {selected['name']}")
        return 0

    selected: dict[str, object] = {}
    window = "Pick application point: click marker, Enter confirms, Esc cancels"

    def nearest_candidate(x: int, y: int) -> dict[str, object]:
        return min(visible_candidates, key=lambda item: (item["pixel"][0] - x) ** 2 + (item["pixel"][1] - y) ** 2)

    def redraw(highlight: dict[str, object] | None = None) -> None:
        canvas = image.copy()
        if highlight is not None:
            x, y = highlight["pixel"]
            cv2.circle(canvas, (x, y), 28, (0, 255, 255), 4, cv2.LINE_AA)
            cv2.putText(canvas, f"selected {highlight['id']}", (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(canvas, f"selected {highlight['id']}", (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(window, canvas)

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            selected.clear()
            selected.update(nearest_candidate(x, y))
            redraw(selected)

    try:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window, on_mouse)
        redraw()

        print("Click a marker in the preview window, then press Enter. Press Esc to cancel.")
        while True:
            key = cv2.waitKey(50)
            if key in (13, 10):
                if selected:
                    break
                print("No point selected yet.")
            if key == 27:
                cv2.destroyWindow(window)
                print("Selection cancelled.")
                return 1

        cv2.destroyWindow(window)
        emit_selected_candidate(data, selected)
        print(f"Selected candidate {selected['id']}: {selected['name']}")
        return 0
    except cv2.error:
        preview_path = save_preview_image(model_dir, output_root, preview)
        selected = prompt_candidate_selection(data, preview_path)
        if selected is None:
            return 1
        emit_selected_candidate(data, selected)
        print(f"Selected candidate {selected['id']}: {selected['name']}")
        return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--joint", default=None)
    parser.add_argument("--link", default=None)
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--preview-points", action="store_true")
    parser.add_argument("--pick-point", action="store_true")
    parser.add_argument("--select-point", type=int, default=None)
    parser.add_argument("--initial-angle", type=float, default=None)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--camera-eye", nargs=3, type=float, default=[-1.45, -1.55, 0.86])
    parser.add_argument("--camera-target", nargs=3, type=float, default=[0.0, -0.04, 0.06])
    args = parser.parse_args()

    if args.preview_points:
        return create_preview(args)
    if args.pick_point:
        return pick_interactively(args)
    if args.select_point is not None:
        return select_candidate(args)
    parser.error("Use --preview-points, --pick-point, or --select-point ID")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
