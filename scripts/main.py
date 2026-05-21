#!/usr/bin/env python3
"""Dispatch an object simulation to the prismatic or revolute pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

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

        joint_name = joint.attrib.get("name", "")
        link_name = child.attrib.get("link", "")
        joint_limit = None
        if limit is not None and "lower" in limit.attrib and "upper" in limit.attrib:
            joint_limit = (float(limit.attrib["lower"]), float(limit.attrib["upper"]))

        return joint_type, joint_name, link_name, joint_limit

    raise RuntimeError(f"No moving joint found in {model_dir / 'mobility.urdf'}")


def default_initial_angle(model_dir: Path, limits: tuple[float, float] | None) -> float:
    if model_dir.name == "11691":
        return -1.5
    if limits is None:
        return 0.0
    lower, upper = limits
    if lower <= 0.0 <= upper:
        return 0.0
    return lower


def drawer_index_from_link(link_name: str) -> str:
    if not link_name.startswith("link_"):
        raise RuntimeError(f"Cannot infer prismatic drawer index from link name: {link_name}")
    return link_name.removeprefix("link_")


def default_direction(model_dir: Path, joint_type: str) -> list[float]:
    if model_dir.name == "45384" and joint_type == "revolute":
        return [-1.0, 0.0, 0.0]
    return [0.0, 0.0, 1.0]


def preferred_joint(model_dir: Path, detected_joint: str, detected_link: str) -> tuple[str, str]:
    if model_dir.name == "44817":
        return "joint_1", "link_1"
    return detected_joint, detected_link


def mesh_has_vertices(mesh_path: Path) -> bool:
    if not mesh_path.exists():
        return False

    with mesh_path.open("r", encoding="utf-8", errors="ignore") as mesh_file:
        for line in mesh_file:
            if line.startswith("v "):
                return True
    return False


def has_valid_handle(model_dir: Path, link_name: str) -> bool:
    root = ET.parse(model_dir / "mobility.urdf").getroot()
    link = root.find(f".//link[@name='{link_name}']")
    if link is None:
        raise RuntimeError(f"Could not find {link_name} in {model_dir / 'mobility.urdf'}")

    for visual in link.findall("visual"):
        if "handle" not in visual.attrib.get("name", "").lower():
            continue

        mesh = visual.find("./geometry/mesh")
        filename = mesh.attrib.get("filename") if mesh is not None else None
        if filename and mesh_has_vertices(model_dir / filename):
            return True

    return False


def contact_point_source(model_dir: Path, link_name: str) -> tuple[str, str]:
    if has_valid_handle(model_dir, link_name):
        return "handle", f"handle on {link_name}"
    return "none", f"no valid handle on {link_name}"


def build_picker_command(
    model_dir: Path,
    joint_type: str,
    joint_name: str,
    link_name: str,
    limits: tuple[float, float] | None,
    args: argparse.Namespace,
    scripts_dir: Path,
    *,
    mode: str,
) -> list[str]:
    command = [
        sys.executable,
        str(scripts_dir / "application_point_picker.py"),
        "--model-dir",
        str(model_dir),
        "--joint",
        joint_name,
        "--link",
        link_name,
        "--output-root",
        args.output_root,
    ]

    if args.initial_angle is not None:
        command += ["--initial-angle", str(args.initial_angle)]
    elif joint_type == "revolute":
        command += ["--initial-angle", str(default_initial_angle(model_dir, limits))]

    if mode == "preview":
        command.append("--preview-points")
    elif mode == "pick":
        command.append("--pick-point")
    elif mode == "select":
        command += ["--select-point", str(args.select_point)]
    else:
        raise RuntimeError(f"Unsupported picker mode: {mode}")

    return command


def run_picker(
    model_dir: Path,
    joint_type: str,
    joint_name: str,
    link_name: str,
    limits: tuple[float, float] | None,
    args: argparse.Namespace,
    scripts_dir: Path,
    *,
    mode: str,
) -> tuple[int, dict[str, object] | None]:
    command = build_picker_command(model_dir, joint_type, joint_name, link_name, limits, args, scripts_dir, mode=mode)
    print(f"Running picker: {' '.join(command)}")
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    selection = None
    for line in completed.stdout.splitlines():
        if line.startswith("CONTACT_POINT_SELECTION_JSON="):
            selection = json.loads(line.split("=", 1)[1])
        else:
            print(line)
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    if mode in {"pick", "select"} and completed.returncode == 0 and selection is None:
        raise RuntimeError("Picker completed without returning a selected contact point.")
    return completed.returncode, selection


def set_selected_contact_point(args: argparse.Namespace, selection: dict[str, object] | None) -> None:
    if selection is None:
        return
    args.contact_point_local = list(selection["local_point"])
    candidate_id = selection.get("candidate_id")
    candidate_name = selection.get("candidate_name", "candidate")
    args.contact_point_strategy = f"manual candidate {candidate_id}: {candidate_name} from interactive picker"


def ensure_contact_point(
    model_dir: Path,
    joint_type: str,
    joint_name: str,
    link_name: str,
    limits: tuple[float, float] | None,
    args: argparse.Namespace,
    scripts_dir: Path,
) -> int:
    source_kind, source_description = contact_point_source(model_dir, link_name)

    if args.contact_point_mode == "manual":
        print("Contact point mode is manual; opening the interactive point picker.")
        picker_exit_code, selection = run_picker(model_dir, joint_type, joint_name, link_name, limits, args, scripts_dir, mode="pick")
        set_selected_contact_point(args, selection)
        return picker_exit_code

    if args.contact_point_mode == "auto":
        if source_kind != "none":
            print(f"Using {source_description}.")
            return 0
        print(f"Found {source_description}; opening the interactive point picker.")
        picker_exit_code, selection = run_picker(model_dir, joint_type, joint_name, link_name, limits, args, scripts_dir, mode="pick")
        set_selected_contact_point(args, selection)
        return picker_exit_code

    if not sys.stdin.isatty():
        raise RuntimeError(
            "Contact-point confirmation requires an interactive terminal. "
            "Use --contact-point-mode auto to use the detected handle, "
            "or --contact-point-mode manual to force the picker."
        )

    if source_kind == "none":
        print(f"Found {source_description}; opening the interactive point picker.")
        picker_exit_code, selection = run_picker(model_dir, joint_type, joint_name, link_name, limits, args, scripts_dir, mode="pick")
        set_selected_contact_point(args, selection)
        return picker_exit_code

    while True:
        print(f"Contact point candidate: {source_description}.")
        response = input(
            "Press Enter to use it, type 'preview' to inspect candidates, "
            "'pick' to choose manually, or 'cancel' to stop: "
        ).strip().lower()
        if response in {"", "y", "yes", "use"}:
            print(f"Using {source_description}.")
            return 0
        if response in {"preview", "v"}:
            preview_exit_code, _selection = run_picker(
                model_dir,
                joint_type,
                joint_name,
                link_name,
                limits,
                args,
                scripts_dir,
                mode="preview",
            )
            if preview_exit_code != 0:
                return preview_exit_code
            continue
        if response in {"pick", "p", "manual", "edit", "change"}:
            picker_exit_code, selection = run_picker(model_dir, joint_type, joint_name, link_name, limits, args, scripts_dir, mode="pick")
            set_selected_contact_point(args, selection)
            return picker_exit_code
        if response in {"cancel", "c", "stop", "skip", "n", "no"}:
            print("Cancelled before simulation.")
            return 1
        print("Unrecognized answer. Use Enter, preview, pick, or cancel.")


def run_object(model_dir_arg: str, args: argparse.Namespace, scripts_dir: Path) -> int:
    args.contact_point_local = None
    args.contact_point_strategy = None
    model_dir = resolve_model_dir(model_dir_arg)
    if not (model_dir / "mobility.urdf").exists():
        raise FileNotFoundError(model_dir / "mobility.urdf")

    detected_type, detected_joint, detected_link, limits = first_moving_joint(model_dir)
    joint_type = detected_type if args.joint_type == "auto" else args.joint_type
    preferred_joint_name, preferred_link_name = preferred_joint(model_dir, detected_joint, detected_link)
    joint_name = args.joint or preferred_joint_name
    link_name = args.link or preferred_link_name
    direction = args.direction or default_direction(model_dir, joint_type)

    if args.preview_points or args.pick_point or args.select_point is not None:
        print(f"Detected {detected_type}: {detected_joint}/{detected_link}")
        print(f"Selected {joint_type}: {joint_name}/{link_name}")
        picker_mode = "preview" if args.preview_points else "pick" if args.pick_point else "select"
        picker_exit_code, selection = run_picker(model_dir, joint_type, joint_name, link_name, limits, args, scripts_dir, mode=picker_mode)
        set_selected_contact_point(args, selection)
        if picker_exit_code != 0 or args.preview_points:
            return picker_exit_code
    else:
        print(f"Detected {detected_type}: {detected_joint}/{detected_link}")
        print(f"Selected {joint_type}: {joint_name}/{link_name}")
        contact_point_exit_code = ensure_contact_point(model_dir, joint_type, joint_name, link_name, limits, args, scripts_dir)
        if contact_point_exit_code != 0:
            return contact_point_exit_code

    command = [sys.executable]
    if joint_type == "prismatic":
        script = scripts_dir / "render_prismatic_video.py"
        command += [
            str(script),
            "--mode",
            args.mode,
            "--model-dir",
            str(model_dir),
        ]
        if args.mode == "render":
            command += ["--drawer", drawer_index_from_link(link_name)]
        else:
            command += ["--joint", joint_name, "--link", link_name]
        command += [
            "--force",
            str(args.force),
            "--seconds",
            str(args.seconds),
            "--fps",
            str(args.fps),
            "--direction",
            str(direction[0]),
            str(direction[1]),
            str(direction[2]),
            "--output-root",
            args.output_root,
        ]
        if args.contact_point_local is not None:
            command += ["--contact-point-local", *(str(value) for value in args.contact_point_local)]
        if args.contact_point_strategy is not None:
            command += ["--contact-point-strategy", args.contact_point_strategy]
    elif joint_type == "revolute":
        script = scripts_dir / "render_revolute_video.py"
        command += [
            str(script),
            "--mode",
            args.mode,
            "--model-dir",
            str(model_dir),
            "--joint",
            joint_name,
            "--link",
            link_name,
            "--force",
            str(args.force),
            "--seconds",
            str(args.seconds),
            "--fps",
            str(args.fps),
            "--direction",
            str(direction[0]),
            str(direction[1]),
            str(direction[2]),
            "--initial-angle",
            str(args.initial_angle if args.initial_angle is not None else default_initial_angle(model_dir, limits)),
            "--output-root",
            args.output_root,
        ]
        if args.contact_point_local is not None:
            command += ["--contact-point-local", *(str(value) for value in args.contact_point_local)]
        if args.contact_point_strategy is not None:
            command += ["--contact-point-strategy", args.contact_point_strategy]
        if args.mode == "render":
            command += ["--closing-force", str(args.force)]
    else:
        raise RuntimeError(f"Unsupported joint type: {joint_type}")

    if args.mode == "render":
        command += ["--end-hold-seconds", str(args.end_hold_seconds)]
    if args.keep_old:
        command.append("--keep-old")

    print(f"Detected {detected_type}: {detected_joint}/{detected_link}")
    print(f"Selected {joint_type}: {joint_name}/{link_name}")
    print(f"Running {joint_type} {args.mode}: {' '.join(command)}")
    return subprocess.run(command, check=False).returncode


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the force/render pipeline on one or more objects.",
        epilog=(
            "Examples:\n"
            "  python3 scripts/main.py 101062\n"
            "  python3 scripts/main.py 11691 44817 45384 101062\n"
            "  python3 scripts/main.py dataset/101062\n"
            "  python3 scripts/main.py 101062 --preview-points\n"
            "  python3 scripts/main.py 101062 --pick-point\n"
            "  python3 scripts/main.py 101062 --select-point 6\n"
            "  python3 scripts/main.py 101062 --contact-point-mode auto"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("objects", nargs="*", help="Object IDs from dataset/ or object paths, e.g. 11691 44817 45384")
    parser.add_argument("--model-dir", default=None, help="Backward-compatible single object path")
    parser.add_argument("--mode", choices=["render", "apply"], default="render")
    parser.add_argument("--joint-type", choices=["auto", "prismatic", "revolute"], default="auto")
    parser.add_argument("--joint", default=None)
    parser.add_argument("--link", default=None)
    parser.add_argument("--force", type=float, default=0.5)
    parser.add_argument("--direction", nargs=3, type=float, default=None)
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--end-hold-seconds", type=float, default=2.0)
    parser.add_argument("--initial-angle", type=float, default=None)
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument(
        "--contact-point-mode",
        choices=["confirm", "manual", "auto"],
        default="confirm",
        help="confirm (default): ask before using a detected handle; manual: always open the picker; auto: use the handle if present, otherwise open the picker",
    )
    parser.add_argument("--preview-points", action="store_true")
    parser.add_argument("--pick-point", action="store_true")
    parser.add_argument("--select-point", type=int, default=None)
    parser.add_argument("--keep-old", action="store_true")
    args = parser.parse_args()

    objects = args.objects
    if args.model_dir is not None:
        objects.append(args.model_dir)
    if not objects:
        parser.error("pass at least one object ID or directory, e.g. python3 scripts/main.py 101062")

    scripts_dir = Path(__file__).resolve().parent
    exit_code = 0
    for index, model_dir in enumerate(objects, start=1):
        if len(objects) > 1:
            print(f"\n[{index}/{len(objects)}] {model_dir}")
        exit_code = run_object(model_dir, args, scripts_dir) or exit_code

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
