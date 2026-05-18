#!/usr/bin/env python3
"""Dispatch an object simulation to the prismatic or revolute pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

try:
    from paths import resolve_model_dir
except ModuleNotFoundError:
    from scripts.paths import resolve_model_dir


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


def run_object(model_dir_arg: str, args: argparse.Namespace, scripts_dir: Path) -> int:
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
        picker = scripts_dir / "application_point_picker.py"
        command = [
            sys.executable,
            str(picker),
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
        if args.preview_points:
            command.append("--preview-points")
        elif args.pick_point:
            command.append("--pick-point")
        else:
            command += ["--select-point", str(args.select_point)]

        print(f"Detected {detected_type}: {detected_joint}/{detected_link}")
        print(f"Selected {joint_type}: {joint_name}/{link_name}")
        print(f"Running picker: {' '.join(command)}")
        picker_exit_code = subprocess.run(command, check=False).returncode
        if picker_exit_code != 0 or args.preview_points or args.select_point is not None:
            return picker_exit_code

    command = [sys.executable]
    if joint_type == "prismatic":
        script = scripts_dir / ("render_prismatic_video.py" if args.mode == "render" else "apply_prismatic_force.py")
        command += [
            str(script),
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
    elif joint_type == "revolute":
        script = scripts_dir / ("render_revolute_video.py" if args.mode == "render" else "apply_revolute_force.py")
        command += [
            str(script),
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
            "  python3 scripts/main.py 101062 --select-point 6"
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
