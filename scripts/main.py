"""Minimal URDF joint-selection helpers used by the production batch runner."""
from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET


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
        bounds = None
        if limit is not None and "lower" in limit.attrib and "upper" in limit.attrib:
            bounds = (float(limit.attrib["lower"]), float(limit.attrib["upper"]))
        return (
            joint_type,
            joint.attrib.get("name", ""),
            child.attrib.get("link", ""),
            bounds,
        )
    raise RuntimeError(f"No moving joint found in {model_dir / 'mobility.urdf'}")


def default_initial_angle(model_dir: Path, limits: tuple[float, float] | None) -> float:
    del model_dir
    if limits is None:
        return 0.0
    lower, upper = limits
    return 0.0 if lower <= 0.0 <= upper else lower


def drawer_index_from_link(link_name: str) -> str:
    if not link_name.startswith("link_"):
        raise RuntimeError(f"Cannot infer prismatic drawer index from link name: {link_name}")
    return link_name.removeprefix("link_")


def preferred_joint(model_dir: Path, detected_joint: str, detected_link: str) -> tuple[str, str]:
    del model_dir
    return detected_joint, detected_link
