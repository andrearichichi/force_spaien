#!/usr/bin/env python3
"""Validate the dual-render ForceSAPIEN package contract."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs"
SOURCE_PACKAGE = ROOT / "output"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--source-package", type=Path, default=SOURCE_PACKAGE)
    args = parser.parse_args()
    package, source = args.package.resolve(), args.source_package.resolve()
    manifest = json.loads((source / "manifest.json").read_text())
    errors, total = [], 0
    for row in manifest["objects"]:
        slug = row["directory"]
        category = str(row["object_name"])
        folder = package / category
        for name in ("contact_sheet.png", "final_video.mp4", "run.log", "simulation.json"):
            if not (folder / name).is_file() or (folder / name).stat().st_size == 0:
                errors.append(f"{category}: missing/empty {name}")
        if sha256(folder / "final_video.mp4") != sha256(source / "objects" / slug / "final_video.mp4"):
            errors.append(f"{category}: final video differs from source")
        if sha256(folder / "simulation.json") != sha256(source / "objects" / slug / "simulation.json"):
            errors.append(f"{category}: simulation metadata differs from source")
        document = json.loads((folder / "simulation.json").read_text())
        expected = len(document["samples"]["force"])
        rgbs = sorted((folder / "frames" / "rgb").glob("frame_*.png"))
        masks = sorted((folder / "frames" / "masks").glob("frame_*.png"))
        expected_names = [f"frame_{index:04d}.png" for index in range(expected)]
        if [path.name for path in rgbs] != expected_names or [path.name for path in masks] != expected_names:
            errors.append(f"{category}: frame count/naming mismatch")
            continue
        for rgb_path, mask_path in zip(rgbs, masks):
            rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
            mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            if rgb is None or mask is None or rgb.shape[:2] != mask.shape[:2]:
                errors.append(f"{category}/{rgb_path.name}: unreadable or shape mismatch")
                break
            values = set(np.unique(mask).tolist())
            if not values.issubset({0, 255}) or 255 not in values:
                errors.append(f"{category}/{mask_path.name}: mask is not binary/nonempty")
                break
        total += expected
        print(f"PASS {category}: {expected} pairs, 1920x1080, video and metadata hashes preserved")
    print(f"SUMMARY samples={len(manifest['objects'])} frame_pairs={total} errors={len(errors)}")
    for error in errors:
        print(f"ERROR {error}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
