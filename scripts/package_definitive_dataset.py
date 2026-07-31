#!/usr/bin/env python3
"""Build one portable definitive package from ten production raw runs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

import cv2
import imageio_ffmpeg
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/production_pipeline.json"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def timing_sheet(video: Path, duration: float, destination: Path) -> None:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    times = (0.0, 1.0, 1.9, 2.0, 2.1, duration / 2.0,
             max(0.0, duration - 1.0 / 30.0))
    frames = []
    with tempfile.TemporaryDirectory() as temporary:
        for index, timestamp in enumerate(times):
            target = Path(temporary) / f"{index}.jpg"
            subprocess.run([ffmpeg, "-y", "-v", "error", "-ss", str(timestamp),
                            "-i", str(video), "-frames:v", "1", str(target)], check=True)
            frame = cv2.imread(str(target))
            cv2.putText(frame, f"t={timestamp:.3f}s", (12, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 0, 0), 2, cv2.LINE_AA)
            frames.append(frame)
    frames.append(np.full_like(frames[0], 255))
    cv2.imwrite(str(destination), np.vstack((np.hstack(frames[:4]), np.hstack(frames[4:]))))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    raw = args.raw_root.resolve()
    output = args.output_root.resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite package: {output}")
    config = json.loads(CONFIG.read_text())
    output.mkdir(parents=True)
    (output / "objects").mkdir()
    (output / "assets").mkdir()
    cards, manifest_objects = [], []
    for object_id in config["object_ids"]:
        row = config["objects"][object_id]
        matches = list(raw.glob(f"*_{object_id}_*/simulation.json"))
        if len(matches) != 1:
            raise RuntimeError(f"{object_id}: expected one raw simulation, found {len(matches)}")
        source = matches[0].parent
        slug = f"{row['name'].lower()}_{object_id}"
        destination = output / "objects" / slug
        destination.mkdir()
        for name in ("final_video.mp4", "simulation.json", "run.log"):
            shutil.copy2(source / name, destination / name)
        document = json.loads((destination / "simulation.json").read_text())
        duration = float(document["metadata"]["timing"]["video_duration_seconds"])
        timing_sheet(destination / "final_video.mp4", duration,
                     destination / "contact_sheet.png")
        cards.append(
            f'<article><h2>{row["name"]} ({object_id})</h2>'
            f'<video controls muted loop preload="metadata" src="objects/{slug}/final_video.mp4"></video>'
            f'<a href="objects/{slug}/contact_sheet.png"><img src="objects/{slug}/contact_sheet.png"></a>'
            f'<p>{row["joint_type"]}; contact <code>{row["contact_label"]}</code>; '
            f'{row["personalized_force_n"]:.12g} N</p>'
            f'<a href="objects/{slug}/simulation.json">JSON</a> '
            f'<a href="objects/{slug}/run.log">log</a></article>'
        )
        manifest_objects.append({
            "object_id": object_id, "object_name": row["name"],
            "contact_label": row["contact_label"], "joint_type": row["joint_type"],
            "personalized_force_n": row["personalized_force_n"],
        })
    style = "body{font-family:sans-serif;margin:2rem;background:#f4f6f8}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:1rem}article{background:white;padding:1rem;border-radius:10px}video,img{width:100%;height:auto}"
    (output / "assets/style.css").write_text(style + "\n")
    (output / "index.html").write_text(
        '<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        '<link rel="stylesheet" href="assets/style.css"><title>ForceSAPIEN definitive dataset</title></head><body>'
        '<h1>70%-travel, zero-joint-friction definitive dataset</h1><p>Gravity disabled; 1/240 s timestep; '
        '2 s physical force pulse; viscous damping only with T_decay=2 s. Cyan trails are recorded application-point history. '
        'Forces are personalized by calibrated object physics. No object-specific physical or actuation overrides.</p><main>'
        + "".join(cards) + '</main></body></html>\n'
    )
    (output / "README.md").write_text(
        "# Portable ForceSAPIEN definitive package\n\nOpen `index.html` through a local HTTP server.\n"
    )
    (output / "manifest.json").write_text(json.dumps({
        "schema_version": 1, "objects": manifest_objects,
        "physics": config["physics"], "rendering": config["rendering"],
    }, indent=2) + "\n")
    files = sorted(path for path in output.rglob("*")
                   if path.is_file() and path.name != "SHA256SUMS")
    (output / "SHA256SUMS").write_text("".join(
        f"{sha(path)}  {path.relative_to(output).as_posix()}\n" for path in files
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
