#!/usr/bin/env python3
"""Generate geometry-only manual-contact candidates and static previews."""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from manual_contact_utils import (
    CONTACT_VERDICTS,
    candidate_id,
    candidate_local_points,
    candidate_strategy,
    initial_contact_geometries,
    link_inertial_properties,
    moving_link_vertices,
)
from run_forcesapien_batch_final_dataset import detect_object, sanitize_name


VIEWS = {
    "front": (0, 2),
    "side": (1, 2),
    "top": (0, 1),
}


def render_preview(path: Path, vertices: np.ndarray, candidates: list[dict[str, object]], view: str) -> None:
    width, height = 1000, 760
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    axes = VIEWS[view]
    cloud = vertices[:, axes]
    lo, hi = cloud.min(axis=0), cloud.max(axis=0)
    span = np.maximum(hi - lo, 1e-8)

    def project(point: list[float] | np.ndarray) -> tuple[int, int]:
        value = np.asarray(point, dtype=float)[list(axes)]
        normalized = (value - lo) / span
        return int(70 + normalized[0] * (width - 140)), int(height - 70 - normalized[1] * (height - 140))

    stride = max(1, len(vertices) // 12000)
    for point in vertices[::stride]:
        x, y = project(point)
        draw.point((x, y), fill=(185, 195, 205))
    for index, candidate in enumerate(candidates):
        x, y = project(candidate["local_point"])
        recommended = bool(candidate["recommended"])
        color = (14, 125, 80) if recommended else (190, 55, 45)
        radius = 8 if recommended else 6
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline="white", width=2)
        draw.text((x + 8, y - 12), str(index + 1), fill=color, font=ImageFont.load_default())
    draw.text((24, 20), f"{view.title()} view — numbered moving-link candidates", fill=(25, 35, 48))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def object_candidates(model_dir: Path, info: dict[str, object], force_magnitude: float = 5.0) -> tuple[np.ndarray, list[dict[str, object]]]:
    object_id = str(info["object_id"])
    vertices = moving_link_vertices(model_dir, str(info["link_name"]))
    selected_link_mass, selected_link_inertia_diag = link_inertial_properties(model_dir, str(info["link_name"]))
    bbox_max_dim = float(np.max(vertices.max(axis=0) - vertices.min(axis=0)))
    local_points = candidate_local_points(vertices, count=20)
    geometries = initial_contact_geometries(
        model_dir,
        str(info["joint_name"]),
        str(info["link_name"]),
        local_points,
        list(info["joint_axis"]),
    )
    records = []
    for index, (local_point, geometry) in enumerate(zip(local_points, geometries)):
        lever = float(geometry["lever_arm_perpendicular"])
        joint_type = str(info["joint_type"])
        projected_force = force_magnitude if joint_type == "prismatic" else None
        expected = projected_force if joint_type == "prismatic" else force_magnitude * lever
        normalized_effect = abs(float(expected)) / max(selected_link_mass * max(bbox_max_dim, 1e-6), 1e-6)
        effect_label = "TOO_WEAK" if normalized_effect < 0.15 else "AGGRESSIVE" if normalized_effect > 25.0 else "OK"
        warning = "" if expected > 1e-5 else "zero/near-zero joint-axis effect"
        records.append(
            {
                "candidate_id": candidate_id(object_id, index),
                "number": index + 1,
                "strategy": candidate_strategy(object_id, index),
                "contact_semantic_verdict": CONTACT_VERDICTS.get(object_id, "CONTACT_WARN"),
                "lever_arm_perpendicular": lever if joint_type == "revolute" else None,
                "expected_torque_or_projection_opening": expected,
                "expected_torque_or_projection_closing": -expected,
                "projected_force": projected_force,
                "selected_link_mass": selected_link_mass,
                "selected_link_inertia_diag": selected_link_inertia_diag,
                "bbox_max_dim": bbox_max_dim,
                "force_effect_label": effect_label,
                "recommended_direction": "prismatic_axis" if joint_type == "prismatic" else "tangent_opening",
                "local_point": local_point.astype(float).tolist(),
                "world_point": geometry["contact_point_world"],
                "joint_axis_world": geometry["joint_axis_world"],
                "joint_origin_world": geometry["joint_origin_world"],
                "tangent_opening_world": geometry["tangent_opening_world"],
                "tangent_closing_world": [-float(v) for v in geometry["tangent_opening_world"]],
                "warning": warning,
                "recommended": False,
            }
        )
    valid = [record for record in records if not record["warning"]]
    if valid:
        if object_id == "100109":
            next(item for item in valid if item["candidate_id"] == "usb_cover_tip")["recommended"] = True
        else:
            max(valid, key=lambda item: abs(float(item["expected_torque_or_projection_opening"])))["recommended"] = True
    return vertices, records


def yaml_snippet(info: dict[str, object], candidate: dict[str, object], direction_mode: str) -> str:
    return "\n".join(
        [
            f"{info['object_id']}:",
            f"  object_name: {info['object_name']}",
            f"  joint_name: {info['joint_name']}",
            f"  link_name: {info['link_name']}",
            "  contact_mode: candidate_id",
            f"  candidate_id: {candidate['candidate_id']}",
            f"  force_direction_mode: {direction_mode}",
            f"  note: \"recommended {candidate['strategy']} candidate\"",
        ]
    )


def generate(dataset_root: Path, output_root: Path, object_ids: list[str], force_magnitude: float = 5.0) -> list[dict[str, object]]:
    output_root.mkdir(parents=True, exist_ok=True)
    all_records: list[dict[str, object]] = []
    sections = []
    for object_id in object_ids:
        info = detect_object(dataset_root / object_id)
        if not info.get("valid"):
            raise RuntimeError(f"{object_id}: {info.get('error_message')}")
        folder_name = f"{object_id}_{sanitize_name(str(info['object_name']), 'object')}"
        folder = output_root / folder_name
        folder.mkdir(parents=True, exist_ok=True)
        candidate_file = folder / "candidates.json"
        vertices, records = object_candidates(dataset_root / object_id, info, force_magnitude)
        recommended = next(record for record in records if record["recommended"])
        for view in VIEWS:
            render_preview(folder / f"preview_{view}.png", vertices, records, view)
        payload = {
            "object_id": object_id,
            "object_name": info["object_name"],
            "joint_name": info["joint_name"],
            "link_name": info["link_name"],
            "joint_type": info["joint_type"],
            "candidates": records,
            "recommended_candidate_id": recommended["candidate_id"],
            "force_magnitude": force_magnitude,
            "yaml_snippet_opening": yaml_snippet(info, recommended, "prismatic_axis" if info["joint_type"] == "prismatic" else "tangent_opening"),
            "yaml_snippet_closing": yaml_snippet(info, recommended, "negative_prismatic_axis" if info["joint_type"] == "prismatic" else "tangent_closing"),
        }
        candidate_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        for record in records:
            all_records.append({"object_id": object_id, "object_name": info["object_name"], **record})
        rows = "".join(
            f"<tr class={'recommended' if r['recommended'] else ''}><td>{r['number']}</td><td><code>{html.escape(str(r['candidate_id']))}</code></td>"
            f"<td>{html.escape(str(r['strategy']))}</td><td>{html.escape(str(r['contact_semantic_verdict']))}</td>"
            f"<td>{r['lever_arm_perpendicular'] if r['lever_arm_perpendicular'] is not None else 'N/A'}</td>"
            f"<td>{r['expected_torque_or_projection_opening']:.6g}</td><td>{r['expected_torque_or_projection_closing']:.6g}</td><td>{r['force_effect_label']}</td><td>{r['selected_link_mass']:.6g}</td><td>{r['selected_link_inertia_diag']}</td><td>{r['bbox_max_dim']:.6g}</td>"
            f"<td><code>{r['recommended_direction']}</code></td><td><code>{html.escape(json.dumps(r['local_point']))}</code></td>"
            f"<td><code>{html.escape(json.dumps(r['world_point']))}</code></td><td>{html.escape(str(r['warning']) or 'none')}</td>"
            f"<td><pre>{html.escape(yaml_snippet(info, r, 'prismatic_axis' if info['joint_type'] == 'prismatic' else 'tangent_opening'))}</pre></td>"
            f"<td><pre>{html.escape(yaml_snippet(info, r, 'negative_prismatic_axis' if info['joint_type'] == 'prismatic' else 'tangent_closing'))}</pre></td></tr>"
            for r in records
        )
        previews = "".join(
            f'<img src="{folder_name}/preview_{view}.png" alt="{view} preview">' for view in VIEWS
        )
        sections.append(
            f"<section><h2>{html.escape(str(info['object_name']))} <small>{object_id}</small></h2>"
            f"<p>Selected <code>{info['joint_name']}/{info['link_name']}</code>. Green is the recommended candidate; semantic confidence is recorded separately.</p>"
            f"<div class='previews'>{previews}</div>"
            f"<table><thead><tr><th>#</th><th>candidate_id</th><th>strategy</th><th>semantic warning</th><th>lever arm</th><th>opening effect @ {force_magnitude:g}</th><th>closing effect @ {force_magnitude:g}</th><th>effect label</th><th>link mass</th><th>inertia diag</th><th>bbox max</th><th>recommended direction</th><th>local point</th><th>world point</th><th>geometry warning</th><th>opening/positive YAML</th><th>closing/negative YAML</th></tr></thead><tbody>{rows}</tbody></table></section>"
        )
    fields = [
        "object_id", "object_name", "candidate_id", "number", "strategy", "contact_semantic_verdict",
        "lever_arm_perpendicular", "projected_force", "expected_torque_or_projection_opening", "expected_torque_or_projection_closing", "force_effect_label", "selected_link_mass", "selected_link_inertia_diag", "bbox_max_dim", "recommended_direction",
        "local_point", "world_point", "joint_axis_world", "joint_origin_world", "tangent_opening_world", "tangent_closing_world", "warning", "recommended",
    ]
    with (output_root / "candidates.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(all_records)
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>ForceSAPIEN manual contact candidates</title>
<style>body{{font:14px system-ui;margin:0;background:#f4f6f8;color:#17202d}}main{{max-width:1500px;margin:auto;padding:28px}}section{{background:white;padding:20px;margin:20px 0;border:1px solid #d8e0e8;border-radius:10px}}.previews{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}}img{{width:100%;border:1px solid #ddd}}table{{width:100%;border-collapse:collapse;font-size:12px}}th,td{{border:1px solid #ddd;padding:6px;vertical-align:top}}th{{background:#edf3f7}}tr.recommended{{background:#e4f7ea}}code,pre{{font-family:monospace}}pre{{background:#f5f7f9;padding:12px;overflow:auto}}</style></head>
<body><main><h1>Manual contact point and direction candidates</h1><p>Geometry-only previews; no dynamics were run. Candidate numbers correspond across front, side, and top views. Each row gives both possible force directions (versi), their signed expected torque/projection at the same fixed {force_magnitude:g} dataset/SAPIEN force units, and copy-paste YAML.</p>{''.join(sections)}</main></body></html>"""
    (output_root / "index.html").write_text(page, encoding="utf-8")
    return all_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_root", default="final_dataset")
    parser.add_argument("--output_root", default="contact_selection")
    parser.add_argument("--object_ids", nargs="+", required=True)
    parser.add_argument("--force-magnitude", type=float, default=5.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = generate(Path(args.dataset_root).resolve(), Path(args.output_root).resolve(), args.object_ids, args.force_magnitude)
    print(f"Wrote {Path(args.output_root).resolve() / 'index.html'}")
    print(f"Generated {len(records)} candidates without running dynamics")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
