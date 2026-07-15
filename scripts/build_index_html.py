#!/usr/bin/env python3
"""Build a polished static ForceSAPIEN / FORCEARTGS dashboard."""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
from typing import Any


DIAGNOSTIC_FILES = [
    "physics_diagnostics.md",
    "physics_timeseries.tsv",
    "force_torque_by_frame.png",
    "q_qdot_qddot_by_frame.png",
    "resistance_by_frame.png",
]

REPORT_FILES = [
    ("Final global impulse-decay report", "final_manual_contact_global_impulse_decay_report.md"),
    ("Final global impulse-decay table", "final_manual_contact_global_impulse_decay_table.tsv"),
    ("Physical consistency report", "physical_consistency_manual_contact_global_impulse_decay.md"),
    ("Physical consistency table", "physical_consistency_manual_contact_global_impulse_decay.tsv"),
    ("Visual review report", "final_visual_review_manual_contact_global_impulse_decay.md"),
    ("Visual review table", "final_visual_review_manual_contact_global_impulse_decay.tsv"),
]

CONTACT_SHEETS = [
    "forcesapien_physics_audit/frames/contact_sheet.png",
    "forcesapien_physics_audit/postfix_contact_sheet.jpg",
]

IMPORTANT_FIELDS = [
    "object_id",
    "object_name",
    "joint_type",
    "selected_joint",
    "selected_link",
    "q_start",
    "q_end",
    "delta_q",
    "force_magnitude",
    "force_application_mode",
    "final_physical_consistency_verdict",
    "contact_verdict",
    "joint_verdict",
    "scale_verdict",
    "physics_verdict",
    "peak_abs_qdot",
    "final_abs_qdot",
    "qdot_decay_ratio",
    "settled",
    "video_frame_count",
    "diagnostics_complete",
    "motion_detected",
]


def esc(value: Any) -> str:
    if value is None or value == "":
        return "N/A"
    return html.escape(str(value), quote=True)


def fmt(value: Any, digits: int = 4) -> str:
    if value is None or value == "":
        return "N/A"
    if isinstance(value, bool):
        return "true" if value else "false"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if digits == 0:
        return str(int(round(number)))
    return f"{number:.{digits}f}".rstrip("0").rstrip(".")


def as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "pass"}:
        return True
    if text in {"false", "0", "no", "fail"}:
        return False
    return None


def rel(path: Path, repo_root: Path) -> str:
    return path.resolve().relative_to(repo_root.resolve()).as_posix()


def nested(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def first_value(*values: Any) -> Any:
    for value in values:
        if value is not None and value != "":
            return value
    return None


def read_summary(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row["object_id"]: row for row in csv.DictReader(handle, delimiter="\t") if row.get("object_id")}


def load_simulation(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle), None
    except Exception as exc:  # noqa: BLE001 - invalid JSON should become an error card
        return None, str(exc)


def object_id_from_sim(sim: dict[str, Any], folder: Path) -> str:
    if str(sim.get("object_id", "")).isdigit():
        return str(sim["object_id"])
    model_dir = nested(sim, "metadata", "object", "model_dir")
    if model_dir:
        return Path(str(model_dir)).name
    for part in folder.name.split("_"):
        if part.isdigit():
            return part
    return folder.name


def object_name_from_folder(folder: Path, object_id: str) -> str:
    prefix = folder.name.split(f"_{object_id}", 1)[0]
    return prefix.replace("_", " ").title() if prefix else "N/A"


def sample_series(sim: dict[str, Any]) -> dict[str, Any]:
    series = nested(sim, "metadata", "summary", "sample_series", "force", default={})
    return series if isinstance(series, dict) else {}


def extract_q(sim: dict[str, Any], joint_type: str, row: dict[str, str]) -> tuple[Any, Any, Any, str]:
    series = sample_series(sim)
    if joint_type == "prismatic":
        return (
            first_value(row.get("q_start"), sim.get("q_start"), series.get("initial_joint_position_m")),
            first_value(row.get("q_end"), sim.get("q_end"), series.get("final_joint_position_m")),
            first_value(row.get("delta_q"), sim.get("delta_q"), series.get("delta_joint_position_m")),
            "m",
        )
    return (
        first_value(row.get("q_start"), sim.get("q_start"), series.get("initial_joint_angle_rad")),
        first_value(row.get("q_end"), sim.get("q_end"), series.get("final_joint_angle_rad")),
        first_value(row.get("delta_q"), sim.get("delta_q"), series.get("delta_joint_angle_rad")),
        "rad",
    )


def force_mode(sim: dict[str, Any], row: dict[str, str]) -> str:
    mode = first_value(row.get("force_application_mode"), sim.get("force_application_mode"), nested(sim, "metadata", "actuation", "force", "force_application_mode"))
    return "generalized_set_qf" if mode == "generalized" else str(mode or "N/A")


def collect_files(folder: Path, repo_root: Path) -> dict[str, str | None]:
    diagnostics = folder / "diagnostics"
    files: dict[str, str | None] = {
        "simulation_json": rel(folder / "simulation.json", repo_root),
        "final_video": rel(folder / "final_video.mp4", repo_root) if (folder / "final_video.mp4").exists() else None,
    }
    for name in DIAGNOSTIC_FILES:
        path = diagnostics / name
        files[name] = rel(path, repo_root) if path.exists() else None
    return files


def make_object(sim_path: Path, repo_root: Path, summary_rows: dict[str, dict[str, str]]) -> dict[str, Any]:
    folder = sim_path.parent
    files = collect_files(folder, repo_root)
    sim, error = load_simulation(sim_path)
    if sim is None:
        object_id = next((part for part in folder.name.split("_") if part.isdigit()), folder.name)
        obj = {
            "object_id": object_id,
            "object_name": object_name_from_folder(folder, object_id),
            "folder": rel(folder, repo_root),
            "status": "error",
            "json_error": error,
            "files": files,
            "missing_files": [name for name, path in files.items() if name != "simulation_json" and not path],
            "missing_fields": IMPORTANT_FIELDS,
        }
        return obj

    object_id = object_id_from_sim(sim, folder)
    row = summary_rows.get(object_id, {})
    joint_type = str(first_value(row.get("joint_type"), nested(sim, "metadata", "object", "joint_type"), sim.get("motion_type")))
    q_start, q_end, delta_q, q_unit = extract_q(sim, joint_type, row)
    force = nested(sim, "metadata", "actuation", "force", default={})
    force = force if isinstance(force, dict) else {}
    mode = force_mode(sim, row)
    warnings = first_value(row.get("remaining_warnings"), sim.get("warning_messages"), nested(sim, "metadata", "validation", "warning_messages"))
    if isinstance(warnings, list):
        warnings = "; ".join(str(item) for item in warnings)

    obj = {
        "object_id": object_id,
        "object_name": first_value(row.get("object_name"), nested(sim, "metadata", "object", "name"), object_name_from_folder(folder, object_id)),
        "folder": rel(folder, repo_root),
        "status": first_value(nested(sim, "status"), "success"),
        "error_message": nested(sim, "error_message"),
        "joint_type": joint_type,
        "selected_joint": first_value(row.get("selected_joint"), sim.get("selected_joint"), nested(sim, "metadata", "object", "joint")),
        "selected_link": first_value(row.get("selected_link"), sim.get("selected_link"), nested(sim, "metadata", "object", "link")),
        "parent_link": nested(sim, "metadata", "object", "parent_link"),
        "child_link": nested(sim, "metadata", "object", "child_link"),
        "joint_axis_world": first_value(sim.get("joint_axis_world_at_pulse"), force.get("joint_axis_world_at_pulse"), force.get("joint_axis_world")),
        "joint_origin_world": first_value(sim.get("joint_origin_world_at_pulse"), force.get("joint_origin_world_at_pulse"), force.get("joint_origin_world")),
        "joint_limits": first_value(nested(sim, "metadata", "actuation", "joint_limits_rad"), nested(sim, "metadata", "actuation", "joint_limits_m")),
        "q_start": q_start,
        "q_end": q_end,
        "delta_q": delta_q,
        "q_unit": q_unit,
        "force_application_mode": mode,
        "calibrated_newtons": False,
        "force_units": "dataset/SAPIEN units",
        "force_magnitude": first_value(sim.get("force_magnitude"), force.get("magnitude_n"), force.get("applied_tangential_force_n"), force.get("applied_linear_force_n")),
        "force_duration_s": first_value(sim.get("force_duration_s"), nested(sim, "metadata", "actuation", "force", "force_duration_s")),
        "force_direction_world": first_value(sim.get("force_direction_world_at_pulse"), force.get("force_direction_world_at_pulse"), force.get("direction_world")),
        "force_application_point_world": first_value(sim.get("force_application_point_world_at_pulse"), force.get("force_application_point_world_at_pulse"), force.get("force_application_point_world")),
        "torque_about_axis": first_value(sim.get("torque_about_axis_at_pulse"), force.get("torque_about_axis_nm"), force.get("generalized_torque_nm")),
        "projected_force_along_axis": first_value(sim.get("raw_projected_force_along_axis_at_pulse"), force.get("raw_projected_force_along_axis")),
        "contact_strategy": first_value(sim.get("contact_strategy"), nested(sim, "metadata", "application_point", "strategy")),
        "contact_semantic_verdict": sim.get("contact_semantic_verdict"),
        "contact_semantic_explanation": sim.get("contact_semantic_explanation"),
        "contact_verdict": first_value(row.get("contact_verdict"), sim.get("contact_verdict"), nested(sim, "metadata", "validation", "contact_verdict")),
        "joint_verdict": first_value(row.get("joint_verdict"), sim.get("joint_verdict"), nested(sim, "metadata", "validation", "joint_verdict")),
        "scale_verdict": first_value(row.get("scale_verdict"), sim.get("scale_verdict"), nested(sim, "metadata", "validation", "scale_verdict")),
        "physics_verdict": first_value(row.get("physics_verdict"), sim.get("physics_verdict"), nested(sim, "metadata", "validation", "physics_verdict")),
        "peak_abs_qdot": sim.get("peak_abs_qdot"),
        "final_abs_qdot": sim.get("final_abs_qdot"),
        "qdot_decay_ratio": sim.get("qdot_decay_ratio"),
        "settled": sim.get("settled"),
        "effective_joint_inertia_proxy": sim.get("effective_joint_inertia_proxy"),
        "final_physical_consistency_verdict": sim.get("final_physical_consistency_verdict"),
        "video_frame_count": first_value(row.get("video_frame_count"), nested(sim, "metadata", "summary", "total_sample_count")),
        "diagnostics_complete": first_value(row.get("diagnostics_complete"), all(files[name] for name in DIAGNOSTIC_FILES)),
        "motion_detected": first_value(row.get("motion_detected"), abs(float(delta_q)) > 1e-9 if delta_q not in (None, "") else None),
        "warning_messages": warnings,
        "files": files,
        "missing_files": [name for name, path in files.items() if name != "simulation_json" and not path],
        "json_error": error,
    }
    obj["missing_fields"] = [field for field in IMPORTANT_FIELDS if obj.get(field) in (None, "")]
    return obj


def verdict_class(value: Any) -> str:
    text = str(value or "").upper()
    if text == "PASS":
        return "pass"
    if text == "WARN":
        return "warn"
    if text == "FAIL":
        return "fail"
    return "na"


def verdict_badge(label: str, value: Any) -> str:
    return f'<span class="verdict {verdict_class(value)}"><span>{esc(label)}</span>{esc(value)}</span>'


def warning_items(raw: Any) -> list[str]:
    if raw is None or raw == "":
        return []
    if isinstance(raw, list):
        parts = [str(item).strip() for item in raw]
    else:
        parts = [part.strip() for part in str(raw).split(";")]
    return [part for part in parts if part]


def warning_labels(raw: Any) -> list[tuple[str, str]]:
    text = " ".join(warning_items(raw)).lower()
    labels: list[tuple[str, str]] = []
    if "max dimension" in text or "outside rough expected" in text:
        labels.append(("Scale", "normalized/dataset-specific scale"))
    if "generalized" in text or "set_qf" in text or "not true external" in text:
        labels.append(("Legacy", "legacy generalized-force warning; final output uses external_link_force"))
    if "motion is still active" in text or "settled video" in text:
        labels.append(("Motion", "motion still active at final frame"))
    if "contact" in text and "not true external" not in text:
        labels.append(("Contact", "contact selection warning"))
    if not labels and text:
        labels.append(("Warning", warning_items(raw)[0]))
    return labels


def warning_summary(raw: Any) -> str:
    labels = warning_labels(raw)
    if not labels:
        return "none"
    return "; ".join(f"{label}: {message}" for label, message in labels)


def warning_badges(raw: Any) -> str:
    labels = warning_labels(raw)
    if not labels:
        return '<span class="warning-chip ok">No remaining warning labels</span>'
    return "".join(f'<span class="warning-chip"><strong>{esc(label)}</strong>{esc(message)}</span>' for label, message in labels)


def truncate(text: Any, limit: int = 120) -> str:
    value = str(text or "N/A")
    if len(value) <= limit:
        return value
    return value[: limit - 1].rstrip() + "..."


def stat_card(label: str, value: Any, note: str) -> str:
    return f'<div class="stat"><span>{esc(label)}</span><strong>{esc(value)}</strong><small>{esc(note)}</small></div>'


def field_list(items: list[tuple[str, str]]) -> str:
    return "".join(f"<li><code>{esc(field)}</code> {esc(description)}</li>" for field, description in items)


def render_file_links(obj: dict[str, Any]) -> str:
    labels = [
        ("simulation_json", "simulation.json"),
        ("final_video", "final_video.mp4"),
        ("physics_timeseries.tsv", "physics_timeseries.tsv"),
        ("physics_diagnostics.md", "physics_diagnostics.md"),
        ("q_qdot_qddot_by_frame.png", "q/qdot/qddot plot"),
        ("force_torque_by_frame.png", "force/torque plot"),
        ("resistance_by_frame.png", "resistance plot"),
    ]
    chips = [
        f'<a class="file-chip" href="{esc(obj["files"][key])}">{esc(label)}</a>'
        for key, label in labels
        if obj.get("files", {}).get(key)
    ]
    return "\n".join(chips) if chips else '<span class="muted">N/A</span>'


def table_links(obj: dict[str, Any]) -> str:
    links = [
        ("simulation_json", "JSON"),
        ("final_video", "Video"),
        ("physics_timeseries.tsv", "TSV"),
        ("q_qdot_qddot_by_frame.png", "Plots"),
    ]
    chips = [
        f'<a class="mini-link" href="{esc(obj["files"][key])}">{esc(label)}</a>'
        for key, label in links
        if obj.get("files", {}).get(key)
    ]
    return "".join(chips) if chips else '<span class="muted">N/A</span>'


def render_thumbnails(obj: dict[str, Any]) -> str:
    labels = [
        ("q_qdot_qddot_by_frame.png", "q, qdot, qddot by frame"),
        ("force_torque_by_frame.png", "force and torque by frame"),
        ("resistance_by_frame.png", "resistance by frame"),
    ]
    thumbs = []
    for key, label in labels:
        path = obj.get("files", {}).get(key)
        if path:
            thumbs.append(f'<figure class="plot-thumb"><a href="{esc(path)}"><img src="{esc(path)}" alt="{esc(label)}" loading="lazy"></a><figcaption>{esc(label)}</figcaption></figure>')
    return "\n".join(thumbs) if thumbs else '<p class="muted compact-note">No diagnostic plot thumbnails found.</p>'


def render_object_card(obj: dict[str, Any]) -> str:
    if obj.get("json_error"):
        return f"""
        <article class="object-card">
          <div class="object-title">
            <div><p class="eyebrow">Object {esc(obj.get('object_id'))}</p><h3>{esc(obj.get('object_name'))}</h3></div>
          </div>
          <p class="bad">Invalid simulation.json: {esc(obj.get('json_error'))}</p>
          <div class="file-row">{render_file_links(obj)}</div>
        </article>
        """

    video = obj.get("files", {}).get("final_video")
    video_html = f'<video controls muted loop preload="metadata" src="{esc(video)}"></video>' if video else '<div class="missing">No final_video.mp4 found.</div>'
    warnings = obj.get("warning_messages") or "N/A"
    return f"""
    <article class="object-card">
      <div class="object-title">
        <div>
          <p class="eyebrow">Object {esc(obj.get('object_id'))}</p>
          <h3>{esc(obj.get('object_name'))}</h3>
        </div>
        <div class="object-badges">
          <span class="joint-badge">{esc(obj.get('joint_type'))}</span>
          {verdict_badge("consistency", "WARN" if obj.get('final_physical_consistency_verdict') == "PHYSICS_PLAUSIBLE_BUT_UNCALIBRATED" else obj.get('physics_verdict'))}
        </div>
      </div>
      {video_html}
      <div class="verdict-strip">
        {verdict_badge("contact", obj.get('contact_verdict'))}
        {verdict_badge("joint", obj.get('joint_verdict'))}
        {verdict_badge("scale", obj.get('scale_verdict'))}
      </div>
      <dl class="metric-grid">
        <div><dt>Selected joint/link</dt><dd>{esc(obj.get('selected_joint'))} / {esc(obj.get('selected_link'))}</dd></div>
        <div><dt>q start</dt><dd>{esc(fmt(obj.get('q_start')))} {esc(obj.get('q_unit'))}</dd></div>
        <div><dt>q end</dt><dd>{esc(fmt(obj.get('q_end')))} {esc(obj.get('q_unit'))}</dd></div>
        <div><dt>delta q</dt><dd>{esc(fmt(obj.get('delta_q')))} {esc(obj.get('q_unit'))}</dd></div>
        <div><dt>Force mode</dt><dd>{esc(obj.get('force_application_mode'))}</dd></div>
        <div><dt>Force magnitude</dt><dd>{esc(fmt(obj.get('force_magnitude')))} dataset/SAPIEN units</dd></div>
        <div><dt>Force duration</dt><dd>{esc(fmt(obj.get('force_duration_s')))} s</dd></div>
        <div><dt>Peak / final |qdot|</dt><dd>{esc(fmt(obj.get('peak_abs_qdot')))} / {esc(fmt(obj.get('final_abs_qdot')))}</dd></div>
        <div><dt>qdot decay ratio</dt><dd>{esc(fmt(obj.get('qdot_decay_ratio')))}</dd></div>
        <div><dt>Settled</dt><dd>{esc(fmt(obj.get('settled')))}</dd></div>
        <div><dt>Pulse torque / projection</dt><dd>{esc(fmt(first_value(obj.get('torque_about_axis'), obj.get('projected_force_along_axis'))))}</dd></div>
        <div><dt>Effective inertia proxy</dt><dd>{esc(fmt(obj.get('effective_joint_inertia_proxy')))}</dd></div>
        <div><dt>Contact semantics</dt><dd>{esc(obj.get('contact_semantic_verdict'))}: {esc(obj.get('contact_strategy'))}</dd></div>
        <div><dt>Physical consistency</dt><dd>{esc(obj.get('final_physical_consistency_verdict'))}</dd></div>
        <div><dt>Force units</dt><dd>dataset/SAPIEN units</dd></div>
        <div><dt>Calibrated Newtons</dt><dd>no</dd></div>
        <div><dt>Video frames</dt><dd>{esc(fmt(obj.get('video_frame_count'), 0))}</dd></div>
      </dl>
      <div class="warnings"><strong>Warnings</strong><div class="warning-list">{warning_badges(warnings)}</div></div>
      <div class="file-row">{render_file_links(obj)}</div>
      <details class="plots" open>
        <summary>Diagnostic plots</summary>
        <div class="thumbs">{render_thumbnails(obj)}</div>
      </details>
    </article>
    """


def table_rows(objects: list[dict[str, Any]]) -> str:
    rows = []
    for obj in objects:
        object_id = esc(obj.get("object_id"))
        row_data = {
            "verdicts": " ".join(str(obj.get(key, "")).upper() for key in ["contact_verdict", "joint_verdict", "scale_verdict", "physics_verdict"]),
            "jointType": str(obj.get("joint_type", "")),
            "forceMode": str(obj.get("force_application_mode", "")),
        }
        data_attr = esc(json.dumps(row_data, ensure_ascii=True))
        warning_text = warning_summary(obj.get("warning_messages"))
        raw_warnings = obj.get("warning_messages") or "N/A"
        diagnostics_text = "complete" if as_bool(obj.get("diagnostics_complete")) is True else "missing"
        video_text = "OK" if obj.get("files", {}).get("final_video") else "missing"
        details = f"""
          <div class="detail-grid">
            <div><strong>Full warnings</strong><p>{esc(raw_warnings)}</p></div>
            <div><strong>Motion</strong><p>q_start {esc(fmt(obj.get('q_start')))} {esc(obj.get('q_unit'))}; q_end {esc(fmt(obj.get('q_end')))} {esc(obj.get('q_unit'))}; delta_q {esc(fmt(obj.get('delta_q')))} {esc(obj.get('q_unit'))}</p></div>
            <div><strong>Joint</strong><p>{esc(obj.get('selected_joint'))} / {esc(obj.get('selected_link'))}</p></div>
            <div><strong>Mode</strong><p>{esc(obj.get('force_application_mode'))}; 5 dataset/SAPIEN force units; calibrated Newtons: no</p></div>
            <div><strong>Final verdict</strong><p>{esc(obj.get('final_physical_consistency_verdict'))}</p></div>
            <div><strong>Files</strong><div class="file-row">{render_file_links(obj)}</div></div>
          </div>
        """
        rows.append(
            f"""
            <tr class="summary-main" data-object="{object_id}" data-row="{data_attr}">
              <td class="sticky-col object-cell"><strong>{esc(obj.get('object_name'))}</strong><span>#{object_id}</span></td>
              <td class="joint-cell"><strong>{esc(obj.get('joint_type'))}</strong><span>{esc(obj.get('selected_joint'))} &middot; {esc(obj.get('selected_link'))}</span></td>
              <td class="motion-cell"><strong>{esc(fmt(obj.get('q_start'), 3))} &rarr; {esc(fmt(obj.get('q_end'), 3))} {esc(obj.get('q_unit'))}</strong><span>&Delta; {esc(fmt(obj.get('delta_q'), 3))}</span></td>
              <td class="mode-cell"><strong>{esc(obj.get('force_application_mode'))}</strong><span>calibrated Newtons: no</span></td>
              <td class="verdict-cell">
                {verdict_badge('contact', obj.get('contact_verdict'))}
                {verdict_badge('joint', obj.get('joint_verdict'))}
                {verdict_badge('scale', obj.get('scale_verdict'))}
                {verdict_badge('consistency', 'WARN' if obj.get('final_physical_consistency_verdict') == 'PHYSICS_PLAUSIBLE_BUT_UNCALIBRATED' else obj.get('physics_verdict'))}
              </td>
              <td class="status-cell"><strong>{esc(fmt(obj.get('video_frame_count'), 0))} frames</strong><span>{esc(video_text)}</span></td>
              <td class="status-cell"><strong>{esc(diagnostics_text)}</strong><span>{'OK' if diagnostics_text == 'complete' else 'check files'}</span></td>
              <td class="warning-cell"><span title="{esc(raw_warnings)}">{esc(truncate(warning_text))}</span></td>
              <td class="links-cell">{table_links(obj)}</td>
            </tr>
            <tr class="summary-detail" data-object="{object_id}" data-row="{data_attr}">
              <td class="sticky-col detail-label">Details</td>
              <td colspan="8"><details><summary>Expand full object details</summary>{details}</details></td>
            </tr>
            """
        )
    return "\n".join(rows)


def render_reports(output_root: Path, repo_root: Path) -> str:
    report_cards = []
    for label, name in REPORT_FILES:
        path = output_root / name
        if path.exists():
            report_cards.append(f'<a class="report-card" href="{esc(rel(path, repo_root))}"><strong>{esc(label)}</strong><span>{esc(name)}</span></a>')
        else:
            report_cards.append(f'<div class="report-card unavailable"><strong>{esc(label)}</strong><span>not available</span></div>')

    sheets = []
    for name in CONTACT_SHEETS:
        path = output_root / name
        if path.exists():
            href = rel(path, repo_root)
            sheets.append(f'<figure class="sheet"><a href="{esc(href)}"><img src="{esc(href)}" alt="{esc(Path(name).name)}" loading="lazy"></a><figcaption>{esc(name)}</figcaption></figure>')
        else:
            sheets.append(f'<div class="missing-sheet">{esc(name)} not available</div>')

    return f"""
    <section id="audit" class="section-block">
      <div class="section-heading">
        <p class="eyebrow">Reports</p>
        <h2>Audit links and contact sheets</h2>
        <p>Open the validated summary, audit tables, and postfix reports directly from the generated output tree.</p>
      </div>
      <div class="report-grid">{''.join(report_cards)}</div>
      <div class="contact-sheets">{''.join(sheets)}</div>
    </section>
    """


def build_html(objects: list[dict[str, Any]], output_root: Path, repo_root: Path, skipped: list[str]) -> str:
    total = len(objects)
    successful = sum(1 for obj in objects if str(obj.get("status", "")).lower() == "success" and not obj.get("json_error"))
    videos = sum(1 for obj in objects if obj.get("files", {}).get("final_video"))
    diagnostics = sum(1 for obj in objects if as_bool(obj.get("diagnostics_complete")) is True)
    pass_count = sum(1 for obj in objects if all(str(obj.get(key, "")).upper() == "PASS" for key in ["contact_verdict", "joint_verdict", "scale_verdict", "physics_verdict"]))
    fail_count = sum(1 for obj in objects if any(str(obj.get(key, "")).upper() == "FAIL" for key in ["contact_verdict", "joint_verdict", "scale_verdict", "physics_verdict"]))
    warn_count = max(0, total - pass_count - fail_count)
    missing_files = sorted({name for obj in objects for name in obj.get("missing_files", [])})
    missing_fields = sorted({name for obj in objects for name in obj.get("missing_fields", [])})
    skipped_note = f'<p class="note-line">Skipped non-final simulation folders: {esc(", ".join(skipped))}</p>' if skipped else ""
    stats = "".join(
        [
            stat_card("Objects", total, "final validated set"),
            stat_card("Videos", videos, "embedded MP4 cards"),
            stat_card("Diagnostics complete", diagnostics, "TSV, MD, PNG plots"),
            stat_card("Failed objects", fail_count, "no remaining failures"),
            stat_card("Mode", "external_link_force", "5 dataset force units"),
            stat_card("Calibrated Newtons", "no", "dataset/SAPIEN units"),
        ]
    )
    cards = "\n".join(render_object_card(obj) for obj in objects)
    reports = render_reports(output_root, repo_root)
    data_json = json.dumps(objects, ensure_ascii=True)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ForceSAPIEN / FORCEARTGS Final Dataset Dashboard</title>
  <style>
    :root {{
      --bg: #f4f6f8;
      --paper: #ffffff;
      --ink: #18212f;
      --muted: #657386;
      --line: #d9e0e8;
      --soft: #f0f4f8;
      --accent: #0d6d80;
      --accent-ink: #074657;
      --pass: #dff5e8;
      --pass-ink: #17643b;
      --warn: #fff2c7;
      --warn-ink: #785400;
      --fail: #ffe2df;
      --fail-ink: #8d261f;
      --code: #f7f9fb;
    }}
    * {{ box-sizing: border-box; }}
    html {{ scroll-behavior: smooth; }}
    body {{ margin: 0; background: var(--bg); color: var(--ink); font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif; line-height: 1.5; }}
    a {{ color: var(--accent-ink); text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    code, pre {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
    code {{ padding: 1px 4px; border-radius: 4px; background: #edf2f7; }}
    pre {{ overflow-x: auto; margin: 12px 0 0; padding: 14px; border: 1px solid var(--line); border-radius: 8px; background: var(--code); font-size: 13px; line-height: 1.45; }}
    .page {{ max-width: 1240px; margin: 0 auto; padding: 28px 22px 56px; }}
    .hero {{ background: linear-gradient(180deg, #fff 0%, #f8fbfc 100%); border-bottom: 1px solid var(--line); }}
    .hero-inner {{ max-width: 1240px; margin: 0 auto; padding: 42px 22px 30px; }}
    .hero-grid {{ display: grid; grid-template-columns: minmax(0, 1.55fr) minmax(320px, 0.9fr); gap: 26px; align-items: end; }}
    h1 {{ margin: 0 0 10px; font-size: clamp(30px, 4.2vw, 50px); line-height: 1.05; letter-spacing: 0; }}
    h2 {{ margin: 0 0 10px; font-size: 25px; line-height: 1.18; letter-spacing: 0; }}
    h3 {{ margin: 0 0 8px; font-size: 18px; letter-spacing: 0; }}
    p {{ margin: 0 0 10px; }}
    .subtitle {{ max-width: 760px; color: var(--muted); font-size: 18px; }}
    .eyebrow {{ margin: 0 0 8px; color: var(--accent); font-weight: 800; font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; }}
    .stats {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }}
    .stat {{ min-height: 92px; border: 1px solid var(--line); border-radius: 8px; padding: 13px; background: var(--paper); box-shadow: 0 8px 24px rgba(28, 42, 60, 0.05); }}
    .stat span {{ display: block; color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.06em; }}
    .stat strong {{ display: block; margin: 4px 0 2px; font-size: 24px; line-height: 1.1; overflow-wrap: anywhere; }}
    .stat small {{ color: var(--muted); }}
    .nav {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 22px; }}
    .nav a {{ padding: 7px 10px; border: 1px solid var(--line); border-radius: 999px; background: #fff; font-size: 13px; color: var(--ink); }}
    .callout {{ margin-top: 22px; border: 1px solid #e8c76b; border-left: 5px solid #cc9200; border-radius: 8px; background: #fff8e8; padding: 15px 16px; max-width: 940px; }}
    .callout strong {{ color: #604300; }}
    .section-block {{ margin: 28px 0; }}
    .section-heading {{ max-width: 820px; margin-bottom: 14px; }}
    .section-heading p:not(.eyebrow) {{ color: var(--muted); }}
    .panel, .object-card {{ background: var(--paper); border: 1px solid var(--line); border-radius: 8px; padding: 20px; box-shadow: 0 1px 2px rgba(25, 38, 55, 0.04); }}
    .two-col {{ display: grid; grid-template-columns: minmax(0, 1fr) minmax(340px, 0.85fr); gap: 18px; align-items: start; }}
    .repo-list {{ margin: 0; padding-left: 20px; }}
    .repo-list li {{ margin: 7px 0; }}
    .tree {{ white-space: pre; font-size: 12.5px; }}
    .note-line {{ margin-top: 12px; color: var(--muted); font-size: 13px; }}
    .muted {{ color: var(--muted); }}
    .flow {{ display: grid; grid-template-columns: repeat(9, minmax(105px, 1fr)); gap: 8px; overflow-x: auto; padding-bottom: 4px; }}
    .flow-step {{ min-height: 112px; border: 1px solid var(--line); border-radius: 8px; background: #fff; padding: 12px; }}
    .flow-step span {{ display: inline-grid; place-items: center; width: 24px; height: 24px; border-radius: 50%; background: #dceff3; color: var(--accent-ink); font-weight: 800; font-size: 12px; }}
    .flow-step strong {{ display: block; margin: 8px 0 3px; font-size: 14px; }}
    .flow-step p {{ margin: 0; color: var(--muted); font-size: 12px; }}
    .fixes {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; margin-top: 14px; }}
    .fix {{ border: 1px solid var(--line); border-radius: 8px; background: #fbfcfd; padding: 12px; }}
    .fix strong {{ display: block; margin-bottom: 3px; }}
    .schema-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }}
    details.schema {{ border: 1px solid var(--line); border-radius: 8px; background: #fff; overflow: hidden; }}
    details.schema summary {{ cursor: pointer; padding: 13px 14px; font-weight: 750; background: #f8fafc; }}
    details.schema div {{ padding: 0 14px 14px; }}
    details.schema ul {{ margin: 8px 0 0; padding-left: 20px; }}
    details.schema li {{ margin: 5px 0; }}
    .example {{ margin-top: 14px; }}
    .object-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 18px; }}
    .object-card {{ overflow: hidden; }}
    .object-title {{ display: flex; justify-content: space-between; gap: 12px; align-items: start; margin-bottom: 12px; }}
    .object-title h3 {{ font-size: 22px; }}
    .object-badges, .verdict-strip, .file-row {{ display: flex; flex-wrap: wrap; gap: 7px; }}
    .joint-badge, .file-chip, .verdict {{ display: inline-flex; align-items: center; gap: 6px; border-radius: 999px; border: 1px solid var(--line); padding: 5px 9px; background: #fff; font-size: 12px; font-weight: 700; white-space: nowrap; }}
    .joint-badge {{ background: #e9f6f8; color: var(--accent-ink); border-color: #b9dce4; }}
    .verdict span {{ color: inherit; opacity: 0.75; font-weight: 650; }}
    .verdict.pass {{ background: var(--pass); color: var(--pass-ink); border-color: #abdcbc; }}
    .verdict.warn {{ background: var(--warn); color: var(--warn-ink); border-color: #e6c66a; }}
    .verdict.fail {{ background: var(--fail); color: var(--fail-ink); border-color: #eba7a1; }}
    .verdict.na {{ background: var(--soft); color: var(--muted); }}
    video {{ display: block; width: 100%; max-height: 300px; background: #0d1218; border-radius: 8px; margin: 12px 0; }}
    .missing {{ display: grid; place-items: center; min-height: 180px; border: 1px dashed var(--line); border-radius: 8px; color: var(--muted); background: #fafbfc; }}
    .metric-grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 8px; margin: 12px 0; }}
    .metric-grid div {{ min-width: 0; border: 1px solid var(--line); border-radius: 8px; padding: 9px; background: #fbfcfe; }}
    dt {{ color: var(--muted); font-size: 12px; }}
    dd {{ margin: 0; overflow-wrap: anywhere; }}
    .warnings {{ margin: 10px 0; color: #4b5665; font-size: 13px; }}
    .warnings strong {{ display: block; margin-bottom: 2px; color: var(--ink); }}
    .warning-list {{ display: flex; flex-wrap: wrap; gap: 6px; }}
    .warning-chip {{ display: inline-flex; gap: 5px; align-items: center; border: 1px solid #e4c96f; border-radius: 999px; padding: 4px 8px; background: #fff8e5; color: #725000; font-size: 12px; }}
    .warning-chip.ok {{ border-color: #bdd8c8; background: #effaf3; color: var(--pass-ink); }}
    .file-chip {{ color: var(--accent-ink); background: #f7fbfc; }}
    .plots {{ margin-top: 12px; }}
    .plots summary {{ cursor: pointer; color: var(--muted); font-weight: 700; }}
    .thumbs, .contact-sheets {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; margin-top: 10px; }}
    figure {{ margin: 0; border: 1px solid var(--line); border-radius: 8px; overflow: hidden; background: #fff; }}
    img {{ display: block; width: 100%; height: auto; }}
    .plot-thumb img {{ aspect-ratio: 16 / 10; object-fit: contain; background: #f8fafc; }}
    figcaption {{ padding: 7px 9px; color: var(--muted); font-size: 12px; }}
    .controls {{ display: flex; flex-wrap: wrap; gap: 6px; align-items: center; margin-bottom: 10px; }}
    input[type="search"] {{ flex: 1 1 240px; min-height: 34px; border: 1px solid var(--line); border-radius: 8px; padding: 6px 10px; font-size: 13px; }}
    button {{ min-height: 34px; border: 1px solid var(--line); border-radius: 8px; background: #fff; color: var(--ink); padding: 5px 9px; cursor: pointer; font-weight: 650; font-size: 12px; }}
    button.active {{ background: var(--accent); border-color: var(--accent); color: #fff; }}
    .table-wrap {{ overflow-x: auto; border: 1px solid var(--line); border-radius: 8px; background: #fff; box-shadow: 0 1px 2px rgba(25, 38, 55, 0.04); }}
    .summary-table {{ min-width: 1120px; width: 100%; border-collapse: separate; border-spacing: 0; table-layout: fixed; font-size: 12.5px; }}
    .summary-table th, .summary-table td {{ border-bottom: 1px solid var(--line); padding: 8px 9px; text-align: left; vertical-align: middle; }}
    .summary-table th {{ position: sticky; top: 0; background: #edf3f7; z-index: 3; color: #425165; font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; }}
    .summary-table tbody tr.summary-main:nth-of-type(4n+1) td {{ background: #fbfcfd; }}
    .summary-table tbody tr.summary-main:hover td {{ background: #f2f8fa; }}
    .summary-table .sticky-col {{ position: sticky; left: 0; z-index: 2; background: #fff; box-shadow: 1px 0 0 var(--line); }}
    .summary-table th.sticky-col {{ z-index: 4; background: #edf3f7; }}
    .summary-main td {{ height: 58px; }}
    .object-cell strong, .joint-cell strong, .motion-cell strong, .mode-cell strong, .status-cell strong {{ display: block; line-height: 1.15; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
    .object-cell span, .joint-cell span, .motion-cell span, .mode-cell span, .status-cell span {{ display: block; margin-top: 3px; color: var(--muted); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
    .verdict-cell {{ display: flex; flex-wrap: wrap; gap: 4px; align-items: center; }}
    .summary-table .verdict {{ padding: 3px 6px; font-size: 10.5px; }}
    .warning-cell span {{ display: block; max-width: 190px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; color: #4d5968; }}
    .links-cell {{ white-space: nowrap; }}
    .mini-link {{ display: inline-flex; align-items: center; justify-content: center; margin: 2px 3px 2px 0; min-width: 42px; border: 1px solid var(--line); border-radius: 999px; padding: 3px 7px; background: #f7fbfc; color: var(--accent-ink); font-size: 11px; font-weight: 750; }}
    .summary-detail td {{ padding: 0 9px 8px; background: #fff; }}
    .summary-detail .detail-label {{ padding-top: 8px; color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; }}
    .summary-detail details {{ border: 1px solid var(--line); border-radius: 8px; background: #fbfcfd; }}
    .summary-detail summary {{ cursor: pointer; padding: 7px 10px; color: var(--accent-ink); font-weight: 750; font-size: 12px; }}
    .detail-grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; padding: 10px; }}
    .detail-grid div {{ min-width: 0; }}
    .detail-grid strong {{ display: block; margin-bottom: 3px; }}
    .detail-grid p {{ margin: 0; color: #4d5968; overflow-wrap: anywhere; }}
    .report-grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; }}
    .report-card {{ display: block; border: 1px solid var(--line); border-radius: 8px; padding: 12px; background: #fff; }}
    .report-card strong, .report-card span {{ display: block; }}
    .report-card span {{ margin-top: 4px; color: var(--muted); font-size: 12px; overflow-wrap: anywhere; }}
    .report-card.unavailable {{ color: var(--muted); background: #f9fafb; }}
    .sheet img {{ max-height: 320px; object-fit: contain; background: #f8fafc; }}
    .missing-sheet {{ display: grid; place-items: center; min-height: 120px; border: 1px dashed var(--line); border-radius: 8px; color: var(--muted); }}
    .future-list {{ display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 8px; margin-top: 12px; }}
    .future-list span {{ border: 1px solid var(--line); border-radius: 8px; padding: 10px; background: #fbfcfe; text-align: center; font-weight: 700; font-size: 13px; }}
    .generator-notes {{ color: var(--muted); font-size: 13px; }}
    .bad {{ color: var(--fail-ink); }}
    @media (max-width: 1040px) {{ .hero-grid, .two-col, .object-grid, .schema-grid {{ grid-template-columns: 1fr; }} .fixes, .report-grid, .future-list {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
    @media (max-width: 720px) {{ .hero-inner, .page {{ padding-left: 16px; padding-right: 16px; }} .stats, .metric-grid, .thumbs, .contact-sheets, .fixes, .report-grid, .future-list {{ grid-template-columns: 1fr; }} .object-title {{ display: block; }} .object-badges {{ margin-top: 8px; }} }}
  </style>
</head>
<body>
  <header class="hero">
    <div class="hero-inner">
      <div class="hero-grid">
        <div>
          <p class="eyebrow">Final dataset report</p>
          <h1>Realistic manual-contact adaptive physics outputs.</h1>
          <p class="subtitle">True SAPIEN external-link-force simulations with one manually selected point and direction per object</p>
          <nav class="nav" aria-label="Page sections">
            <a href="#repo">Repository</a>
            <a href="#pipeline">Pipeline</a>
            <a href="#simulation-json">simulation.json</a>
            <a href="#objects">Objects</a>
            <a href="#summary">Table</a>
            <a href="#audit">Reports</a>
            <a href="#physics-mode">Physics mode</a>
          </nav>
          <div class="callout">
            <strong>Interpretation warning.</strong>
            Final mode: <code>manual_contact_global_impulse_decay</code>. Every object receives the same short 0.10-second push, then evolves passively with globally shared damping and friction until settled plus one second. Only the semantic contact point and opening direction vary. True SAPIEN external-link-force; dataset/SAPIEN units; calibrated Newtons: no.
          </div>
        </div>
        <div class="stats">{stats}</div>
      </div>
    </div>
  </header>
  <main class="page">
    <section id="repo" class="section-block two-col">
      <div class="panel">
        <p class="eyebrow">Repository</p>
        <h2>What this repository does</h2>
        <ul class="repo-list">
          <li>Loads articulated object assets and metadata from <code>final_dataset/</code>.</li>
          <li>Detects URDF joints and movable links for revolute, prismatic, and screw-style articulation.</li>
          <li>Selects semantic joints and contact strategies for difficult objects.</li>
          <li>Renders object articulation videos into <code>outputs/</code>.</li>
          <li>Writes <code>simulation.json</code> with identity, joint, motion, force-debug, and validation metadata.</li>
          <li>Writes diagnostics, validation tables, audit reports, and contact sheets.</li>
        </ul>
        {skipped_note}
      </div>
      <div class="panel">
        <h2>Folder tree</h2>
        <pre class="tree">repo/
  final_dataset/
    &lt;object_id&gt;/
  scripts/
    run_forcesapien_batch_final_dataset.py
    render_revolute_video.py
    render_prismatic_video.py
    render_screw_video.py
    build_index_html.py
  outputs/
    &lt;object&gt;_&lt;id&gt;_&lt;joint&gt;_manual_contact_global_impulse_decay_check/
      simulation.json
      final_video.mp4
      diagnostics/
        physics_timeseries.tsv
        physics_diagnostics.md
        q_qdot_qddot_by_frame.png
        force_torque_by_frame.png
        resistance_by_frame.png</pre>
      </div>
    </section>

    <section id="pipeline" class="section-block">
      <div class="section-heading">
        <p class="eyebrow">Pipeline</p>
        <h2>From dataset object to validated output</h2>
        <p>The current pipeline discovers an object, chooses its joint and contact semantics, renders a video, and writes machine-readable diagnostics for audit.</p>
      </div>
      <div class="flow">
        <div class="flow-step"><span>1</span><strong>Dataset object</strong><p>Read from <code>final_dataset/</code>.</p></div>
        <div class="flow-step"><span>2</span><strong>URDF / metadata</strong><p>Parse joints and moving links.</p></div>
        <div class="flow-step"><span>3</span><strong>Joint selection</strong><p>Select the intended DOF.</p></div>
        <div class="flow-step"><span>4</span><strong>Manual contact</strong><p>Select the application point and force direction.</p></div>
        <div class="flow-step"><span>5</span><strong>Renderer</strong><p>Use revolute, prismatic, or screw renderer.</p></div>
        <div class="flow-step"><span>6</span><strong>Video</strong><p>Generate <code>final_video.mp4</code>.</p></div>
        <div class="flow-step"><span>7</span><strong>simulation.json</strong><p>Write run metadata and measurements.</p></div>
        <div class="flow-step"><span>8</span><strong>Diagnostics</strong><p>Write TSV, markdown, and plots.</p></div>
        <div class="flow-step"><span>9</span><strong>Validation</strong><p>Summaries and audit reports.</p></div>
      </div>
      <div class="fixes">
        <div class="fix"><strong>Stapler 103111</strong>Uses <code>joint_1/link_1</code> so the lid opens upward.</div>
        <div class="fix"><strong>Refrigerator 10905</strong>Uses a plausible door free-edge or handle-like contact.</div>
        <div class="fix"><strong>Folding chair 102255</strong>Uses robust moving-surface contact.</div>
      </div>
    </section>

    <section id="simulation-json" class="section-block">
      <div class="section-heading">
        <p class="eyebrow">Schema guide</p>
        <h2>How to read simulation.json</h2>
        <p>Each object folder contains a JSON record for a true SAPIEN external-link-force simulation, tying together object identity, selected articulation, pulse geometry, joint-space motion, and physical-consistency validation.</p>
      </div>
      <div class="schema-grid">
        <details class="schema" open><summary>A. Identity and run status</summary><div><p>Use these fields to confirm which object/run produced the artifacts and whether the run completed cleanly.</p><ul>{field_list([("object_id", "dataset object identifier."), ("object_name", "human-readable object name."), ("status", "run completion state."), ("error_message", "error text if the run failed."), ("output_folder", "folder containing the generated artifacts.")])}</ul></div></details>
        <details class="schema" open><summary>B. Joint/articulation metadata</summary><div><p>These fields describe the selected degree of freedom. Revolute <code>q</code> is radians; prismatic <code>q</code> is meters. For torque/projection, interpret the joint axis in world frame.</p><ul>{field_list([("joint_type", "revolute, prismatic, or screw-style articulation."), ("selected_joint / joint_name / joint_index", "the chosen URDF joint."), ("selected_link / moving_link", "the link expected to move."), ("parent_link / child_link", "joint hierarchy when available."), ("joint_axis_urdf_local", "axis in URDF-local coordinates."), ("joint_axis_world", "axis transformed into world coordinates."), ("joint_origin_world", "world-space joint origin."), ("joint_limits", "allowed joint range."), ("q_unit", "radians for revolute, meters for prismatic.")])}</ul></div></details>
        <details class="schema"><summary>C. Motion data</summary><div><p><code>q</code> is joint-space motion, not pixel/image motion. <code>qdot</code> and <code>qddot</code> describe joint velocity and acceleration over frames.</p><ul>{field_list([("q_start", "initial joint position."), ("q_end", "final joint position."), ("delta_q", "total joint-space displacement."), ("per-frame q", "sampled joint position over time."), ("qdot", "joint velocity."), ("qddot", "joint acceleration.")])}</ul></div></details>
        <details class="schema"><summary>D. Force/contact data</summary><div><p>The final outputs use true SAPIEN external-link forces: the same 5-unit, 0.10 s pulse for every object, at a manually selected point and direction, followed only by passive dynamics. Damping is 2.0 and friction is 0.30 globally. Calibrated Newtons: no.</p><ul>{field_list([("final_mode", "manual_contact_global_impulse_decay."), ("force_policy", "fixed_global_impulse_decay; no per-object dynamics adaptation."), ("force_application_mode", "external_link_force for every final object."), ("force_magnitude / actual_force_magnitude", "5 dataset/SAPIEN force units for every object."), ("contact_point_world_at_pulse", "manually selected world-space point on the moving link."), ("force_direction_world_at_pulse", "manually selected direction/sign."), ("true_external_force_used", "true for every final output."), ("fallback_used", "false for every final output.")])}</ul></div></details>
        <details class="schema"><summary>E. Validation fields</summary><div><p>The strict final verdict is <code>PHYSICS_PLAUSIBLE_BUT_UNCALIBRATED</code> for all ten objects. Remaining warnings describe contact semantics or normalized dataset scale—not use of fake or generalized motion. A max-duration stop is always a hard failure.</p><ul>{field_list([("final_physical_consistency_verdict", "strict final physical-consistency result."), ("contact_semantic_verdict", "semantic confidence in the selected contact region."), ("torque_projection_verdict", "world-space recomputation check."), ("effective_joint_inertia_proxy", "diagnostic response proxy, not calibrated inertia."), ("qdot_decay_ratio", "final absolute qdot divided by peak absolute qdot."), ("settled", "whether final velocity meets the settle threshold."), ("duration_verdict", "settled-plus-hold pass or hard max-duration failure."), ("final_acceptance", "final pass/fail after enforcing duration policy."), ("warning_messages", "remaining dataset-scale or contact caveats.")])}</ul></div></details>
        <details class="schema"><summary>F. Interpretation rule</summary><div><p>The final dataset does not use <code>generalized_set_qf</code>. That mode is legacy/background only. Every displayed final result uses <code>external_link_force</code>, with force reported in dataset/SAPIEN units and calibrated Newtons explicitly marked <strong>no</strong>.</p></div></details>
      </div>
      <div class="panel example">
        <h3>Readable example</h3>
        <pre>{{
  "object_id": "103111",
  "object_name": "Stapler",
  "joint_type": "revolute",
  "selected_joint": "joint_1",
  "selected_link": "link_1",
  "q_unit": "radians",
  "final_mode": "manual_contact_global_impulse_decay",
  "force_application_mode": "external_link_force",
  "force_policy": "fixed_magnitude",
  "force_magnitude": 5.0,
  "actual_force_magnitude": 5.0,
  "per_object_force_adaptation": false,
  "force_duration_s": 0.2,
  "true_external_force_used": true,
  "fallback_used": false,
  "contact_strategy": "stapler_lid_front_top",
  "contact_source": "candidate_id",
  "force_direction_mode": "tangent_opening",
  "calibrated_newtons": false,
  "force_units": "dataset/SAPIEN units"
}}</pre>
      </div>
    </section>

    <section id="objects" class="section-block">
      <div class="section-heading">
        <p class="eyebrow">Visual results</p>
        <h2>Object result cards</h2>
        <p>Each card embeds the existing video, summarizes the selected articulation and validation verdicts, and links to the JSON, timeseries, diagnostics, and plots.</p>
      </div>
      <div class="object-grid">{cards}</div>
    </section>

    <section id="summary" class="section-block panel">
      <div class="section-heading">
        <p class="eyebrow">Search</p>
        <h2>Summary table</h2>
        <p>Filter by validation state, joint type, or force mode. All final objects use dataset/SAPIEN force units and calibrated Newtons are no.</p>
      </div>
      <div class="controls">
        <input id="search" type="search" placeholder="Search objects, joints, warnings, modes">
        <button class="active" data-filter="all">All</button>
        <button data-filter="PASS">PASS</button>
        <button data-filter="WARN">WARN</button>
        <button data-filter="FAIL">FAIL</button>
        <button data-filter="revolute">Revolute</button>
        <button data-filter="prismatic">Prismatic</button>
        <button data-filter="external_link_force">external_link_force</button>
      </div>
      <div class="table-wrap">
        <table id="object-table" class="summary-table">
          <colgroup>
            <col style="width: 150px">
            <col style="width: 160px">
            <col style="width: 135px">
            <col style="width: 165px">
            <col style="width: 215px">
            <col style="width: 85px">
            <col style="width: 100px">
            <col style="width: 210px">
            <col style="width: 135px">
          </colgroup>
          <thead><tr>
            <th class="sticky-col">Object</th><th>Joint</th><th>Motion</th><th>Mode</th><th>Verdicts</th><th>Video</th><th>Diagnostics</th><th>Main warning</th><th>Links</th>
          </tr></thead>
          <tbody>{table_rows(objects)}</tbody>
        </table>
      </div>
    </section>

    {reports}

    <section id="physics-mode" class="section-block panel">
      <p class="eyebrow">Physics status</p>
      <h2>Validated manual-contact fixed-force mode</h2>
      <p>All final outputs use <code>manual_contact_global_impulse_decay</code> with true <code>external_link_force</code>: the same 5-unit, 0.10-second pulse, damping 2.0, and friction 0.30 for every object. Force is exactly zero after the pulse and motion decelerates passively. There is no fixed six-second minimum: recording continues until settling plus one second, while reaching the 10-second maximum is a hard failure.</p>
      <p>The same magnitude 5.0 dataset/SAPIEN force units is used for every object. The force magnitude is fixed globally rather than adapted per object. Calibrated Newtons: no.</p>
      <p>Refrigerator can move more than Scissors despite slightly lower pulse torque because the loaded dataset articulation has a lower effective joint-inertia proxy. This comparison describes the loaded SAPIEN assets, not real appliances or tools.</p>
      <div class="future-list">
        <span>initial force pulse</span>
        <span>force removed</span>
        <span>passive motion</span>
        <span>damping/friction</span>
        <span>qdot decay and settling</span>
      </div>
    </section>

    <section class="section-block panel generator-notes">
      <h2>Generator notes</h2>
      <p>Objects included: {total}. Objects with videos: {videos}. Objects with complete diagnostics: {diagnostics}. Successful outputs: {successful}. PASS/WARN/FAIL object states: {pass_count}/{warn_count}/{fail_count}.</p>
      <p>Missing optional files among included objects: {esc(', '.join(missing_files) if missing_files else 'none')}.</p>
      <p>Missing important fields among included objects: {esc(', '.join(missing_fields) if missing_fields else 'none')}.</p>
      <p>This page is self-contained and uses relative links from the repository root. Open <code>index.html</code> locally; no web server or external CDN is required.</p>
    </section>
  </main>
  <script>
    const objects = {data_json};
    const rows = Array.from(document.querySelectorAll("#object-table tbody tr"));
    const search = document.getElementById("search");
    const buttons = Array.from(document.querySelectorAll("[data-filter]"));
    let activeFilter = "all";

    function rowMatches(row) {{
      const text = row.textContent.toLowerCase();
      const data = JSON.parse(row.dataset.row);
      const query = search.value.trim().toLowerCase();
      const queryOk = !query || text.includes(query);
      let filterOk = true;
      if (activeFilter === "PASS") filterOk = data.verdicts.split(" ").every(v => v === "PASS");
      else if (activeFilter === "WARN") filterOk = data.verdicts.includes("WARN");
      else if (activeFilter === "FAIL") filterOk = data.verdicts.includes("FAIL");
      else if (activeFilter !== "all") filterOk = data.jointType === activeFilter || data.forceMode === activeFilter;
      return queryOk && filterOk;
    }}

    function updateRows() {{
      rows.forEach(row => {{ row.style.display = rowMatches(row) ? "" : "none"; }});
    }}

    search.addEventListener("input", updateRows);
    buttons.forEach(button => {{
      button.addEventListener("click", () => {{
        activeFilter = button.dataset.filter;
        buttons.forEach(b => b.classList.toggle("active", b === button));
        updateRows();
      }});
    }});
  </script>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="outputs", type=Path)
    parser.add_argument("--index-path", default="index.html", type=Path)
    args = parser.parse_args()

    repo_root = Path.cwd()
    output_root = args.output_root if args.output_root.is_absolute() else repo_root / args.output_root
    index_path = args.index_path if args.index_path.is_absolute() else repo_root / args.index_path
    summary_rows = read_summary(output_root / "final_manual_contact_global_impulse_decay_table.tsv")
    canonical_ids = set(summary_rows)

    objects: list[dict[str, Any]] = []
    skipped: list[str] = []
    for sim_path in sorted(output_root.glob("*_manual_contact_global_impulse_decay_check/simulation.json")):
        sim, _error = load_simulation(sim_path)
        object_id = next((part for part in sim_path.parent.name.split("_") if part.isdigit()), sim_path.parent.name)
        if sim is not None:
            object_id = object_id_from_sim(sim, sim_path.parent)
        if canonical_ids and object_id not in canonical_ids:
            skipped.append(rel(sim_path.parent, repo_root))
            continue
        objects.append(make_object(sim_path, repo_root, summary_rows))

    objects.sort(key=lambda obj: int(obj["object_id"]) if str(obj.get("object_id", "")).isdigit() else str(obj.get("object_id", "")))
    rendered = build_html(objects, output_root, repo_root, skipped)
    rendered = "\n".join(line.rstrip() for line in rendered.splitlines()) + "\n"
    index_path.write_text(rendered, encoding="utf-8")

    missing_files = sorted({name for obj in objects for name in obj.get("missing_files", [])})
    missing_fields = sorted({name for obj in objects for name in obj.get("missing_fields", [])})
    print(f"Wrote {rel(index_path, repo_root)}")
    print(f"Included objects: {len(objects)}")
    print(f"Objects with videos: {sum(1 for obj in objects if obj.get('files', {}).get('final_video'))}")
    print(f"Objects with complete diagnostics: {sum(1 for obj in objects if as_bool(obj.get('diagnostics_complete')) is True)}")
    print(f"Missing files: {', '.join(missing_files) if missing_files else 'none'}")
    print(f"Missing fields: {', '.join(missing_fields) if missing_fields else 'none'}")
    print(f"Skipped non-final simulation folders: {len(skipped)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
