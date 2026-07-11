"""
JSON schema definitions and serialization for video analysis outputs.

Phase 0: extended analysis.json (bbox + keypoints per frame), version fields,
and empty stub templates for calibration / sections / metrics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, TypedDict

from .pose_analyzer import CameraAngle, FrameAnalysis, KeypointData, KP

# ── Version constants ─────────────────────────────────────────────────────────

SCHEMA_VERSION = 2
DERIVED_VERSION_INITIAL = 0

# Stable keypoint order for JSON arrays (11 COCO points used by the pipeline).
KEYPOINT_NAMES: tuple[str, ...] = tuple(KP.keys())

# Phase labels for sections.json (filled by section_analyzer in Phase 3).
SECTION_PHASES: tuple[str, ...] = (
    "approach",
    "hop_1",
    "hop_2",
    "hop_3",
    "hop_4",
    "final_jump",
    "landing",
)


# ── TypedDict stubs (documentation + type hints) ──────────────────────────────


class KeypointEntry(TypedDict):
    name: str
    x: float
    y: float
    conf: float


class FrameRecord(TypedDict, total=False):
    frame_idx: int
    timestamp_s: float
    person_detected: bool
    person_bbox: Optional[list[float]]
    keypoints: list[KeypointEntry]
    track_overlap: Optional[float]
    athlete_state: Optional[str]
    position_s: Optional[float]
    predicted_bbox: Optional[list[float]]
    track_id: Optional[int]
    camera_angle: str
    shoulder_ratio: float
    angle_confidence: float
    keypoints_valid: int
    quality_score: float
    usable_for_analysis: bool
    has_mask: bool
    mask_area_px: int
    torso_height_px: float
    shoulder_width_px: float
    body_height_px: float
    appearance_sim: float
    tracking_source: str
    manually_corrected: bool
    correction_source: str


class CalibrationJson(TypedDict, total=False):
    """Track geometry — populated via UI calibration (Phase 1)."""

    schema_version: int
    track_polygon: list[list[float]]   # [[x,y], ...] closed polygon
    corridor: list[list[float]]        # optional inner/outer bounds
    landing_zone: list[list[float]]    # sand pit polygon
    axis: dict[str, Any]               # e.g. origin, direction for 1D s
    keyframes: dict[str, int]          # named landmarks → frame_idx


class ContactRecord(TypedDict, total=False):
    index: int                         # 1–5
    frame_idx: int
    timestamp_s: float
    phase: str
    type: str                          # hop | landing
    surface: str                       # track | sand | unknown
    position_s: Optional[float]
    confidence: float
    foot: Optional[str]


class PhaseMarkerRecord(TypedDict, total=False):
    frame_idx: int
    phase: str
    timestamp_s: float
    pose_tag: str          # hop_contact | hop_flight | final_takeoff | feet_together
    source: str            # manual | propagated | auto
    confidence: float


class SectionsJson(TypedDict, total=False):
    """Phase segmentation and contact events (Phase 3)."""

    schema_version: int
    derived_version: int
    athlete_id: str
    phases: dict[str, dict[str, int]]  # phase → {start_frame, end_frame}
    contacts: list[ContactRecord]
    phase_markers: list[PhaseMarkerRecord]
    confidence: float
    notes: str


class MetricsJson(TypedDict, total=False):
    """Derived hop lengths, speeds, scale, consistency (Phase 4)."""

    schema_version: int
    derived_version: int
    athlete_id: str
    video_name: str
    hop_lengths_m: list[Optional[float]]   # hop 1–4 + final jump (compat)
    hop_lengths_px: list[Optional[float]]
    total_hops_px: Optional[float]
    total_hops_m: Optional[float]
    segments: list[dict[str, Any]]
    timing: dict[str, Any]
    scale: dict[str, Any]
    consistency: dict[str, Any]
    comparison: dict[str, Any]
    overrides: dict[str, Any]
    partial: dict[str, Any]


# ── Empty stub templates ──────────────────────────────────────────────────────


def empty_calibration() -> CalibrationJson:
    return {
        "schema_version": 1,
        "track_polygon": [],
        "corridor": [],
        "landing_zone": [],
        "axis": {},
        "keyframes": [],
    }


def empty_sections() -> SectionsJson:
    phases = {name: {"start_frame": None, "end_frame": None} for name in SECTION_PHASES}
    return {
        "schema_version": 2,
        "derived_version": DERIVED_VERSION_INITIAL,
        "phases": phases,
        "contacts": [],
        "phase_markers": [],
        "confidence": 0.0,
        "notes": "",
    }


def empty_metrics() -> MetricsJson:
    return {
        "schema_version": 2,
        "derived_version": DERIVED_VERSION_INITIAL,
        "hop_lengths_m": [None, None, None, None, None],
        "hop_lengths_px": [None, None, None, None, None],
        "total_hops_px": None,
        "total_hops_m": None,
        "segments": [],
        "timing": {
            "contact_intervals_s": [],
            "phase_durations_s": {},
            "contact_timestamps_s": [],
        },
        "scale": {
            "m_per_px": None,
            "source": "none",
            "notes": [],
            "meters_estimated": True,
            "hops_corridor_m": 10.0,
        },
        "consistency": {},
        "comparison": {},
        "overrides": {},
        "partial": {},
    }


# ── Serialization ─────────────────────────────────────────────────────────────


def serialize_keypoints(keypoints: dict[str, KeypointData]) -> list[KeypointEntry]:
    out: list[KeypointEntry] = []
    for name in KEYPOINT_NAMES:
        kp = keypoints.get(name)
        if kp is not None:
            out.append({
                "name": name,
                "x": round(kp.x, 2),
                "y": round(kp.y, 2),
                "conf": round(kp.conf, 4),
            })
        else:
            out.append({"name": name, "x": 0.0, "y": 0.0, "conf": 0.0})
    return out


def serialize_bbox(bbox: Optional[tuple]) -> Optional[list[float]]:
    if bbox is None:
        return None
    return [round(float(v), 2) for v in bbox]


def frame_analysis_to_dict(
    fa: FrameAnalysis,
    *,
    appearance_sim: float = 0.0,
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Convert a FrameAnalysis to an analysis.json frame record."""
    record: dict[str, Any] = {
        "frame_idx":           fa.frame_idx,
        "timestamp_s":         round(fa.timestamp_s, 3),
        "person_detected":     fa.person_detected,
        "person_bbox":         serialize_bbox(fa.person_bbox),
        "keypoints":           serialize_keypoints(fa.keypoints),
        "track_overlap":       None,
        "athlete_state":       None,
        "position_s":          None,
        "predicted_bbox":      None,
        "track_id":            fa.track_id,
        "camera_angle":        fa.camera_angle.value,
        "shoulder_ratio":      round(fa.shoulder_ratio, 4),
        "angle_confidence":    round(fa.angle_confidence, 4),
        "keypoints_valid":     fa.keypoints_valid_count,
        "quality_score":       fa.quality_score,
        "usable_for_analysis": fa.usable_for_analysis,
        "has_mask":            fa.has_mask,
        "mask_area_px":        fa.mask_area_px,
        "torso_height_px":     round(fa.torso_height_px, 1),
        "shoulder_width_px":   round(fa.shoulder_width_px, 1),
        "body_height_px":      round(fa.body_height_px, 1),
        "appearance_sim":      appearance_sim,
        "tracking_source":     fa.tracking_source,
        "manually_corrected":  fa.manually_corrected,
        "correction_source":   fa.correction_source,
    }
    if extra:
        record.update(extra)
    return record


def read_analysis_version(output_dir: Path) -> int:
    path = output_dir / "analysis.json"
    if not path.exists():
        return 0
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return int(data.get("analysis_version", 0))


def next_analysis_version(output_dir: Path) -> int:
    return read_analysis_version(output_dir) + 1


def build_analysis_document(
    *,
    video_path: str,
    video_info: dict,
    config: dict,
    summary: dict,
    frames: list[dict],
    output_dir: Path,
    analysis_pass: Optional[str] = None,
) -> dict[str, Any]:
    doc: dict[str, Any] = {
        "schema_version":   SCHEMA_VERSION,
        "analysis_version": next_analysis_version(output_dir),
        "derived_version":  DERIVED_VERSION_INITIAL,
        "video":            video_path,
        "video_info":       video_info,
        "config":           config,
        "summary":          summary,
        "frames":           frames,
    }
    if analysis_pass is not None:
        doc["pass"] = analysis_pass
    return doc


def write_json_if_missing(path: Path, data: dict) -> None:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)


def write_derived_stubs(output_dir: Path) -> None:
    """Create empty calibration/sections/metrics JSON if not yet present."""
    write_json_if_missing(output_dir / "calibration.json", empty_calibration())
    write_json_if_missing(output_dir / "sections.json", empty_sections())
    write_json_if_missing(output_dir / "metrics.json", empty_metrics())


def load_keypoints_from_record(frame_data: dict) -> dict[str, KeypointData]:
    """Restore keypoints dict from a stored frame record (backward compatible)."""
    keypoints: dict[str, KeypointData] = {}
    raw = frame_data.get("keypoints")
    if not raw:
        return keypoints
    for entry in raw:
        name = entry.get("name")
        if not name:
            continue
        keypoints[name] = KeypointData(
            name=name,
            x=float(entry.get("x", 0)),
            y=float(entry.get("y", 0)),
            conf=float(entry.get("conf", 0)),
        )
    return keypoints


def frame_record_to_analysis(frame_data: dict) -> FrameAnalysis:
    """Reconstruct FrameAnalysis from analysis.json frame entry."""
    from .pose_analyzer import _compute_quality, _estimate_camera_angle

    fa = FrameAnalysis(
        frame_idx=frame_data.get("frame_idx", 0),
        timestamp_s=frame_data.get("timestamp_s", 0.0),
    )
    fa.person_detected     = frame_data.get("person_detected", False)
    fa.track_id            = frame_data.get("track_id")
    fa.has_mask            = frame_data.get("has_mask", False)
    fa.mask_area_px        = frame_data.get("mask_area_px", 0)
    fa.quality_score       = frame_data.get("quality_score", 0.0)
    fa.usable_for_analysis = frame_data.get("usable_for_analysis", False)
    fa.manually_corrected  = frame_data.get("manually_corrected", False)
    fa.correction_source   = frame_data.get("correction_source", "auto")
    fa.tracking_source     = frame_data.get("tracking_source", "bytetrack")
    fa.shoulder_ratio      = frame_data.get("shoulder_ratio", 0.0)
    fa.angle_confidence    = frame_data.get("angle_confidence", 0.0)
    fa.torso_height_px     = frame_data.get("torso_height_px", 0.0)
    fa.shoulder_width_px   = frame_data.get("shoulder_width_px", 0.0)
    fa.body_height_px      = frame_data.get("body_height_px", 0.0)

    try:
        fa.camera_angle = CameraAngle(frame_data.get("camera_angle", "UNKNOWN"))
    except ValueError:
        fa.camera_angle = CameraAngle.UNKNOWN

    bbox = frame_data.get("person_bbox")
    if bbox and len(bbox) >= 4:
        fa.person_bbox = tuple(bbox[:4])
        fa.person_bbox_area = float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))

    fa.keypoints = load_keypoints_from_record(frame_data)
    fa.keypoints_valid_count = frame_data.get(
        "keypoints_valid",
        sum(1 for kp in fa.keypoints.values() if kp.valid),
    )

    if fa.keypoints and fa.person_detected and fa.camera_angle == CameraAngle.UNKNOWN:
        _estimate_camera_angle(fa)
        _compute_quality(fa)

    return fa
