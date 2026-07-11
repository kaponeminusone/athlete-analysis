"""
Phase 4 metrics — speeds, hop lengths, scale, technique consistency.

Reads sections.json + analysis.json (+ optional calibration.json / overrides).
Works with contact timestamps alone; meters appear when scale or overrides exist.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .calibration import load_calibration
from .gt_contacts import score_pose_tag
from .phase_classifier import ATHLETES_ROOT, load_athlete_templates, load_prototypes
from .pose_features import FEATURE_NAMES, extract_pose_features
from .schemas import empty_metrics
from .track_scorer import _interp_keyframe, _project_position_s

METRICS_SCHEMA_VERSION = 2
ANKLE_CONF_MIN = 0.4

# Known track distance from first hop contact → landing in sand (same for all athletes).
DEFAULT_HOPS_CORRIDOR_M = 10.0
DEFAULT_VENUE_ID = "default"
VENUE_SCALE_PATH = Path("data") / "venue_scale.json"
GT_CONTACTS_PATH = Path("data") / "gt_contacts.json"

# hop_lengths_* slots: hop_1..hop_4 + final (final == last contact→landing when 5 contacts)
HOP_SEGMENT_IDS = ("hop_1", "hop_2", "hop_3", "hop_4", "final")
COMPARE_SEGMENT_IDS = ("hop_1", "hop_2", "hop_3", "hop_4")

# Relative |delta|/mean below this → ≈ ; else + (above mean) / − (below mean)
GENERAL_NEAR_REL = 0.08
GENERAL_LIMITED_MIN_SESSIONS = 2
POSE_LABEL_BUENA = 0.75
POSE_LABEL_REGULAR = 0.55
FINAL_FLIGHT_MAX_SAMPLES = 12

# How length_px was measured
LENGTH_METHOD_POSITION_S = "position_s"
LENGTH_METHOD_FOOT_AXIS = "foot_axis"
LENGTH_METHOD_FOOT_DX = "foot_dx"
LENGTH_METHOD_BBOX = "bbox_center"


def load_venue_hops_corridor(venue_id: str = DEFAULT_VENUE_ID) -> float:
    """Venue-default corridor length (m). Falls back to profile.json then 10.0."""
    scale_path = Path("venues") / venue_id / "scale.json"
    for path in (scale_path, VENUE_SCALE_PATH):
        data = _load_json(path)
        if data and data.get("hops_corridor_m") is not None:
            try:
                val = float(data["hops_corridor_m"])
                if val > 0:
                    return val
            except (TypeError, ValueError):
                pass
    # Optional field on venue color profile
    profile_path = Path("venues") / venue_id / "profile.json"
    profile = _load_json(profile_path)
    if profile and profile.get("hops_corridor_m") is not None:
        try:
            val = float(profile["hops_corridor_m"])
            if val > 0:
                return val
        except (TypeError, ValueError):
            pass
    return DEFAULT_HOPS_CORRIDOR_M


def save_venue_hops_corridor(hops_corridor_m: float, venue_id: str = DEFAULT_VENUE_ID) -> Path:
    """Persist corridor meters as venue default (scale.json + profile field if present)."""
    val = float(hops_corridor_m)
    scale_path = Path("venues") / venue_id / "scale.json"
    existing = _load_json(scale_path) or {}
    existing["hops_corridor_m"] = val
    existing["venue_id"] = venue_id
    _write_json(scale_path, existing)

    # Keep data/venue_scale.json in sync for simple tooling
    data_scale = _load_json(VENUE_SCALE_PATH) or {}
    data_scale["hops_corridor_m"] = val
    data_scale["venue_id"] = venue_id
    _write_json(VENUE_SCALE_PATH, data_scale)

    profile_path = Path("venues") / venue_id / "profile.json"
    profile = _load_json(profile_path)
    if profile is not None:
        profile["hops_corridor_m"] = val
        _write_json(profile_path, profile)
    return scale_path


def resolve_hops_corridor_m(overrides: Optional[dict] = None, venue_id: str = DEFAULT_VENUE_ID) -> float:
    """Override → venue default → 10.0 m."""
    overrides = overrides or {}
    raw = overrides.get("hops_corridor_m")
    if raw is None:
        # Legacy alias: total_length_m meant the same corridor
        raw = overrides.get("total_length_m")
    if raw is not None:
        try:
            val = float(raw)
            if val > 0:
                return val
        except (TypeError, ValueError):
            pass
    return load_venue_hops_corridor(venue_id)


def _foot_point(frame: Optional[dict]) -> Optional[tuple[float, float]]:
    """Ankle midpoint, else bbox foot proxy (bottom-center)."""
    if not frame:
        return None
    kps = {k["name"]: k for k in frame.get("keypoints") or []}
    ankles: list[tuple[float, float]] = []
    for name in ("l_ankle", "r_ankle"):
        kp = kps.get(name)
        if kp and float(kp.get("conf", 0)) >= ANKLE_CONF_MIN:
            ankles.append((float(kp["x"]), float(kp["y"])))
    if ankles:
        return (
            sum(p[0] for p in ankles) / len(ankles),
            sum(p[1] for p in ankles) / len(ankles),
        )
    bbox = frame.get("person_bbox")
    if bbox and len(bbox) >= 4:
        return ((bbox[0] + bbox[2]) * 0.5, bbox[3])
    return None


def _load_json(path: Path) -> Optional[dict[str, Any]]:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _round(v: Optional[float], nd: int = 4) -> Optional[float]:
    if v is None:
        return None
    return round(float(v), nd)


def _safe_div(num: Optional[float], den: Optional[float]) -> Optional[float]:
    if num is None or den is None or abs(den) < 1e-12:
        return None
    return float(num) / float(den)


def _frame_map(frames: list[dict]) -> dict[int, dict]:
    return {int(f["frame_idx"]): f for f in frames if f.get("frame_idx") is not None}


def _bbox_center(frame: Optional[dict]) -> Optional[tuple[float, float]]:
    if not frame:
        return None
    bbox = frame.get("person_bbox")
    if not bbox or len(bbox) < 4:
        return None
    return ((bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5)


def _axis_unit_px(axis: Optional[dict], width: int, height: int) -> Optional[tuple[float, float, float, float]]:
    """Return (ox, oy, ux, uy) in pixels, or None."""
    if not axis:
        return None
    origin = axis.get("origin")
    direction = axis.get("direction")
    if not origin or not direction or len(origin) < 2 or len(direction) < 2:
        return None
    ox, oy = float(origin[0]) * width, float(origin[1]) * height
    dx, dy = float(direction[0]), float(direction[1])
    norm = math.hypot(dx, dy)
    if norm < 1e-9:
        return None
    return ox, oy, dx / norm, dy / norm


def _project_xy_s(x: float, y: float, axis: Optional[dict], width: int, height: int) -> Optional[float]:
    unit = _axis_unit_px(axis, width, height)
    if unit is None:
        return None
    ox, oy, ux, uy = unit
    return (x - ox) * ux + (y - oy) * uy


def _axis_at(keyframes: list[dict], frame_idx: int) -> Optional[dict]:
    if not keyframes:
        return None
    kf = _interp_keyframe(keyframes, frame_idx)
    return kf.get("axis") if isinstance(kf, dict) else None


def _scale_at(keyframes: list[dict], frame_idx: int) -> Optional[dict]:
    if not keyframes:
        return None
    kf = _interp_keyframe(keyframes, frame_idx)
    return kf.get("scale") if isinstance(kf, dict) else None


def _approx_axis_from_polygon(cal: dict, width: int, height: int) -> Optional[dict]:
    """Fallback axis from track_polygon / first keyframe polygon (along long side)."""
    poly = None
    for kf in cal.get("keyframes") or []:
        poly = kf.get("track_polygon") or kf.get("corridor_polygon")
        if poly and len(poly) >= 2:
            break
    if not poly:
        poly = cal.get("track_polygon") or []
    if not poly or len(poly) < 2:
        return None
    pts = [(float(p[0]), float(p[1])) for p in poly if len(p) >= 2]
    if len(pts) < 2:
        return None
    # Prefer image-normalized polygons (0–1); treat as normalized if max <= 1.5
    max_v = max(max(abs(x), abs(y)) for x, y in pts)
    if max_v > 1.5:
        pts = [(x / width, y / height) for x, y in pts]
    # Longest edge direction
    best_len = -1.0
    best_dir = (1.0, 0.0)
    origin = pts[0]
    for i in range(len(pts)):
        a = pts[i]
        b = pts[(i + 1) % len(pts)]
        dx, dy = b[0] - a[0], b[1] - a[1]
        length = math.hypot(dx, dy)
        if length > best_len:
            best_len = length
            best_dir = (dx, dy)
            origin = a
    return {"origin": [origin[0], origin[1]], "direction": [best_dir[0], best_dir[1]]}


def resolve_m_per_px(
    cal: Optional[dict],
    overrides: Optional[dict],
    *,
    hop_lengths_px: Optional[list[Optional[float]]] = None,
    contact_foot_points: Optional[list[Optional[tuple[float, float]]]] = None,
    width: int = 1280,
    height: int = 720,
    keyframes: Optional[list[dict]] = None,
    ref_frame_idx: int = 0,
    hops_corridor_m: Optional[float] = None,
    venue_id: str = DEFAULT_VENUE_ID,
) -> dict[str, Any]:
    """
    Resolve meters-per-pixel scale.

    Priority:
      1. hops_corridor_m / total hops px (1st hop contact → landing) — primary
      2. Legacy per-hop hop_lengths_m overrides (API tolerance)
      3. Explicit overrides.m_per_px / known_distance_m with points
      4. Calibration keyframe scale.known_distance_m + point_a/b
      5. else unknown (m_per_px = None)

    Returns dict: m_per_px, source, notes, hops_corridor_m
    """
    overrides = overrides or {}
    keyframes = keyframes or ((cal or {}).get("keyframes") or [])
    notes: list[str] = []

    corridor = hops_corridor_m
    if corridor is None:
        corridor = resolve_hops_corridor_m(overrides, venue_id=venue_id)

    px_parts = [p for p in (hop_lengths_px or [])[:4] if p is not None]
    total_px = float(sum(px_parts)) if px_parts else None

    # 1. Primary: known corridor meters / total hops pixels
    if corridor is not None and corridor > 0 and total_px is not None and total_px > 1e-3:
        return {
            "m_per_px": float(corridor) / total_px,
            "source": "hops_corridor",
            "notes": [
                f"Escala: {corridor:g} m de 1er hop -> aterrizaje / {total_px:.1f} px",
            ],
            "hops_corridor_m": float(corridor),
        }

    # 2. Legacy per-hop overrides → average implied scale
    ov_hops = overrides.get("hop_lengths_m")
    if ov_hops and hop_lengths_px:
        ratios: list[float] = []
        for m_val, px_val in zip(ov_hops, hop_lengths_px):
            if m_val is None or px_val is None:
                continue
            try:
                m_f, px_f = float(m_val), float(px_val)
            except (TypeError, ValueError):
                continue
            if px_f > 1e-3 and m_f > 0:
                ratios.append(m_f / px_f)
        if ratios:
            return {
                "m_per_px": float(np.mean(ratios)),
                "source": "override_hop_lengths",
                "notes": [f"scale from {len(ratios)} hop override(s)"],
                "hops_corridor_m": float(corridor) if corridor else None,
            }

    # 3. Direct m_per_px or known_distance with points in overrides
    if overrides.get("m_per_px") is not None:
        try:
            return {
                "m_per_px": float(overrides["m_per_px"]),
                "source": "override_m_per_px",
                "notes": notes,
                "hops_corridor_m": float(corridor) if corridor else None,
            }
        except (TypeError, ValueError):
            pass

    known = overrides.get("known_distance_m")
    pa = overrides.get("point_a")
    pb = overrides.get("point_b")
    if known is not None and pa and pb and len(pa) >= 2 and len(pb) >= 2:
        # Points may be normalized (0–1) or pixels
        ax, ay = float(pa[0]), float(pa[1])
        bx, by = float(pb[0]), float(pb[1])
        if max(abs(ax), abs(ay), abs(bx), abs(by)) <= 1.5:
            ax, ay, bx, by = ax * width, ay * height, bx * width, by * height
        dist_px = math.hypot(bx - ax, by - ay)
        if dist_px > 1e-3:
            return {
                "m_per_px": float(known) / dist_px,
                "source": "override_known_distance",
                "notes": notes,
                "hops_corridor_m": float(corridor) if corridor else None,
            }

    # 4. Calibration keyframe scale
    scale = _scale_at(keyframes, ref_frame_idx)
    if scale and scale.get("known_distance_m") is not None:
        pa = scale.get("point_a")
        pb = scale.get("point_b")
        if pa and pb and len(pa) >= 2 and len(pb) >= 2:
            ax, ay = float(pa[0]) * width, float(pa[1]) * height
            bx, by = float(pb[0]) * width, float(pb[1]) * height
            dist_px = math.hypot(bx - ax, by - ay)
            if dist_px > 1e-3:
                return {
                    "m_per_px": float(scale["known_distance_m"]) / dist_px,
                    "source": "calibration_scale",
                    "notes": notes,
                    "hops_corridor_m": float(corridor) if corridor else None,
                }

    return {
        "m_per_px": None,
        "source": "none",
        "notes": ["no scale; need 5 contacts (hop px) or calibration"],
        "hops_corridor_m": float(corridor) if corridor else DEFAULT_HOPS_CORRIDOR_M,
    }


def _contact_length_px(
    frame_a: Optional[dict],
    frame_b: Optional[dict],
    *,
    pos_a: Optional[float],
    pos_b: Optional[float],
    axis: Optional[dict],
    width: int,
    height: int,
) -> tuple[Optional[float], str]:
    """Length between two contact frames in pixels (+ method)."""
    # Prefer projected track position
    if pos_a is not None and pos_b is not None:
        return abs(float(pos_b) - float(pos_a)), LENGTH_METHOD_POSITION_S

    foot_a = _foot_point(frame_a) if frame_a else None
    foot_b = _foot_point(frame_b) if frame_b else None

    if foot_a and foot_b and axis:
        sa = _project_xy_s(foot_a[0], foot_a[1], axis, width, height)
        sb = _project_xy_s(foot_b[0], foot_b[1], axis, width, height)
        if sa is not None and sb is not None:
            return abs(sb - sa), LENGTH_METHOD_FOOT_AXIS

    if foot_a and foot_b:
        # Approximate along track: prefer |Δx| (lateral camera); else Euclidean
        dx = abs(foot_b[0] - foot_a[0])
        dy = abs(foot_b[1] - foot_a[1])
        if dx >= dy * 0.5:
            return dx, LENGTH_METHOD_FOOT_DX
        return math.hypot(dx, dy), LENGTH_METHOD_FOOT_DX

    ca = _bbox_center(frame_a)
    cb = _bbox_center(frame_b)
    if ca and cb:
        return math.hypot(cb[0] - ca[0], cb[1] - ca[1]), LENGTH_METHOD_BBOX

    return None, "none"


def _pose_vector(frame: Optional[dict]) -> Optional[list[float]]:
    if not frame:
        return None
    feat = extract_pose_features(frame)
    if not feat.valid or not feat.vector:
        return None
    return [round(float(v), 4) for v in feat.vector]


def _cosine_sim(a: list[float], b: list[float]) -> Optional[float]:
    if len(a) != len(b) or not a:
        return None
    va = np.asarray(a, dtype=float)
    vb = np.asarray(b, dtype=float)
    na, nb = float(np.linalg.norm(va)), float(np.linalg.norm(vb))
    if na < 1e-9 or nb < 1e-9:
        return None
    return float(np.dot(va, vb) / (na * nb))


def _similarity_0_1(a: list[float], b: list[float]) -> Optional[float]:
    """Map cosine similarity from [-1,1] to [0,1], with Euclidean fallback blend."""
    cos = _cosine_sim(a, b)
    if cos is None:
        return None
    # Features are mostly [0,1]-ish; cosine already informative
    return max(0.0, min(1.0, (cos + 1.0) * 0.5)) if cos < 0 else max(0.0, min(1.0, cos))


def _load_athlete_sessions(athlete_id: str, exclude_video: Optional[str] = None) -> list[dict]:
    sessions_dir = ATHLETES_ROOT / athlete_id / "sessions"
    if not sessions_dir.exists():
        return []
    out: list[dict] = []
    for path in sorted(sessions_dir.glob("*.json")):
        if exclude_video and path.stem == exclude_video:
            continue
        data = _load_json(path)
        if data:
            out.append(data)
    return out


def _save_athlete_session(athlete_id: str, video_name: str, snapshot: dict) -> Path:
    path = ATHLETES_ROOT / athlete_id / "sessions" / f"{video_name}.json"
    _write_json(path, snapshot)
    return path


def _compute_consistency(
    *,
    athlete_id: Optional[str],
    video_name: str,
    contact_poses: list[Optional[list[float]]],
    contact_phases: list[str],
    intervals_s: list[Optional[float]],
) -> dict[str, Any]:
    """Compare pose at contacts + timing vs templates and prior sessions."""
    result: dict[str, Any] = {
        "athlete_id": athlete_id,
        "pose_scores": [],
        "timing": {},
        "overall": None,
        "sessions_compared": 0,
        "notes": [],
    }
    if not athlete_id:
        result["notes"].append("sin athlete_id; consistencia omitida")
        return result

    templates = load_athlete_templates(athlete_id)
    phases_t = (templates or {}).get("phases") or {}
    pose_tags_t = (templates or {}).get("pose_tags") or {}

    sessions = _load_athlete_sessions(athlete_id, exclude_video=video_name)
    result["sessions_compared"] = len(sessions)

    pose_scores: list[Optional[float]] = []
    for i, (vec, phase) in enumerate(zip(contact_poses, contact_phases)):
        entry: dict[str, Any] = {
            "contact_index": i + 1,
            "phase": phase,
            "vs_template": None,
            "vs_history": None,
            "score": None,
        }
        if not vec:
            pose_scores.append(None)
            result["pose_scores"].append(entry)
            continue

        scores: list[float] = []
        # Template centroid for phase (or hop_contact tag)
        centroid = None
        if phase in phases_t and phases_t[phase].get("centroid"):
            centroid = phases_t[phase]["centroid"]
        elif pose_tags_t.get("hop_contact", {}).get("centroid"):
            centroid = pose_tags_t["hop_contact"]["centroid"]
        if centroid and len(centroid) == len(vec):
            sim = _similarity_0_1(vec, [float(x) for x in centroid])
            entry["vs_template"] = _round(sim, 3)
            if sim is not None:
                scores.append(sim)

        # History: mean similarity to same contact index in prior sessions
        hist_sims: list[float] = []
        for sess in sessions:
            poses = sess.get("pose_vectors") or []
            if i < len(poses) and poses[i]:
                sim = _similarity_0_1(vec, poses[i])
                if sim is not None:
                    hist_sims.append(sim)
        if hist_sims:
            entry["vs_history"] = _round(float(np.mean(hist_sims)), 3)
            scores.append(float(np.mean(hist_sims)))

        score = float(np.mean(scores)) if scores else None
        entry["score"] = _round(score, 3)
        pose_scores.append(score)
        result["pose_scores"].append(entry)

    # Timing vs history means
    hist_intervals: list[list[float]] = []
    for sess in sessions:
        iv = sess.get("intervals_s") or []
        if iv:
            hist_intervals.append([float(x) if x is not None else float("nan") for x in iv])

    timing_devs: list[Optional[float]] = []
    timing_z: list[Optional[float]] = []
    if hist_intervals:
        arr = np.array(hist_intervals, dtype=float)
        means = np.nanmean(arr, axis=0)
        stds = np.nanstd(arr, axis=0)
        for j, dt in enumerate(intervals_s):
            if dt is None or j >= len(means) or math.isnan(means[j]):
                timing_devs.append(None)
                timing_z.append(None)
                continue
            mean = float(means[j])
            std = float(stds[j]) if j < len(stds) else 0.0
            pct = abs(dt - mean) / mean if abs(mean) > 1e-6 else None
            timing_devs.append(_round(pct, 3) if pct is not None else None)
            z = (dt - mean) / std if std > 1e-6 else 0.0
            timing_z.append(_round(z, 3))
    else:
        result["notes"].append("sin historial de sesiones para timing")

    # Timing score: 1 - mean relative deviation (clamped)
    valid_devs = [d for d in timing_devs if d is not None]
    timing_score = None
    if valid_devs:
        timing_score = max(0.0, min(1.0, 1.0 - float(np.mean(valid_devs))))

    result["timing"] = {
        "interval_deviation_pct": timing_devs,
        "interval_z": timing_z,
        "score": _round(timing_score, 3),
    }

    pose_valid = [s for s in pose_scores if s is not None]
    parts = []
    if pose_valid:
        parts.append(float(np.mean(pose_valid)))
    if timing_score is not None:
        parts.append(timing_score)
    if parts:
        result["overall"] = _round(float(np.mean(parts)), 3)
    elif not sessions and not phases_t:
        result["notes"].append("sin plantillas ni historial")

    return result


def _pose_quality_label(score: Optional[float]) -> Optional[str]:
    if score is None:
        return None
    if score >= POSE_LABEL_BUENA:
        return "buena"
    if score >= POSE_LABEL_REGULAR:
        return "regular"
    return "débil"


def _indicator_vs_mean(current: Optional[float], mean: Optional[float]) -> Optional[str]:
    """+ above mean, − below mean, ~ near mean (relative)."""
    if current is None or mean is None:
        return None
    if abs(mean) < 1e-9:
        if abs(current) < 1e-9:
            return "~"
        return "+" if current > 0 else "−"
    rel = (current - mean) / abs(mean)
    if abs(rel) <= GENERAL_NEAR_REL:
        return "~"
    return "+" if rel > 0 else "−"


def _load_all_athlete_sessions(exclude_video: Optional[str] = None) -> list[dict]:
    """Aggregate session snapshots across all athletes (general baseline)."""
    if not ATHLETES_ROOT.exists():
        return []
    out: list[dict] = []
    for athlete_dir in sorted(ATHLETES_ROOT.iterdir()):
        if not athlete_dir.is_dir():
            continue
        sessions_dir = athlete_dir / "sessions"
        if not sessions_dir.exists():
            continue
        for path in sorted(sessions_dir.glob("*.json")):
            if exclude_video and path.stem == exclude_video:
                continue
            data = _load_json(path)
            if data:
                out.append(data)
    return out


def _gt_interval_means_from_frames(
    gt_path: Path = GT_CONTACTS_PATH,
    fps_default: float = 30.0,
    exclude_video: Optional[str] = None,
) -> dict[str, list[float]]:
    """
    Derive contact-interval samples (s) from GT frame_idx groups per video.

    Uses fps_default when analysis timestamps are unavailable. Returns lists
    keyed by segment id hop_1..hop_4 (4 intervals for 5 contacts).
    """
    gt = _load_json(gt_path)
    if not gt:
        return {sid: [] for sid in COMPARE_SEGMENT_IDS}
    samples = gt.get("samples") or []
    by_video: dict[str, dict[str, int]] = {}
    for s in samples:
        vid = str(s.get("video_id") or s.get("analysis_dir") or "")
        if exclude_video and vid == exclude_video:
            continue
        phase = str(s.get("phase") or "")
        fidx = s.get("frame_idx")
        if not vid or fidx is None or phase not in ("hop_1", "hop_2", "hop_3", "hop_4", "landing"):
            continue
        by_video.setdefault(vid, {})[phase] = int(fidx)

    buckets: dict[str, list[float]] = {sid: [] for sid in COMPARE_SEGMENT_IDS}
    phase_order = ("hop_1", "hop_2", "hop_3", "hop_4", "landing")
    for vid, phases in by_video.items():
        if not all(p in phases for p in phase_order):
            continue
        # Prefer analysis timestamps if present
        analysis = _load_json(Path("output") / vid / "analysis.json") or {}
        fmap = _frame_map(analysis.get("frames") or [])
        fps = float((analysis.get("video_info") or {}).get("fps") or fps_default)
        times: list[float] = []
        for p in phase_order:
            fidx = phases[p]
            fr = fmap.get(fidx)
            if fr and fr.get("timestamp_s") is not None:
                times.append(float(fr["timestamp_s"]))
            else:
                times.append(fidx / fps)
        # 4 intervals: hop_1..hop_3 + hop_4→landing
        for i, sid in enumerate(COMPARE_SEGMENT_IDS):
            if i + 1 >= len(times):
                break
            dt = float(times[i + 1]) - float(times[i])
            if dt > 0:
                buckets[sid].append(dt)
    return buckets


def _build_general_baseline(exclude_video: Optional[str] = None) -> dict[str, Any]:
    """
    General reference means from all athlete sessions + GT contact intervals.

    Prefer session means for timing/speed/length; supplement timing from GT
    when session count is low.
    """
    sessions = _load_all_athlete_sessions(exclude_video=exclude_video)
    dt_buckets: dict[str, list[float]] = {sid: [] for sid in COMPARE_SEGMENT_IDS}
    speed_buckets: dict[str, list[float]] = {sid: [] for sid in COMPARE_SEGMENT_IDS}
    length_buckets: dict[str, list[float]] = {sid: [] for sid in COMPARE_SEGMENT_IDS}

    for sess in sessions:
        intervals = sess.get("intervals_s") or []
        speeds = sess.get("speeds_m_s") or []
        lengths = sess.get("lengths_m") or []
        for i, sid in enumerate(COMPARE_SEGMENT_IDS):
            if i < len(intervals) and intervals[i] is not None:
                try:
                    v = float(intervals[i])
                    if v > 0:
                        dt_buckets[sid].append(v)
                except (TypeError, ValueError):
                    pass
            if i < len(speeds) and speeds[i] is not None:
                try:
                    v = float(speeds[i])
                    if math.isfinite(v) and v > 0:
                        speed_buckets[sid].append(v)
                except (TypeError, ValueError):
                    pass
            # lengths_m: hop_1..hop_4 (+ optional final slot at index 4)
            if i < len(lengths) and lengths[i] is not None:
                try:
                    v = float(lengths[i])
                    if math.isfinite(v) and v > 0:
                        length_buckets[sid].append(v)
                except (TypeError, ValueError):
                    pass

    gt_dts = _gt_interval_means_from_frames(exclude_video=exclude_video)
    session_videos = {str(s.get("video_name") or "") for s in sessions}
    for sid, vals in gt_dts.items():
        # Supplement timing from GT when few session samples for this segment
        if len(dt_buckets[sid]) < GENERAL_LIMITED_MIN_SESSIONS:
            if not session_videos or len(dt_buckets[sid]) == 0:
                dt_buckets[sid].extend(vals)
            elif len(dt_buckets[sid]) < GENERAL_LIMITED_MIN_SESSIONS:
                dt_buckets[sid].extend(vals)

    means: dict[str, dict[str, Optional[float]]] = {}
    sample_counts: dict[str, int] = {}
    for sid in COMPARE_SEGMENT_IDS:
        n_dt = len(dt_buckets[sid])
        sample_counts[sid] = n_dt
        means[sid] = {
            "dt_s": _round(float(np.mean(dt_buckets[sid])), 4) if n_dt else None,
            "speed_m_s": (
                _round(float(np.mean(speed_buckets[sid])), 3) if speed_buckets[sid] else None
            ),
            "length_m": (
                _round(float(np.mean(length_buckets[sid])), 4) if length_buckets[sid] else None
            ),
        }

    n_sessions = len(sessions)
    max_samples = max(sample_counts.values()) if sample_counts else 0
    # Limited when few independent takes (sessions or GT videos)
    limited = max_samples < GENERAL_LIMITED_MIN_SESSIONS

    return {
        "sessions": n_sessions,
        "sample_counts": sample_counts,
        "means": means,
        "limited": limited,
        "source": "sessions+gt" if any(gt_dts[s] for s in COMPARE_SEGMENT_IDS) else "sessions",
    }


def _compute_vs_general(
    segments: list[dict[str, Any]],
    *,
    exclude_video: Optional[str] = None,
) -> dict[str, Any]:
    """Compare current take segments against general baseline means."""
    baseline = _build_general_baseline(exclude_video=exclude_video)
    means = baseline.get("means") or {}
    seg_by_id = {}
    for s in segments:
        sid = s.get("id")
        if sid in COMPARE_SEGMENT_IDS:
            seg_by_id[sid] = s
        elif sid == "final":
            # Last interval may be labeled "final" instead of hop_4
            seg_by_id.setdefault("hop_4", s)

    out_segments: list[dict[str, Any]] = []
    closeness: list[float] = []

    for sid in COMPARE_SEGMENT_IDS:
        seg = seg_by_id.get(sid) or {}
        m = means.get(sid) or {}
        dt = seg.get("dt_s")
        speed = seg.get("speed_m_s")
        length = seg.get("length_m")
        mean_dt = m.get("dt_s")
        mean_speed = m.get("speed_m_s")
        mean_length = m.get("length_m")

        dt_delta = _round(dt - mean_dt, 4) if dt is not None and mean_dt is not None else None
        speed_delta = (
            _round(speed - mean_speed, 3)
            if speed is not None and mean_speed is not None
            else None
        )
        length_delta = (
            _round(length - mean_length, 4)
            if length is not None and mean_length is not None
            else None
        )

        # Prefer timing indicator; fall back to speed then length
        indicator = _indicator_vs_mean(dt, mean_dt)
        if indicator is None:
            indicator = _indicator_vs_mean(speed, mean_speed)
        if indicator is None:
            indicator = _indicator_vs_mean(length, mean_length)

        if dt is not None and mean_dt is not None and abs(mean_dt) > 1e-9:
            closeness.append(max(0.0, min(1.0, 1.0 - abs(dt - mean_dt) / abs(mean_dt))))
        elif speed is not None and mean_speed is not None and abs(mean_speed) > 1e-9:
            closeness.append(max(0.0, min(1.0, 1.0 - abs(speed - mean_speed) / abs(mean_speed))))

        out_segments.append({
            "id": sid,
            "dt_s": _round(dt, 4) if dt is not None else None,
            "dt_mean_s": mean_dt,
            "dt_delta_s": dt_delta,
            "speed_m_s": _round(speed, 3) if speed is not None else None,
            "speed_mean_m_s": mean_speed,
            "speed_delta_ms": speed_delta,
            "length_m": _round(length, 4) if length is not None else None,
            "length_mean_m": mean_length,
            "length_delta_m": length_delta,
            "indicator": indicator,
        })

    overall = _round(float(np.mean(closeness)), 3) if closeness else None
    notes: list[str] = []
    if baseline.get("limited"):
        notes.append("Baseline general limitado")
    if baseline.get("sessions", 0) == 0 and not any(
        (baseline.get("sample_counts") or {}).get(s, 0) for s in COMPARE_SEGMENT_IDS
    ):
        notes.append("sin datos generales de timing")

    return {
        "overall": overall,
        "segments": out_segments,
        "limited": bool(baseline.get("limited")),
        "sessions_used": baseline.get("sessions", 0),
        "sample_counts": baseline.get("sample_counts") or {},
        "source": baseline.get("source"),
        "notes": notes,
    }


def _prototype_centroid(phase_or_tag: str) -> Optional[list[float]]:
    """Centroid from phase_prototypes phases or pose_tags."""
    proto = load_prototypes()
    phases = proto.get("phases") or {}
    tags = proto.get("pose_tags") or {}
    entry = phases.get(phase_or_tag) or tags.get(phase_or_tag)
    if not entry or not entry.get("centroid"):
        return None
    return [float(x) for x in entry["centroid"]]


def _score_pose_vs_prototype(
    vec: Optional[list[float]],
    *,
    phase: Optional[str] = None,
    pose_tag: Optional[str] = None,
) -> Optional[float]:
    """Blend GT pose_tag score with phase/tag centroid similarity."""
    if not vec:
        return None
    scores: list[float] = []
    if pose_tag:
        try:
            tag_s = float(score_pose_tag(vec, pose_tag))
            if tag_s > 0:
                scores.append(tag_s)
        except Exception:
            pass
    centroid = None
    if phase:
        centroid = _prototype_centroid(phase)
    if centroid is None and pose_tag:
        centroid = _prototype_centroid(pose_tag)
    if centroid and len(centroid) == len(vec):
        sim = _similarity_0_1(vec, centroid)
        if sim is not None:
            scores.append(sim)
    if not scores:
        return None
    return _round(float(np.mean(scores)), 3)


def _compute_pose_quality(
    *,
    contact_infos: list[dict[str, Any]],
    fmap: dict[int, dict],
) -> dict[str, Any]:
    """
    Pose quality at hop contacts (vs hop_contact / phase prototype) and along
    final flight (hop_4 → landing) vs hop_flight.
    """
    hops: list[dict[str, Any]] = []
    hop_contacts = [
        ci for ci in contact_infos
        if str(ci.get("phase") or "").startswith("hop") or ci.get("type") == "hop"
    ]
    # Prefer first 4 contacts as hops when typed loosely
    if len(hop_contacts) < 4 and len(contact_infos) >= 4:
        hop_contacts = contact_infos[:4]

    for i, ci in enumerate(hop_contacts[:4]):
        phase = str(ci.get("phase") or f"hop_{i + 1}")
        if phase == "hop" or not phase.startswith("hop"):
            phase = f"hop_{i + 1}"
        vec = _pose_vector(ci.get("frame"))
        score = _score_pose_vs_prototype(vec, phase=phase, pose_tag="hop_contact")
        hops.append({
            "phase": phase,
            "score": score,
            "label": _pose_quality_label(score),
            "frame_idx": ci.get("frame_idx"),
        })

    final_flight: dict[str, Any] = {
        "score": None,
        "label": None,
        "from_frame": None,
        "to_frame": None,
        "samples": [],
    }

    # last hop contact → landing contact
    landing_ci = None
    for ci in contact_infos:
        if str(ci.get("phase") or "") == "landing" or ci.get("type") == "landing":
            landing_ci = ci
            break
    if landing_ci is None and len(contact_infos) >= 5:
        landing_ci = contact_infos[4]

    hop4_ci = hop_contacts[3] if len(hop_contacts) >= 4 else (
        contact_infos[3] if len(contact_infos) >= 4 else None
    )

    if hop4_ci is not None and landing_ci is not None:
        f0 = int(hop4_ci["frame_idx"])
        f1 = int(landing_ci["frame_idx"])
        final_flight["from_frame"] = f0
        final_flight["to_frame"] = f1
        if f1 > f0:
            span = f1 - f0
            step = max(1, span // FINAL_FLIGHT_MAX_SAMPLES)
            sample_idxs = list(range(f0 + step, f1, step))[:FINAL_FLIGHT_MAX_SAMPLES]
            # Always include a mid-flight and near-landing (not post-landing)
            mid = f0 + span // 2
            near_land = max(f0 + 1, f1 - max(1, step // 2))
            for extra in (mid, near_land):
                if f0 < extra < f1 and extra not in sample_idxs:
                    sample_idxs.append(extra)
            sample_idxs = sorted(set(sample_idxs))

            sample_scores: list[float] = []
            samples_out: list[dict[str, Any]] = []
            for fidx in sample_idxs:
                frame = fmap.get(fidx)
                vec = _pose_vector(frame)
                sc = _score_pose_vs_prototype(vec, pose_tag="hop_flight")
                # Soft blend with biomechanical flight_score when available
                if frame and vec:
                    feat = extract_pose_features(frame)
                    if feat.valid and feat.flight_score is not None:
                        flight_bio = float(feat.flight_score)
                        if sc is not None:
                            sc = _round(0.7 * sc + 0.3 * flight_bio, 3)
                        else:
                            sc = _round(flight_bio, 3)
                if sc is not None:
                    sample_scores.append(sc)
                    samples_out.append({"frame_idx": fidx, "score": sc})
            final_flight["samples"] = samples_out
            if sample_scores:
                avg = float(np.mean(sample_scores))
                final_flight["score"] = _round(avg, 3)
                final_flight["label"] = _pose_quality_label(avg)

    hop_scores = [h["score"] for h in hops if h.get("score") is not None]
    parts = list(hop_scores)
    if final_flight.get("score") is not None:
        parts.append(float(final_flight["score"]))
    overall = _round(float(np.mean(parts)), 3) if parts else None

    notes: list[str] = []
    if not _prototype_centroid("hop_contact") and not (load_prototypes().get("pose_tags") or {}).get("hop_contact"):
        notes.append("sin prototipo hop_contact")
    if not hops and not final_flight.get("score"):
        notes.append("sin poses válidas en contactos")

    return {
        "overall": overall,
        "hops": hops,
        "final_flight": final_flight,
        "notes": notes,
    }


def compute_metrics(
    output_dir: Path | str,
    athlete_id: Optional[str] = None,
    *,
    save: bool = True,
    persist_history: bool = True,
) -> dict[str, Any]:
    """
    Compute Phase 4 metrics from sections + analysis (+ calibration / overrides).

    Writes metrics.json when save=True. Snapshots athlete session history when
    athlete_id is set and persist_history=True.
    """
    output_dir = Path(output_dir)
    video_name = output_dir.name

    analysis = _load_json(output_dir / "analysis.json") or {}
    sections = _load_json(output_dir / "sections.json") or {}
    cal = load_calibration(output_dir) or {}
    prev_metrics = _load_json(output_dir / "metrics.json") or empty_metrics()
    overrides = dict(prev_metrics.get("overrides") or {})

    frames = analysis.get("frames") or []
    fmap = _frame_map(frames)
    vi = analysis.get("video_info") or {}
    width = int(vi.get("width", 1280))
    height = int(vi.get("height", 720))

    derived = int(sections.get("derived_version") or analysis.get("derived_version") or 0)
    aid = athlete_id or sections.get("athlete_id")

    keyframes = sorted(cal.get("keyframes") or [], key=lambda k: k.get("frame_idx", 0))
    fallback_axis = _approx_axis_from_polygon(cal, width, height)

    contacts = sorted(
        list(sections.get("contacts") or []),
        key=lambda c: (float(c.get("timestamp_s", 0)), int(c.get("frame_idx", 0))),
    )

    # Enrich contact positions
    contact_infos: list[dict[str, Any]] = []
    foot_points: list[Optional[tuple[float, float]]] = []
    for c in contacts:
        fidx = int(c.get("frame_idx", -1))
        frame = fmap.get(fidx)
        axis = _axis_at(keyframes, fidx) or fallback_axis
        pos_s = c.get("position_s")
        if pos_s is None and frame is not None:
            pos_s = frame.get("position_s")
        if pos_s is None and frame is not None:
            bbox = frame.get("person_bbox")
            if bbox and len(bbox) >= 4 and axis:
                pos_s = _project_position_s(tuple(bbox[:4]), axis, width, height)
            foot = _foot_point(frame)
            if pos_s is None and foot and axis:
                pos_s = _project_xy_s(foot[0], foot[1], axis, width, height)
        foot = _foot_point(frame) if frame else None
        foot_points.append(foot)
        contact_infos.append({
            "index": c.get("index"),
            "frame_idx": fidx,
            "timestamp_s": float(c.get("timestamp_s", frame.get("timestamp_s", 0) if frame else 0)),
            "phase": c.get("phase") or ("landing" if c.get("type") == "landing" else "hop"),
            "type": c.get("type"),
            "position_s": float(pos_s) if pos_s is not None else None,
            "axis": axis,
            "frame": frame,
        })

    # Phase durations from sections.phases
    phases = sections.get("phases") or {}
    phase_durations_s: dict[str, Optional[float]] = {}
    for name, bounds in phases.items():
        if not isinstance(bounds, dict):
            phase_durations_s[name] = None
            continue
        sf, ef = bounds.get("start_frame"), bounds.get("end_frame")
        if sf is None or ef is None:
            phase_durations_s[name] = None
            continue
        fa, fb = fmap.get(int(sf)), fmap.get(int(ef))
        if fa and fb:
            phase_durations_s[name] = _round(
                float(fb.get("timestamp_s", 0)) - float(fa.get("timestamp_s", 0)), 4,
            )
        else:
            # Fallback via fps
            fps = float(vi.get("fps") or 30)
            phase_durations_s[name] = _round((int(ef) - int(sf)) / fps, 4)

    # Build segments: optional approach + consecutive contact pairs
    segments: list[dict[str, Any]] = []
    contact_intervals_s: list[Optional[float]] = []

    # Approach: phase start → first contact
    approach_bounds = phases.get("approach") or {}
    if contact_infos and approach_bounds.get("start_frame") is not None:
        start_f = fmap.get(int(approach_bounds["start_frame"]))
        end_info = contact_infos[0]
        if start_f is not None:
            dt = float(end_info["timestamp_s"]) - float(start_f.get("timestamp_s", 0))
            axis = end_info.get("axis") or _axis_at(keyframes, end_info["frame_idx"]) or fallback_axis
            pos_a = start_f.get("position_s")
            if pos_a is None and start_f.get("person_bbox") and axis:
                pos_a = _project_position_s(tuple(start_f["person_bbox"][:4]), axis, width, height)
            length_px, method = _contact_length_px(
                start_f, end_info.get("frame"),
                pos_a=pos_a, pos_b=end_info.get("position_s"),
                axis=axis, width=width, height=height,
            )
            segments.append({
                "id": "approach",
                "label": "approach",
                "from_frame": int(approach_bounds["start_frame"]),
                "to_frame": end_info["frame_idx"],
                "dt_s": _round(max(0.0, dt), 4),
                "length_px": _round(length_px, 2),
                "length_m": None,  # filled after scale
                "speed_px_s": None,
                "speed_m_s": None,
                "length_method": method,
            })

    # Contact→contact pairs → hop_1..hop_4 (and final = last pair)
    hop_lengths_px: list[Optional[float]] = [None, None, None, None, None]
    hop_lengths_m: list[Optional[float]] = [None, None, None, None, None]
    hop_methods: list[Optional[str]] = [None, None, None, None, None]

    for i in range(len(contact_infos) - 1):
        a, b = contact_infos[i], contact_infos[i + 1]
        dt = float(b["timestamp_s"]) - float(a["timestamp_s"])
        contact_intervals_s.append(_round(dt, 4))
        axis = b.get("axis") or a.get("axis") or fallback_axis
        length_px, method = _contact_length_px(
            a.get("frame"), b.get("frame"),
            pos_a=a.get("position_s"), pos_b=b.get("position_s"),
            axis=axis, width=width, height=height,
        )
        seg_id = HOP_SEGMENT_IDS[i] if i < 4 else f"interval_{i + 1}"
        # Map first 4 pairs to hop_1..hop_4; last pair also fills "final"
        if i < 4:
            hop_lengths_px[i] = _round(length_px, 2)
            hop_methods[i] = method
        if i == len(contact_infos) - 2:
            # final jump slot = last contact interval (typically hop_4 → landing)
            hop_lengths_px[4] = _round(length_px, 2)
            hop_methods[4] = method
            seg_id = "final" if i >= 3 else seg_id

        label = HOP_SEGMENT_IDS[i] if i < 4 else "final"
        if i == len(contact_infos) - 2 and i < 3:
            label = "final"
        segments.append({
            "id": label if i < 4 else "final",
            "label": label if i < 4 else "final",
            "from_frame": a["frame_idx"],
            "to_frame": b["frame_idx"],
            "from_contact": a.get("index"),
            "to_contact": b.get("index"),
            "dt_s": _round(max(0.0, dt), 4),
            "length_px": _round(length_px, 2),
            "length_m": None,
            "speed_px_s": None,
            "speed_m_s": None,
            "length_method": method,
        })

    # Deduplicate if hop_4 and final would both appear as separate identical rows:
    # keep hop_1..hop_{n-1} + final for the last interval when n contacts == 5
    # (segments already has one row per pair; hop_4 row for i=3 is the final interval)
    # Relabel last hop segment as both hop_4 and final in hop_lengths; UI uses segments.

    # Ensure hop_4 slot filled when we have 5 contacts (4 intervals): slots 0..3 + final@4
    if len(contact_infos) >= 5 and hop_lengths_px[3] is not None:
        hop_lengths_px[4] = hop_lengths_px[3]  # final == hop_4→landing

    # Venue / override corridor (default 10 m). Primary scale source.
    corridor_m = resolve_hops_corridor_m(overrides)
    if "hops_corridor_m" not in overrides:
        overrides["hops_corridor_m"] = corridor_m

    # Resolve scale (corridor_m / total hops px when contacts exist)
    scale_info = resolve_m_per_px(
        cal,
        overrides,
        hop_lengths_px=hop_lengths_px,
        contact_foot_points=foot_points,
        width=width,
        height=height,
        keyframes=keyframes,
        ref_frame_idx=contact_infos[0]["frame_idx"] if contact_infos else 0,
        hops_corridor_m=corridor_m,
    )
    m_per_px = scale_info.get("m_per_px")
    corridor_m = float(scale_info.get("hops_corridor_m") or corridor_m)

    # Apply meters + speeds (proportional to px via uniform m_per_px)
    for seg in segments:
        px = seg.get("length_px")
        dt = seg.get("dt_s")
        if m_per_px is not None and px is not None:
            seg["length_m"] = _round(px * m_per_px, 4)
        seg["speed_px_s"] = _round(_safe_div(px, dt), 2)
        seg["speed_m_s"] = _round(_safe_div(seg.get("length_m"), dt), 3)

    for i in range(5):
        px = hop_lengths_px[i]
        if m_per_px is not None and px is not None:
            hop_lengths_m[i] = _round(px * m_per_px, 4)

    # Legacy: explicit per-hop meters still win for display if present
    if overrides.get("hop_lengths_m"):
        for i, m_val in enumerate(overrides["hop_lengths_m"]):
            if i < 5 and m_val is not None:
                try:
                    hop_lengths_m[i] = round(float(m_val), 4)
                except (TypeError, ValueError):
                    pass

    # Totals from first hop contact → landing (= corridor reference when scaled)
    unique_hop_px = hop_lengths_px[:4]
    total_hops_px = _round(sum(p for p in unique_hop_px if p is not None), 2) if any(
        p is not None for p in unique_hop_px
    ) else None
    total_hops_m = None
    if total_hops_px is not None and len(contact_infos) >= 5:
        total_hops_m = _round(corridor_m, 4)
    elif m_per_px is not None and total_hops_px is not None:
        total_hops_m = _round(total_hops_px * m_per_px, 4)
    elif all(m is not None for m in hop_lengths_m[:4]):
        total_hops_m = _round(sum(hop_lengths_m[:4]), 4)  # type: ignore[arg-type]

    # Pose vectors at contacts
    contact_poses = [_pose_vector(ci.get("frame")) for ci in contact_infos]
    contact_phases = [str(ci.get("phase") or "hop") for ci in contact_infos]

    consistency = _compute_consistency(
        athlete_id=aid,
        video_name=video_name,
        contact_poses=contact_poses,
        contact_phases=contact_phases,
        intervals_s=contact_intervals_s,
    )

    vs_general = _compute_vs_general(segments, exclude_video=video_name)
    pose_quality = _compute_pose_quality(contact_infos=contact_infos, fmap=fmap)
    comparison = {
        "vs_general": vs_general,
        "pose_quality": pose_quality,
    }

    src = scale_info.get("source") or ""
    meters_estimated = src not in ("hops_corridor", "override_hop_lengths", "override_m_per_px", "override_known_distance")
    if m_per_px is None:
        meters_estimated = True

    metrics: dict[str, Any] = {
        "schema_version": METRICS_SCHEMA_VERSION,
        "derived_version": derived,
        "athlete_id": aid,
        "video_name": video_name,
        "hop_lengths_m": hop_lengths_m,
        "hop_lengths_px": hop_lengths_px,
        "total_hops_px": total_hops_px,
        "total_hops_m": total_hops_m,
        "segments": segments,
        "timing": {
            "contact_intervals_s": contact_intervals_s,
            "phase_durations_s": phase_durations_s,
            "contact_timestamps_s": [_round(ci["timestamp_s"], 4) for ci in contact_infos],
        },
        "scale": {
            "m_per_px": _round(m_per_px, 8) if m_per_px is not None else None,
            "source": scale_info.get("source"),
            "notes": scale_info.get("notes") or [],
            "meters_estimated": meters_estimated,
            "hops_corridor_m": _round(corridor_m, 4),
        },
        "consistency": consistency,
        "comparison": comparison,
        "overrides": overrides,
        "partial": {
            "contact_count": len(contact_infos),
            "expected_contacts": 5,
            "has_axis": any(ci.get("axis") for ci in contact_infos) or fallback_axis is not None,
            "has_scale": m_per_px is not None,
            "length_methods": hop_methods,
            "feature_names": list(FEATURE_NAMES),
        },
    }

    if save:
        _write_json(output_dir / "metrics.json", metrics)

    if persist_history and aid and contact_infos:
        snapshot = {
            "schema_version": 1,
            "video_name": video_name,
            "athlete_id": aid,
            "derived_version": derived,
            "timestamps_s": [_round(ci["timestamp_s"], 4) for ci in contact_infos],
            "intervals_s": contact_intervals_s,
            "lengths_px": hop_lengths_px,
            "lengths_m": hop_lengths_m,
            "total_hops_px": total_hops_px,
            "total_hops_m": total_hops_m,
            "speeds_px_s": [s.get("speed_px_s") for s in segments if s.get("id") != "approach"],
            "speeds_m_s": [s.get("speed_m_s") for s in segments if s.get("id") != "approach"],
            "pose_vectors": contact_poses,
            "contact_phases": contact_phases,
            "consistency_overall": consistency.get("overall"),
            "feature_names": list(FEATURE_NAMES),
        }
        _save_athlete_session(aid, video_name, snapshot)

    return metrics


def apply_overrides(
    output_dir: Path | str,
    overrides_update: dict[str, Any],
    *,
    athlete_id: Optional[str] = None,
    update_calibration: bool = True,
    persist_venue_scale: bool = True,
    venue_id: str = DEFAULT_VENUE_ID,
) -> dict[str, Any]:
    """
    Merge overrides into metrics.json, optionally persist venue corridor,
    then recompute metrics.

    Preferred UI field: ``hops_corridor_m`` (single corridor length in meters).
    Legacy fields (hop_lengths_m, total_length_m, …) remain accepted.
    """
    output_dir = Path(output_dir)
    prev = _load_json(output_dir / "metrics.json") or empty_metrics()
    merged = dict(prev.get("overrides") or {})

    for key in (
        "hops_corridor_m",
        "hop_lengths_m",
        "total_length_m",
        "known_distance_m",
        "point_a",
        "point_b",
        "m_per_px",
        "notes",
    ):
        if key in overrides_update and overrides_update[key] is not None:
            merged[key] = overrides_update[key]

    # Clear keys explicitly set to null via clear flags
    if overrides_update.get("clear"):
        for k in overrides_update["clear"]:
            merged.pop(k, None)

    # Prefer corridor; drop stale total_length_m if corridor set (same meaning)
    if merged.get("hops_corridor_m") is not None and "total_length_m" in merged:
        # Keep both for back-compat readers, but sync values
        try:
            merged["total_length_m"] = float(merged["hops_corridor_m"])
        except (TypeError, ValueError):
            pass

    prev["overrides"] = merged
    _write_json(output_dir / "metrics.json", prev)

    if persist_venue_scale and merged.get("hops_corridor_m") is not None:
        try:
            save_venue_hops_corridor(float(merged["hops_corridor_m"]), venue_id=venue_id)
        except (TypeError, ValueError):
            pass

    # Optionally stamp calibration scale from known_distance points
    if update_calibration and merged.get("known_distance_m") and merged.get("point_a") and merged.get("point_b"):
        cal = load_calibration(output_dir)
        if cal is not None:
            kfs = cal.get("keyframes") or []
            scale_block = {
                "known_distance_m": float(merged["known_distance_m"]),
                "point_a": list(merged["point_a"]),
                "point_b": list(merged["point_b"]),
            }
            if kfs:
                # Attach to nearest / first keyframe
                kfs[0]["scale"] = scale_block
            else:
                cal.setdefault("keyframes", []).append({
                    "frame_idx": 0,
                    "track_polygon": [],
                    "scale": scale_block,
                })
            from .calibration import save_calibration
            save_calibration(output_dir, cal)

    aid = athlete_id or overrides_update.get("athlete_id")
    return compute_metrics(output_dir, athlete_id=aid)


def load_metrics(output_dir: Path | str) -> dict[str, Any]:
    output_dir = Path(output_dir)
    data = _load_json(output_dir / "metrics.json")
    return data if data is not None else empty_metrics()
