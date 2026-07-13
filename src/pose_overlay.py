"""
Visual pose superposition overlays for hop quality (Análisis mode).

Renders a person crop at contact (or mid-flight) with:
  - current take skeleton (cyan) — legs + hips
  - general/reference skeleton (amber) from GT contact samples

Reference preference: GT sample keypoints from output/<video_id>/analysis.json,
aligned at mid-hip and scaled by hip–ankle length. Falls back to a geometric
stick figure from the hop_contact prototype feature centroid when GT keypoints
are unavailable.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from .gt_contacts import (
    OUTPUT_ROOT,
    VIDEO_ANALYSIS_CANDIDATES,
    load_gt_contacts,
)
from .pose_features import CONF_MIN, FEATURE_NAMES, extract_pose_features

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROTOTYPES_PATH = PROJECT_ROOT / "data" / "phase_prototypes.json"

OVERLAY_PHASES = ("hop_1", "hop_2", "hop_3", "hop_4", "final_flight")

LEG_JOINTS = ("l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle")
LEG_EDGES = (
    ("l_hip", "r_hip"),
    ("l_hip", "l_knee"),
    ("r_hip", "r_knee"),
    ("l_knee", "l_ankle"),
    ("r_knee", "r_ankle"),
)

# BGR
COLOR_CURRENT = (255, 220, 40)   # cyan-ish
COLOR_REF = (40, 160, 255)       # amber/orange
COLOR_FAINT = (90, 90, 90)

PHASE_TITLES = {
    "hop_1": "Hop 1 — contacto",
    "hop_2": "Hop 2 — contacto",
    "hop_3": "Hop 3 — contacto",
    "hop_4": "Hop 4 — contacto",
    "final_flight": "Vuelo final",
}

USABLE_ANGLES = frozenset({"LATERAL", "SEMI_BACK", "SEMI_FRONT"})


# ─── IO helpers ───────────────────────────────────────────────────────────────

def _load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _frame_map(frames: list[dict]) -> dict[int, dict]:
    return {int(f["frame_idx"]): f for f in frames if f.get("frame_idx") is not None}


def _nearest_frame(fmap: dict[int, dict], frame_idx: int, max_delta: int = 4) -> Optional[dict]:
    if frame_idx in fmap:
        return fmap[frame_idx]
    best = None
    best_d = max_delta + 1
    for k, fr in fmap.items():
        d = abs(k - frame_idx)
        if d < best_d:
            best_d = d
            best = fr
    return best


def _load_analysis_for_video(video_id: str, analysis_dir: Optional[str] = None) -> Optional[dict]:
    candidates: list[str] = []
    if analysis_dir:
        candidates.append(analysis_dir)
    candidates.extend(VIDEO_ANALYSIS_CANDIDATES.get(video_id, [video_id]))
    if video_id not in candidates:
        candidates.append(video_id)
    seen: set[str] = set()
    for name in candidates:
        if name in seen:
            continue
        seen.add(name)
        data = _load_json(OUTPUT_ROOT / name / "analysis.json")
        if data and data.get("frames"):
            return data
    return None


def _load_frame_image(
    out_dir: Path,
    frame_idx: int,
    video_path: Optional[str] = None,
) -> Optional[np.ndarray]:
    path = out_dir / "frames" / f"frame_{frame_idx:06d}.jpg"
    if path.exists():
        img = cv2.imread(str(path))
        if img is not None:
            return img
    # Decodificar el frame exacto del video fuente (funciona sin JPEG en disco).
    # video_path se resuelve, si no se pasa, del analysis.json del out_dir.
    try:
        from .frame_io import read_frame_bgr
        img = read_frame_bgr(out_dir.name, frame_idx, out_dir.parent, video_path=video_path)
        if img is not None:
            return img
    except Exception:
        pass
    # nearest saved frame on disk (secundario, sólo si no se pudo decodificar)
    frames_dir = out_dir / "frames"
    if not frames_dir.exists():
        return None
    best_path = None
    best_d = 10**9
    for p in frames_dir.glob("frame_*.jpg"):
        try:
            idx = int(p.stem.split("_")[1])
        except (IndexError, ValueError):
            continue
        d = abs(idx - frame_idx)
        if d < best_d:
            best_d = d
            best_path = p
    if best_path is not None and best_d <= 6:
        return cv2.imread(str(best_path))
    return None


# ─── Keypoints ────────────────────────────────────────────────────────────────

def _kp_xy(frame: dict, conf_min: float = CONF_MIN) -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    for kp in frame.get("keypoints") or []:
        name = kp.get("name")
        if not name:
            continue
        if float(kp.get("conf", 0)) < conf_min:
            continue
        out[name] = (float(kp["x"]), float(kp["y"]))
    return out


def _mid_hip(kps: dict[str, tuple[float, float]]) -> Optional[tuple[float, float]]:
    lh, rh = kps.get("l_hip"), kps.get("r_hip")
    if not lh or not rh:
        return None
    return ((lh[0] + rh[0]) * 0.5, (lh[1] + rh[1]) * 0.5)


def _pose_scale(kps: dict[str, tuple[float, float]], mid: tuple[float, float]) -> float:
    """Scale = max distance from mid-hip to ankles (prefer) or shoulders."""
    dists: list[float] = []
    for name in ("l_ankle", "r_ankle", "l_knee", "r_knee", "l_shoulder", "r_shoulder"):
        p = kps.get(name)
        if p:
            dists.append(math.hypot(p[0] - mid[0], p[1] - mid[1]))
    if not dists:
        return 0.0
    return max(dists)


def align_reference_to_current(
    ref_kps: dict[str, tuple[float, float]],
    cur_kps: dict[str, tuple[float, float]],
) -> dict[str, tuple[float, float]]:
    """
    Place reference stick figure into current image coordinates:
    same mid-hip, scale by current hip–ankle (pose) length.
    """
    mid_c = _mid_hip(cur_kps)
    mid_r = _mid_hip(ref_kps)
    if mid_c is None or mid_r is None:
        return {}
    scale_c = _pose_scale(cur_kps, mid_c)
    scale_r = _pose_scale(ref_kps, mid_r)
    if scale_c < 8 or scale_r < 8:
        return {}
    ratio = scale_c / scale_r
    aligned: dict[str, tuple[float, float]] = {}
    for name, (x, y) in ref_kps.items():
        nx = (x - mid_r[0]) * ratio
        ny = (y - mid_r[1]) * ratio
        aligned[name] = (mid_c[0] + nx, mid_c[1] + ny)
    return aligned


# ─── GT reference selection ───────────────────────────────────────────────────

def _vec_distance(a: list[float], b: list[float]) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return 1e9
    aa = np.asarray(a[:n], dtype=float)
    bb = np.asarray(b[:n], dtype=float)
    return float(np.linalg.norm(aa - bb))


def _pick_gt_sample(
    phase: str,
    *,
    exclude_video: Optional[str] = None,
    current_vec: Optional[list[float]] = None,
) -> Optional[dict[str, Any]]:
    store = load_gt_contacts()
    samples = list(store.get("samples") or [])
    if not samples:
        return None

    if phase == "final_flight":
        # Prefer landing contacts (sand) as visual reference for end of flight;
        # hop_4 also useful as takeoff reference — use landing first.
        pool = [s for s in samples if s.get("phase") == "landing"]
        if not pool:
            pool = [s for s in samples if s.get("phase") == "hop_4"]
    else:
        pool = [s for s in samples if s.get("phase") == phase]
        if not pool:
            pool = [s for s in samples if s.get("pose_tag") == "hop_contact"]

    if exclude_video:
        others = [s for s in pool if s.get("video_id") != exclude_video]
        if others:
            pool = others

    usable = [s for s in pool if s.get("camera_angle") in USABLE_ANGLES]
    if usable:
        pool = usable

    if not pool:
        return None

    # Prefer higher quality; among ties, closest feature vector to current
    def sort_key(s: dict) -> tuple:
        q = -float(s.get("quality_score") or 0)
        dist = 0.0
        if current_vec and s.get("feature_vector"):
            dist = _vec_distance(current_vec, s["feature_vector"])
        return (q, dist)

    pool.sort(key=sort_key)
    return pool[0]


def _load_gt_keypoints(sample: dict[str, Any]) -> Optional[dict[str, tuple[float, float]]]:
    video_id = sample.get("video_id")
    frame_idx = sample.get("frame_idx")
    if video_id is None or frame_idx is None:
        return None
    analysis = _load_analysis_for_video(str(video_id), sample.get("analysis_dir"))
    if not analysis:
        return None
    fmap = _frame_map(analysis.get("frames") or [])
    fr = _nearest_frame(fmap, int(frame_idx), max_delta=6)
    if not fr:
        return None
    kps = _kp_xy(fr)
    if _mid_hip(kps) is None:
        return None
    return kps


def _mid_flight_gt_keypoints(
    *,
    exclude_video: Optional[str] = None,
) -> Optional[dict[str, tuple[float, float]]]:
    """Load mid-flight frame between hop_4 and landing from a GT video."""
    store = load_gt_contacts()
    samples = list(store.get("samples") or [])
    by_video: dict[str, dict[str, dict]] = {}
    for s in samples:
        vid = s.get("video_id")
        ph = s.get("phase")
        if not vid or not ph:
            continue
        if exclude_video and vid == exclude_video:
            continue
        by_video.setdefault(vid, {})[ph] = s

    for vid, phases in by_video.items():
        h4 = phases.get("hop_4")
        land = phases.get("landing")
        if not h4 or not land:
            continue
        f0, f1 = int(h4["frame_idx"]), int(land["frame_idx"])
        if f1 <= f0 + 2:
            continue
        mid = f0 + (f1 - f0) // 2
        analysis = _load_analysis_for_video(vid, (h4.get("analysis_dir") or land.get("analysis_dir")))
        if not analysis:
            continue
        fmap = _frame_map(analysis.get("frames") or [])
        fr = _nearest_frame(fmap, mid, max_delta=8)
        if not fr:
            continue
        kps = _kp_xy(fr)
        if _mid_hip(kps) is not None and (kps.get("l_ankle") or kps.get("r_ankle")):
            return kps
    return None


def _reconstruct_from_prototype(
    pose_tag: str = "hop_contact",
) -> dict[str, tuple[float, float]]:
    """
    Approximate stick figure in normalized coords (mid-hip origin, unit scale)
    from prototype feature centroid. Limited accuracy — knees from angles,
    ankles from sep / asymmetry. Documented fallback only.
    """
    centroid = None
    if PROTOTYPES_PATH.exists():
        proto = _load_json(PROTOTYPES_PATH) or {}
        entry = (proto.get("pose_tags") or {}).get(pose_tag) or {}
        centroid = entry.get("centroid")
    if not centroid or len(centroid) < len(FEATURE_NAMES):
        # Neutral hop-ish defaults
        feats = {n: 0.5 for n in FEATURE_NAMES}
        feats["l_knee_angle"] = 140.0
        feats["r_knee_angle"] = 155.0
        feats["ankle_sep_norm"] = 0.35
        feats["foot_asymmetry"] = 0.25
        feats["torso_lean"] = 8.0
    else:
        feats = {FEATURE_NAMES[i]: float(centroid[i]) for i in range(len(FEATURE_NAMES))}

    # Feature angles are stored as degrees in extract_pose_features (raw),
    # but prototypes may hold normalized-ish values — clamp sensibly.
    l_ang = float(feats.get("l_knee_angle", 140))
    r_ang = float(feats.get("r_knee_angle", 155))
    if l_ang <= 3.5:  # likely 0–1 normalized leftover
        l_ang = 90 + l_ang * 90
    if r_ang <= 3.5:
        r_ang = 90 + r_ang * 90
    l_ang = max(60.0, min(175.0, l_ang))
    r_ang = max(60.0, min(175.0, r_ang))

    sep = float(feats.get("ankle_sep_norm", 0.35))
    sep = max(0.1, min(1.0, sep if sep > 0.05 else 0.35))
    asym = float(feats.get("foot_asymmetry", 0.2))
    asym = max(0.0, min(0.8, asym if asym < 2 else 0.25))

    # Unit stick: hips at ±0.12, thighs length ~0.45, shanks ~0.45
    hip_w = 0.14
    thigh = 0.42
    shank = 0.45

    def leg(side: str, knee_angle: float, ankle_x: float) -> dict[str, tuple[float, float]]:
        hip = (-hip_w if side == "l" else hip_w, 0.0)
        # Knee: hang mostly downward; bend via interior angle at knee
        # Place knee below hip; ankle from knee using (180 - knee_angle) lean
        bend = math.radians(180.0 - knee_angle)
        # Prefer forward (positive x for contact asymmetry)
        dir_sign = 1.0 if side == "l" else -1.0
        knee = (hip[0] + dir_sign * thigh * math.sin(bend * 0.35), hip[1] + thigh * math.cos(bend * 0.15))
        # Ankle further down with sep
        ankle = (ankle_x, knee[1] + shank * 0.95)
        prefix = "l_" if side == "l" else "r_"
        return {
            f"{prefix}hip": hip,
            f"{prefix}knee": knee,
            f"{prefix}ankle": ankle,
        }

    ax_l = -sep * 0.5 - asym * 0.15
    ax_r = sep * 0.5 + asym * 0.15
    kps: dict[str, tuple[float, float]] = {}
    kps.update(leg("l", l_ang, ax_l))
    kps.update(leg("r", r_ang, ax_r))
    return kps


def _place_normalized(
    norm_kps: dict[str, tuple[float, float]],
    mid: tuple[float, float],
    scale: float,
) -> dict[str, tuple[float, float]]:
    return {
        name: (mid[0] + x * scale, mid[1] + y * scale)
        for name, (x, y) in norm_kps.items()
    }


# ─── Contact / flight frame resolution ────────────────────────────────────────

def _resolve_current_frame_idx(
    phase: str,
    out_dir: Path,
    analysis: dict,
) -> Optional[int]:
    sections = _load_json(out_dir / "sections.json") or {}
    metrics = _load_json(out_dir / "metrics.json") or {}
    pq = ((metrics.get("comparison") or {}).get("pose_quality") or {})

    if phase == "final_flight":
        ff = pq.get("final_flight") or {}
        samples = ff.get("samples") or []
        if samples:
            # Prefer sample closest to mid of span
            f0 = ff.get("from_frame")
            f1 = ff.get("to_frame")
            if f0 is not None and f1 is not None:
                mid = int(f0) + (int(f1) - int(f0)) // 2
                best = min(samples, key=lambda s: abs(int(s.get("frame_idx", 0)) - mid))
                return int(best["frame_idx"])
            return int(samples[len(samples) // 2]["frame_idx"])
        if ff.get("from_frame") is not None and ff.get("to_frame") is not None:
            return int(ff["from_frame"]) + (int(ff["to_frame"]) - int(ff["from_frame"])) // 2
        # sections: hop_4 → landing
        contacts = sections.get("contacts") or []
        h4 = next((c for c in contacts if c.get("phase") == "hop_4"), None)
        land = next((c for c in contacts if c.get("phase") == "landing"), None)
        if h4 and land:
            return int(h4["frame_idx"]) + (int(land["frame_idx"]) - int(h4["frame_idx"])) // 2
        return None

    # hop_N from metrics pose_quality or sections contacts
    for hop in pq.get("hops") or []:
        if hop.get("phase") == phase and hop.get("frame_idx") is not None:
            return int(hop["frame_idx"])
    for c in sections.get("contacts") or []:
        if c.get("phase") == phase and c.get("frame_idx") is not None:
            return int(c["frame_idx"])
    return None


# ─── Crop & draw ──────────────────────────────────────────────────────────────

def _leg_crop_box(
    img_shape: tuple[int, ...],
    kps: dict[str, tuple[float, float]],
    person_bbox: Optional[list[float]] = None,
    pad: float = 0.18,
) -> tuple[int, int, int, int]:
    h, w = img_shape[:2]
    xs: list[float] = []
    ys: list[float] = []
    for name in LEG_JOINTS:
        p = kps.get(name)
        if p:
            xs.append(p[0])
            ys.append(p[1])
    mid = _mid_hip(kps)
    if mid:
        # Include a bit of lower torso above hips
        xs.extend([mid[0] - 20, mid[0] + 20])
        ys.append(mid[1] - 40)

    if person_bbox and len(person_bbox) >= 4:
        bx1, by1, bx2, by2 = [float(v) for v in person_bbox[:4]]
        # Prefer lower 2/3 of person bbox
        top = by1 + (by2 - by1) * (1.0 / 3.0)
        xs.extend([bx1, bx2])
        ys.extend([top, by2])

    if not xs or not ys:
        return 0, 0, w, h

    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    bw = max(20.0, x2 - x1)
    bh = max(20.0, y2 - y1)
    x1 -= bw * pad
    x2 += bw * pad
    y1 -= bh * pad * 0.6
    y2 += bh * pad
    ix1 = max(0, int(math.floor(x1)))
    iy1 = max(0, int(math.floor(y1)))
    ix2 = min(w, int(math.ceil(x2)))
    iy2 = min(h, int(math.ceil(y2)))
    if ix2 - ix1 < 40 or iy2 - iy1 < 40:
        return 0, 0, w, h
    return ix1, iy1, ix2, iy2


def _draw_legs(
    img: np.ndarray,
    kps: dict[str, tuple[float, float]],
    color: tuple[int, int, int],
    *,
    thickness: int = 2,
    origin: tuple[int, int] = (0, 0),
) -> None:
    ox, oy = origin
    for a, b in LEG_EDGES:
        pa, pb = kps.get(a), kps.get(b)
        if not pa or not pb:
            continue
        cv2.line(
            img,
            (int(pa[0] - ox), int(pa[1] - oy)),
            (int(pb[0] - ox), int(pb[1] - oy)),
            color,
            thickness,
            cv2.LINE_AA,
        )
    for name in LEG_JOINTS:
        p = kps.get(name)
        if not p:
            continue
        pt = (int(p[0] - ox), int(p[1] - oy))
        cv2.circle(img, pt, 5, color, -1, cv2.LINE_AA)
        cv2.circle(img, pt, 6, (0, 0, 0), 1, cv2.LINE_AA)

    # Optional mid-hip marker / short lower-torso stub upward
    mid = _mid_hip(kps)
    if mid:
        mx, my = int(mid[0] - ox), int(mid[1] - oy)
        cv2.line(img, (mx, my), (mx, max(0, my - 28)), color, max(1, thickness - 1), cv2.LINE_AA)
        cv2.circle(img, (mx, my), 4, color, -1, cv2.LINE_AA)


def _draw_legend(img: np.ndarray, title: str) -> None:
    h, w = img.shape[:2]
    bar_h = 36
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.72, img, 0.28, 0, img)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, title, (8, 16), font, 0.45, (230, 230, 230), 1, cv2.LINE_AA)
    # Legend swatches
    y = 30
    cv2.circle(img, (12, y), 4, COLOR_REF, -1)
    cv2.putText(img, "General", (20, y + 4), font, 0.38, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.circle(img, (95, y), 4, COLOR_CURRENT, -1)
    cv2.putText(img, "Esta toma", (103, y + 4), font, 0.38, (200, 200, 200), 1, cv2.LINE_AA)


def _draw_error_placeholder(message: str, size: tuple[int, int] = (320, 240)) -> np.ndarray:
    w, h = size
    img = np.full((h, w, 3), 32, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, "Sin overlay", (16, h // 2 - 8), font, 0.55, (160, 160, 160), 1, cv2.LINE_AA)
    cv2.putText(img, message[:48], (16, h // 2 + 18), font, 0.4, (120, 120, 120), 1, cv2.LINE_AA)
    return img


# ─── Public API ───────────────────────────────────────────────────────────────

def render_pose_overlay(
    output_dir: Path | str,
    phase: str,
    *,
    video_name: Optional[str] = None,
    use_cache: bool = True,
    force: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Build pose overlay image for a phase.

    Returns (bgr_image, meta).
    """
    out_dir = Path(output_dir)
    phase = str(phase).strip()
    if phase not in OVERLAY_PHASES:
        raise ValueError(f"phase must be one of {OVERLAY_PHASES}")

    cache_path = out_dir / "overlays" / f"{phase}.png"
    meta: dict[str, Any] = {"phase": phase, "source": None, "cached": False}

    analysis = _load_json(out_dir / "analysis.json")
    if not analysis:
        img = _draw_error_placeholder("Sin analysis.json")
        return img, {**meta, "error": "no_analysis"}

    derived = analysis.get("derived_version")
    metrics = _load_json(out_dir / "metrics.json") or {}
    metrics_derived = metrics.get("derived_version")
    cache_tag = f"{derived}:{metrics_derived}"

    if use_cache and not force and cache_path.exists():
        tag_path = cache_path.with_suffix(".tag")
        if tag_path.exists() and tag_path.read_text(encoding="utf-8").strip() == cache_tag:
            cached = cv2.imread(str(cache_path))
            if cached is not None:
                meta["cached"] = True
                meta["source"] = "cache"
                return cached, meta

    vid = video_name or out_dir.name
    frame_idx = _resolve_current_frame_idx(phase, out_dir, analysis)
    if frame_idx is None:
        img = _draw_error_placeholder("Sin frame de contacto")
        return img, {**meta, "error": "no_contact_frame"}

    fmap = _frame_map(analysis.get("frames") or [])
    cur_frame = _nearest_frame(fmap, frame_idx, max_delta=6)
    if not cur_frame:
        img = _draw_error_placeholder(f"Frame {frame_idx} ausente")
        return img, {**meta, "error": "frame_missing", "frame_idx": frame_idx}

    actual_idx = int(cur_frame.get("frame_idx", frame_idx))
    img_full = _load_frame_image(out_dir, actual_idx, video_path=analysis.get("video"))
    if img_full is None:
        img = _draw_error_placeholder("Sin imagen de frame")
        return img, {**meta, "error": "no_image", "frame_idx": actual_idx}

    cur_kps = _kp_xy(cur_frame)
    mid_c = _mid_hip(cur_kps)
    if mid_c is None:
        img = _draw_error_placeholder("Caderas no detectadas")
        return img, {**meta, "error": "no_hips", "frame_idx": actual_idx}

    scale_c = _pose_scale(cur_kps, mid_c)
    if scale_c < 8:
        img = _draw_error_placeholder("Escala de pose inválida")
        return img, {**meta, "error": "bad_scale", "frame_idx": actual_idx}

    # Current pose feature vector for GT matching
    current_vec = None
    feat = extract_pose_features(cur_frame)
    if feat.valid:
        current_vec = list(feat.vector)

    ref_kps: Optional[dict[str, tuple[float, float]]] = None
    ref_source = "none"

    if phase == "final_flight":
        ref_kps = _mid_flight_gt_keypoints(exclude_video=vid)
        if ref_kps:
            ref_source = "gt_mid_flight"
        else:
            sample = _pick_gt_sample("final_flight", exclude_video=vid, current_vec=current_vec)
            if sample:
                ref_kps = _load_gt_keypoints(sample)
                if ref_kps:
                    ref_source = f"gt:{sample.get('video_id')}:{sample.get('frame_idx')}"
    else:
        sample = _pick_gt_sample(phase, exclude_video=vid, current_vec=current_vec)
        if sample:
            ref_kps = _load_gt_keypoints(sample)
            if ref_kps:
                ref_source = f"gt:{sample.get('video_id')}:{sample.get('frame_idx')}"

    if ref_kps is None:
        # Geometric fallback from prototype features
        tag = "hop_flight" if phase == "final_flight" else "hop_contact"
        # hop_flight may not exist — reconstruct uses hop_contact angles
        norm = _reconstruct_from_prototype("hop_contact" if tag != "landing" else "landing")
        if phase == "final_flight":
            # Stretch legs a bit for flight look
            for name in list(norm.keys()):
                x, y = norm[name]
                if "ankle" in name or "knee" in name:
                    norm[name] = (x * 1.05, y * 1.08)
        ref_aligned = _place_normalized(norm, mid_c, scale_c)
        ref_source = "prototype_reconstruct"
    else:
        ref_aligned = align_reference_to_current(ref_kps, cur_kps)
        if not ref_aligned:
            norm = _reconstruct_from_prototype("hop_contact")
            ref_aligned = _place_normalized(norm, mid_c, scale_c)
            ref_source = "prototype_reconstruct"

    bbox = cur_frame.get("person_bbox")
    x1, y1, x2, y2 = _leg_crop_box(img_full.shape, cur_kps, bbox)
    crop = img_full[y1:y2, x1:x2].copy()
    if crop.size == 0:
        crop = img_full.copy()
        x1, y1 = 0, 0

    # Darken slightly so skeletons pop
    crop = cv2.convertScaleAbs(crop, alpha=0.88, beta=-8)

    _draw_legs(crop, ref_aligned, COLOR_REF, thickness=2, origin=(x1, y1))
    _draw_legs(crop, cur_kps, COLOR_CURRENT, thickness=2, origin=(x1, y1))

    title = PHASE_TITLES.get(phase, phase)
    _draw_legend(crop, title)

    # Upscale small crops for UI readability
    ch, cw = crop.shape[:2]
    if max(ch, cw) < 220:
        scale = 220 / max(ch, cw)
        crop = cv2.resize(crop, (int(cw * scale), int(ch * scale)), interpolation=cv2.INTER_LINEAR)

    meta.update({
        "frame_idx": actual_idx,
        "source": ref_source,
        "cache_tag": cache_tag,
    })

    if use_cache:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(cache_path), crop)
            cache_path.with_suffix(".tag").write_text(cache_tag, encoding="utf-8")
        except OSError as exc:
            logger.warning("Could not cache pose overlay: %s", exc)

    return crop, meta


def render_pose_overlay_png(
    output_dir: Path | str,
    phase: str,
    **kwargs: Any,
) -> tuple[bytes, dict[str, Any]]:
    img, meta = render_pose_overlay(output_dir, phase, **kwargs)
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("PNG encode failed")
    return buf.tobytes(), meta
