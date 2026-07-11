"""
Extract normalized pose features from YOLO keypoints for phase classification.

Features are scale- and position-invariant (centered on mid-hip, scaled by torso).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

from .pose_analyzer import KP

CONF_MIN = 0.35
FEATURE_NAMES: tuple[str, ...] = (
    "l_knee_angle",
    "r_knee_angle",
    "torso_lean",
    "ankle_sep_norm",
    "foot_asymmetry",
    "leg_extension_asym",
    "avg_foot_height",
    "arch_score",
    "hop_contact_score",
    "flight_score",
    "running_score",
    "landing_score",
    "feet_together_score",
    "body_extension",
)


@dataclass
class PoseFeatures:
    valid: bool = False
    quality: float = 0.0
    l_knee_angle: float = 0.0
    r_knee_angle: float = 0.0
    torso_lean: float = 0.0
    ankle_sep_norm: float = 0.0
    foot_asymmetry: float = 0.0
    leg_extension_asym: float = 0.0
    avg_foot_height: float = 0.0
    arch_score: float = 0.0
    hop_contact_score: float = 0.0
    flight_score: float = 0.0
    running_score: float = 0.0
    landing_score: float = 0.0
    feet_together_score: float = 0.0
    body_extension: float = 0.0
    vector: list[float] = field(default_factory=list)

    def as_dict(self) -> dict[str, float]:
        return {name: getattr(self, name) for name in FEATURE_NAMES}


def _kp_map(frame: dict) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for kp in frame.get("keypoints") or []:
        name = kp.get("name")
        if not name:
            continue
        conf = float(kp.get("conf", 0))
        if conf < CONF_MIN:
            continue
        out[name] = {"x": float(kp["x"]), "y": float(kp["y"]), "conf": conf}
    return out


def _pt(kps: dict[str, dict[str, float]], name: str) -> Optional[tuple[float, float]]:
    kp = kps.get(name)
    if not kp:
        return None
    return kp["x"], kp["y"]


def _angle_deg(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
    bax, bay = a[0] - b[0], a[1] - b[1]
    bcx, bcy = c[0] - b[0], c[1] - b[1]
    dot = bax * bcx + bay * bcy
    mag = math.hypot(bax, bay) * math.hypot(bcx, bcy)
    if mag < 1e-6:
        return 0.0
    cos_a = max(-1.0, min(1.0, dot / mag))
    return math.degrees(math.acos(cos_a))


def _mid(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    return ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5)


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def extract_pose_features(frame: dict) -> PoseFeatures:
    """Build feature vector from a single analysis.json frame record."""
    kps = _kp_map(frame)
    if len(kps) < 6:
        return PoseFeatures(valid=False)

    l_hip = _pt(kps, "l_hip")
    r_hip = _pt(kps, "r_hip")
    l_knee = _pt(kps, "l_knee")
    r_knee = _pt(kps, "r_knee")
    l_ankle = _pt(kps, "l_ankle")
    r_ankle = _pt(kps, "r_ankle")
    l_sh = _pt(kps, "l_shoulder")
    r_sh = _pt(kps, "r_shoulder")

    if not all([l_hip, r_hip]):
        return PoseFeatures(valid=False)

    mid_hip = _mid(l_hip, r_hip)
    scale_pts = [p for p in (l_sh, r_sh, l_ankle, r_ankle) if p]
    if not scale_pts:
        return PoseFeatures(valid=False)

    torso_h = max(_dist(mid_hip, p) for p in scale_pts)
    if torso_h < 8:
        return PoseFeatures(valid=False)

    def norm(p: tuple[float, float]) -> tuple[float, float]:
        return ((p[0] - mid_hip[0]) / torso_h, (p[1] - mid_hip[1]) / torso_h)

    quality = min(1.0, len(kps) / 9.0)

    l_knee_angle = (
        _angle_deg(norm(l_hip), norm(l_knee), norm(l_ankle))
        if l_knee and l_ankle else 150.0
    )
    r_knee_angle = (
        _angle_deg(norm(r_hip), norm(r_knee), norm(r_ankle))
        if r_knee and r_ankle else 150.0
    )

    mid_sh = _mid(l_sh, r_sh) if l_sh and r_sh else mid_hip
    torso_vec = (mid_sh[0] - mid_hip[0], mid_sh[1] - mid_hip[1])
    torso_lean = math.degrees(math.atan2(torso_vec[0], max(abs(torso_vec[1]), 1e-3)))

    ankle_sep = _dist(l_ankle, r_ankle) / torso_h if l_ankle and r_ankle else 0.0
    ankle_sep_norm = min(1.5, ankle_sep)

    foot_asymmetry = 0.0
    if l_ankle and r_ankle:
        foot_asymmetry = abs(l_ankle[0] - r_ankle[0]) / torso_h

    leg_extension_asym = abs(l_knee_angle - r_knee_angle) / 180.0

    foot_ys = [p[1] for p in (l_ankle, r_ankle) if p]
    hip_y = mid_hip[1]
    avg_foot_height = (
        sum((fy - hip_y) / torso_h for fy in foot_ys) / len(foot_ys)
        if foot_ys else 0.0
    )

    knee_angles = [a for a in (l_knee_angle, r_knee_angle) if a > 0]
    body_extension = (
        sum(abs(180 - a) for a in knee_angles) / max(len(knee_angles), 1) / 90.0
    )

    min_knee = min(l_knee_angle, r_knee_angle)
    max_knee = max(l_knee_angle, r_knee_angle)
    arch_score = min(1.0, body_extension * 0.6 + max(0.0, torso_lean) / 45.0 * 0.4)

    hop_contact_score = min(
        1.0,
        foot_asymmetry * 0.5 + (1.0 - min_knee / 180.0) * 0.35 + max(0.0, avg_foot_height) * 0.15,
    )

    flight_score = min(
        1.0,
        leg_extension_asym * 0.45 + max(0.0, -avg_foot_height) * 0.35 + foot_asymmetry * 0.2,
    )

    running_score = min(
        1.0,
        (1.0 - leg_extension_asym) * 0.4
        + (1.0 - abs(ankle_sep_norm - 0.35)) * 0.35
        + max(0.0, avg_foot_height) * 0.25,
    )

    landing_score = min(
        1.0,
        (1.0 - body_extension) * 0.4
        + max(0.0, avg_foot_height) * 0.35
        + (1.0 - flight_score) * 0.25,
    )

    feet_together_score = min(1.0, max(0.0, 1.0 - ankle_sep_norm * 1.8) * (0.5 + body_extension * 0.5))

    vec = [
        l_knee_angle / 180.0,
        r_knee_angle / 180.0,
        (torso_lean + 90) / 180.0,
        ankle_sep_norm,
        foot_asymmetry,
        leg_extension_asym,
        (avg_foot_height + 1) / 2.0,
        arch_score,
        hop_contact_score,
        flight_score,
        running_score,
        landing_score,
        feet_together_score,
        body_extension,
    ]

    return PoseFeatures(
        valid=True,
        quality=round(quality, 3),
        l_knee_angle=round(l_knee_angle, 2),
        r_knee_angle=round(r_knee_angle, 2),
        torso_lean=round(torso_lean, 2),
        ankle_sep_norm=round(ankle_sep_norm, 3),
        foot_asymmetry=round(foot_asymmetry, 3),
        leg_extension_asym=round(leg_extension_asym, 3),
        avg_foot_height=round(avg_foot_height, 3),
        arch_score=round(arch_score, 3),
        hop_contact_score=round(hop_contact_score, 3),
        flight_score=round(flight_score, 3),
        running_score=round(running_score, 3),
        landing_score=round(landing_score, 3),
        feet_together_score=round(feet_together_score, 3),
        body_extension=round(body_extension, 3),
        vector=vec,
    )


def extract_features_for_frames(frames: list[dict]) -> dict[int, PoseFeatures]:
    out: dict[int, PoseFeatures] = {}
    for f in frames:
        if not f.get("person_detected"):
            continue
        fidx = int(f.get("frame_idx", 0))
        feat = extract_pose_features(f)
        if feat.valid:
            out[fidx] = feat
    return out
