"""
Track scorer — pista↔atleta (Phase 2).

Computes per-frame track_overlap, athlete_state, position_s, predicted_bbox
and optional selection scores when calibration.json exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from .calibration import load_calibration
from .mask_utils import athlete_mask_overlap, load_mask_png

ATHLETE_STATES = ("GROUND", "AIR", "OFF_TRACK_NEAR", "FINAL_FLIGHT", "LOST")

HIGH_OVERLAP = 0.5
LOW_OVERLAP = 0.15
LANDING_OVERLAP = 0.25
PREDICTION_HISTORY = 5
LOST_AFTER_FRAMES = 10


def _as_polygon(points: list[list[float]], width: int, height: int) -> Optional[np.ndarray]:
    if not points or len(points) < 3:
        return None
    pts = np.array([[p[0] * width, p[1] * height] for p in points], dtype=np.float32)
    return pts


def _bbox_bottom_half(bbox: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    mid_y = y1 + (y2 - y1) * 0.5
    return (x1, mid_y, x2, y2)


def _rect_area(bbox: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _fraction_inside(bbox: tuple[float, float, float, float],
                     polygon: np.ndarray) -> float:
    """Fraction of bbox (bottom-half) area inside polygon."""
    area = _rect_area(bbox)
    if area <= 0:
        return 0.0
    x1, y1, x2, y2 = bbox
    w = max(2, int(x2 - x1))
    h = max(2, int(y2 - y1))
    mask_rect = np.zeros((h, w), dtype=np.uint8)
    mask_rect[:] = 255
    mask_poly = np.zeros((h, w), dtype=np.uint8)
    shifted = polygon.copy()
    shifted[:, 0] -= x1
    shifted[:, 1] -= y1
    cv2.fillPoly(mask_poly, [shifted.astype(np.int32)], 255)
    inter = cv2.bitwise_and(mask_rect, mask_poly)
    return float(inter.sum()) / float(mask_rect.sum() * 255)


def _bbox_polygon_overlap(bbox: tuple[float, float, float, float],
                          polygon: Optional[np.ndarray]) -> float:
    if polygon is None:
        return 0.0
    bottom = _bbox_bottom_half(bbox)
    return _fraction_inside(bottom, polygon)


def _project_position_s(bbox: tuple[float, float, float, float],
                        axis: Optional[dict[str, Any]],
                        width: int, height: int) -> Optional[float]:
    if not axis:
        return None
    origin = axis.get("origin")
    direction = axis.get("direction")
    if not origin or not direction or len(origin) < 2 or len(direction) < 2:
        return None
    ox, oy = float(origin[0]) * width, float(origin[1]) * height
    dx, dy = float(direction[0]), float(direction[1])
    norm = (dx * dx + dy * dy) ** 0.5
    if norm <= 1e-6:
        return None
    dx /= norm
    dy /= norm
    cx = (bbox[0] + bbox[2]) * 0.5
    cy = bbox[3]  # foot proxy
    return float((cx - ox) * dx + (cy - oy) * dy)


def _nearest_keyframe(keyframes: list[dict], frame_idx: int) -> Optional[dict]:
    if not keyframes:
        return None
    best = min(keyframes, key=lambda k: abs(k["frame_idx"] - frame_idx))
    return best


def _interp_keyframe(keyframes: list[dict], frame_idx: int) -> dict[str, Any]:
    """Linear interp between bracketing keyframes when point counts match."""
    if not keyframes:
        return {}
    if len(keyframes) == 1:
        return keyframes[0]

    before = [k for k in keyframes if k["frame_idx"] <= frame_idx]
    after = [k for k in keyframes if k["frame_idx"] >= frame_idx]

    if not before:
        return keyframes[0]
    if not after:
        return keyframes[-1]

    k0 = before[-1]
    k1 = after[0]
    if k0["frame_idx"] == k1["frame_idx"]:
        return k0

    t = (frame_idx - k0["frame_idx"]) / max(k1["frame_idx"] - k0["frame_idx"], 1)

    def _lerp_poly(key: str) -> list[list[float]]:
        p0 = k0.get(key) or []
        p1 = k1.get(key) or []
        if len(p0) != len(p1) or not p0:
            return p0 or p1
        out: list[list[float]] = []
        for a, b in zip(p0, p1):
            out.append([a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])])
        return out

    result: dict[str, Any] = {
        "frame_idx": frame_idx,
        "track_polygon": _lerp_poly("track_polygon"),
        "corridor_polygon": _lerp_poly("corridor_polygon"),
        "landing_zone": _lerp_poly("landing_zone"),
    }
    if k0.get("axis") and k1.get("axis"):
        a0, a1 = k0["axis"], k1["axis"]
        o0, o1 = a0.get("origin"), a1.get("origin")
        d0, d1 = a0.get("direction"), a1.get("direction")
        if o0 and o1 and d0 and d1:
            result["axis"] = {
                "origin": [
                    o0[0] + t * (o1[0] - o0[0]),
                    o0[1] + t * (o1[1] - o0[1]),
                ],
                "direction": [
                    d0[0] + t * (d1[0] - d0[0]),
                    d0[1] + t * (d1[1] - d0[1]),
                ],
            }
    elif k0.get("axis"):
        result["axis"] = k0["axis"]
    return result


@dataclass
class TrackGeometry:
    track_polygon: Optional[np.ndarray] = None
    landing_zone: Optional[np.ndarray] = None
    axis: Optional[dict[str, Any]] = None
    track_mask: Optional[np.ndarray] = None
    sand_mask: Optional[np.ndarray] = None


@dataclass
class TrackScorerContext:
    """Stateful scorer for one video pass."""

    width: int
    height: int
    keyframes: list[dict] = field(default_factory=list)
    mask_mode: bool = False
    mask_frames: dict[str, dict[str, Any]] = field(default_factory=dict)
    masks_dir: Optional[Path] = None
    _mask_cache: dict[int, tuple[Optional[np.ndarray], Optional[np.ndarray]]] = field(
        default_factory=dict, repr=False,
    )
    athlete_state: str = "LOST"
    _history: list[tuple[int, float, float, float, float]] = field(default_factory=list)
    _frames_low_overlap: int = 0
    _predicted_bbox: Optional[tuple[float, float, float, float]] = None

    @classmethod
    def from_output_dir(cls, output_dir: Path,
                        width: int, height: int) -> Optional["TrackScorerContext"]:
        cal = load_calibration(output_dir)
        if cal is None:
            return None
        mask_mode = cal.get("mode") == "color_masks"
        kfs = sorted(cal.get("keyframes") or [], key=lambda k: k["frame_idx"])
        if not kfs and not mask_mode:
            return None
        mask_frames = cal.get("mask_frames") or {}
        return cls(
            width=width,
            height=height,
            keyframes=kfs,
            mask_mode=mask_mode and bool(mask_frames),
            mask_frames=mask_frames,
            masks_dir=output_dir if mask_mode else None,
        )

    def _load_masks(self, frame_idx: int) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if frame_idx in self._mask_cache:
            return self._mask_cache[frame_idx]
        track_mask = sand_mask = None
        entry = self.mask_frames.get(str(frame_idx))
        if entry and self.masks_dir:
            track_rel = entry.get("track")
            sand_rel = entry.get("sand")
            if track_rel:
                track_mask = load_mask_png(self.masks_dir / track_rel)
            if sand_rel:
                sand_mask = load_mask_png(self.masks_dir / sand_rel)
        self._mask_cache[frame_idx] = (track_mask, sand_mask)
        return track_mask, sand_mask

    def geometry_at(self, frame_idx: int) -> TrackGeometry:
        track_mask = sand_mask = None
        if self.mask_mode:
            track_mask, sand_mask = self._load_masks(frame_idx)
        kf = _interp_keyframe(self.keyframes, frame_idx) if self.keyframes else {}
        return TrackGeometry(
            track_polygon=_as_polygon(kf.get("track_polygon") or [], self.width, self.height),
            landing_zone=_as_polygon(kf.get("landing_zone") or [], self.width, self.height),
            axis=kf.get("axis"),
            track_mask=track_mask,
            sand_mask=sand_mask,
        )

    def _predict_bbox(self, frame_idx: int) -> Optional[tuple[float, float, float, float]]:
        hist = self._history[-PREDICTION_HISTORY:]
        if len(hist) < 2:
            return None
        f0, cx0, cy0, bw, bh = hist[-2]
        f1, cx1, cy1, _, _ = hist[-1]
        dt = f1 - f0
        if dt <= 0:
            return None
        vx = (cx1 - cx0) / dt
        vy = (cy1 - cy0) / dt
        steps = frame_idx - f1
        pcx = cx1 + vx * steps
        pcy = cy1 + vy * steps
        return (pcx - bw * 0.5, pcy - bh, pcx + bw * 0.5, pcy)

    def _update_state(self, track_overlap: float, landing_overlap: float,
                      has_bbox: bool) -> str:
        if not has_bbox:
            self._frames_low_overlap += 1
            if self._frames_low_overlap >= LOST_AFTER_FRAMES:
                return "LOST"
            if self.athlete_state in ("AIR", "FINAL_FLIGHT"):
                return self.athlete_state
            return "LOST"

        if landing_overlap >= LANDING_OVERLAP:
            self._frames_low_overlap = 0
            return "FINAL_FLIGHT"

        if track_overlap >= HIGH_OVERLAP:
            self._frames_low_overlap = 0
            return "GROUND"

        if track_overlap >= LOW_OVERLAP:
            self._frames_low_overlap = 0
            if self.athlete_state in ("AIR", "FINAL_FLIGHT"):
                return "AIR"
            return "GROUND"

        # low overlap
        self._frames_low_overlap += 1
        if self.athlete_state in ("GROUND", "AIR", "FINAL_FLIGHT", "OFF_TRACK_NEAR"):
            if self.athlete_state == "FINAL_FLIGHT":
                return "FINAL_FLIGHT"
            if self._frames_low_overlap <= LOST_AFTER_FRAMES:
                return "AIR"
            return "OFF_TRACK_NEAR"
        if self._frames_low_overlap >= LOST_AFTER_FRAMES:
            return "LOST"
        return "OFF_TRACK_NEAR"

    def score_bbox(
        self,
        bbox: Optional[tuple[float, float, float, float]],
        frame_idx: int,
        *,
        update_state: bool = True,
        person_seg_mask: Optional[np.ndarray] = None,
    ) -> dict[str, Any]:
        """Compute track fields for one athlete bbox."""
        empty: dict[str, Any] = {
            "track_overlap": None,
            "athlete_state": "LOST" if update_state else None,
            "position_s": None,
            "predicted_bbox": None,
        }
        if bbox is None:
            if update_state:
                self.athlete_state = self._update_state(0.0, 0.0, False)
                empty["athlete_state"] = self.athlete_state
            return empty

        geo = self.geometry_at(frame_idx)
        if geo.track_mask is not None:
            track_overlap = athlete_mask_overlap(
                bbox, person_seg_mask, geo.track_mask, self.width, self.height,
            )
        else:
            track_overlap = _bbox_polygon_overlap(bbox, geo.track_polygon)

        if geo.sand_mask is not None:
            landing_overlap = athlete_mask_overlap(
                bbox, person_seg_mask, geo.sand_mask, self.width, self.height,
            )
        else:
            landing_overlap = _bbox_polygon_overlap(bbox, geo.landing_zone)
        position_s = _project_position_s(bbox, geo.axis, self.width, self.height)

        cx = (bbox[0] + bbox[2]) * 0.5
        cy = (bbox[1] + bbox[3]) * 0.5
        bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
        self._history.append((frame_idx, cx, cy, bw, bh))
        if len(self._history) > PREDICTION_HISTORY * 2:
            self._history.pop(0)

        if update_state:
            self.athlete_state = self._update_state(track_overlap, landing_overlap, True)

        state = self.athlete_state if update_state else None
        predicted = None
        if update_state and self.athlete_state in ("AIR", "FINAL_FLIGHT"):
            predicted = self._predict_bbox(frame_idx)
            self._predicted_bbox = predicted

        return {
            "track_overlap": round(track_overlap, 4),
            "athlete_state": state,
            "position_s": round(position_s, 3) if position_s is not None else None,
            "predicted_bbox": (
                [round(v, 2) for v in predicted] if predicted else None
            ),
        }

    @property
    def in_air_mode(self) -> bool:
        return self.athlete_state in ("AIR", "FINAL_FLIGHT")

    def candidate_selection_score(
        self,
        bbox: tuple[float, float, float, float],
        frame_idx: int,
        appearance_sim: float,
        motion_px: float = 0.0,
    ) -> float:
        """Higher = more likely to be the athlete."""
        geo = self.geometry_at(frame_idx)
        if geo.track_mask is not None:
            track_overlap = athlete_mask_overlap(
                bbox, None, geo.track_mask, self.width, self.height,
            )
        else:
            track_overlap = _bbox_polygon_overlap(bbox, geo.track_polygon)
        if geo.sand_mask is not None:
            landing_overlap = athlete_mask_overlap(
                bbox, None, geo.sand_mask, self.width, self.height,
            )
        else:
            landing_overlap = _bbox_polygon_overlap(bbox, geo.landing_zone)
        score = 0.0

        if track_overlap > HIGH_OVERLAP and motion_px > 5:
            score += 2.0 + track_overlap

        if landing_overlap >= LANDING_OVERLAP:
            score += 1.5 + landing_overlap

        if self.in_air_mode:
            score += appearance_sim * 1.2
            pred = self._predicted_bbox or self._predict_bbox(frame_idx)
            if pred:
                pcx = (pred[0] + pred[2]) * 0.5
                pcy = (pred[1] + pred[3]) * 0.5
                cx = (bbox[0] + bbox[2]) * 0.5
                cy = (bbox[1] + bbox[3]) * 0.5
                dist = ((cx - pcx) ** 2 + (cy - pcy) ** 2) ** 0.5
                score += max(0.0, 1.5 - dist / max(self.height * 0.15, 1))
        else:
            score += track_overlap * 0.8 + appearance_sim * 0.5
            if motion_px > 3:
                score += min(motion_px / 200.0, 0.5)

        return score


def recompute_frames_track_fields(
    frames: list[dict],
    output_dir: Path,
    *,
    width: int,
    height: int,
) -> tuple[list[dict], int]:
    """
    Fill track_overlap / athlete_state / position_s / predicted_bbox
    on existing frame records. Returns (updated_frames, count_updated).
    """
    ctx = TrackScorerContext.from_output_dir(output_dir, width, height)
    if ctx is None:
        return frames, 0

    updated = 0
    out: list[dict] = []
    for rec in frames:
        rec = dict(rec)
        bbox_raw = rec.get("person_bbox")
        bbox = tuple(bbox_raw[:4]) if bbox_raw and len(bbox_raw) >= 4 else None
        fields = ctx.score_bbox(bbox, int(rec.get("frame_idx", 0)))
        for key, val in fields.items():
            if val is not None:
                rec[key] = val
        updated += 1
        out.append(rec)
    return out, updated
