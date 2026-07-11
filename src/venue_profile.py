"""Venue track/sand learning from manual calibration for auto-detection."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

DEFAULT_VENUE_ID = "default"
VENUE_ROOT = Path("venues")
PROFILE_VERSION = 3

TRACK_H_PRIOR = (35, 100)
TRACK_S_MIN = 10
TRACK_V_MIN = 80
SAND_H_PRIOR = (10, 35)
SAND_S_MIN = 40

ERODE_PX = 10
HSV_K_LEARN = 1.5
HSV_K_DETECT = 2.0

MAX_TRACK_AREA = 0.25
MAX_SAND_AREA = 0.20


def profile_path(venue_id: str = DEFAULT_VENUE_ID) -> Path:
    return VENUE_ROOT / venue_id / "profile.json"


def load_profile(venue_id: str = DEFAULT_VENUE_ID) -> Optional[dict[str, Any]]:
    path = profile_path(venue_id)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_profile(data: dict[str, Any], venue_id: str = DEFAULT_VENUE_ID) -> Path:
    path = profile_path(venue_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return path


def _color_stats_block(profile: dict[str, Any], *, sand: bool = False) -> Optional[dict[str, Any]]:
    """Read track/sand HSV stats from v3 (track_color) or v2 (track_hsv) profile."""
    if sand:
        block = profile.get("sand_color") or profile.get("sand_hsv")
    else:
        block = profile.get("track_color") or profile.get("track_hsv")
    if not block:
        return None
    mean = block.get("mean_hsv") or block.get("mean")
    std = block.get("std_hsv") or block.get("std")
    if not mean or not std:
        return None
    return {
        "mean": list(mean),
        "std": list(std),
        "sample_count": int(block.get("sample_count", 0)),
        "negative_samples_count": int(block.get("negative_samples_count", 0)),
    }


def _write_color_block(stats: dict[str, Any], *, sand: bool = False) -> dict[str, Any]:
    return {
        "mean_hsv": stats["mean"],
        "std_hsv": stats["std"],
        "sample_count": int(stats.get("sample_count", 0)),
        "negative_samples_count": int(stats.get("negative_samples_count", 0)),
    }


def _merge_hsv_stats(
    existing: Optional[dict[str, Any]],
    new_pixels: np.ndarray,
) -> Optional[dict[str, Any]]:
    new_stats = _hsv_stats(new_pixels)
    if new_stats is None:
        return existing
    if existing is None:
        return new_stats
    n_old = int(existing.get("sample_count", 0))
    n_new = int(new_stats["sample_count"])
    if n_new <= 0:
        return existing
    mean_old = np.array(existing["mean"], dtype=np.float64)
    std_old = np.array(existing["std"], dtype=np.float64)
    mean_new = new_pixels.astype(np.float64).mean(axis=0)
    var_new = new_pixels.astype(np.float64).var(axis=0)
    n_total = n_old + n_new
    mean_total = (mean_old * n_old + mean_new * n_new) / n_total
    var_old = std_old ** 2
    var_total = (
        (var_old + mean_old ** 2) * n_old
        + (var_new + mean_new ** 2) * n_new
    ) / n_total - mean_total ** 2
    var_total = np.maximum(var_total, 0.0)
    return {
        "mean": [float(v) for v in mean_total],
        "std": [float(v) for v in np.sqrt(var_total)],
        "sample_count": n_total,
        "negative_samples_count": int(existing.get("negative_samples_count", 0)),
    }


def _selection_to_masks(
    selection: dict[str, Any],
    width: int,
    height: int,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    from .mask_utils import decode_mask_sample, erode_mask, polygon_to_mask

    track_mask = None
    sand_mask = None
    if selection.get("track_mask") is not None:
        track_mask = decode_mask_sample(selection["track_mask"], width, height)
    elif selection.get("track_polygon"):
        track_mask = polygon_to_mask(selection["track_polygon"], width, height)
    if selection.get("sand_mask") is not None:
        sand_mask = decode_mask_sample(selection["sand_mask"], width, height)
    elif selection.get("sand_polygon"):
        sand_mask = polygon_to_mask(selection["sand_polygon"], width, height)
    elif selection.get("landing_zone"):
        sand_mask = polygon_to_mask(selection["landing_zone"], width, height)
    if track_mask is not None:
        track_mask = erode_mask(track_mask, ERODE_PX)
    if sand_mask is not None:
        sand_mask = erode_mask(sand_mask, ERODE_PX)
    return track_mask, sand_mask


def _clamp_norm(x: float, y: float) -> list[float]:
    return [max(0.0, min(1.0, x)), max(0.0, min(1.0, y))]


def _polygon_mask(
    points: list[list[float]], width: int, height: int,
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    if len(points) < 3:
        return mask
    pts = np.array(
        [[int(p[0] * width), int(p[1] * height)] for p in points],
        dtype=np.int32,
    )
    cv2.fillPoly(mask, [pts], 255)
    return mask


def _eroded_polygon_mask(
    points: list[list[float]], width: int, height: int, erode_px: int = ERODE_PX,
) -> np.ndarray:
    mask = _polygon_mask(points, width, height)
    if erode_px <= 0:
        return mask
    k = erode_px * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.erode(mask, kernel, iterations=1)


def _hsv_stats(pixels_hsv: np.ndarray) -> Optional[dict[str, Any]]:
    if pixels_hsv is None or len(pixels_hsv) < 50:
        return None
    h = pixels_hsv[:, 0].astype(np.float32)
    s = pixels_hsv[:, 1].astype(np.float32)
    v = pixels_hsv[:, 2].astype(np.float32)
    return {
        "mean": [float(h.mean()), float(s.mean()), float(v.mean())],
        "std": [float(h.std()), float(s.std()), float(v.std())],
        "sample_count": int(len(pixels_hsv)),
    }


def _apply_hsv_priors(
    low: list[int], high: list[int], *, track: bool,
) -> tuple[list[int], list[int]]:
    if track:
        low[0] = max(low[0], TRACK_H_PRIOR[0])
        high[0] = min(high[0], TRACK_H_PRIOR[1])
        low[1] = max(low[1], TRACK_S_MIN)
        low[2] = max(low[2], TRACK_V_MIN)
    else:
        low[0] = max(low[0], SAND_H_PRIOR[0])
        high[0] = min(high[0], SAND_H_PRIOR[1])
        low[1] = max(low[1], SAND_S_MIN)
    return low, high


def _hsv_ranges_from_stats(
    stats: dict[str, Any],
    *,
    k: float = HSV_K_LEARN,
    track: bool = True,
) -> dict[str, list[int]]:
    """Build tight OpenCV inRange bounds from mean ± k*std."""
    mean = stats["mean"]
    std = stats["std"]
    low = [
        max(0, int(round(mean[0] - k * std[0]))),
        max(0, int(round(mean[1] - k * std[1]))),
        max(0, int(round(mean[2] - k * std[2]))),
    ]
    high = [
        min(179, int(round(mean[0] + k * std[0]))),
        min(255, int(round(mean[1] + k * std[1]))),
        min(255, int(round(mean[2] + k * std[2]))),
    ]
    low, high = _apply_hsv_priors(low, high, track=track)
    if low[0] > high[0] or low[1] > high[1] or low[2] > high[2]:
        mid = stats["mean"]
        low = [max(0, int(mid[0] - k)), max(0, int(mid[1] - k)), max(0, int(mid[2] - k))]
        high = [
            min(179, int(mid[0] + k)),
            min(255, int(mid[1] + k)),
            min(255, int(mid[2] + k)),
        ]
        low, high = _apply_hsv_priors(low, high, track=track)
    return {"low": low, "high": high}


def _polygon_shape_stats(polygon: list[list[float]]) -> dict[str, float]:
    pts = np.array(polygon, dtype=np.float32)
    if len(pts) < 3:
        return {}
    area = float(cv2.contourArea(pts))
    rect = cv2.minAreaRect(pts)
    w, h = rect[1]
    aspect = max(w, h) / max(min(w, h), 1e-6)
    return {
        "area_norm": area,
        "aspect_ratio": float(aspect),
        "center_x": float(pts[:, 0].mean()),
        "center_y": float(pts[:, 1].mean()),
    }


def _keyframe_source(kf: dict[str, Any]) -> str:
    explicit = kf.get("source")
    if explicit in ("manual", "venue_auto"):
        return explicit
    if kf.get("venue_confidence") is not None:
        return "venue_auto"
    poly = kf.get("track_polygon") or []
    if len(poly) > 4:
        return "manual"
    stats = _polygon_shape_stats(poly)
    if stats.get("area_norm", 0.0) > 0.5:
        return "venue_auto"
    return "manual"


def _select_learn_keyframes(keyframes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """All keyframes with a usable track polygon (exclude full-frame venue-auto quads)."""
    return [
        k for k in keyframes
        if len(k.get("track_polygon") or []) >= 3
        and _polygon_shape_stats(k["track_polygon"]).get("area_norm", 1.0) <= 0.5
    ]


def _lane_orientation_hint(frame_bgr: np.ndarray, mask: np.ndarray) -> Optional[float]:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.bitwise_and(edges, mask)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=40, maxLineGap=8)
    if lines is None:
        return None
    angles: list[float] = []
    for line in lines[:40]:
        x1, y1, x2, y2 = line[0]
        ang = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        angles.append(float(ang))
    if not angles:
        return None
    return float(np.median(angles))


def _order_quad_corners(pts: np.ndarray) -> np.ndarray:
    if len(pts) != 4:
        return pts
    pts = np.array(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _fit_quad_from_contour(contour: np.ndarray, width: int, height: int) -> list[list[float]]:
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
    if len(approx) == 4:
        pts = approx.reshape(-1, 2).astype(np.float32)
    else:
        rect = cv2.minAreaRect(contour)
        pts = cv2.boxPoints(rect).astype(np.float32)
    pts = _order_quad_corners(pts)
    out: list[list[float]] = []
    for x, y in pts:
        out.append(_clamp_norm(x / width, y / height))
    return out


def _gaussian(x: float, mu: float, sigma: float) -> float:
    if sigma <= 0:
        return 1.0 if abs(x - mu) < 1e-9 else 0.0
    return math.exp(-0.5 * ((x - mu) / sigma) ** 2)


def _spatial_roi_mask(
    width: int, height: int, profile: dict[str, Any], *, sand: bool = False,
) -> np.ndarray:
    roi = np.zeros((height, width), dtype=np.uint8)
    center = profile.get("expected_center")
    if center is None:
        ss = profile.get("shape_stats", {})
        center = [ss.get("center_x", 0.5), ss.get("center_y", 0.6)]
    cx, cy = center[0], center[1]
    if sand:
        y0 = int(max(0, (cy - 0.05) * height))
        y1 = height
        x0 = int(max(0, (cx - 0.35) * width))
    else:
        y0 = int(max(0, (cy - 0.22) * height))
        y1 = int(min(height, (cy + 0.22) * height))
        x0 = int(max(0, (cx - 0.35) * width))
    roi[y0:y1, x0:width] = 255
    return roi


def _build_color_mask(
    frame_hsv: np.ndarray,
    profile: dict[str, Any],
    *,
    sand: bool = False,
) -> np.ndarray:
    stats = _color_stats_block(profile, sand=sand)
    range_key = "sand_hsv_range" if sand else "track_hsv_range"
    if stats is None and not profile.get(range_key):
        return np.zeros(frame_hsv.shape[:2], dtype=np.uint8)

    h, s, v = cv2.split(frame_hsv)
    if sand and stats:
        mean = np.array(stats["mean"], dtype=np.float32)
        std = np.array(stats["std"], dtype=np.float32)
        k = HSV_K_DETECT
        low = mean - k * std
        high = mean + k * std
        low[0] = max(low[0], SAND_H_PRIOR[0])
        high[0] = min(high[0], SAND_H_PRIOR[1])
        low[1] = max(low[1], SAND_S_MIN)
        mask = (
            (h >= low[0]) & (h <= high[0])
            & (s >= low[1]) & (s <= high[1])
            & (v >= low[2]) & (v <= high[2])
        )
        out = (mask.astype(np.uint8) * 255)
    elif sand:
        hsv_range = profile.get("sand_hsv_range")
        if not hsv_range:
            return np.zeros(frame_hsv.shape[:2], dtype=np.uint8)
        low = np.array(hsv_range["low"], dtype=np.uint8)
        high = np.array(hsv_range["high"], dtype=np.uint8)
        out = cv2.inRange(frame_hsv, low, high)
    elif stats:
        mean_h = stats["mean"][0]
        std_h = stats["std"][0]
        mean_s = stats["mean"][1]
        std_s = stats["std"][1]
        mean_v = stats["mean"][2]
        std_v = stats["std"][2]
        k = HSV_K_DETECT
        h_lo = max(TRACK_H_PRIOR[0], mean_h - k * std_h)
        h_hi = min(TRACK_H_PRIOR[1], mean_h + k * std_h)
        s_lo = max(TRACK_S_MIN, mean_s - k * std_s)
        s_hi = min(255, mean_s + k * std_s)
        v_lo = max(TRACK_V_MIN, mean_v - k * std_v)
        out = (
            (h >= h_lo) & (h <= h_hi)
            & (s >= s_lo) & (s <= s_hi)
            & (v >= v_lo)
        ).astype(np.uint8) * 255
    else:
        hsv_range = profile.get("track_hsv_range")
        if not hsv_range:
            return np.zeros(frame_hsv.shape[:2], dtype=np.uint8)
        low = np.array(hsv_range["low"], dtype=np.uint8)
        high = np.array(hsv_range["high"], dtype=np.uint8)
        out = cv2.inRange(frame_hsv, low, high)

    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21 if not sand else 11, 21 if not sand else 11))
    open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    out = cv2.morphologyEx(out, cv2.MORPH_OPEN, open_k, iterations=1)
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, close_k, iterations=2)
    return out


def _score_contour(
    contour: np.ndarray,
    width: int,
    height: int,
    profile: dict[str, Any],
    *,
    sand: bool = False,
) -> float:
    area_norm = cv2.contourArea(contour) / (width * height)
    max_area = MAX_SAND_AREA if sand else MAX_TRACK_AREA
    if area_norm > max_area:
        return -1.0

    expected_area = profile.get("expected_area_norm")
    if expected_area is None:
        expected_area = profile.get("shape_stats", {}).get("area_norm", 0.12)
    expected_aspect = profile.get("expected_aspect_ratio")
    if expected_aspect is None:
        expected_aspect = profile.get("shape_stats", {}).get("aspect_ratio", 4.0)
    expected_center = profile.get("expected_center")
    if expected_center is None:
        ss = profile.get("shape_stats", {})
        expected_center = [ss.get("center_x", 0.5), ss.get("center_y", 0.6)]

    rect = cv2.minAreaRect(contour)
    rw, rh = rect[1]
    aspect = max(rw, rh) / max(min(rw, rh), 1e-6)
    M = cv2.moments(contour)
    if M["m00"] > 0:
        cx = (M["m10"] / M["m00"]) / width
        cy = (M["m01"] / M["m00"]) / height
    else:
        cx, cy = expected_center[0], expected_center[1]

    score = (
        0.45 * _gaussian(area_norm, expected_area, 0.04)
        + 0.30 * _gaussian(aspect, expected_aspect, 2.0)
        + 0.25 * _gaussian(cy, expected_center[1], 0.15)
    )
    if area_norm > min(max_area, expected_area * 2.5):
        return -1.0
    if area_norm < max(0.015, expected_area * 0.2):
        return -1.0
    return score


def _pick_best_contour(
    mask: np.ndarray,
    width: int,
    height: int,
    profile: dict[str, Any],
    *,
    sand: bool = False,
    roi_y_min: int = 0,
) -> tuple[Optional[np.ndarray], float, float]:
    if roi_y_min > 0:
        roi = np.zeros_like(mask)
        roi[roi_y_min:, :] = 255
        mask = cv2.bitwise_and(mask, roi)

    spatial = _spatial_roi_mask(width, height, profile, sand=sand)
    mask = cv2.bitwise_and(mask, spatial)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return None, 0.0, 0.0

    expected_center = profile.get("expected_center")
    if expected_center is None:
        ss = profile.get("shape_stats", {})
        expected_center = [ss.get("center_x", 0.5), ss.get("center_y", 0.6)]

    best: Optional[np.ndarray] = None
    best_score = -1.0
    best_area = 0.0
    for label_idx in range(1, num_labels):
        area_px = stats[label_idx, cv2.CC_STAT_AREA]
        if area_px <= 0:
            continue
        comp = (labels == label_idx).astype(np.uint8) * 255
        contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        cy = centroids[label_idx][1] / height
        if not sand and abs(cy - expected_center[1]) > 0.18:
            continue
        score = _score_contour(contour, width, height, profile, sand=sand)
        area_norm = area_px / (width * height)
        if score > best_score:
            best_score = score
            best = contour
            best_area = area_norm

    if best is None or best_score <= 0:
        return None, 0.0, 0.0
    return best, best_score, best_area


def _largest_blob_mask(
    mask: np.ndarray,
    *,
    roi_y_min: int = 0,
    max_area_norm: float = MAX_TRACK_AREA,
    min_area_norm: float = 0.005,
) -> tuple[Optional[np.ndarray], float]:
    """Fallback: keep largest connected component within area bounds."""
    h, w = mask.shape[:2]
    work = mask.copy()
    if roi_y_min > 0:
        roi = np.zeros_like(work)
        roi[roi_y_min:, :] = 255
        work = cv2.bitwise_and(work, roi)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(work, connectivity=8)
    if num_labels <= 1:
        return None, 0.0
    best_label = -1
    best_area_px = 0
    frame_area = w * h
    for label_idx in range(1, num_labels):
        area_px = stats[label_idx, cv2.CC_STAT_AREA]
        area_norm = area_px / frame_area
        if area_norm < min_area_norm or area_norm > max_area_norm:
            continue
        if area_px > best_area_px:
            best_area_px = area_px
            best_label = label_idx
    if best_label < 0:
        return None, 0.0
    out = (labels == best_label).astype(np.uint8) * 255
    return out, best_area_px / frame_area


def learn_from_selections(
    video_path: str | Path,
    selections: list[dict[str, Any]],
    *,
    accumulate: bool = True,
    venue_id: str = DEFAULT_VENUE_ID,
    output_dir: Optional[str | Path] = None,
) -> dict[str, Any]:
    """
    Learn or accumulate HSV color models from brush masks and/or polygons.
    selections: [{frame_idx, track_mask|track_polygon, sand_mask|sand_polygon, source}]
    """
    video_path = Path(video_path)
    if not selections:
        raise ValueError("No selections provided for venue learning.")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)

    track_pixels: list[np.ndarray] = []
    sand_pixels: list[np.ndarray] = []
    shape_stats: list[dict[str, float]] = []
    lane_angles: list[float] = []
    frames_used = 0
    sand_frames = 0

    for sel in selections:
        frame_idx = int(sel["frame_idx"])
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        h, w = frame.shape[:2]
        track_mask, sand_mask = _selection_to_masks(sel, w, h)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        if track_mask is not None and (track_mask > 0).sum() >= 50:
            track_sel = hsv[track_mask > 0]
            track_pixels.append(track_sel)
            contours, _ = cv2.findContours(track_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cnt = max(contours, key=cv2.contourArea)
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
                poly = [[p[0] / w, p[1] / h] for p in approx.reshape(-1, 2)]
                if len(poly) >= 3:
                    shape_stats.append(_polygon_shape_stats(poly))
            ang = _lane_orientation_hint(frame, track_mask)
            if ang is not None:
                lane_angles.append(ang)
            frames_used += 1

        if sand_mask is not None and (sand_mask > 0).sum() >= 30:
            sand_sel = hsv[sand_mask > 0]
            sand_pixels.append(sand_sel)
            sand_frames += 1

    cap.release()

    if not track_pixels:
        raise ValueError("Could not extract track pixels from selections.")

    all_track = np.vstack(track_pixels)
    track_stats = _hsv_stats(all_track)
    if track_stats is None:
        raise ValueError("Insufficient track pixel samples for HSV model.")

    sand_stats = None
    if sand_pixels:
        all_sand = np.vstack(sand_pixels)
        sand_stats = _hsv_stats(all_sand)

    existing = load_profile(venue_id) if accumulate else None
    if existing and accumulate:
        prev_track = _color_stats_block(existing, sand=False)
        prev_sand = _color_stats_block(existing, sand=True)
        track_stats = _merge_hsv_stats(prev_track, all_track) or track_stats
        if sand_stats and sand_pixels:
            all_sand = np.vstack(sand_pixels)
            sand_stats = _merge_hsv_stats(prev_sand, all_sand) or sand_stats
        elif prev_sand:
            sand_stats = prev_sand

    areas = [s["area_norm"] for s in shape_stats if "area_norm" in s]
    aspects = [s["aspect_ratio"] for s in shape_stats if "aspect_ratio" in s]
    centers_x = [s["center_x"] for s in shape_stats if "center_x" in s]
    centers_y = [s["center_y"] for s in shape_stats if "center_y" in s]

    videos = list(existing.get("videos_contributed", [])) if existing else []
    if existing and existing.get("source_video") and existing["source_video"] not in videos:
        videos.append(existing["source_video"])
    if video_path.name not in videos:
        videos.append(video_path.name)

    prev_frames = int(existing.get("frames_used", 0)) if existing and accumulate else 0
    prev_sand_frames = int(existing.get("sand_frames_used", 0)) if existing and accumulate else 0

    profile: dict[str, Any] = {
        "version": PROFILE_VERSION,
        "venue_id": venue_id,
        "videos_contributed": videos,
        "accumulated_samples": accumulate and existing is not None,
        "source_video": video_path.name,
        "learned_at": datetime.now(timezone.utc).isoformat(),
        "frames_used": prev_frames + frames_used,
        "sand_frames_used": prev_sand_frames + sand_frames,
        "learned_from_manual": True,
        "video_size": [width, height],
        "track_color": _write_color_block(track_stats),
        "track_hsv": track_stats,
        "track_hsv_range": _hsv_ranges_from_stats(track_stats, k=HSV_K_LEARN, track=True),
        "expected_area_norm": float(np.median(areas)) if areas else (
            existing.get("expected_area_norm", 0.12) if existing else 0.12
        ),
        "expected_aspect_ratio": float(np.median(aspects)) if aspects else (
            existing.get("expected_aspect_ratio", 4.0) if existing else 4.0
        ),
        "expected_center": [
            float(np.median(centers_x)) if centers_x else (
                existing.get("expected_center", [0.5, 0.6])[0] if existing else 0.5
            ),
            float(np.median(centers_y)) if centers_y else (
                existing.get("expected_center", [0.5, 0.6])[1] if existing else 0.6
            ),
        ],
        "shape_stats": {
            "area_norm": float(np.median(areas)) if areas else (
                existing.get("shape_stats", {}).get("area_norm", 0.12) if existing else 0.12
            ),
            "aspect_ratio": float(np.median(aspects)) if aspects else (
                existing.get("shape_stats", {}).get("aspect_ratio", 4.0) if existing else 4.0
            ),
            "center_x": float(np.median(centers_x)) if centers_x else (
                existing.get("shape_stats", {}).get("center_x", 0.5) if existing else 0.5
            ),
            "center_y": float(np.median(centers_y)) if centers_y else (
                existing.get("shape_stats", {}).get("center_y", 0.6) if existing else 0.6
            ),
        },
        "lane_angle_deg": float(np.median(lane_angles)) if lane_angles else (
            existing.get("lane_angle_deg") if existing else None
        ),
    }
    if existing and existing.get("hops_corridor_m") is not None:
        profile["hops_corridor_m"] = existing["hops_corridor_m"]
    if sand_stats:
        profile["sand_color"] = _write_color_block(sand_stats)
        profile["sand_hsv"] = sand_stats
        profile["sand_hsv_range"] = _hsv_ranges_from_stats(sand_stats, k=HSV_K_LEARN, track=False)
    elif existing and existing.get("sand_color"):
        profile["sand_color"] = existing["sand_color"]
        profile["sand_hsv"] = existing.get("sand_hsv")
        profile["sand_hsv_range"] = existing.get("sand_hsv_range")

    save_profile(profile, venue_id)
    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "venue_profile_learned.json", "w", encoding="utf-8") as f:
            json.dump(profile, f, indent=2)
    return profile


def _selections_from_calibration(cal: dict[str, Any]) -> list[dict[str, Any]]:
    selections: list[dict[str, Any]] = []
    for kf in _select_learn_keyframes(cal.get("keyframes") or []):
        sel: dict[str, Any] = {
            "frame_idx": int(kf["frame_idx"]),
            "source": kf.get("source", "manual"),
        }
        if kf.get("track_polygon"):
            sel["track_polygon"] = kf["track_polygon"]
        if kf.get("landing_zone"):
            sel["sand_polygon"] = kf["landing_zone"]
        selections.append(sel)
    return selections


def selections_from_calibration(cal: dict[str, Any]) -> list[dict[str, Any]]:
    """Public helper to build learn selections from calibration keyframes."""
    return _selections_from_calibration(cal)


def learn_from_calibration(
    video_path: str | Path,
    calibration: dict[str, Any] | Path,
    output_dir: Optional[str | Path] = None,
    *,
    venue_id: str = DEFAULT_VENUE_ID,
    accumulate: bool = True,
    extra_samples: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    """
    Learn HSV color models from calibration keyframes (and optional brush samples).
    """
    if isinstance(calibration, Path):
        with open(calibration, encoding="utf-8") as f:
            cal = json.load(f)
    else:
        cal = calibration

    selections = _selections_from_calibration(cal)
    if extra_samples:
        by_frame = {int(s["frame_idx"]): s for s in selections}
        for sample in extra_samples:
            fidx = int(sample["frame_idx"])
            if fidx in by_frame:
                merged = {**by_frame[fidx], **sample}
                by_frame[fidx] = merged
            else:
                by_frame[fidx] = sample
        selections = list(by_frame.values())

    if not selections:
        raise ValueError("No usable keyframes with track_polygon found in calibration.")

    return learn_from_selections(
        video_path,
        selections,
        accumulate=accumulate,
        venue_id=venue_id,
        output_dir=output_dir,
    )


def segment_frame_masks(
    frame_bgr: np.ndarray,
    profile: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Per-frame HSV segmentation → full-resolution track/sand masks and confidence.
    """
    h, w = frame_bgr.shape[:2]
    empty = np.zeros((h, w), dtype=np.uint8)
    if not _color_stats_block(profile, sand=False) and not profile.get("track_hsv_range"):
        return empty, empty, 0.0

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    track_raw = _build_color_mask(hsv, profile, sand=False)

    lower_roi = np.zeros((h, w), dtype=np.uint8)
    lower_roi[int(h * 0.3):, :] = 255
    track_raw = cv2.bitwise_and(track_raw, lower_roi)

    best, track_score, track_area = _pick_best_contour(track_raw, w, h, profile, sand=False)
    track_mask = empty.copy()
    if best is not None and track_area <= MAX_TRACK_AREA:
        cv2.drawContours(track_mask, [best], -1, 255, -1)
    else:
        blob, blob_area = _largest_blob_mask(track_raw, min_area_norm=0.005, max_area_norm=MAX_TRACK_AREA)
        if blob is not None:
            track_mask = blob
            track_area = blob_area
            track_score = 0.55
        else:
            return empty, empty, 0.0

    if track_area > MAX_TRACK_AREA or (track_mask > 0).sum() == 0:
        return empty, empty, 0.0

    expected_area = profile.get("expected_area_norm", profile.get("shape_stats", {}).get("area_norm", 0.12))
    confidence = float(min(1.0, track_score * 0.6 + _gaussian(track_area, expected_area, 0.04) * 0.4))

    sand_mask = empty.copy()
    if _color_stats_block(profile, sand=True) or profile.get("sand_hsv_range"):
        sand_raw = _build_color_mask(hsv, profile, sand=True)
        M = cv2.moments(best)
        ty = int(M["m01"] / M["m00"]) if M["m00"] > 0 else int(h * 0.55)
        sand_best, sand_score, sand_area = _pick_best_contour(
            sand_raw, w, h, profile, sand=True, roi_y_min=ty,
        )
        if sand_best is not None and sand_area <= MAX_SAND_AREA:
            cv2.drawContours(sand_mask, [sand_best], -1, 255, -1)
            confidence = float(min(1.0, confidence * 0.7 + sand_score * 0.3))
        else:
            sand_blob, sand_area = _largest_blob_mask(
                sand_raw, roi_y_min=ty, max_area_norm=MAX_SAND_AREA, min_area_norm=0.003,
            )
            if sand_blob is not None:
                sand_mask = sand_blob
                confidence = float(min(1.0, confidence * 0.85 + 0.1))

    return track_mask, sand_mask, confidence


def apply_masks_to_output(
    output_dir: Path,
    video_path: Path,
    profile: Optional[dict[str, Any]] = None,
    *,
    venue_id: str = DEFAULT_VENUE_ID,
    prefer_keyframes: bool = True,
) -> dict[str, Any]:
    """Segment every analysis frame, save mask PNGs, update calibration.json."""
    from .calibration import default_calibration, load_calibration, save_calibration
    from .calibration_propagator import target_frames_from_analysis
    from .mask_utils import mask_area_norm, save_mask_png
    from .venue_corrector import _read_gray, _warp_mask_flow
    from .venue_masks import build_masks_from_keyframes, should_use_keyframe_pipeline
    from .venue_seg_infer import has_trained_seg_model, infer_frame_masks, load_venue_seg_model

    output_dir = Path(output_dir)
    video_path = Path(video_path)
    frame_indices = target_frames_from_analysis(output_dir)
    if not frame_indices:
        raise ValueError("No analysis frames found. Run /analyze first.")

    cal = load_calibration(output_dir) or default_calibration(video_path.stem, video_path.name)

    if has_trained_seg_model(venue_id):
        seg_model = load_venue_seg_model(venue_id)
        if seg_model is not None:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise FileNotFoundError(f"Cannot open video: {video_path}")

            masks_dir = output_dir / "venue_masks"
            masks_dir.mkdir(parents=True, exist_ok=True)
            mask_frames: dict[str, dict[str, Any]] = {}
            confidences: list[float] = []

            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                h, w = frame.shape[:2]
                track_mask, sand_mask, confidence = infer_frame_masks(frame, seg_model)
                track_area = mask_area_norm(track_mask, w, h)
                sand_area = mask_area_norm(sand_mask, w, h)

                track_rel = f"venue_masks/track_{frame_idx:06d}.png"
                sand_rel = f"venue_masks/sand_{frame_idx:06d}.png"
                save_mask_png(output_dir / track_rel, track_mask)
                save_mask_png(output_dir / sand_rel, sand_mask)

                mask_frames[str(frame_idx)] = {
                    "track": track_rel,
                    "sand": sand_rel,
                    "source": "cnn",
                    "confidence": round(confidence, 3),
                    "track_area_norm": round(track_area, 4),
                    "sand_area_norm": round(sand_area, 4),
                }
                if confidence > 0:
                    confidences.append(confidence)

            cap.release()
            if not mask_frames:
                raise ValueError("CNN mask segmentation produced no frames.")

            mean_conf = float(sum(confidences) / len(confidences)) if confidences else 0.0
            cal["version"] = max(int(cal.get("version", 1)), 3)
            cal["mode"] = "cnn_masks"
            cal["video"] = video_path.name
            cal["mask_frames"] = mask_frames
            cal["venue_profile"] = {
                "venue_id": venue_id,
                "applied_at": datetime.now(timezone.utc).isoformat(),
                "frames_detected": len(mask_frames),
                "mode": "cnn_masks",
                "mean_confidence": round(mean_conf, 3),
            }
            save_calibration(output_dir, cal)
            return cal

    if should_use_keyframe_pipeline(cal, prefer_keyframes=prefer_keyframes):
        result = build_masks_from_keyframes(
            video_path,
            cal,
            frame_indices,
            output_dir,
            use_flow_refinement=True,
        )
        cal["version"] = max(int(cal.get("version", 1)), 3)
        cal["mode"] = "keyframe_masks"
        cal["video"] = video_path.name
        cal["mask_frames"] = result["mask_frames"]
        cal["venue_profile"] = {
            "venue_id": venue_id,
            "applied_at": datetime.now(timezone.utc).isoformat(),
            "frames_detected": result["frames_applied"],
            "mode": "keyframe_masks",
            "keyframes_used": result["keyframes_used"],
            "source_counts": result["source_counts"],
        }
        save_calibration(output_dir, cal)
        return cal

    profile = profile or load_profile(venue_id)
    if profile is None:
        raise FileNotFoundError("No venue profile found. Run learn first or add manual keyframes.")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    masks_dir = output_dir / "venue_masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    segmented: dict[int, dict[str, Any]] = {}

    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        h, w = frame.shape[:2]
        track_mask, sand_mask, confidence = segment_frame_masks(frame, profile)
        track_area = mask_area_norm(track_mask, w, h)
        sand_area = mask_area_norm(sand_mask, w, h)
        segmented[frame_idx] = {
            "track_mask": track_mask,
            "sand_mask": sand_mask,
            "confidence": confidence,
            "track_area": track_area,
            "sand_area": sand_area,
            "width": w,
            "height": h,
        }

    successful_indices = [
        idx for idx, data in segmented.items() if data["track_area"] > 0
    ]
    if not successful_indices:
        cap.release()
        raise ValueError("Mask segmentation produced no frames.")

    def _interpolate_from_nearest(frame_idx: int, data: dict[str, Any]) -> dict[str, Any]:
        if data["track_area"] > 0:
            return data
        nearest = min(successful_indices, key=lambda k: abs(k - frame_idx))
        source = segmented[nearest]
        track_mask = source["track_mask"].copy()
        sand_mask = source["sand_mask"].copy()
        confidence = float(source["confidence"]) * 0.85
        if nearest != frame_idx:
            nearest_gray = _read_gray(cap, nearest)
            target_gray = _read_gray(cap, frame_idx)
            if nearest_gray is not None and target_gray is not None:
                if nearest < frame_idx:
                    track_mask = _warp_mask_flow(nearest_gray, target_gray, track_mask)
                    sand_mask = _warp_mask_flow(nearest_gray, target_gray, sand_mask)
                else:
                    track_mask = _warp_mask_flow(target_gray, nearest_gray, track_mask)
                    sand_mask = _warp_mask_flow(target_gray, nearest_gray, sand_mask)
        w, h = data["width"], data["height"]
        return {
            "track_mask": track_mask,
            "sand_mask": sand_mask,
            "confidence": confidence,
            "track_area": mask_area_norm(track_mask, w, h),
            "sand_area": mask_area_norm(sand_mask, w, h),
            "width": w,
            "height": h,
            "interpolated": True,
        }

    mask_frames: dict[str, dict[str, Any]] = {}
    keyframes: list[dict[str, Any]] = []

    for frame_idx in frame_indices:
        if frame_idx not in segmented:
            if not successful_indices:
                continue
            nearest = min(successful_indices, key=lambda k: abs(k - frame_idx))
            source = segmented[nearest]
            w, h = source["width"], source["height"]
            segmented[frame_idx] = {
                "track_mask": source["track_mask"].copy(),
                "sand_mask": source["sand_mask"].copy(),
                "confidence": float(source["confidence"]) * 0.8,
                "track_area": 0.0,
                "sand_area": 0.0,
                "width": w,
                "height": h,
            }
        data = _interpolate_from_nearest(frame_idx, segmented[frame_idx])
        track_mask = data["track_mask"]
        sand_mask = data["sand_mask"]
        confidence = data["confidence"]
        track_area = data["track_area"]
        sand_area = data["sand_area"]
        w, h = data["width"], data["height"]

        track_rel = f"venue_masks/track_{frame_idx:06d}.png"
        sand_rel = f"venue_masks/sand_{frame_idx:06d}.png"
        save_mask_png(output_dir / track_rel, track_mask)
        save_mask_png(output_dir / sand_rel, sand_mask)

        entry: dict[str, Any] = {
            "track": track_rel,
            "sand": sand_rel,
            "source": "color",
            "confidence": round(confidence, 3),
            "track_area_norm": round(track_area, 4),
            "sand_area_norm": round(sand_area, 4),
        }
        if data.get("interpolated"):
            entry["interpolated"] = True
        mask_frames[str(frame_idx)] = entry

        track_polygon: list[list[float]] = []
        contours, _ = cv2.findContours(track_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            track_polygon = _fit_quad_from_contour(max(contours, key=cv2.contourArea), w, h)

        landing_zone: list[list[float]] = []
        sand_contours, _ = cv2.findContours(sand_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if sand_contours:
            landing_zone = _fit_quad_from_contour(max(sand_contours, key=cv2.contourArea), w, h)
            if len(landing_zone) < 3:
                landing_zone = []

        kf: dict[str, Any] = {
            "frame_idx": frame_idx,
            "track_polygon": track_polygon if len(track_polygon) >= 4 else [],
            "corridor_polygon": [],
            "landing_zone": landing_zone,
            "venue_confidence": round(confidence, 3),
            "venue_area_norm": round(track_area, 4),
            "source": "venue_auto",
        }
        keyframes.append(kf)

    cap.release()
    if not mask_frames:
        raise ValueError("Mask segmentation produced no frames.")

    cal = load_calibration(output_dir) or default_calibration(video_path.stem, video_path.name)
    cal["version"] = max(int(cal.get("version", 1)), 3)
    cal["mode"] = "color_masks"
    cal["video"] = video_path.name
    cal["mask_frames"] = mask_frames
    existing_keyframes = cal.get("keyframes") or []
    manual_by_idx = {
        int(k["frame_idx"]): k
        for k in existing_keyframes
        if _keyframe_source(k) == "manual"
    }
    if manual_by_idx:
        merged = {int(k["frame_idx"]): k for k in keyframes}
        for fidx, manual_kf in manual_by_idx.items():
            merged[fidx] = manual_kf
        cal["keyframes"] = sorted(merged.values(), key=lambda k: int(k["frame_idx"]))
    else:
        cal["keyframes"] = keyframes
    cal["venue_profile"] = {
        "venue_id": venue_id,
        "applied_at": datetime.now(timezone.utc).isoformat(),
        "frames_detected": len(keyframes),
        "mode": "color_masks",
    }

    save_calibration(output_dir, cal)
    return cal


def detect_track_sand_frame(
    frame_bgr: np.ndarray,
    profile: dict[str, Any],
) -> tuple[list[list[float]], list[list[float]], float, float]:
    """
    Detect track and sand quads in one frame using learned profile.
    Returns normalized quads, confidence score, and track area ratio.
    """
    h, w = frame_bgr.shape[:2]
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    if not _color_stats_block(profile, sand=False) and not profile.get("track_hsv_range"):
        return [], [], 0.0, 0.0

    track_mask = _build_color_mask(hsv, profile, sand=False)
    best, track_score, track_area = _pick_best_contour(track_mask, w, h, profile, sand=False)
    if best is None:
        return [], [], 0.0, 0.0

    track_polygon = _fit_quad_from_contour(best, w, h)
    if len(track_polygon) < 4:
        return [], [], 0.0, 0.0

    expected_area = profile.get("expected_area_norm", profile.get("shape_stats", {}).get("area_norm", 0.12))
    shape_conf = _gaussian(track_area, expected_area, 0.04)
    confidence = float(min(1.0, track_score * 0.6 + shape_conf * 0.4))

    landing_zone: list[list[float]] = []
    if _color_stats_block(profile, sand=True) or profile.get("sand_hsv_range"):
        sand_mask = _build_color_mask(hsv, profile, sand=True)
        track_pts = np.array([[p[0] * w, p[1] * h] for p in track_polygon], dtype=np.int32)
        ty_max = int(track_pts[:, 1].max())
        sand_best, sand_score, sand_area = _pick_best_contour(
            sand_mask, w, h, profile, sand=True, roi_y_min=ty_max,
        )
        if sand_best is not None and sand_area <= MAX_SAND_AREA:
            landing_zone = _fit_quad_from_contour(sand_best, w, h)
            if len(landing_zone) >= 3:
                confidence = float(min(1.0, confidence * 0.7 + sand_score * 0.3))

    return track_polygon, landing_zone, confidence, track_area


def _manual_keyframes_for_propagation(cal: dict[str, Any]) -> list[dict[str, Any]]:
    keyframes = cal.get("keyframes") or []
    manual = [
        k for k in keyframes
        if len(k.get("track_polygon") or []) >= 4
        and _keyframe_source(k) == "manual"
        and _polygon_shape_stats(k["track_polygon"]).get("area_norm", 1.0) <= 0.5
    ]
    if manual:
        return manual
    return [
        k for k in keyframes
        if len(k.get("track_polygon") or []) >= 4
        and _polygon_shape_stats(k["track_polygon"]).get("area_norm", 1.0) <= 0.5
    ]


def _refine_polygon_with_color(
    frame_bgr: np.ndarray,
    polygon: list[list[float]],
    profile: dict[str, Any],
    *,
    radius_px: int = 12,
) -> list[list[float]]:
    """Snap vertices slightly toward nearest color edge within radius."""
    if len(polygon) < 3:
        return polygon
    h, w = frame_bgr.shape[:2]
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = _build_color_mask(hsv, profile, sand=False)
    edges = cv2.Canny(mask, 50, 150)
    refined: list[list[float]] = []
    for nx, ny in polygon:
        cx, cy = int(nx * w), int(ny * h)
        x0, y0 = max(0, cx - radius_px), max(0, cy - radius_px)
        x1, y1 = min(w, cx + radius_px + 1), min(h, cy + radius_px + 1)
        roi = edges[y0:y1, x0:x1]
        ys, xs = np.where(roi > 0)
        if len(xs) == 0:
            refined.append([nx, ny])
            continue
        dist = (xs + x0 - cx) ** 2 + (ys + y0 - cy) ** 2
        best = int(np.argmin(dist))
        refined.append(_clamp_norm((xs[best] + x0) / w, (ys[best] + y0) / h))
    return refined


def propagate_with_profile(
    video_path: str | Path,
    frame_indices: list[int],
    profile: dict[str, Any],
) -> list[dict[str, Any]]:
    """Auto-detect track/sand per frame using venue profile."""
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    keyframes: list[dict[str, Any]] = []
    for frame_idx in sorted(set(frame_indices)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        track_poly, landing, conf, area = detect_track_sand_frame(frame, profile)
        if len(track_poly) < 4:
            continue
        kf: dict[str, Any] = {
            "frame_idx": frame_idx,
            "track_polygon": track_poly,
            "corridor_polygon": [],
            "landing_zone": landing if len(landing) >= 3 else [],
            "venue_confidence": round(conf, 3),
            "venue_area_norm": round(area, 4),
            "source": "venue_auto",
        }
        keyframes.append(kf)

    cap.release()
    return keyframes


def apply_profile_to_output(
    output_dir: Path,
    video_path: Path,
    profile: Optional[dict[str, Any]] = None,
    *,
    venue_id: str = DEFAULT_VENUE_ID,
    merge_existing: bool = True,
    prefer_propagation: bool = True,
) -> dict[str, Any]:
    """Generate calibration keyframes from profile and save to output_dir."""
    from .calibration import default_calibration, load_calibration, normalize_calibration, save_calibration
    from .calibration_propagator import propagate_calibration, target_frames_from_analysis

    profile = profile or load_profile(venue_id)
    if profile is None:
        raise FileNotFoundError("No venue profile found. Run learn first.")

    frame_indices = target_frames_from_analysis(output_dir)
    if not frame_indices:
        raise ValueError("No analysis frames found. Run /analyze first.")

    cal = load_calibration(output_dir) or default_calibration(video_path.stem, video_path.name)
    detected: list[dict[str, Any]] = []
    propagation_used = False

    manual_seeds = _manual_keyframes_for_propagation(cal) if prefer_propagation else []
    if manual_seeds:
        seeds = [
            {
                "frame_idx": int(k["frame_idx"]),
                "track_polygon": k.get("track_polygon") or [],
                "landing_zone": k.get("landing_zone") or [],
            }
            for k in manual_seeds
        ]
        try:
            prop = propagate_calibration(video_path, seeds, frame_indices)
            cap = cv2.VideoCapture(str(video_path))
            for kf in prop.get("keyframes") or []:
                fidx = int(kf["frame_idx"])
                cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
                ok, frame = cap.read()
                track_poly = kf.get("track_polygon") or []
                landing = kf.get("landing_zone") or []
                conf, area = 0.85, _polygon_shape_stats(track_poly).get("area_norm", 0.0)
                if ok and frame is not None and track_poly:
                    track_poly = _refine_polygon_with_color(frame, track_poly, profile)
                    conf = 0.9
                detected.append({
                    "frame_idx": fidx,
                    "track_polygon": track_poly,
                    "corridor_polygon": [],
                    "landing_zone": landing,
                    "venue_confidence": round(conf, 3),
                    "venue_area_norm": round(area, 4),
                    "source": "venue_auto",
                })
            cap.release()
            propagation_used = True
        except Exception:
            detected = []

    covered = {k["frame_idx"] for k in detected if k.get("track_polygon")}
    missing = [f for f in frame_indices if f not in covered]
    if missing:
        color_kfs = propagate_with_profile(video_path, missing, profile)
        detected.extend(color_kfs)

    if not detected:
        raise ValueError("Profile detection produced no keyframes.")

    cal["version"] = max(int(cal.get("version", 1)), 2)
    cal["mode"] = "venue_profile"
    cal["video"] = video_path.name
    cal["venue_profile"] = {
        "venue_id": venue_id,
        "applied_at": datetime.now(timezone.utc).isoformat(),
        "frames_detected": len(detected),
        "propagation_used": propagation_used,
        "prefer_propagation": prefer_propagation,
    }

    if merge_existing and cal.get("keyframes"):
        by_idx = {int(k["frame_idx"]): k for k in cal["keyframes"]}
        for kf in detected:
            existing = by_idx.get(int(kf["frame_idx"]))
            if existing and _keyframe_source(existing) == "manual":
                continue
            by_idx[int(kf["frame_idx"])] = kf
        cal["keyframes"] = sorted(by_idx.values(), key=lambda k: int(k["frame_idx"]))
    else:
        cal["keyframes"] = detected

    data = normalize_calibration(cal)
    save_calibration(output_dir, data)
    return data


def save_debug_frames(
    video_path: Path,
    profile: dict[str, Any],
    output_dir: Path,
    frame_indices: list[int],
    max_frames: int = 5,
) -> list[Path]:
    """Save overlay debug images for venue detection."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    saved: list[Path] = []
    for frame_idx in frame_indices[:max_frames]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        track_poly, landing, conf, area = detect_track_sand_frame(frame, profile)
        vis = frame.copy()
        h, w = vis.shape[:2]
        if len(track_poly) >= 4:
            pts = np.array([[int(p[0] * w), int(p[1] * h)] for p in track_poly], dtype=np.int32)
            cv2.polylines(vis, [pts], True, (0, 255, 0), 2)
        if len(landing) >= 3:
            pts = np.array([[int(p[0] * w), int(p[1] * h)] for p in landing], dtype=np.int32)
            cv2.polylines(vis, [pts], True, (0, 200, 255), 2)
        cv2.putText(
            vis, f"conf={conf:.2f} area={area:.3f} f={frame_idx}", (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
        )
        out_path = output_dir / f"venue_detect_{frame_idx:06d}.jpg"
        cv2.imwrite(str(out_path), vis)
        saved.append(out_path)
    cap.release()
    return saved
