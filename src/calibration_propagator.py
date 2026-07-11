"""Seed-based track calibration propagation via Lucas-Kanade optical flow."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)

MIN_VALID_POINTS_RATIO = 0.5
SNAP_SEARCH_RADIUS_PX = 18


def _clamp_norm(x: float, y: float) -> list[float]:
    return [max(0.0, min(1.0, x)), max(0.0, min(1.0, y))]


def _as_point_list(raw: Any) -> list[list[float]]:
    if not isinstance(raw, list):
        return []
    points: list[list[float]] = []
    for item in raw:
        if (
            isinstance(item, (list, tuple))
            and len(item) >= 2
            and isinstance(item[0], (int, float))
            and isinstance(item[1], (int, float))
        ):
            points.append(_clamp_norm(float(item[0]), float(item[1])))
    return points


def polygon_from_seed(seed: dict[str, Any]) -> list[list[float]]:
    """Build track polygon from seed keyframe data."""
    poly = _as_point_list(seed.get("track_polygon"))
    if len(poly) >= 3:
        return poly

    points = _as_point_list(seed.get("seed_points"))
    labels = seed.get("labels") or []
    if len(points) >= 4:
        corner_labels = {"corner_tl", "corner_tr", "corner_br", "corner_bl"}
        if corner_labels.issubset(set(labels)):
            order = ["corner_tl", "corner_tr", "corner_br", "corner_bl"]
            label_to_pt = {lab: pt for lab, pt in zip(labels, points)}
            return [label_to_pt[lab] for lab in order if lab in label_to_pt]
        return points[:4]
    return points


def landing_zone_from_seed(seed: dict[str, Any]) -> list[list[float]]:
    zone = _as_point_list(seed.get("landing_zone"))
    if len(zone) >= 3:
        return zone

    points = _as_point_list(seed.get("seed_points"))
    labels = seed.get("labels") or []
    arena_pts: list[list[float]] = []
    for pt, lab in zip(points, labels):
        if lab in ("arena_tl", "arena_tr", "arena_br", "arena_bl", "arena"):
            arena_pts.append(pt)
    return arena_pts if len(arena_pts) >= 3 else []


def _read_gray(cap: cv2.VideoCapture, frame_idx: int) -> Optional[np.ndarray]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def _norm_to_px(points: list[list[float]], width: int, height: int) -> np.ndarray:
    pts = np.array([[p[0] * width, p[1] * height] for p in points], dtype=np.float32)
    return pts.reshape(-1, 1, 2)


def _px_to_norm(pts_px: np.ndarray, width: int, height: int) -> list[list[float]]:
    flat = pts_px.reshape(-1, 2)
    out: list[list[float]] = []
    for x, y in flat:
        out.append(_clamp_norm(x / width, y / height))
    return out


def _track_points_lk(
    prev_gray: np.ndarray,
    next_gray: np.ndarray,
    pts_px: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, next_gray, pts_px, None, **LK_PARAMS,
    )
    if next_pts is None or status is None:
        return pts_px.copy(), np.zeros((pts_px.shape[0], 1), dtype=np.uint8)
    return next_pts, status


def _snap_point_to_lines(
    gray: np.ndarray,
    x: float,
    y: float,
    radius: int = SNAP_SEARCH_RADIUS_PX,
) -> tuple[float, float]:
    """Snap a pixel point to nearest Hough line segment within radius."""
    h, w = gray.shape[:2]
    ix, iy = int(round(x)), int(round(y))
    x0 = max(0, ix - radius)
    y0 = max(0, iy - radius)
    x1 = min(w, ix + radius + 1)
    y1 = min(h, iy + radius + 1)
    roi = gray[y0:y1, x0:x1]
    if roi.size == 0:
        return x, y

    edges = cv2.Canny(roi, 50, 150)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=20,
        minLineLength=max(8, radius // 2), maxLineGap=6,
    )
    if lines is None:
        return x, y

    best_dist = float(radius + 1)
    best_pt = (x, y)
    for seg in lines[:, 0]:
        lx0, ly0, lx1, ly1 = seg
        gx0, gy0 = lx0 + x0, ly0 + y0
        gx1, gy1 = lx1 + x0, ly1 + y0
        # Project point onto segment
        dx, dy = gx1 - gx0, gy1 - gy0
        seg_len2 = dx * dx + dy * dy
        if seg_len2 <= 1e-6:
            continue
        t = max(0.0, min(1.0, ((x - gx0) * dx + (y - gy0) * dy) / seg_len2))
        px = gx0 + t * dx
        py = gy0 + t * dy
        dist = ((px - x) ** 2 + (py - y) ** 2) ** 0.5
        if dist < best_dist:
            best_dist = dist
            best_pt = (px, py)
    return best_pt


def _snap_polygon(
    gray: np.ndarray,
    polygon: list[list[float]],
    width: int,
    height: int,
) -> list[list[float]]:
    snapped: list[list[float]] = []
    for nx, ny in polygon:
        x, y = nx * width, ny * height
        sx, sy = _snap_point_to_lines(gray, x, y)
        snapped.append(_clamp_norm(sx / width, sy / height))
    return snapped


def _propagate_chain(
    cap: cv2.VideoCapture,
    width: int,
    height: int,
    start_frame: int,
    polygon: list[list[float]],
    landing: list[list[float]],
    frame_indices: list[int],
    direction: int,
    *,
    snap_to_lines: bool = False,
) -> dict[int, dict[str, list[list[float]]]]:
    """Propagate polygon along consecutive frames in given direction (+1 or -1)."""
    if not polygon:
        return {}

    target_set = set(frame_indices)
    if start_frame not in target_set and direction > 0:
        pass  # still propagate through targets

    results: dict[int, dict[str, list[list[float]]]] = {
        start_frame: {"track_polygon": [p[:] for p in polygon], "landing_zone": [p[:] for p in landing]},
    }

    track_pts = _norm_to_px(polygon, width, height)
    land_pts = _norm_to_px(landing, width, height) if landing else None

    prev_gray = _read_gray(cap, start_frame)
    if prev_gray is None:
        return results

    if snap_to_lines:
        results[start_frame]["track_polygon"] = _snap_polygon(prev_gray, polygon, width, height)
        track_pts = _norm_to_px(results[start_frame]["track_polygon"], width, height)

    cur_frame = start_frame
    sorted_targets = sorted(frame_indices)
    min_f, max_f = sorted_targets[0], sorted_targets[-1]

    if direction > 0:
        stop = max_f
        step_range = range(start_frame + 1, stop + 1)
    else:
        stop = min_f
        step_range = range(start_frame - 1, stop - 1, -1)

    for nxt in step_range:
        nxt_gray = _read_gray(cap, nxt)
        if nxt_gray is None:
            break

        track_pts, track_status = _track_points_lk(prev_gray, nxt_gray, track_pts)
        valid_ratio = float(track_status.sum()) / max(len(track_status), 1)
        if valid_ratio < MIN_VALID_POINTS_RATIO:
            break

        cur_poly = _px_to_norm(track_pts, width, height)
        cur_land: list[list[float]] = []
        if land_pts is not None and len(land_pts) > 0:
            land_pts, land_status = _track_points_lk(prev_gray, nxt_gray, land_pts)
            if float(land_status.sum()) / max(len(land_status), 1) >= MIN_VALID_POINTS_RATIO:
                cur_land = _px_to_norm(land_pts, width, height)
            else:
                land_pts = None

        if snap_to_lines:
            cur_poly = _snap_polygon(nxt_gray, cur_poly, width, height)
            track_pts = _norm_to_px(cur_poly, width, height)

        if nxt in target_set:
            results[nxt] = {
                "track_polygon": cur_poly,
                "landing_zone": cur_land,
            }

        prev_gray = nxt_gray
        cur_frame = nxt

    return results


def _interp_polygons(
    keyframes: list[dict[str, Any]],
    frame_idx: int,
) -> dict[str, list[list[float]]]:
    """Linear interpolation between bracketing keyframes."""
    if not keyframes:
        return {"track_polygon": [], "landing_zone": []}
    if len(keyframes) == 1:
        k = keyframes[0]
        return {
            "track_polygon": k.get("track_polygon") or [],
            "landing_zone": k.get("landing_zone") or [],
        }

    before = [k for k in keyframes if k["frame_idx"] <= frame_idx]
    after = [k for k in keyframes if k["frame_idx"] >= frame_idx]
    if not before:
        k = keyframes[0]
        return {"track_polygon": k.get("track_polygon") or [], "landing_zone": k.get("landing_zone") or []}
    if not after:
        k = keyframes[-1]
        return {"track_polygon": k.get("track_polygon") or [], "landing_zone": k.get("landing_zone") or []}

    k0, k1 = before[-1], after[0]
    if k0["frame_idx"] == k1["frame_idx"]:
        return {
            "track_polygon": k0.get("track_polygon") or [],
            "landing_zone": k0.get("landing_zone") or [],
        }

    t = (frame_idx - k0["frame_idx"]) / max(k1["frame_idx"] - k0["frame_idx"], 1)

    def _lerp(key: str) -> list[list[float]]:
        p0 = k0.get(key) or []
        p1 = k1.get(key) or []
        if len(p0) != len(p1) or not p0:
            return p0 or p1
        return [
            [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]
            for a, b in zip(p0, p1)
        ]

    return {"track_polygon": _lerp("track_polygon"), "landing_zone": _lerp("landing_zone")}


def propagate_calibration(
    video_path: str | Path,
    seeds: list[dict[str, Any]],
    target_frame_indices: list[int],
    *,
    snap_to_lines: bool = False,
    from_frame: Optional[int] = None,
) -> dict[str, Any]:
    """
    Propagate seed polygons to target frames using optical flow.

    Returns dict with keyframes list, per_frame_polygons map, and propagation metadata.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    if not seeds:
        raise ValueError("No seeds provided")
    if not target_frame_indices:
        raise ValueError("No target frame indices")

    target_frame_indices = sorted(set(int(f) for f in target_frame_indices))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)

    # If from_frame set, treat matching seed/keyframe as sole origin
    active_seeds = seeds
    if from_frame is not None:
        active_seeds = [s for s in seeds if int(s.get("frame_idx", -1)) == from_frame]
        if not active_seeds:
            active_seeds = [{"frame_idx": from_frame, **seeds[0]}]
            active_seeds[0]["frame_idx"] = from_frame

    merged: dict[int, dict[str, Any]] = {}

    for seed in active_seeds:
        frame_idx = int(seed.get("frame_idx", 0))
        if from_frame is not None and frame_idx != from_frame:
            continue

        polygon = polygon_from_seed(seed)
        landing = landing_zone_from_seed(seed)
        if len(polygon) < 3:
            continue

        for direction in (1, -1):
            partial = _propagate_chain(
                cap, width, height,
                frame_idx, polygon, landing,
                target_frame_indices,
                direction,
                snap_to_lines=snap_to_lines,
            )
            for fidx, geo in partial.items():
                if fidx not in merged:
                    merged[fidx] = geo
                    merged[fidx]["_seed_dist"] = abs(fidx - frame_idx)
                else:
                    dist = abs(fidx - frame_idx)
                    if dist < merged[fidx].get("_seed_dist", 10**9):
                        merged[fidx] = geo
                        merged[fidx]["_seed_dist"] = dist

    cap.release()

    # Fill gaps via interpolation from successful propagations
    anchor_keyframes = [
        {
            "frame_idx": fidx,
            "track_polygon": geo.get("track_polygon") or [],
            "landing_zone": geo.get("landing_zone") or [],
        }
        for fidx, geo in sorted(merged.items())
        if geo.get("track_polygon")
    ]

    per_frame: dict[int, list[list[float]]] = {}
    keyframes_out: list[dict[str, Any]] = []

    for fidx in target_frame_indices:
        if fidx in merged and merged[fidx].get("track_polygon"):
            poly = merged[fidx]["track_polygon"]
            landing = merged[fidx].get("landing_zone") or []
        else:
            interp = _interp_polygons(anchor_keyframes, fidx)
            poly = interp.get("track_polygon") or []
            landing = interp.get("landing_zone") or []

        per_frame[fidx] = poly
        if poly:
            kf: dict[str, Any] = {
                "frame_idx": fidx,
                "track_polygon": poly,
                "corridor_polygon": [],
                "landing_zone": landing,
            }
            keyframes_out.append(kf)

    propagation_meta = {
        "method": "optical_flow",
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "frame_count": len(keyframes_out),
        "snap_to_lines": snap_to_lines,
        "from_frame": from_frame,
    }

    return {
        "keyframes": keyframes_out,
        "per_frame_polygons": {str(k): v for k, v in per_frame.items()},
        "propagation": propagation_meta,
    }


def target_frames_from_analysis(output_dir: Path) -> list[int]:
    """Read frame indices from analysis.json if present."""
    import json

    path = output_dir / "analysis.json"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    frames = data.get("frames") or []
    return sorted(int(fr.get("frame_idx", i)) for i, fr in enumerate(frames))


def target_frames_from_video(
    video_path: Path,
    *,
    stride: int = 1,
    max_frames: Optional[int] = None,
) -> list[int]:
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    limit = total if max_frames is None else min(total, max_frames)
    return list(range(0, limit, max(1, stride)))
