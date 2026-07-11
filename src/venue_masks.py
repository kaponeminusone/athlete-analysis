"""Keyframe-first venue mask generation from manual calibration polygons."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from .calibration_propagator import _interp_polygons
from .mask_utils import mask_area_norm, polygon_to_mask, save_mask_png
from .venue_corrector import _read_gray, _warp_mask_flow
from .venue_profile import _keyframe_source

MIN_KEYFRAMES_FOR_PIPELINE = 5


def polygon_keyframes(calibration: dict[str, Any]) -> list[dict[str, Any]]:
    """Keyframes with a usable track polygon, preferring manual over venue_auto."""
    keyframes = calibration.get("keyframes") or []
    usable = [
        k for k in keyframes
        if len(k.get("track_polygon") or []) >= 3
        and _keyframe_source(k) != "venue_auto"
    ]
    if usable:
        return sorted(usable, key=lambda k: int(k["frame_idx"]))
    return sorted(
        [k for k in keyframes if len(k.get("track_polygon") or []) >= 3],
        key=lambda k: int(k["frame_idx"]),
    )


def count_polygon_keyframes(calibration: dict[str, Any]) -> int:
    return len(polygon_keyframes(calibration))


def should_use_keyframe_pipeline(
    calibration: dict[str, Any],
    *,
    prefer_keyframes: bool = True,
) -> bool:
    if not prefer_keyframes:
        return False
    return count_polygon_keyframes(calibration) >= MIN_KEYFRAMES_FOR_PIPELINE


def _exact_keyframe(keyframes: list[dict[str, Any]], frame_idx: int) -> Optional[dict[str, Any]]:
    for kf in keyframes:
        if int(kf["frame_idx"]) == frame_idx:
            return kf
    return None


def _polygons_for_frame(
    keyframes: list[dict[str, Any]],
    frame_idx: int,
) -> tuple[list[list[float]], list[list[float]], str]:
    """Return track polygon, landing zone, and source tag for one frame."""
    exact = _exact_keyframe(keyframes, frame_idx)
    if exact and len(exact.get("track_polygon") or []) >= 3:
        track = exact.get("track_polygon") or []
        landing = exact.get("landing_zone") or []
        return track, landing, "keyframe"

    interp = _interp_polygons(keyframes, frame_idx)
    track = interp.get("track_polygon") or []
    landing = interp.get("landing_zone") or []
    return track, landing, "interpolated"


def _nearest_keyframe_index(keyframes: list[dict[str, Any]], frame_idx: int) -> int:
    return min(range(len(keyframes)), key=lambda i: abs(int(keyframes[i]["frame_idx"]) - frame_idx))


def _refine_with_flow(
    cap: cv2.VideoCapture,
    keyframes: list[dict[str, Any]],
    frame_idx: int,
    track_mask: np.ndarray,
    sand_mask: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Warp rasterized mask from nearest keyframe using optical flow."""
    if not keyframes:
        return track_mask, sand_mask, False

    nearest_i = _nearest_keyframe_index(keyframes, frame_idx)
    nearest = keyframes[nearest_i]
    nearest_idx = int(nearest["frame_idx"])
    if nearest_idx == frame_idx:
        return track_mask, sand_mask, False

    nearest_track = polygon_to_mask(nearest.get("track_polygon") or [], width, height)
    nearest_sand = polygon_to_mask(nearest.get("landing_zone") or [], width, height)
    if (nearest_track > 0).sum() == 0:
        return track_mask, sand_mask, False

    nearest_gray = _read_gray(cap, nearest_idx)
    target_gray = _read_gray(cap, frame_idx)
    if nearest_gray is None or target_gray is None:
        return track_mask, sand_mask, False

    if nearest_idx < frame_idx:
        warped_track = _warp_mask_flow(nearest_gray, target_gray, nearest_track)
        warped_sand = _warp_mask_flow(nearest_gray, target_gray, nearest_sand)
    else:
        warped_track = _warp_mask_flow(target_gray, nearest_gray, nearest_track)
        warped_sand = _warp_mask_flow(target_gray, nearest_gray, nearest_sand)

    if (warped_track > 0).sum() == 0:
        return track_mask, sand_mask, False
    return warped_track, warped_sand, True


def build_masks_from_keyframes(
    video_path: Path | str,
    calibration: dict[str, Any],
    frame_indices: list[int],
    output_dir: Path | str,
    *,
    use_flow_refinement: bool = True,
) -> dict[str, Any]:
    """
    Rasterize calibration polygons (exact, interpolated, or flow-warped) to mask PNGs.

    Returns dict with mask_frames map and metadata. Does not mutate calibration keyframes.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    keyframes = polygon_keyframes(calibration)
    if len(keyframes) < MIN_KEYFRAMES_FOR_PIPELINE:
        raise ValueError(
            f"Need at least {MIN_KEYFRAMES_FOR_PIPELINE} keyframes with track_polygon; got {len(keyframes)}"
        )
    if not frame_indices:
        raise ValueError("No frame indices provided")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    masks_dir = output_dir / "venue_masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    mask_frames: dict[str, dict[str, Any]] = {}
    source_counts = {"keyframe": 0, "interpolated": 0, "flow_warp": 0}

    for frame_idx in sorted(set(int(f) for f in frame_indices)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        height, width = frame.shape[:2]

        track_poly, landing_poly, source = _polygons_for_frame(keyframes, frame_idx)
        track_mask = polygon_to_mask(track_poly, width, height)
        sand_mask = polygon_to_mask(landing_poly, width, height)

        if (
            use_flow_refinement
            and source == "interpolated"
            and len(keyframes) >= 2
        ):
            flow_track, flow_sand, used_flow = _refine_with_flow(
                cap, keyframes, frame_idx, track_mask, sand_mask, width, height,
            )
            if used_flow:
                track_mask = flow_track
                sand_mask = flow_sand
                source = "flow_warp"

        track_area = mask_area_norm(track_mask, width, height)
        sand_area = mask_area_norm(sand_mask, width, height)

        track_rel = f"venue_masks/track_{frame_idx:06d}.png"
        sand_rel = f"venue_masks/sand_{frame_idx:06d}.png"
        save_mask_png(output_dir / track_rel, track_mask)
        save_mask_png(output_dir / sand_rel, sand_mask)

        confidence = 1.0 if source == "keyframe" else 0.92 if source == "flow_warp" else 0.88
        mask_frames[str(frame_idx)] = {
            "track": track_rel,
            "sand": sand_rel,
            "source": source,
            "confidence": round(confidence, 3),
            "track_area_norm": round(track_area, 4),
            "sand_area_norm": round(sand_area, 4),
        }
        source_counts[source] = source_counts.get(source, 0) + 1

    cap.release()

    if not mask_frames:
        raise ValueError("Keyframe mask pipeline produced no frames.")

    return {
        "mask_frames": mask_frames,
        "keyframes_used": len(keyframes),
        "frames_applied": len(mask_frames),
        "source_counts": source_counts,
    }
