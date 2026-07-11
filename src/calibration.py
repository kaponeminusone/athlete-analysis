"""Track calibration persistence (Phase 1 — geometry + seed auto v2)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

CALIBRATION_VERSION = 2
CALIBRATION_VERSION_MANUAL = 1

CORNER_LABELS = ["corner_tl", "corner_tr", "corner_br", "corner_bl"]
OPTIONAL_LABELS = ["foul_board", "arena_tl", "arena_tr", "arena_br", "arena_bl"]


def calibration_path(output_dir: Path) -> Path:
    return output_dir / "calibration.json"


def default_calibration(video_name: str, video_file: str | None = None) -> dict[str, Any]:
    return {
        "version": CALIBRATION_VERSION,
        "mode": "seed_auto",
        "video": video_file or f"{video_name}.mp4",
        "seeds": [],
        "keyframes": [],
    }


def load_calibration(output_dir: Path) -> Optional[dict[str, Any]]:
    path = calibration_path(output_dir)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return normalize_calibration(data)


def save_calibration(output_dir: Path, data: dict[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    normalized = normalize_calibration(data)
    path = calibration_path(output_dir)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(normalized, f, indent=2)
    return path


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
            points.append([
                max(0.0, min(1.0, float(item[0]))),
                max(0.0, min(1.0, float(item[1]))),
            ])
    return points


def _normalize_seed(raw: dict[str, Any]) -> dict[str, Any]:
    frame_idx = int(raw.get("frame_idx", 0))
    seed: dict[str, Any] = {
        "frame_idx": frame_idx,
        "seed_points": _as_point_list(raw.get("seed_points")),
    }
    labels = raw.get("labels")
    if isinstance(labels, list):
        seed["labels"] = [str(l) for l in labels]
    if raw.get("track_polygon"):
        seed["track_polygon"] = _as_point_list(raw.get("track_polygon"))
    if raw.get("landing_zone"):
        seed["landing_zone"] = _as_point_list(raw.get("landing_zone"))
    return seed


def _normalize_keyframe(raw: dict[str, Any]) -> dict[str, Any]:
    frame_idx = int(raw.get("frame_idx", 0))
    keyframe: dict[str, Any] = {
        "frame_idx": frame_idx,
        "track_polygon": _as_point_list(raw.get("track_polygon")),
        "corridor_polygon": _as_point_list(raw.get("corridor_polygon")),
        "landing_zone": _as_point_list(raw.get("landing_zone")),
    }
    axis = raw.get("axis")
    if isinstance(axis, dict):
        origin = axis.get("origin")
        direction = axis.get("direction")
        if (
            isinstance(origin, (list, tuple))
            and len(origin) >= 2
            and isinstance(direction, (list, tuple))
            and len(direction) >= 2
        ):
            keyframe["axis"] = {
                "origin": [float(origin[0]), float(origin[1])],
                "direction": [float(direction[0]), float(direction[1])],
            }
    scale = raw.get("scale")
    if isinstance(scale, dict):
        point_a = scale.get("point_a")
        point_b = scale.get("point_b")
        known = scale.get("known_distance_m")
        if (
            known is not None
            and isinstance(point_a, (list, tuple))
            and len(point_a) >= 2
            and isinstance(point_b, (list, tuple))
            and len(point_b) >= 2
        ):
            keyframe["scale"] = {
                "known_distance_m": float(known),
                "point_a": [float(point_a[0]), float(point_a[1])],
                "point_b": [float(point_b[0]), float(point_b[1])],
            }
    return keyframe


def normalize_calibration(data: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize calibration payload before save."""
    version = int(data.get("version", CALIBRATION_VERSION_MANUAL))
    video = str(data.get("video", ""))

    keyframes_raw = data.get("keyframes", [])
    keyframes: list[dict[str, Any]] = []
    if isinstance(keyframes_raw, list):
        for item in keyframes_raw:
            if isinstance(item, dict):
                keyframes.append(_normalize_keyframe(item))
    keyframes.sort(key=lambda k: k["frame_idx"])

    result: dict[str, Any] = {
        "version": version,
        "video": video,
        "keyframes": keyframes,
    }

    mode = data.get("mode")
    if mode:
        result["mode"] = str(mode)
    elif version >= 2:
        result["mode"] = "seed_auto" if data.get("seeds") else "manual"

    seeds_raw = data.get("seeds")
    if isinstance(seeds_raw, list):
        seeds = [_normalize_seed(s) for s in seeds_raw if isinstance(s, dict)]
        seeds.sort(key=lambda s: s["frame_idx"])
        result["seeds"] = seeds

    propagation = data.get("propagation")
    if isinstance(propagation, dict):
        result["propagation"] = propagation

    mask_frames = data.get("mask_frames")
    if isinstance(mask_frames, dict):
        result["mask_frames"] = mask_frames

    venue_profile = data.get("venue_profile")
    if isinstance(venue_profile, dict):
        result["venue_profile"] = venue_profile

    return result


def has_seeds(cal: dict[str, Any]) -> bool:
    seeds = cal.get("seeds") or []
    return any(len(s.get("seed_points") or []) >= 4 or len(s.get("track_polygon") or []) >= 3 for s in seeds)


def keyframes_incomplete(cal: dict[str, Any], target_frames: list[int]) -> bool:
    """True when propagated keyframes don't cover analysis frames."""
    if not target_frames:
        return False
    keyframes = cal.get("keyframes") or []
    if not keyframes:
        return True
    covered = {k["frame_idx"] for k in keyframes if k.get("track_polygon")}
    if len(covered) < len(target_frames) * 0.5:
        return True
    # Missing more than 10% of target frames
    missing = sum(1 for f in target_frames if f not in covered)
    return missing > max(1, len(target_frames) // 10)


def run_propagation_for_output(
    output_dir: Path,
    video_path: Path,
    *,
    snap_to_lines: bool = False,
    from_frame: Optional[int] = None,
) -> dict[str, Any]:
    """Load calibration, propagate seeds, save expanded keyframes."""
    from .calibration_propagator import (
        propagate_calibration,
        target_frames_from_analysis,
        target_frames_from_video,
    )

    cal = load_calibration(output_dir) or default_calibration(video_path.stem)
    seeds = list(cal.get("seeds") or [])

    if from_frame is not None:
        kf = next(
            (k for k in cal.get("keyframes") or [] if int(k.get("frame_idx", -1)) == from_frame),
            None,
        )
        if kf and kf.get("track_polygon"):
            seeds = [{
                "frame_idx": from_frame,
                "track_polygon": kf.get("track_polygon") or [],
                "landing_zone": kf.get("landing_zone") or [],
            }]
        else:
            seeds = [s for s in seeds if int(s.get("frame_idx", -1)) == from_frame]
            if not seeds:
                raise ValueError(f"No seed or keyframe at frame {from_frame}")

    if not seeds:
        raise ValueError("No seeds in calibration.json")

    targets = target_frames_from_analysis(output_dir)
    if not targets:
        targets = target_frames_from_video(video_path)

    result = propagate_calibration(
        video_path,
        seeds,
        targets,
        snap_to_lines=snap_to_lines,
        from_frame=from_frame,
    )

    cal["version"] = max(int(cal.get("version", 1)), CALIBRATION_VERSION)
    cal["mode"] = cal.get("mode") or "seed_auto"
    if from_frame is not None and cal.get("keyframes"):
        merged = {int(k["frame_idx"]): k for k in cal["keyframes"]}
        for kf in result["keyframes"]:
            merged[int(kf["frame_idx"])] = kf
        cal["keyframes"] = sorted(merged.values(), key=lambda k: k["frame_idx"])
    else:
        cal["keyframes"] = result["keyframes"]
    cal["propagation"] = result["propagation"]
    save_calibration(output_dir, cal)
    return cal
