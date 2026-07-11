"""Manual venue mask correction and optical-flow propagation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional

import cv2
import numpy as np

from .calibration import load_calibration, save_calibration
from .calibration_propagator import target_frames_from_analysis
from .mask_utils import decode_mask_sample, load_mask_png, mask_area_norm, save_mask_png

Layer = Literal["track", "sand"]
Direction = Literal["both", "forward", "backward"]
Operation = Literal["add", "remove"]


def _read_gray(cap: cv2.VideoCapture, frame_idx: int) -> Optional[np.ndarray]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def _warp_mask_flow(prev_gray: np.ndarray, next_gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Warp mask from prev frame coords to next frame using Farneback flow."""
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, next_gray, None,
        pyr_scale=0.5, levels=3, winsize=21,
        iterations=3, poly_n=5, poly_sigma=1.1, flags=0,
    )
    h, w = mask.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = grid_x + flow[..., 0]
    map_y = grid_y + flow[..., 1]
    warped = cv2.remap(mask, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return (warped > 0).astype(np.uint8) * 255


def _apply_delta(mask: np.ndarray, delta: np.ndarray, operation: Operation) -> np.ndarray:
    out = mask.copy()
    if operation == "add":
        out[delta > 0] = 255
    else:
        out[delta > 0] = 0
    return out


def _mask_entry_paths(output_dir: Path, entry: dict[str, Any], layer: Layer) -> Path:
    rel = entry.get(layer)
    if not rel:
        raise FileNotFoundError(f"No {layer} mask path in calibration entry")
    return output_dir / rel


def _update_mask_entry(
    output_dir: Path,
    cal: dict[str, Any],
    frame_idx: int,
    layer: Layer,
    mask: np.ndarray,
    width: int,
    height: int,
) -> None:
    mask_frames = cal.setdefault("mask_frames", {})
    key = str(frame_idx)
    entry = mask_frames.get(key)
    track_rel = f"venue_masks/track_{frame_idx:06d}.png"
    sand_rel = f"venue_masks/sand_{frame_idx:06d}.png"
    (output_dir / "venue_masks").mkdir(parents=True, exist_ok=True)

    if entry is None:
        entry = {"track": track_rel, "sand": sand_rel}
        mask_frames[key] = entry
        if layer == "track":
            save_mask_png(output_dir / sand_rel, np.zeros((height, width), np.uint8))
        else:
            save_mask_png(output_dir / track_rel, np.zeros((height, width), np.uint8))

    path = _mask_entry_paths(output_dir, entry, layer)
    save_mask_png(path, mask)

    if layer == "track":
        entry["track_area_norm"] = round(mask_area_norm(mask, width, height), 4)
    else:
        entry["sand_area_norm"] = round(mask_area_norm(mask, width, height), 4)


def _propagate_mask_chain(
    cap: cv2.VideoCapture,
    output_dir: Path,
    cal: dict[str, Any],
    source_idx: int,
    source_mask: np.ndarray,
    layer: Layer,
    radius: int,
    direction: Direction,
    width: int,
    height: int,
    target_frames: set[int],
) -> list[int]:
    affected: list[int] = []
    mask_frames = cal.get("mask_frames") or {}

    def _step_chain(step_dir: int) -> None:
        nonlocal source_mask
        prev_idx = source_idx
        prev_gray = _read_gray(cap, prev_idx)
        if prev_gray is None:
            return
        current_mask = source_mask
        for step in range(1, radius + 1):
            next_idx = source_idx + step_dir * step
            if next_idx < 0:
                break
            if next_idx not in target_frames:
                continue
            next_gray = _read_gray(cap, next_idx)
            if next_gray is None:
                break
            if step_dir > 0:
                warped = _warp_mask_flow(prev_gray, next_gray, current_mask)
            else:
                warped = _warp_mask_flow(next_gray, prev_gray, current_mask)
            entry = mask_frames.get(str(next_idx))
            if entry:
                existing = load_mask_png(_mask_entry_paths(output_dir, entry, layer))
                if existing is not None and existing.shape == warped.shape:
                    warped = cv2.bitwise_or(warped, existing) if layer == "track" else warped
            _update_mask_entry(output_dir, cal, next_idx, layer, warped, width, height)
            affected.append(next_idx)
            prev_gray = next_gray
            current_mask = warped

    if direction in ("both", "forward"):
        _step_chain(+1)
    if direction in ("both", "backward"):
        _step_chain(-1)
    return affected


def apply_mask_correction(
    video_path: Path | str,
    output_dir: Path | str,
    frame_idx: int,
    layer: Layer,
    mask_grid: list[list[int]] | None = None,
    *,
    operation: Operation = "add",
    radius: int = 15,
    direction: Direction = "both",
    full_mask: Optional[list[list[int]]] = None,
) -> dict[str, Any]:
    """
    Apply brush delta to a venue mask layer and propagate to neighboring frames.

    Returns summary dict with affected frame indices.
    """
    output_dir = Path(output_dir)
    video_path = Path(video_path)
    cal = load_calibration(output_dir)
    if cal is None or cal.get("mode") != "color_masks":
        raise ValueError("color_masks calibration required. Apply venue profile first.")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        raise ValueError(f"Cannot read frame {frame_idx}")
    height, width = frame.shape[:2]

    mask_frames = cal.get("mask_frames") or {}
    entry = mask_frames.get(str(frame_idx))
    if entry:
        existing = load_mask_png(_mask_entry_paths(output_dir, entry, layer))
    else:
        existing = np.zeros((height, width), dtype=np.uint8)

    if existing is None:
        existing = np.zeros((height, width), dtype=np.uint8)

    if full_mask is not None:
        corrected = decode_mask_sample(full_mask, width, height)
        if corrected is None:
            cap.release()
            raise ValueError("Invalid full_mask payload")
    else:
        if not mask_grid:
            cap.release()
            raise ValueError("mask_grid or full_mask required")
        delta = decode_mask_sample(mask_grid, width, height)
        if delta is None:
            cap.release()
            raise ValueError("Invalid mask_grid payload")
        corrected = _apply_delta(existing, delta, operation)

    _update_mask_entry(output_dir, cal, frame_idx, layer, corrected, width, height)
    cal["mode"] = "color_masks"
    save_calibration(output_dir, cal)

    target_frames = set(target_frames_from_analysis(output_dir))
    affected = _propagate_mask_chain(
        cap, output_dir, cal, frame_idx, corrected, layer,
        radius, direction, width, height, target_frames,
    )
    cap.release()
    save_calibration(output_dir, cal)

    return {
        "frame_idx": frame_idx,
        "layer": layer,
        "operation": operation,
        "radius": radius,
        "direction": direction,
        "frames_corrected": [frame_idx] + affected,
        "total_affected": 1 + len(affected),
    }
