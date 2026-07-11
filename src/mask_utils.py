"""Mask helpers for venue segmentation and athlete overlap scoring."""

from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

MASK_GRID_W = 160
MASK_GRID_H = 90


def load_mask_png(path: Path | str) -> Optional[np.ndarray]:
    path = Path(path)
    if not path.exists():
        return None
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return img


def save_mask_png(path: Path | str, mask: np.ndarray) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = mask if mask.dtype == np.uint8 else (mask > 0).astype(np.uint8) * 255
    cv2.imwrite(str(path), out)
    return path


def mask_area_norm(mask: np.ndarray, width: int, height: int) -> float:
    if mask is None or mask.size == 0:
        return 0.0
    return float((mask > 0).sum()) / float(max(width * height, 1))


def grid_mask_to_full(
    grid: list[list[int]] | np.ndarray,
    width: int,
    height: int,
    *,
    grid_w: int = MASK_GRID_W,
    grid_h: int = MASK_GRID_H,
) -> np.ndarray:
    """Upsample normalized brush grid to full-frame uint8 mask."""
    arr = np.array(grid, dtype=np.uint8)
    if arr.ndim != 2:
        raise ValueError("grid mask must be 2D")
    if arr.shape != (grid_h, grid_w):
        arr = cv2.resize(arr, (grid_w, grid_h), interpolation=cv2.INTER_NEAREST)
    full = cv2.resize(arr, (width, height), interpolation=cv2.INTER_NEAREST)
    return (full > 0).astype(np.uint8) * 255


def decode_mask_sample(
    sample: Any,
    width: int,
    height: int,
) -> Optional[np.ndarray]:
    """Decode brush mask from API payload (grid, base64 PNG, or flat list)."""
    if sample is None:
        return None
    if isinstance(sample, str):
        raw = base64.b64decode(sample)
        arr = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        if img.shape[:2] != (height, width):
            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_NEAREST)
        return (img > 0).astype(np.uint8) * 255
    if isinstance(sample, list):
        if not sample:
            return None
        if isinstance(sample[0], list):
            return grid_mask_to_full(sample, width, height)
        flat = np.array(sample, dtype=np.uint8)
        if flat.size == MASK_GRID_W * MASK_GRID_H:
            grid = flat.reshape(MASK_GRID_H, MASK_GRID_W)
            return grid_mask_to_full(grid, width, height)
        if flat.size == width * height:
            return (flat.reshape(height, width) > 0).astype(np.uint8) * 255
    return None


def polygon_to_mask(
    points: list[list[float]],
    width: int,
    height: int,
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


def erode_mask(mask: np.ndarray, erode_px: int = 10) -> np.ndarray:
    if erode_px <= 0:
        return mask
    k = erode_px * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.erode(mask, kernel, iterations=1)


def athlete_mask_overlap(
    person_bbox: Optional[tuple[float, float, float, float]],
    person_seg_mask: Optional[np.ndarray],
    region_mask: Optional[np.ndarray],
    width: int,
    height: int,
) -> float:
    """
    Fraction of athlete (bottom-half bbox or seg mask) overlapping region_mask.
    """
    if region_mask is None or region_mask.size == 0:
        return 0.0

    if person_seg_mask is not None and person_seg_mask.size > 0:
        seg = person_seg_mask
        if seg.shape[:2] != (height, width):
            seg = cv2.resize(seg, (width, height), interpolation=cv2.INTER_NEAREST)
        athlete = (seg > 0).astype(np.uint8)
        athlete_area = int(athlete.sum())
        if athlete_area <= 0:
            return 0.0
        region = (region_mask > 0).astype(np.uint8)
        inter = int(cv2.bitwise_and(athlete, region).sum())
        return float(inter) / float(athlete_area)

    if person_bbox is None:
        return 0.0

    x1, y1, x2, y2 = person_bbox
    mid_y = y1 + (y2 - y1) * 0.5
    bx1 = max(0, int(x1))
    by1 = max(0, int(mid_y))
    bx2 = min(width, int(x2))
    by2 = min(height, int(y2))
    if bx2 <= bx1 or by2 <= by1:
        return 0.0

    roi = region_mask[by1:by2, bx1:bx2]
    if roi.size == 0:
        return 0.0
    bbox_area = (bx2 - bx1) * (by2 - by1)
    inside = int((roi > 0).sum())
    return float(inside) / float(max(bbox_area, 1))


def composite_mask_overlay(
    track_mask: Optional[np.ndarray],
    sand_mask: Optional[np.ndarray],
    *,
    track_color: tuple[int, int, int] = (0, 255, 0),
    sand_color: tuple[int, int, int] = (0, 220, 255),
    alpha: float = 0.45,
) -> np.ndarray:
    """BGR overlay image (H,W,3) with colored semi-transparent masks."""
    h = w = 0
    if track_mask is not None:
        h, w = track_mask.shape[:2]
    elif sand_mask is not None:
        h, w = sand_mask.shape[:2]
    else:
        return np.zeros((1, 1, 3), dtype=np.uint8)

    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    if track_mask is not None:
        m = track_mask > 0
        overlay[m] = track_color
    if sand_mask is not None:
        m = sand_mask > 0
        overlay[m] = sand_color
    return overlay
