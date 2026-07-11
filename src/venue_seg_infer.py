"""YOLO11-seg inference for venue track/sand masks (Tier 2)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from .venue_profile import VENUE_ROOT

CLASS_NAMES = ("track", "sand")
CONF_THRESHOLD = 0.05
_MODEL_CACHE: dict[str, Any] = {}


def model_json_path(venue_id: str) -> Path:
    return VENUE_ROOT / venue_id / "model.json"


def load_model_meta(venue_id: str) -> Optional[dict[str, Any]]:
    path = model_json_path(venue_id)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def has_trained_seg_model(venue_id: str) -> bool:
    meta = load_model_meta(venue_id)
    if not meta:
        return False
    weights = meta.get("weights")
    if not weights:
        return False
    wpath = Path(weights)
    if not wpath.is_absolute():
        wpath = Path(__file__).resolve().parents[1] / weights
    return wpath.exists()


def _resolve_weights(meta: dict[str, Any]) -> Optional[Path]:
    weights = meta.get("weights")
    if not weights:
        return None
    wpath = Path(weights)
    if not wpath.is_absolute():
        wpath = Path(__file__).resolve().parents[1] / weights
    return wpath if wpath.exists() else None


def load_venue_seg_model(venue_id: str):
    """Load cached YOLO seg model for venue_id, or None if not trained."""
    if venue_id in _MODEL_CACHE:
        return _MODEL_CACHE[venue_id]

    meta = load_model_meta(venue_id)
    if meta is None:
        return None
    wpath = _resolve_weights(meta)
    if wpath is None:
        return None

    from ultralytics import YOLO

    model = YOLO(str(wpath))
    _MODEL_CACHE[venue_id] = model
    return model


def clear_model_cache(venue_id: Optional[str] = None) -> None:
    if venue_id is None:
        _MODEL_CACHE.clear()
    else:
        _MODEL_CACHE.pop(venue_id, None)


def infer_frame_masks(
    frame_bgr: np.ndarray,
    model,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Run segmentation on one BGR frame.
    Returns uint8 full-resolution masks (0/255) and mean class confidence.
    """
    h, w = frame_bgr.shape[:2]
    empty = np.zeros((h, w), dtype=np.uint8)

    if model is None:
        return empty, empty, 0.0

    results = model(frame_bgr, verbose=False, conf=CONF_THRESHOLD)
    if not results:
        return empty, empty, 0.0

    result = results[0]
    if result.masks is None or result.boxes is None or len(result.boxes) == 0:
        return empty, empty, 0.0

    track_mask = np.zeros((h, w), dtype=np.uint8)
    sand_mask = np.zeros((h, w), dtype=np.uint8)
    confidences: list[float] = []

    names = result.names or {}
    masks_data = result.masks.data.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)
    confs = result.boxes.conf.cpu().numpy()

    for i, cls_id in enumerate(classes):
        name = names.get(int(cls_id), str(cls_id)).lower()
        if name not in CLASS_NAMES:
            continue
        mask_small = masks_data[i]
        mask_full = cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_NEAREST)
        mask_bin = (mask_full > 0.5).astype(np.uint8) * 255
        confidences.append(float(confs[i]))
        if name == "track":
            track_mask = np.maximum(track_mask, mask_bin)
        elif name == "sand":
            sand_mask = np.maximum(sand_mask, mask_bin)

    if not confidences:
        return empty, empty, 0.0

    confidence = float(sum(confidences) / len(confidences))
    return track_mask, sand_mask, confidence
