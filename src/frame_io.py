"""
frame_io — lectura unificada de frames BGR (independiente de dependencias pesadas).

Estrategia de resolución de un frame (en orden):
  1) Leer frames/frame_{idx:06d}.jpg del disco si existe.
  2) Si no, decodificar del video fuente vía cv2.VideoCapture (seek por índice).
  3) Si tampoco se puede, devolver None.

Esto permite operar en modo "lean" (sin JPEG de frames en disco): los endpoints
y overlays pueden reconstruir cualquier frame decodificando del video original.

El video_path se resuelve, si no se pasa, del campo "video" de analysis.json
(igual que hace POST /correct al recibir req.video_path).

Cache LRU en memoria OPCIONAL (OrderedDict, máx ~32 frames), desactivada por
defecto (env TJ_FRAME_CACHE). Sólo cachea frames decodificados del video.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from . import opt_flags

_CACHE_MAX = 32
_frame_cache: "OrderedDict[tuple[str, int], np.ndarray]" = OrderedDict()


def _cache_get(key: tuple[str, int]) -> Optional[np.ndarray]:
    img = _frame_cache.get(key)
    if img is not None:
        _frame_cache.move_to_end(key)
    return img


def _cache_put(key: tuple[str, int], img: np.ndarray) -> None:
    _frame_cache[key] = img
    _frame_cache.move_to_end(key)
    while len(_frame_cache) > _CACHE_MAX:
        _frame_cache.popitem(last=False)


def _resolve_video_path(video_name: str, output_root: Path) -> Optional[str]:
    """Resolver el video fuente desde analysis.json (campo "video")."""
    try:
        analysis_path = Path(output_root) / video_name / "analysis.json"
        if analysis_path.exists():
            with open(analysis_path, encoding="utf-8") as f:
                data = json.load(f)
            vp = data.get("video")
            if vp and Path(vp).exists():
                return str(vp)
    except (OSError, ValueError, json.JSONDecodeError):
        pass
    return None


def read_frame_bgr(
    video_name: str,
    frame_idx: int,
    output_root,
    video_path: Optional[str] = None,
) -> Optional[np.ndarray]:
    """Return BGR ndarray for a frame:
      (1) frames/frame_{idx:06d}.jpg if exists,
      (2) else decode from source video via VideoCapture seek,
      (3) else None.
    """
    frame_idx = int(frame_idx)
    output_root = Path(output_root)

    # (1) JPEG en disco (ruta rápida y compatible)
    frame_path = output_root / video_name / "frames" / f"frame_{frame_idx:06d}.jpg"
    if frame_path.exists():
        img = cv2.imread(str(frame_path))
        if img is not None:
            return img

    use_cache = opt_flags.frame_cache()
    cache_key = (str(video_name), frame_idx)
    if use_cache:
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached

    # (2) decodificar del video fuente
    if video_path is None:
        video_path = _resolve_video_path(video_name, output_root)
    if video_path and Path(video_path).exists():
        cap = cv2.VideoCapture(str(video_path))
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, img = cap.read()
        finally:
            cap.release()
        if ret and img is not None:
            if use_cache:
                _cache_put(cache_key, img)
            return img

    # (3) no disponible
    return None
