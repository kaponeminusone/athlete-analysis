"""
opt_flags — banderas de optimización de almacenamiento (fuente única de verdad).

Todas las banderas se leen de variables de entorno EN CADA LLAMADA, de modo que
el comportamiento por defecto (sin variables de entorno) es idéntico al de hoy:
sigue escribiendo `frames/*.jpg` y `annotated/*.jpg` en disco.

El ahorro de espacio es OPT-IN: hay que exportar explícitamente las variables.

Variables y valores por defecto:
  TJ_WRITE_ANNOTATED         = "1"  → escribir annotated/*.jpg (pipeline, refine, /correct)
  TJ_PERSIST_FRAMES          = "1"  → escribir frames/*.jpg (pipeline, refine)
  TJ_CORRECTION_WRITE_FRAMES = "1"  → propagate_correction escribe frames/*.jpg
  TJ_ANNOTATED_CACHE         = "0"  → LRU en memoria para bytes anotados en get_frame
  TJ_FRAME_CACHE             = "0"  → LRU en memoria para frames decodificados (frame_io)
  TJ_FRAME_CACHE_MAX         = "128" → tamaño LRU de frames BGR (si TJ_FRAME_CACHE=1)
  TJ_ANNOTATED_CACHE_MAX     = "64"  → tamaño LRU de JPEG anotados (si TJ_ANNOTATED_CACHE=1)
  TJ_JPEG_QUALITY            = "95" → calidad JPEG al codificar en memoria (cv2.imencode)

Poner cualquiera a "0"/"false"/"no"/"off" desactiva la escritura correspondiente.
"""

from __future__ import annotations

import os

_TRUE = {"1", "true", "yes", "on"}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in _TRUE


def _env_int(name: str, default: int, *, minimum: int = 1, maximum: int = 4096) -> int:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        value = int(str(raw).strip())
    except ValueError:
        return default
    return max(minimum, min(maximum, value))


# ─── Escritores en disco (default = comportamiento actual, TRUE) ───────────────

def write_annotated() -> bool:
    """Cuando FALSE, los escritores omiten annotated/*.jpg."""
    return _env_bool("TJ_WRITE_ANNOTATED", True)


def persist_frames() -> bool:
    """Cuando FALSE, pipeline/reanalyzer omiten frames/*.jpg. Default TRUE."""
    return _env_bool("TJ_PERSIST_FRAMES", True)


def correction_write_frames() -> bool:
    """Cuando FALSE, propagate_correction omite el imwrite de frames."""
    return _env_bool("TJ_CORRECTION_WRITE_FRAMES", True)


# ─── Caches opcionales en memoria (default = OFF) ─────────────────────────────

def annotated_cache() -> bool:
    """LRU opcional para bytes anotados servidos por get_frame."""
    return _env_bool("TJ_ANNOTATED_CACHE", False)


def frame_cache() -> bool:
    """LRU opcional para frames BGR decodificados desde video."""
    return _env_bool("TJ_FRAME_CACHE", False)


def frame_cache_max() -> int:
    """Tamaño máximo del LRU de frames BGR (TJ_FRAME_CACHE_MAX, default 128)."""
    return _env_int("TJ_FRAME_CACHE_MAX", 128, minimum=8, maximum=2048)


def annotated_cache_max() -> int:
    """Tamaño máximo del LRU de JPEG anotados (TJ_ANNOTATED_CACHE_MAX, default 64)."""
    return _env_int("TJ_ANNOTATED_CACHE_MAX", 64, minimum=8, maximum=2048)


# ─── Calidad de codificación en memoria ───────────────────────────────────────

def jpeg_quality() -> int:
    """Calidad JPEG para cv2.imencode en memoria. 95 = default de cv2.imwrite."""
    try:
        return int(os.getenv("TJ_JPEG_QUALITY", "95"))
    except ValueError:
        return 95
