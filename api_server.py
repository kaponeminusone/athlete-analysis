"""
API Server — bridge entre la UI y el pipeline de analisis.

Endpoints:
  GET  /status                       → estado del servidor y modelos cargados
  GET  /analysis/{video_name}        → cargar analysis.json existente
  GET  /api/calibration/{video_name} → cargar calibration.json
  POST /api/calibration/{video_name} → guardar calibration.json
  POST /api/calibration/{video_name}/seeds → guardar semillas de pista
  POST /api/calibration/{video_name}/propagate → propagar calibracion (optical flow)
  POST /api/recompute-tracking/{video_name} → recalcular campos track sin YOLO
  POST /api/sections/analyze/{video_name}   → detectar fases (pose + tobillos)
  POST /api/sections/mark/{video_name}      → marcar fase en frame
  DELETE /api/sections/mark/{video_name}/{frame_idx}
  POST /api/sections/propagate/{video_name} → propagar hops desde ancla
  GET  /api/sections/pose-scores/{video_name}
  GET  /api/sections/{video_name}          → cargar sections.json
  GET  /api/metrics/{video_name}           → cargar metrics.json
  POST /api/metrics/{video_name}/compute   → recalcular métricas
  POST /api/metrics/{video_name}/overrides → overrides de escala (hops_corridor_m)
  POST /api/metrics/{video_name}/scale     → longitud corredor hops (m) + venue default
  GET  /api/metrics/{video_name}/pose-overlay/{phase} → PNG superposición pose (hop/vuelo)
  GET  /api/venue/profile              → perfil de venue aprendido
  POST /api/venue/learn                → aprender colores de pista/arena + exportar dataset CNN
  POST /api/venue/train                → exportar dataset + entrenar CNN pista/arena
  GET  /api/venue/model                → estado del modelo CNN de venue
  GET  /api/venue/dataset              → manifiesto del dataset CNN multi-video
  POST /api/venue/apply/{video_name}   → auto-calibrar con perfil + recomputar
  POST /api/venue/correct/{video_name} → correccion manual de mascaras pista/arena
  POST /analyze                      → correr el pipeline completo en un video
  POST /correct                      → aplicar correccion manual a un frame
  GET  /frame/{video_name}/{frame}   → imagen del frame crudo o anotado
  GET  /mask/{video_name}/{frame}    → detecciones del frame (para click-selection)

Instalar dependencias extra:
  pip install fastapi uvicorn python-multipart

Correr:
  python api_server.py
  → http://localhost:8000
"""

import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks
    from fastapi.responses import FileResponse, JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import uvicorn
except ImportError:
    print("[ERROR] Run: pip install fastapi uvicorn")
    sys.exit(1)

from src.frame_extractor  import get_video_info
from src.pose_analyzer    import FrameAnalysis, analyze_frame_from_tracker
from src.athlete_tracker  import TrackState, run_tracked_frame
from src.correction       import Correction, apply_correction, propagate_correction, detections_for_frame
from src.sot              import create_sot
from src.visualizer       import annotate_frame, annotate_frame_array
from src.frame_io         import read_frame_bgr
from src                  import opt_flags
from src.pipeline         import PipelineConfig, run_pipeline
from src.job_store        import create_job, get_job, list_jobs
from src.reanalyzer       import ReanalysisConfig, run_reanalysis
from src.schemas          import frame_analysis_to_dict, frame_record_to_analysis
from src.calibration      import (
    default_calibration,
    has_seeds,
    keyframes_incomplete,
    load_calibration,
    normalize_calibration,
    run_propagation_for_output,
    save_calibration,
)
from src.calibration_propagator import target_frames_from_analysis
from src.track_scorer     import recompute_frames_track_fields
from src.section_analyzer import (
    mark_phase_on_frame,
    move_phase_marker_on_frame,
    phase_at_frame,
    run_phase_propagation,
    run_section_analysis,
    unmark_phase_frame,
)
from src.phase_classifier import pose_classify_frames
from src.metrics import apply_overrides, compute_metrics, load_metrics
from src.pose_overlay import OVERLAY_PHASES, render_pose_overlay_png
from src.venue_profile    import (
    DEFAULT_VENUE_ID,
    apply_masks_to_output,
    apply_profile_to_output,
    learn_from_calibration,
    learn_from_selections,
    load_profile,
    save_debug_frames,
    segment_frame_masks,
    selections_from_calibration,
)
from src.venue_masks import should_use_keyframe_pipeline
from src.venue_seg_infer import has_trained_seg_model, load_model_meta

OUTPUT_ROOT = Path(os.getenv("HOPLAB_OUTPUT_ROOT", "output"))
VENUE_ROOT  = Path(os.getenv("HOPLAB_VENUE_ROOT",  "venues"))
VIDEO_ROOT  = Path(os.getenv("HOPLAB_VIDEO_ROOT",  "."))
app = FastAPI(title="Triple Jump Analyzer API")
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Global model cache (load once, reuse) ────────────────────────────────────
_models: dict = {}
_track_states: dict = {}      # video_name → TrackState
_analyses_cache: dict = {}    # video_name → list[FrameAnalysis]

# ─── LRU opcional para bytes anotados servidos por get_frame ───────────────────
# Guardado por opt_flags.annotated_cache() (env TJ_ANNOTATED_CACHE, default OFF).
from collections import OrderedDict as _OrderedDict

_ANNOTATED_CACHE_MAX = 32
_annotated_bytes_cache: "_OrderedDict[tuple, bytes]" = _OrderedDict()


def _analysis_mtime(video_name: str) -> float:
    path = OUTPUT_ROOT / video_name / "analysis.json"
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _annotated_cache_get(key: tuple) -> Optional[bytes]:
    data = _annotated_bytes_cache.get(key)
    if data is not None:
        _annotated_bytes_cache.move_to_end(key)
    return data


def _annotated_cache_put(key: tuple, data: bytes) -> None:
    _annotated_bytes_cache[key] = data
    _annotated_bytes_cache.move_to_end(key)
    while len(_annotated_bytes_cache) > _ANNOTATED_CACHE_MAX:
        _annotated_bytes_cache.popitem(last=False)


def _get_models():
    if "pose" not in _models:
        from ultralytics import YOLO
        print("[API] Loading models...")
        _models["pose"] = YOLO("yolo11s-pose.pt")
        _models["seg"]  = YOLO("yolo11s-seg.pt")
        print("[API] Models ready")
    return _models["pose"], _models["seg"]


def _video_name(video_path: str) -> str:
    return Path(video_path).stem


def _output_dir(video_path: str) -> Path:
    return OUTPUT_ROOT / _video_name(video_path)


def _video_duration(path: Path) -> float:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()
    return round(float(total / fps), 3) if fps else 0.0


def _resolve_output_dir(video_path: str, output_dir: Optional[str] = None) -> Path:
    if output_dir:
        return Path(output_dir)
    return _output_dir(video_path)


def _load_analysis_dir(out_dir: Path) -> Optional[dict]:
    path = out_dir / "analysis.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_analysis(video_name: str) -> Optional[dict]:
    return _load_analysis_dir(OUTPUT_ROOT / video_name)


def _save_analysis_dir(out_dir: Path, frames_data: list[dict], extra: Optional[dict] = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "analysis.json"
    existing: dict = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            existing = json.load(f)
    existing["frames"] = frames_data
    if extra:
        existing.update(extra)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)


def _save_analysis(video_name: str, frames_data: list[dict], extra: Optional[dict] = None) -> None:
    _save_analysis_dir(OUTPUT_ROOT / video_name, frames_data, extra=extra)


def _merge_frame_correction(existing: dict, update: dict) -> dict:
    """Merge corrected pose into analysis.json without wiping track fields."""
    merged = {**existing, **update}
    for key in ("track_overlap", "athlete_state", "position_s", "predicted_bbox"):
        if update.get(key) is None and existing.get(key) is not None:
            merged[key] = existing[key]
    merged.pop("annotated_image", None)
    return merged


def _append_correction_log(out_dir: Path, record: dict) -> None:
    path = out_dir / "corrections.json"
    data: dict = {"schema_version": 1, "corrections": []}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    corrections = list(data.get("corrections") or [])
    corrections.append(record)
    data["corrections"] = corrections
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# ─── Request / Response models ────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    video_path: str
    stride:     int   = 1
    start_sec:  float = 0.0
    end_sec:    Optional[float] = None


class CorrectionRequest(BaseModel):
    video_path:          str
    frame_idx:           int
    correction_type:     Optional[str] = None   # "bbox_correction" | "click_selection" | "mask_correction"
    type:                Optional[str] = None   # UI alias for correction_type
    data:                dict  # {"x1","y1","x2","y2"} or {"x","y"} or {"mask":[...]}
    propagation_radius:  int   = 15
    propagation_end_frame: Optional[int] = None  # specific end frame for forward pass
    sot_backend:         str   = "none"         # "none" | "csrt" | "sam2"
    output_dir:          Optional[str] = None


class CorrectionResponse(BaseModel):
    corrected_frame:  dict
    updated_frames:   list[dict]
    total_affected:   int
    pose_warning:     Optional[str] = None


class CalibrationPayload(BaseModel):
    version:   int = 2
    video:     str
    keyframes: list = []
    mode:      Optional[str] = None
    seeds:     Optional[list] = None
    propagation: Optional[dict] = None


class SeedsPayload(BaseModel):
    seeds:       list
    video_path:  Optional[str] = None
    mode:        str = "seed_auto"


class PropagatePayload(BaseModel):
    video_path:    Optional[str] = None
    snap_to_lines: bool = False
    from_frame:    Optional[int] = None


class VenueLearnPayload(BaseModel):
    video_name:  str
    video_path:  Optional[str] = None
    venue_id:    str = DEFAULT_VENUE_ID
    accumulate:  bool = True
    samples:     Optional[list] = None


class VenueApplyPayload(BaseModel):
    video_path:  Optional[str] = None
    venue_id:    str = DEFAULT_VENUE_ID
    merge:       bool = True
    prefer_propagation: bool = True
    use_masks:   bool = True
    prefer_keyframes: bool = True


class VenueTrainPayload(BaseModel):
    video_name:  Optional[str] = None
    venue_id:    str = DEFAULT_VENUE_ID
    video_path:  Optional[str] = None
    epochs:      int = 40
    imgsz:       int = 640
    model:       str = "yolo11n-seg.pt"


class VenueCorrectPayload(BaseModel):
    frame_idx:       int
    layer:           str = "track"          # track | sand
    mask_grid:       Optional[list] = None
    full_mask:       Optional[list] = None
    operation:       str = "add"            # add | remove
    radius:          int = 15
    direction:       str = "both"           # both | forward | backward
    video_path:      Optional[str] = None


class PhaseMarkPayload(BaseModel):
    frame_idx:       int
    phase:           str
    pose_tag:        Optional[str] = None   # hop_contact | hop_flight | final_takeoff | feet_together
    athlete_id:      Optional[str] = None
    update_template: bool = True


class PhaseMovePayload(BaseModel):
    from_frame_idx:  int
    to_frame_idx:    int


class PhasePropagatePayload(BaseModel):
    athlete_id:      Optional[str] = None


class MetricsOverridesPayload(BaseModel):
    hops_corridor_m:   Optional[float] = None  # known m: 1st hop contact → landing (default 10)
    hop_lengths_m:     Optional[list] = None   # legacy: up to 5 hop1–4 + final
    total_length_m:    Optional[float] = None  # legacy alias of hops_corridor_m
    known_distance_m:  Optional[float] = None
    point_a:           Optional[list] = None   # [x,y] norm or px
    point_b:           Optional[list] = None
    m_per_px:          Optional[float] = None
    notes:             Optional[str] = None
    athlete_id:        Optional[str] = None
    clear:             Optional[list] = None   # keys to remove from overrides


class MetricsScalePayload(BaseModel):
    hops_corridor_m: float
    athlete_id:      Optional[str] = None


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/status")
def status():
    models_loaded = "pose" in _models
    return {
        "status":        "ok",
        "models_loaded": models_loaded,
        "cached_videos": list(_analyses_cache.keys()),
    }


@app.get("/analysis/{video_name}")
def get_analysis(video_name: str):
    data = _load_analysis(video_name)
    if data is None:
        raise HTTPException(404, f"No analysis found for '{video_name}'. Run /analyze first.")
    return JSONResponse(data)


@app.get("/api/calibration/{video_name}")
def get_calibration(video_name: str):
    out_dir = OUTPUT_ROOT / video_name
    data = load_calibration(out_dir)
    if data is None:
        return JSONResponse(default_calibration(video_name))
    return JSONResponse(data)


@app.post("/api/calibration/{video_name}")
def post_calibration(video_name: str, payload: CalibrationPayload):
    out_dir = OUTPUT_ROOT / video_name
    out_dir.mkdir(parents=True, exist_ok=True)
    data = normalize_calibration(payload.model_dump(exclude_none=True))
    if not data.get("video"):
        data["video"] = f"{video_name}.mp4"
    save_calibration(out_dir, data)
    return JSONResponse({"ok": True, "calibration": data})


def _resolve_video_path(video_name: str, out_dir: Path,
                        video_path: Optional[str] = None) -> Path:
    if video_path:
        p = Path(video_path)
        if p.exists():
            return p
    cal = load_calibration(out_dir)
    if cal and cal.get("video"):
        for candidate in (
            Path(cal["video"]),
            Path(video_name).parent / cal["video"],
            Path(video_name) / cal["video"],
            Path(".") / Path(cal["video"]).name,
            Path(".") / cal["video"],
        ):
            if candidate.exists():
                return candidate
    for candidate in (
        Path(f"{video_name}.mp4"),
        Path(f"{video_name}.mov"),
        Path(".") / f"{video_name}.mp4",
    ):
        if candidate.exists():
            return candidate
    raise HTTPException(404, f"Video file not found for '{video_name}'. Pass video_path.")


def _recompute_tracking_if_analysis(out_dir: Path, video_name: str) -> int:
    analysis_path = out_dir / "analysis.json"
    if not analysis_path.exists():
        return 0
    with open(analysis_path, encoding="utf-8") as f:
        data = json.load(f)
    frames = data.get("frames", [])
    if not frames:
        return 0
    vi = data.get("video_info", {})
    width = int(vi.get("width", 1280))
    height = int(vi.get("height", 720))
    updated_frames, count = recompute_frames_track_fields(
        frames, out_dir, width=width, height=height,
    )
    data["frames"] = updated_frames
    with open(analysis_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return count


def _maybe_auto_propagate_calibration(video_name: str, video_path: str) -> None:
    out_dir = OUTPUT_ROOT / video_name
    cal = load_calibration(out_dir)
    targets = target_frames_from_analysis(out_dir)

    try:
        if cal is not None and has_seeds(cal):
            if targets and keyframes_incomplete(cal, targets):
                print(f"[API] Auto-propagating calibration for {video_name}...")
                run_propagation_for_output(out_dir, Path(video_path))
        elif cal is not None and cal.get("keyframes"):
            pass  # manual keyframes already present

        cal_after = load_calibration(out_dir)
        if cal_after and cal_after.get("keyframes"):
            count = _recompute_tracking_if_analysis(out_dir, video_name)
            print(f"[API] Auto calibration done · tracking updated on {count} frames")
    except Exception as exc:
        print(f"[API] Auto calibration failed for {video_name}: {exc}")


@app.post("/api/calibration/{video_name}/seeds")
def post_calibration_seeds(video_name: str, payload: SeedsPayload):
    out_dir = OUTPUT_ROOT / video_name
    out_dir.mkdir(parents=True, exist_ok=True)
    cal = load_calibration(out_dir) or default_calibration(video_name)
    cal["version"] = max(int(cal.get("version", 1)), 2)
    cal["mode"] = payload.mode
    cal["seeds"] = payload.seeds
    if payload.video_path:
        cal["video"] = Path(payload.video_path).name
    elif not cal.get("video"):
        cal["video"] = f"{video_name}.mp4"
    data = normalize_calibration(cal)
    save_calibration(out_dir, data)
    return JSONResponse({"ok": True, "calibration": data})


@app.post("/api/calibration/{video_name}/propagate")
def post_calibration_propagate(video_name: str, payload: PropagatePayload):
    out_dir = OUTPUT_ROOT / video_name
    out_dir.mkdir(parents=True, exist_ok=True)
    cal = load_calibration(out_dir)
    if cal is None or not has_seeds(cal):
        raise HTTPException(
            400,
            f"No seeds for '{video_name}'. POST /api/calibration/{video_name}/seeds first.",
        )

    video_file = _resolve_video_path(video_name, out_dir, payload.video_path)

    try:
        data = run_propagation_for_output(
            out_dir,
            video_file,
            snap_to_lines=payload.snap_to_lines,
            from_frame=payload.from_frame,
        )
    except Exception as exc:
        raise HTTPException(500, f"Propagation failed: {exc}") from exc

    frames_updated = _recompute_tracking_if_analysis(out_dir, video_name)

    return JSONResponse({
        "ok": True,
        "calibration": data,
        "frames_propagated": data.get("propagation", {}).get("frame_count", 0),
        "tracking_frames_updated": frames_updated,
    })


@app.post("/api/recompute-tracking/{video_name}")
def recompute_tracking(video_name: str):
    """
    Recompute track_overlap / athlete_state / position_s / predicted_bbox
    on existing analysis.json using calibration.json (no YOLO re-run).
    """
    out_dir = OUTPUT_ROOT / video_name
    analysis_path = out_dir / "analysis.json"
    if not analysis_path.exists():
        raise HTTPException(
            404,
            f"No analysis found for '{video_name}'. Run /analyze first.",
        )
    cal = load_calibration(out_dir)
    if cal is None:
        raise HTTPException(
            400,
            f"No calibration for '{video_name}'. Apply venue masks in Pista mode first.",
        )
    mask_mode = cal.get("mode") == "color_masks" and bool(cal.get("mask_frames"))
    if not mask_mode and not cal.get("keyframes"):
        raise HTTPException(
            400,
            f"No track calibration for '{video_name}'. "
            "Apply venue profile in Pista mode first.",
        )

    with open(analysis_path, encoding="utf-8") as f:
        data = json.load(f)
    frames = data.get("frames", [])
    if not frames:
        raise HTTPException(400, "analysis.json has no frames.")

    vi = data.get("video_info", {})
    width = int(vi.get("width", 1280))
    height = int(vi.get("height", 720))

    updated_frames, count = recompute_frames_track_fields(
        frames, out_dir, width=width, height=height,
    )
    data["frames"] = updated_frames
    with open(analysis_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return JSONResponse({
        "ok": True,
        "video_name": video_name,
        "frames_updated": count,
    })


@app.post("/api/sections/analyze/{video_name}")
def analyze_sections_endpoint(video_name: str, use_pose: bool = True):
    """Detect approach/hops/landing from ankles + pose; write sections.json."""
    out_dir = OUTPUT_ROOT / video_name
    analysis_path = out_dir / "analysis.json"
    if not analysis_path.exists():
        raise HTTPException(
            404,
            f"No analysis found for '{video_name}'. Run /analyze first.",
        )

    try:
        sections = run_section_analysis(out_dir, use_pose=use_pose)
    except Exception as exc:
        raise HTTPException(500, f"Section analysis failed: {exc}") from exc

    return JSONResponse({
        "ok": True,
        "video_name": video_name,
        "sections": sections,
        "contacts_found": len(sections.get("contacts", [])),
        "markers_count": len(sections.get("phase_markers", [])),
        "confidence": sections.get("confidence", 0),
        "derived_version": sections.get("derived_version", 0),
    })


@app.post("/api/sections/mark/{video_name}")
def mark_phase_endpoint(video_name: str, payload: PhaseMarkPayload):
    """Assign phase/pose to a frame on the timeline."""
    out_dir = OUTPUT_ROOT / video_name
    if not (out_dir / "analysis.json").exists():
        raise HTTPException(404, f"No analysis for '{video_name}'.")

    try:
        sections = mark_phase_on_frame(
            out_dir,
            payload.frame_idx,
            payload.phase,
            pose_tag=payload.pose_tag,
            athlete_id=payload.athlete_id,
            update_template=payload.update_template,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, f"Phase mark failed: {exc}") from exc

    return JSONResponse({
        "ok": True,
        "video_name": video_name,
        "sections": sections,
        "markers_count": len(sections.get("phase_markers", [])),
        "contacts_found": len(sections.get("contacts", [])),
    })


@app.post("/api/sections/mark/{video_name}/move")
def move_phase_marker_endpoint(video_name: str, payload: PhaseMovePayload):
    """Drag a phase marker to another frame (swaps if destination occupied)."""
    out_dir = OUTPUT_ROOT / video_name
    if not (out_dir / "analysis.json").exists():
        raise HTTPException(404, f"No analysis for '{video_name}'.")

    try:
        sections = move_phase_marker_on_frame(
            out_dir, payload.from_frame_idx, payload.to_frame_idx,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, f"Phase move failed: {exc}") from exc

    return JSONResponse({
        "ok": True,
        "video_name": video_name,
        "sections": sections,
        "markers_count": len(sections.get("phase_markers", [])),
        "contacts_found": len(sections.get("contacts", [])),
    })


@app.delete("/api/sections/mark/{video_name}/{frame_idx}")
def unmark_phase_endpoint(video_name: str, frame_idx: int):
    out_dir = OUTPUT_ROOT / video_name
    if not (out_dir / "analysis.json").exists():
        raise HTTPException(404, f"No analysis for '{video_name}'.")

    try:
        sections = unmark_phase_frame(out_dir, frame_idx)
    except Exception as exc:
        raise HTTPException(500, f"Phase unmark failed: {exc}") from exc

    return JSONResponse({
        "ok": True,
        "video_name": video_name,
        "sections": sections,
    })


@app.post("/api/sections/propagate/{video_name}")
def propagate_phases_endpoint(video_name: str, payload: PhasePropagatePayload = PhasePropagatePayload()):
    """Backward-propagate hop markers from final_jump/landing anchor."""
    out_dir = OUTPUT_ROOT / video_name
    if not (out_dir / "analysis.json").exists():
        raise HTTPException(404, f"No analysis for '{video_name}'.")

    try:
        sections = run_phase_propagation(out_dir)
        if payload.athlete_id:
            sections["athlete_id"] = payload.athlete_id
            from src.section_analyzer import write_sections
            write_sections(out_dir, sections)
    except Exception as exc:
        raise HTTPException(500, f"Phase propagation failed: {exc}") from exc

    return JSONResponse({
        "ok": True,
        "video_name": video_name,
        "sections": sections,
        "markers_count": len(sections.get("phase_markers", [])),
        "contacts_found": len(sections.get("contacts", [])),
    })


@app.get("/api/sections/pose-scores/{video_name}")
def pose_scores_endpoint(video_name: str, athlete_id: Optional[str] = None):
    """Per-frame pose phase scores (debug / preview)."""
    out_dir = OUTPUT_ROOT / video_name
    analysis_path = out_dir / "analysis.json"
    if not analysis_path.exists():
        raise HTTPException(404, f"No analysis for '{video_name}'.")

    with open(analysis_path, encoding="utf-8") as f:
        data = json.load(f)
    frames = data.get("frames") or []
    scores = pose_classify_frames(frames, athlete_id=athlete_id)
    return JSONResponse({"ok": True, "scores": scores})


@app.get("/api/sections/{video_name}")
def get_sections(video_name: str):
    out_dir = OUTPUT_ROOT / video_name
    sections_path = out_dir / "sections.json"
    if not sections_path.exists():
        raise HTTPException(404, f"No sections.json for '{video_name}'.")
    with open(sections_path, encoding="utf-8") as f:
        data = json.load(f)
    return JSONResponse(data)


@app.get("/api/metrics/{video_name}")
def get_metrics(video_name: str):
    out_dir = OUTPUT_ROOT / video_name
    if not (out_dir / "analysis.json").exists():
        raise HTTPException(404, f"No analysis for '{video_name}'.")
    metrics_path = out_dir / "metrics.json"
    if not metrics_path.exists():
        return JSONResponse(load_metrics(out_dir))
    with open(metrics_path, encoding="utf-8") as f:
        return JSONResponse(json.load(f))


@app.post("/api/metrics/{video_name}/compute")
def compute_metrics_endpoint(video_name: str, athlete_id: Optional[str] = None):
    """Recompute metrics from sections + analysis (+ calibration / overrides)."""
    out_dir = OUTPUT_ROOT / video_name
    if not (out_dir / "analysis.json").exists():
        raise HTTPException(404, f"No analysis for '{video_name}'.")
    if not (out_dir / "sections.json").exists():
        raise HTTPException(404, f"No sections.json for '{video_name}'. Analyze sections first.")
    try:
        metrics = compute_metrics(out_dir, athlete_id=athlete_id)
    except Exception as exc:
        raise HTTPException(500, f"Metrics compute failed: {exc}") from exc
    return JSONResponse({"ok": True, "video_name": video_name, "metrics": metrics})


@app.post("/api/metrics/{video_name}/overrides")
def metrics_overrides_endpoint(video_name: str, payload: MetricsOverridesPayload):
    """Save scale overrides (prefer hops_corridor_m), recompute metrics."""
    out_dir = OUTPUT_ROOT / video_name
    if not (out_dir / "analysis.json").exists():
        raise HTTPException(404, f"No analysis for '{video_name}'.")
    data = payload.model_dump(exclude_none=True)
    # Normalize legacy total_length_m → hops_corridor_m
    if data.get("hops_corridor_m") is None and data.get("total_length_m") is not None:
        data["hops_corridor_m"] = data["total_length_m"]
    try:
        metrics = apply_overrides(
            out_dir,
            data,
            athlete_id=payload.athlete_id,
        )
    except Exception as exc:
        raise HTTPException(500, f"Metrics overrides failed: {exc}") from exc
    return JSONResponse({"ok": True, "video_name": video_name, "metrics": metrics})


@app.post("/api/metrics/{video_name}/scale")
def metrics_scale_endpoint(video_name: str, payload: MetricsScalePayload):
    """Set hops corridor length (m) and recompute metrics. Persists as venue default."""
    out_dir = OUTPUT_ROOT / video_name
    if not (out_dir / "analysis.json").exists():
        raise HTTPException(404, f"No analysis for '{video_name}'.")
    if payload.hops_corridor_m <= 0:
        raise HTTPException(400, "hops_corridor_m must be > 0.")
    try:
        metrics = apply_overrides(
            out_dir,
            {"hops_corridor_m": float(payload.hops_corridor_m)},
            athlete_id=payload.athlete_id,
        )
    except Exception as exc:
        raise HTTPException(500, f"Metrics scale failed: {exc}") from exc
    return JSONResponse({"ok": True, "video_name": video_name, "metrics": metrics})


@app.get("/api/metrics/{video_name}/pose-overlay/{phase}")
def pose_overlay_endpoint(
    video_name: str,
    phase: str,
    output_dir: Optional[str] = None,
    force: bool = False,
):
    """
    PNG overlay: reference (General) vs current take legs at hop contact / final flight.
    phase: hop_1|hop_2|hop_3|hop_4|final_flight
    Optional output_dir for refined analysis folders.
    """
    if phase not in OVERLAY_PHASES:
        raise HTTPException(400, f"phase must be one of: {', '.join(OVERLAY_PHASES)}")
    out_dir = Path(output_dir) if output_dir else OUTPUT_ROOT / video_name
    if not (out_dir / "analysis.json").exists():
        raise HTTPException(404, f"No analysis for '{video_name}' in {out_dir}.")
    try:
        png_bytes, meta = render_pose_overlay_png(
            out_dir,
            phase,
            video_name=video_name,
            use_cache=True,
            force=force,
        )
    except Exception as exc:
        raise HTTPException(500, f"Pose overlay failed: {exc}") from exc
    from fastapi.responses import Response
    headers = {}
    if meta.get("source"):
        headers["X-Pose-Overlay-Source"] = str(meta["source"])
    if meta.get("frame_idx") is not None:
        headers["X-Pose-Overlay-Frame"] = str(meta["frame_idx"])
    return Response(content=png_bytes, media_type="image/png", headers=headers)


@app.get("/api/venue/profile")
def get_venue_profile(venue_id: str = DEFAULT_VENUE_ID):
    profile = load_profile(venue_id)
    if profile is None:
        return JSONResponse({
            "learned": False,
            "venue_id": venue_id,
            "profile": None,
        })
    return JSONResponse({
        "learned": True,
        "venue_id": venue_id,
        "frames_used": profile.get("frames_used", 0),
        "sand_frames_used": profile.get("sand_frames_used", 0),
        "source_video": profile.get("source_video"),
        "videos_contributed": profile.get("videos_contributed", []),
        "sample_count": (
            profile.get("track_color", {}).get("sample_count")
            or profile.get("track_hsv", {}).get("sample_count", 0)
        ),
        "learned_at": profile.get("learned_at"),
        "hops_corridor_m": profile.get("hops_corridor_m", 10.0),
        "profile": profile,
    })


@app.post("/api/venue/learn")
def post_venue_learn(payload: VenueLearnPayload):
    video_name = payload.video_name
    out_dir = OUTPUT_ROOT / video_name
    cal_path = out_dir / "calibration.json"
    if not payload.samples and not cal_path.exists():
        raise HTTPException(
            400,
            f"No calibration.json for '{video_name}'. Paint brush samples or calibrate first.",
        )
    video_file = _resolve_video_path(video_name, out_dir, payload.video_path)
    try:
        if payload.samples:
            with open(cal_path, encoding="utf-8") as f:
                cal = json.load(f)
            cal_selections = selections_from_calibration(cal)
            by_frame = {int(s["frame_idx"]): s for s in cal_selections}
            for sample in payload.samples:
                fidx = int(sample["frame_idx"])
                if fidx in by_frame:
                    by_frame[fidx] = {**by_frame[fidx], **sample}
                else:
                    by_frame[fidx] = sample
            selections = list(by_frame.values())
            profile = learn_from_selections(
                video_file,
                selections,
                accumulate=payload.accumulate,
                venue_id=payload.venue_id,
            )
        else:
            with open(cal_path, encoding="utf-8") as f:
                cal = json.load(f)
            profile = learn_from_calibration(
                video_file,
                cal,
                venue_id=payload.venue_id,
                accumulate=payload.accumulate,
            )
    except Exception as exc:
        raise HTTPException(500, f"Venue learn failed: {exc}") from exc

    dataset_result: dict = {}
    try:
        from scripts.train_venue_seg import export_dataset_append, get_dataset_info
        from src.venue_masks import polygon_keyframes

        with open(cal_path, encoding="utf-8") as f:
            cal_for_export = json.load(f)
        if polygon_keyframes(cal_for_export):
            dataset_result = export_dataset_append(
                video_name,
                cal_path,
                video_path=video_file,
                venue_id=payload.venue_id,
            )
        else:
            info = get_dataset_info(payload.venue_id)
            dataset_result = {
                "frames_exported": 0,
                "videos_in_dataset": info["video_count"],
                "total_dataset_frames": info["total_frames"],
                "dataset_manifest": info["manifest"],
            }
    except Exception as exc:
        raise HTTPException(500, f"Venue learn ok but dataset export failed: {exc}") from exc

    manifest = dataset_result.get("dataset_manifest", {})
    return JSONResponse({
        "ok": True,
        "venue_id": payload.venue_id,
        "frames_used": profile.get("frames_used", 0),
        "sand_frames_used": profile.get("sand_frames_used", 0),
        "videos_contributed": profile.get("videos_contributed", []),
        "sample_count": profile.get("track_color", {}).get("sample_count", 0),
        "profile_path": str(VENUE_ROOT / payload.venue_id / "profile.json"),
        "frames_exported": dataset_result.get("frames_exported", 0),
        "videos_in_dataset": dataset_result.get("videos_in_dataset", 0),
        "total_dataset_frames": dataset_result.get("total_dataset_frames", 0),
        "dataset_manifest": manifest,
    })


@app.get("/api/venue/model")
def get_venue_model(venue_id: str = DEFAULT_VENUE_ID):
    meta = load_model_meta(venue_id)
    trained = has_trained_seg_model(venue_id)
    return JSONResponse({
        "venue_id": venue_id,
        "trained": trained,
        "model": meta,
        "weights": meta.get("weights") if meta else None,
        "metrics": meta.get("metrics") if meta else None,
        "trained_at": meta.get("trained_at") if meta else None,
        "classes": meta.get("classes", ["track", "sand"]) if meta else ["track", "sand"],
    })


@app.get("/api/venue/dataset")
def get_venue_dataset(venue_id: str = DEFAULT_VENUE_ID):
    from scripts.train_venue_seg import get_dataset_info

    return JSONResponse(get_dataset_info(venue_id))


@app.post("/api/venue/train")
def post_venue_train(payload: VenueTrainPayload):
    """Rebuild combined dataset from manifest and fine-tune CNN seg model."""
    venue_id = payload.venue_id or DEFAULT_VENUE_ID
    epochs = max(1, min(payload.epochs, 200))
    imgsz = max(320, min(payload.imgsz, 1280))

    from scripts.train_venue_seg import get_dataset_info, load_dataset_manifest

    info = get_dataset_info(venue_id)
    if info["total_frames"] < 5:
        raise HTTPException(
            400,
            f"Dataset CNN insuficiente ({info['total_frames']} frames). "
            "Usa 'Aprender de este video' en al menos un video con poligonos.",
        )

    job = create_job()
    job.update({"result_video_name": venue_id})
    job.start()

    def _run():
        import warnings
        warnings.filterwarnings("ignore")
        try:
            from scripts.train_venue_seg import export_dataset_rebuild, train_venue_model

            def on_progress(event: dict):
                job.update(event)

            manifest = load_dataset_manifest(venue_id)
            export_dataset_rebuild(venue_id, manifest=manifest, on_progress=on_progress)
            meta = train_venue_model(
                venue_id,
                epochs=epochs,
                imgsz=imgsz,
                model_name=payload.model,
                on_progress=on_progress,
            )
            job.update({
                "message": f"CNN entrenado: {meta.get('weights', '')}",
                "percent": 100.0,
            })
            job.finish(venue_id)
        except Exception as exc:
            job.fail(str(exc))

    threading.Thread(target=_run, daemon=True).start()

    return JSONResponse({
        "ok": True,
        "job_id": job.job_id,
        "venue_id": venue_id,
        "epochs": epochs,
        "total_dataset_frames": info["total_frames"],
        "videos_in_dataset": info["video_count"],
        "status": "started",
        "poll": f"/api/jobs/{job.job_id}",
        "warning": "El entrenamiento CNN puede tardar varios minutos.",
    })


@app.post("/api/venue/apply/{video_name}")
def post_venue_apply(video_name: str, payload: VenueApplyPayload):
    out_dir = OUTPUT_ROOT / video_name
    analysis_path = out_dir / "analysis.json"
    if not analysis_path.exists():
        raise HTTPException(
            404,
            f"No analysis found for '{video_name}'. Run /analyze first.",
        )
    from src.calibration import load_calibration

    cal_existing = load_calibration(out_dir)
    use_cnn = payload.use_masks and has_trained_seg_model(payload.venue_id)
    use_keyframe_pipeline = (
        payload.use_masks
        and not use_cnn
        and cal_existing is not None
        and should_use_keyframe_pipeline(cal_existing, prefer_keyframes=payload.prefer_keyframes)
    )
    profile = load_profile(payload.venue_id)
    if not use_cnn and not use_keyframe_pipeline and profile is None:
        raise HTTPException(
            400,
            "No venue profile learned. POST /api/venue/learn first, train CNN, or add manual keyframes.",
        )
    video_file = _resolve_video_path(video_name, out_dir, payload.video_path)
    use_masks = payload.use_masks and (
        use_cnn
        or use_keyframe_pipeline
        or (profile and int(profile.get("version", 2)) >= 3)
    )
    try:
        if use_masks:
            cal = apply_masks_to_output(
                out_dir,
                video_file,
                profile=profile,
                venue_id=payload.venue_id,
                prefer_keyframes=payload.prefer_keyframes,
            )
        else:
            cal = apply_profile_to_output(
                out_dir,
                video_file,
                venue_id=payload.venue_id,
                merge_existing=payload.merge,
                prefer_propagation=payload.prefer_propagation,
            )
    except Exception as exc:
        raise HTTPException(500, f"Venue apply failed: {exc}") from exc

    frames_updated = 0
    return JSONResponse({
        "ok": True,
        "video_name": video_name,
        "mode": cal.get("mode", "venue_profile"),
        "keyframes_applied": len(cal.get("keyframes", [])),
        "mask_frames_applied": len(cal.get("mask_frames", {})),
        "tracking_frames_updated": frames_updated,
        "calibration": cal,
    })


@app.post("/api/venue/correct/{video_name}")
def post_venue_correct(video_name: str, payload: VenueCorrectPayload):
    """Apply manual track/sand mask correction and propagate via optical flow."""
    out_dir = OUTPUT_ROOT / video_name
    cal = load_calibration(out_dir)
    if cal is None or cal.get("mode") != "color_masks":
        raise HTTPException(
            400,
            f"No color_masks calibration for '{video_name}'. "
            "Apply venue profile in Pista mode first.",
        )
    video_file = _resolve_video_path(video_name, out_dir, payload.video_path)
    layer = payload.layer if payload.layer in ("track", "sand") else "track"
    operation = payload.operation if payload.operation in ("add", "remove") else "add"
    direction = payload.direction if payload.direction in ("both", "forward", "backward") else "both"
    try:
        result = apply_mask_correction(
            video_file,
            out_dir,
            payload.frame_idx,
            layer,
            mask_grid=payload.mask_grid,
            operation=operation,
            radius=max(0, min(payload.radius, 120)),
            direction=direction,
            full_mask=payload.full_mask,
        )
    except Exception as exc:
        raise HTTPException(500, f"Venue correction failed: {exc}") from exc
    return JSONResponse({"ok": True, **result})


@app.get("/api/venue/masks/{video_name}/{frame_idx}")
def get_venue_masks(
    video_name: str,
    frame_idx: int,
    format: str = "json",
    video_path: Optional[str] = None,
):
    """Return venue track/sand mask info or composite overlay PNG."""
    from src.mask_utils import composite_mask_overlay, load_mask_png, mask_area_norm

    out_dir = OUTPUT_ROOT / video_name
    cal = load_calibration(out_dir)
    mask_modes = ("color_masks", "keyframe_masks", "cnn_masks")
    if cal is None or cal.get("mode") not in mask_modes:
        raise HTTPException(404, f"No venue masks calibration for '{video_name}'.")

    entry = (cal.get("mask_frames") or {}).get(str(frame_idx))
    if not entry:
        raise HTTPException(404, f"No venue masks for frame {frame_idx}.")

    track_path = out_dir / entry["track"]
    sand_path = out_dir / entry["sand"]
    track_mask = load_mask_png(track_path)
    sand_mask = load_mask_png(sand_path)

    if format == "overlay":
        overlay = composite_mask_overlay(track_mask, sand_mask)
        background = None
        try:
            vp = _resolve_video_path(video_name, out_dir, video_path)
            cap = cv2.VideoCapture(str(vp))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            cap.release()
            if ok and frame is not None:
                background = frame
        except HTTPException:
            pass
        if background is None:
            frame_path = out_dir / "frames" / f"frame_{frame_idx:06d}.jpg"
            if frame_path.exists():
                background = cv2.imread(str(frame_path))
        if background is not None:
            if overlay.shape[:2] != background.shape[:2]:
                overlay = cv2.resize(
                    overlay, (background.shape[1], background.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            blended = cv2.addWeighted(background, 1.0, overlay, 0.6, 0)
            _, buf = cv2.imencode(".png", blended)
            from fastapi.responses import Response
            return Response(content=buf.tobytes(), media_type="image/png")
        _, buf = cv2.imencode(".png", overlay)
        from fastapi.responses import Response
        return Response(content=buf.tobytes(), media_type="image/png")

    vi_w = vi_h = 0
    analysis_path = out_dir / "analysis.json"
    if analysis_path.exists():
        with open(analysis_path, encoding="utf-8") as f:
            vi = json.load(f).get("video_info", {})
        vi_w = int(vi.get("width", 0))
        vi_h = int(vi.get("height", 0))

    track_cov = mask_area_norm(track_mask, vi_w or 1280, vi_h or 720) if track_mask is not None else 0.0
    sand_cov = mask_area_norm(sand_mask, vi_w or 1280, vi_h or 720) if sand_mask is not None else 0.0

    return JSONResponse({
        "frame_idx": frame_idx,
        "confidence": entry.get("confidence"),
        "track_coverage": round(track_cov, 4),
        "sand_coverage": round(sand_cov, 4),
        "track_url": _media_url(track_path),
        "sand_url": _media_url(sand_path),
        "overlay_url": (
            f"/api/venue/masks/{video_name}/{frame_idx}?format=overlay"
            + (f"&video_path={video_path}" if video_path else "")
        ),
    })


def _media_url(path: Path | None) -> str:
    if not path:
        return ""
    return f"/media?path={path.resolve().as_posix()}"


def _indexed_images(folder: Path, prefix: str) -> dict[str, str]:
    images: dict[str, str] = {}
    if not folder.exists():
        return images
    for path in folder.glob("*.jpg"):
        stem = path.stem.replace(prefix, "")
        key = stem.lstrip("0") or "0"
        padded = stem if stem.isdigit() else key
        images[key] = _media_url(path)
        images[padded] = _media_url(path)
    return images


def _project_payload(video_path: str, output_dir: Optional[str] = None) -> dict:
    video_file = Path(video_path)
    video_name = _video_name(video_path)
    out_dir = Path(output_dir) if output_dir else OUTPUT_ROOT / video_name
    duration_s = _video_duration(video_file) if video_file.exists() else 0.0
    analysis_path = out_dir / "analysis.json"
    sections_path = out_dir / "sections.json"
    metrics_path = out_dir / "metrics.json"
    chart_path = out_dir / "charts" / "camera_angle_timeline.png"
    analysis_data = None
    sections_data = None
    metrics_data = None
    if analysis_path.exists():
        with open(analysis_path, encoding="utf-8") as f:
            analysis_data = json.load(f)
    if sections_path.exists():
        with open(sections_path, encoding="utf-8") as f:
            sections_data = json.load(f)
    if metrics_path.exists():
        with open(metrics_path, encoding="utf-8") as f:
            metrics_data = json.load(f)

    frames = analysis_data.get("frames", []) if analysis_data else []
    if sections_data and frames:
        for frame in frames:
            fidx = frame.get("frame_idx")
            if fidx is not None:
                frame["phase"] = phase_at_frame(sections_data, int(fidx))

    return {
        "video": {
            "path": str(video_file),
            "name": video_file.name or f"{video_name}.mp4",
            "video_name": video_name,
            "exists": video_file.exists(),
            "url": _media_url(video_file) if video_file.exists() else "",
            "duration_s": duration_s,
        },
        "output": {"path": str(out_dir), "exists": out_dir.exists()},
        "analysis": {
            "path": str(analysis_path),
            "exists": analysis_path.exists(),
            "data": analysis_data,
            "frames": frames,
        },
        "sections": {
            "path": str(sections_path),
            "exists": sections_path.exists(),
            "data": sections_data,
        },
        "metrics": {
            "path": str(metrics_path),
            "exists": metrics_path.exists(),
            "data": metrics_data,
        },
        "assets": {
            "annotated": _indexed_images(out_dir / "annotated", "annotated_"),
            "frames": _indexed_images(out_dir / "frames", "frame_"),
            "chart": _media_url(chart_path) if chart_path.exists() else "",
        },
    }


@app.get("/api/project")
def get_project(video_path: str, output_dir: Optional[str] = None):
    return JSONResponse(_project_payload(video_path, output_dir=output_dir))


@app.get("/api/demo")
def get_demo():
    demo_analysis = next(OUTPUT_ROOT.glob("*/analysis.json"), None)
    if not demo_analysis:
        raise HTTPException(404, "No demo analysis found in output/")
    demo_video = demo_analysis.parent.name + ".mp4"
    return JSONResponse(_project_payload(demo_video))


@app.get("/api/videos")
def list_videos():
    videos = []
    ignored_parts = {"venv", "node_modules", ".git", "__pycache__", "dist"}
    for path in VIDEO_ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        if any(part in ignored_parts for part in path.parts):
            continue
        resolved = path.resolve()
        video_name = _video_name(str(resolved))
        analysis_path  = OUTPUT_ROOT / video_name / "analysis.json"
        refined_dir    = OUTPUT_ROOT / f"{video_name}_refined"
        refined_path   = refined_dir / "analysis.json"
        videos.append({
            "name":             path.name,
            "path":             str(resolved),
            "video_name":       video_name,
            "duration_s":       _video_duration(resolved),
            "has_analysis":     analysis_path.exists(),
            "has_refined":      refined_path.exists(),
            "refined_output_dir": str(refined_dir.resolve()) if refined_path.exists() else None,
            "url":              _media_url(resolved),
        })
    videos.sort(key=lambda item: (not item["has_analysis"], item["name"].lower()))
    return {"videos": videos}


@app.get("/api/debug_coords")
def debug_coords(
    video_path: str,
    frame_idx:  int,
    x1: float, y1: float, x2: float, y2: float,
):
    """
    Draw the UI-sent bbox on the actual raw frame and return the JPEG.
    Use this to verify coordinate mapping: if the red box matches the
    drawn selection on screen, mediaMetrics() is correct.
    """
    from fastapi.responses import Response as RawResponse

    video_name = _video_name(video_path)
    frame_path = OUTPUT_ROOT / video_name / "frames" / f"frame_{frame_idx:06d}.jpg"

    if frame_path.exists():
        img = cv2.imread(str(frame_path))
    else:
        if not Path(video_path).exists():
            raise HTTPException(404, f"Video not found: {video_path}")
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, img = cap.read()
        cap.release()
        if not ret or img is None:
            raise HTTPException(400, f"Cannot read frame {frame_idx}")

    if img is None:
        raise HTTPException(500, "Failed to load frame image")

    h, w = img.shape[:2]
    ix1 = int(np.clip(x1, 0, w - 1))
    iy1 = int(np.clip(y1, 0, h - 1))
    ix2 = int(np.clip(x2, 0, w))
    iy2 = int(np.clip(y2, 0, h))

    font = cv2.FONT_HERSHEY_SIMPLEX

    # Draw a high-contrast bounding box visible on any background:
    # thick black outline then bright yellow interior line
    cv2.rectangle(img, (ix1, iy1), (ix2, iy2), (0, 0, 0), 6)       # black shadow
    cv2.rectangle(img, (ix1, iy1), (ix2, iy2), (0, 230, 255), 3)    # yellow border
    # corner markers
    corner_len = max(10, (ix2 - ix1) // 6)
    for cx, cy, dx, dy in [
        (ix1, iy1,  1,  1), (ix2, iy1, -1,  1),
        (ix1, iy2,  1, -1), (ix2, iy2, -1, -1),
    ]:
        cv2.line(img, (cx, cy), (cx + dx * corner_len, cy), (0, 230, 255), 4)
        cv2.line(img, (cx, cy), (cx, cy + dy * corner_len), (0, 230, 255), 4)

    # Corner coordinate labels
    for label, pt in [
        (f"({ix1},{iy1})", (max(0, ix1),     max(20, iy1 - 6))),
        (f"({ix2},{iy2})", (max(0, ix2 - 90), min(h - 6, iy2 + 18))),
    ]:
        (tw, th), _ = cv2.getTextSize(label, font, 0.55, 2)
        lx, ly = pt
        cv2.rectangle(img, (lx-2, ly-th-3), (lx+tw+2, ly+3), (0,0,0), -1)
        cv2.putText(img, label, (lx, ly), font, 0.55, (0, 230, 255), 2, cv2.LINE_AA)

    # Header banner — always visible at top
    banner = f"Frame {frame_idx}  src={w}x{h}  box=({ix1},{iy1})->({ix2},{iy2})  sz={ix2-ix1}x{iy2-iy1}"
    (bw, bh), _ = cv2.getTextSize(banner, font, 0.55, 2)
    cv2.rectangle(img, (0, 0), (w, bh + 12), (0, 0, 0), -1)
    cv2.putText(img, banner, (6, bh + 5), font, 0.55, (0, 230, 255), 2, cv2.LINE_AA)

    # Encode to memory — no temp files needed
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise HTTPException(500, "Failed to encode image")
    return RawResponse(content=buf.tobytes(), media_type="image/jpeg")


@app.get("/media")
def get_media(path: str):
    media_path = Path(path)
    if not media_path.exists() or not media_path.is_file():
        raise HTTPException(404, f"Media not found: {path}")
    return FileResponse(str(media_path))


def _start_analysis_job(req: AnalyzeRequest) -> dict:
    """Create a job, start the pipeline in a background thread, return job info."""
    if not Path(req.video_path).exists():
        raise HTTPException(404, f"Video not found: {req.video_path}")

    video_name = _video_name(req.video_path)
    job        = create_job()
    job.update({"result_video_name": video_name})
    job.start()

    config = PipelineConfig(
        video_path=req.video_path,
        output_dir=str(OUTPUT_ROOT / video_name),
        stride=req.stride,
        start_sec=req.start_sec,
        end_sec=req.end_sec,
        use_seg=True,
        annotate_every=1,
    )

    def _run():
        import warnings
        warnings.filterwarnings("ignore")
        try:
            def on_progress(event: dict):
                job.update(event)

            run_pipeline(config, on_progress=on_progress)
            _maybe_auto_propagate_calibration(video_name, req.video_path)
            job.finish(video_name)
        except Exception as exc:
            job.fail(str(exc))

    threading.Thread(target=_run, daemon=True).start()

    return {
        "job_id":     job.job_id,
        "video_name": video_name,
        "status":     "started",
        "poll":       f"/api/jobs/{job.job_id}",
    }


@app.post("/analyze")
def analyze_video(req: AnalyzeRequest):
    return _start_analysis_job(req)


@app.post("/api/analyze")
def analyze_video_api(req: AnalyzeRequest):
    return _start_analysis_job(req)


class ReanalyzeRequest(BaseModel):
    video_path:        str
    stride:            int   = 1
    start_sec:         float = 0.0
    end_sec:           Optional[float] = None
    seed_frames:       int   = 15
    seed_start_frame:  Optional[int] = None   # frame_idx interval for seeding
    seed_end_frame:    Optional[int] = None
    use_cnn_masks:     bool  = False
    refine_v2:         bool  = False          # experimental improved refine


@app.post("/api/reanalyze")
def reanalyze_video(req: ReanalyzeRequest):
    """
    Second-pass reanalysis: seeds AppearanceModel from first-pass best
    frames, then re-runs seg+pose on the full video using appearance-only
    athlete selection (no ByteTrack).  Results go to output/<name>_refined/.
    Requires a completed first-pass analysis.json.

    When refine_v2=True: temporal consistency, safer seeds, conservative
    online updates, and post-refine section analysis. Classic path unchanged
    when refine_v2=False.
    """
    if not Path(req.video_path).exists():
        raise HTTPException(404, f"Video not found: {req.video_path}")

    video_name      = _video_name(req.video_path)
    original_outdir = str(OUTPUT_ROOT / video_name)
    refined_outdir  = str(OUTPUT_ROOT / f"{video_name}_refined")

    if not (Path(original_outdir) / "analysis.json").exists():
        raise HTTPException(
            400,
            f"No first-pass analysis found at {original_outdir}/analysis.json. "
            "Run /api/analyze first."
        )

    job = create_job()
    job.update({"result_video_name": f"{video_name}_refined"})
    job.start()

    config = ReanalysisConfig(
        video_path=req.video_path,
        original_output_dir=original_outdir,
        output_dir=refined_outdir,
        stride=req.stride,
        start_sec=req.start_sec,
        end_sec=req.end_sec,
        seed_frames=req.seed_frames,
        annotate_every=1,
        seed_start_frame=req.seed_start_frame,
        seed_end_frame=req.seed_end_frame,
        use_cnn_masks=req.use_cnn_masks,
        refine_v2=req.refine_v2,
    )

    def _run():
        import warnings
        warnings.filterwarnings("ignore")
        try:
            run_reanalysis(config, on_progress=job.update)
            job.finish(f"{video_name}_refined")
        except Exception as exc:
            job.fail(str(exc))

    threading.Thread(target=_run, daemon=True).start()

    return {
        "job_id":     job.job_id,
        "video_name": f"{video_name}_refined",
        "status":     "started",
        "poll":       f"/api/jobs/{job.job_id}",
        "output_dir": refined_outdir,
    }


@app.get("/api/jobs/{job_id}")
def get_job_status(job_id: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(404, f"Job '{job_id}' not found")
    return JSONResponse(job.to_dict())


@app.get("/api/jobs")
def get_all_jobs():
    return JSONResponse(list_jobs())


@app.post("/correct", response_model=CorrectionResponse)
def correct_frame(req: CorrectionRequest):
    """
    Apply a manual correction to a specific frame and propagate to neighbors.

    Steps:
      1. Load the raw frame image.
      2. Re-compute pose using the corrected region.
      3. Update AppearanceModel with this as ground truth.
      4. Re-analyze frames in [frame_idx ± radius].
      5. Update analysis.json and return changed frames.
    """
    if not Path(req.video_path).exists():
        raise HTTPException(404, f"Video not found: {req.video_path}")

    correction_type = req.correction_type or req.type
    if correction_type not in {"bbox_correction", "click_selection", "mask_correction"}:
        raise HTTPException(400, "Invalid correction type")

    model_pose, model_seg = _get_models()
    video_name = _video_name(req.video_path)
    out_dir    = _resolve_output_dir(req.video_path, req.output_dir)
    frames_dir = str(out_dir / "frames")

    # Load or create track state for this video
    if video_name not in _track_states:
        _track_states[video_name] = TrackState()
    state = _track_states[video_name]

    # Find raw frame on disk
    frame_path = str(out_dir / "frames" / f"frame_{req.frame_idx:06d}.jpg")
    if not Path(frame_path).exists():
        # Extract it on the fly
        info = get_video_info(req.video_path)
        cap  = cv2.VideoCapture(req.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, req.frame_idx)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise HTTPException(400, f"Cannot read frame {req.frame_idx} from video")
        Path(frames_dir).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

    correction = Correction(
        frame_idx=req.frame_idx,
        correction_type=correction_type,
        data=req.data,
        propagation_radius=req.propagation_radius,
    )

    # For click_selection: get all detections in this frame first
    all_dets = None
    if correction_type == "click_selection":
        frame_img = cv2.imread(frame_path)
        all_dets  = detections_for_frame(frame_img, model_seg)

    # Apply correction
    try:
        corrected_fa, corrected_mask = apply_correction(
            correction=correction,
            frame_path=frame_path,
            model_pose=model_pose,
            track_state=state,
            all_detections=all_dets,
        )
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

    # Save annotated frame (sólo si la bandera de anotados está ON)
    if opt_flags.write_annotated():
        annotated_path = str(out_dir / "annotated" / f"annotated_{req.frame_idx:06d}.jpg")
        annotate_frame(frame_path, corrected_fa, annotated_path)

    # Build optional SOT backend for forward propagation
    sot = None
    if req.sot_backend in ("csrt", "sam2"):
        try:
            sot = create_sot(req.sot_backend)
        except (RuntimeError, FileNotFoundError) as sot_err:
            raise HTTPException(
                status_code=422,
                detail=f"SOT backend '{req.sot_backend}' not available: {sot_err}",
            )
        except Exception as sot_err:
            raise HTTPException(
                status_code=500,
                detail=f"SOT backend error: {sot_err}",
            )

    # mask_correction with ByteTrack: apply to this frame only, no propagation
    effective_radius = req.propagation_radius
    if correction_type == "mask_correction" and req.sot_backend == "none":
        effective_radius = 0

    # Propagate to adjacent frames
    info = get_video_info(req.video_path)
    updated_fas = propagate_correction(
        corrected_frame_idx=req.frame_idx,
        corrected_fa=corrected_fa,
        video_path=req.video_path,
        fps=info["fps"],
        model_seg=model_seg,
        model_pose=model_pose,
        track_state=state,
        radius=effective_radius,
        frames_dir=frames_dir,
        sot=sot,
        init_mask=corrected_mask,
        end_frame=req.propagation_end_frame,
    )

    # Re-annotate updated frames (usa read_frame_bgr → funciona en modo lean,
    # decodificando del video si el JPEG del frame no está en disco).
    if opt_flags.write_annotated():
        for fa in updated_fas:
            ap = str(out_dir / "annotated" / f"annotated_{fa.frame_idx:06d}.jpg")
            img = read_frame_bgr(
                out_dir.name, fa.frame_idx, out_dir.parent, video_path=req.video_path,
            )
            if img is not None:
                annotated_img = annotate_frame_array(img.copy(), fa)
                Path(ap).parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(ap, annotated_img)

    # Serialize to JSON-safe dicts
    def _fa_to_dict(fa: FrameAnalysis) -> dict:
        d = frame_analysis_to_dict(fa)
        d["annotated_image"] = f"/frame/{video_name}/{fa.frame_idx}/annotated"
        return d

    corrected_dict = _fa_to_dict(corrected_fa)
    updated_dicts  = [_fa_to_dict(fa) for fa in updated_fas]

    # Patch analysis.json with updated frames (same output dir as project)
    analysis = _load_analysis_dir(out_dir)
    saved_count = 0
    if analysis and "frames" in analysis:
        idx_map = {int(f["frame_idx"]): i for i, f in enumerate(analysis["frames"])}
        for d in [corrected_dict] + updated_dicts:
            i = idx_map.get(int(d["frame_idx"]))
            if i is not None:
                analysis["frames"][i] = _merge_frame_correction(analysis["frames"][i], d)
                saved_count += 1
        if saved_count:
            derived = int(analysis.get("derived_version", 0)) + 1
            _save_analysis_dir(out_dir, analysis["frames"], extra={"derived_version": derived})
            _append_correction_log(out_dir, {
                "frame_idx": req.frame_idx,
                "correction_type": correction_type,
                "frames_updated": saved_count,
                "propagation_radius": effective_radius,
                "sot_backend": req.sot_backend,
            })
        else:
            print(f"  [Correction] WARNING: frame {req.frame_idx} not found in "
                  f"{out_dir / 'analysis.json'} — correction not persisted")
    else:
        print(f"  [Correction] WARNING: no analysis.json in {out_dir} — correction not persisted")

    pose_warning = None
    if not corrected_fa.keypoints_valid_count:
        pose_warning = (
            "No se detecto pose en la region seleccionada. "
            "Se guardo el bbox/mascara; prueba un area mas grande o usa bbox/click."
        )

    return CorrectionResponse(
        corrected_frame=corrected_dict,
        updated_frames=updated_dicts,
        total_affected=1 + len(updated_dicts),
        pose_warning=pose_warning,
    )


def _build_fa_from_json(frame_data: dict) -> "FrameAnalysis":
    """Reconstruct a FrameAnalysis from a stored analysis.json frame entry."""
    return frame_record_to_analysis(frame_data)


def _jpeg_response(img: "np.ndarray"):
    """Codificar una imagen BGR a JPEG en memoria y devolverla como Response."""
    from fastapi.responses import Response as RawResponse
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, opt_flags.jpeg_quality()])
    if not ok:
        raise HTTPException(500, "Failed to encode frame")
    return RawResponse(content=buf.tobytes(), media_type="image/jpeg")


@app.get("/frame/{video_name}/{frame_idx}")
def get_frame(video_name: str, frame_idx: int, annotated: bool = True):
    """
    Serve a frame image. If annotated=True and the pre-generated annotated
    image doesn't exist, regenerate it (in memory, or on disk when
    TJ_WRITE_ANNOTATED is on) from analysis.json data. The raw frame is
    resolved via read_frame_bgr, so it works even without frames/*.jpg on disk
    (decoding from the source video as a fallback).
    """
    from fastapi.responses import Response as RawResponse

    if annotated:
        ann_path = OUTPUT_ROOT / video_name / "annotated" / f"annotated_{frame_idx:06d}.jpg"
        # 1) Pre-generado en disco → servir tal cual (compatibilidad, sin cambios)
        if ann_path.exists():
            return FileResponse(str(ann_path), media_type="image/jpeg")

        # 2) LRU opcional de bytes anotados (default OFF)
        use_cache = opt_flags.annotated_cache()
        cache_key = (video_name, frame_idx, _analysis_mtime(video_name))
        if use_cache:
            cached = _annotated_cache_get(cache_key)
            if cached is not None:
                return RawResponse(content=cached, media_type="image/jpeg")

        # 3) Regenerar: frame crudo (disco o video) + datos de analysis.json
        img = read_frame_bgr(video_name, frame_idx, OUTPUT_ROOT)
        analysis = _load_analysis(video_name)
        if img is not None and analysis and "frames" in analysis:
            frame_data = next(
                (f for f in analysis["frames"] if f.get("frame_idx") == frame_idx),
                None,
            )
            if frame_data:
                try:
                    fa = _build_fa_from_json(frame_data)
                    annotated_img = annotate_frame_array(img.copy(), fa)
                    if opt_flags.write_annotated():
                        # Paridad con hoy: cachear en disco cuando la bandera está ON
                        ann_path.parent.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(ann_path), annotated_img)
                        return FileResponse(str(ann_path), media_type="image/jpeg")
                    # Modo lean: servir bytes sin escribir a disco
                    ok, buf = cv2.imencode(
                        ".jpg", annotated_img,
                        [cv2.IMWRITE_JPEG_QUALITY, opt_flags.jpeg_quality()],
                    )
                    if ok:
                        data = buf.tobytes()
                        if use_cache:
                            _annotated_cache_put(cache_key, data)
                        return RawResponse(content=data, media_type="image/jpeg")
                except Exception:
                    pass  # cae al frame crudo

        # 4) Fallback final: frame crudo (disco tal cual, o decodificado del video)
        raw_path = OUTPUT_ROOT / video_name / "frames" / f"frame_{frame_idx:06d}.jpg"
        if raw_path.exists():
            return FileResponse(str(raw_path), media_type="image/jpeg")
        if img is not None:
            return _jpeg_response(img)
        raise HTTPException(404, f"Frame {frame_idx} not found for {video_name}")

    # raw
    raw_path = OUTPUT_ROOT / video_name / "frames" / f"frame_{frame_idx:06d}.jpg"
    if raw_path.exists():
        return FileResponse(str(raw_path), media_type="image/jpeg")
    img = read_frame_bgr(video_name, frame_idx, OUTPUT_ROOT)
    if img is None:
        raise HTTPException(404, f"Frame {frame_idx} not found for {video_name}")
    return _jpeg_response(img)


@app.get("/frame/{video_name}/{frame_idx}/annotated")
def get_annotated_frame(video_name: str, frame_idx: int):
    return get_frame(video_name, frame_idx, annotated=True)


@app.get("/mask/{video_name}/{frame_idx}")
def get_frame_detections(video_name: str, frame_idx: int, video_path: str):
    """
    Return all person detections in a frame (for click-selection UI).
    The UI uses this to show bounding boxes and let the user click one.
    """
    _, model_seg = _get_models()
    frame = read_frame_bgr(video_name, frame_idx, OUTPUT_ROOT, video_path=video_path)
    if frame is None:
        raise HTTPException(404, f"Frame {frame_idx} not found")

    dets = detections_for_frame(frame, model_seg)

    return {
        "frame_idx":  frame_idx,
        "detections": [
            {
                "detection_idx": i,
                "bbox":  d["bbox"],
                "conf":  round(d["conf"], 3),
                "track_id": d["track_id"],
            }
            for i, d in enumerate(dets)
        ]
    }


if __name__ == "__main__":
    print("Starting Triple Jump Analyzer API ...")
    print("Docs: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
