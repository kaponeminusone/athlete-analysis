"""
API Server — bridge entre la UI y el pipeline de analisis.

Endpoints:
  GET  /status                       → estado del servidor y modelos cargados
  GET  /analysis/{video_name}        → cargar analysis.json existente
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
from src.visualizer       import annotate_frame
from src.pipeline         import PipelineConfig, run_pipeline
from src.job_store        import create_job, get_job, list_jobs
from src.reanalyzer       import ReanalysisConfig, run_reanalysis

OUTPUT_ROOT = Path("output")
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


def _load_analysis(video_name: str) -> Optional[list[dict]]:
    path = OUTPUT_ROOT / video_name / "analysis.json"
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    return data


def _save_analysis(video_name: str, frames_data: list[dict], extra: dict = {}):
    out = OUTPUT_ROOT / video_name
    out.mkdir(parents=True, exist_ok=True)
    path = out / "analysis.json"
    existing = {}
    if path.exists():
        with open(path) as f:
            existing = json.load(f)
    existing["frames"] = frames_data
    existing.update(extra)
    with open(path, "w") as f:
        json.dump(existing, f, indent=2)


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


class CorrectionResponse(BaseModel):
    corrected_frame:  dict
    updated_frames:   list[dict]
    total_affected:   int


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


def _project_payload(video_path: str) -> dict:
    video_file = Path(video_path)
    video_name = _video_name(video_path)
    out_dir = OUTPUT_ROOT / video_name
    analysis_path = out_dir / "analysis.json"
    chart_path = out_dir / "charts" / "camera_angle_timeline.png"
    analysis_data = None
    if analysis_path.exists():
        with open(analysis_path, encoding="utf-8") as f:
            analysis_data = json.load(f)

    return {
        "video": {
            "path": str(video_file),
            "name": video_file.name or f"{video_name}.mp4",
            "video_name": video_name,
            "exists": video_file.exists(),
            "url": _media_url(video_file) if video_file.exists() else "",
        },
        "output": {"path": str(out_dir), "exists": out_dir.exists()},
        "analysis": {
            "path": str(analysis_path),
            "exists": analysis_path.exists(),
            "data": analysis_data,
            "frames": analysis_data.get("frames", []) if analysis_data else [],
        },
        "assets": {
            "annotated": _indexed_images(out_dir / "annotated", "annotated_"),
            "frames": _indexed_images(out_dir / "frames", "frame_"),
            "chart": _media_url(chart_path) if chart_path.exists() else "",
        },
    }


@app.get("/api/project")
def get_project(video_path: str):
    return JSONResponse(_project_payload(video_path))


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
    for path in Path(".").rglob("*"):
        if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        if any(part in ignored_parts for part in path.parts):
            continue
        resolved = path.resolve()
        video_name = _video_name(str(resolved))
        analysis_path = OUTPUT_ROOT / video_name / "analysis.json"
        videos.append({
            "name": path.name,
            "path": str(resolved),
            "video_name": video_name,
            "duration_s": _video_duration(resolved),
            "has_analysis": analysis_path.exists(),
            "url": _media_url(resolved),
        })
    videos.sort(key=lambda item: (not item["has_analysis"], item["name"].lower()))
    return {"videos": videos}


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
    video_path: str
    stride:     int   = 1
    start_sec:  float = 0.0
    end_sec:    Optional[float] = None
    seed_frames: int  = 15


@app.post("/api/reanalyze")
def reanalyze_video(req: ReanalyzeRequest):
    """
    Second-pass reanalysis: seeds AppearanceModel from first-pass best
    frames, then re-runs seg+pose on the full video using appearance-only
    athlete selection (no ByteTrack).  Results go to output/<name>_refined/.
    Requires a completed first-pass analysis.json.
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
    out_dir    = _output_dir(req.video_path)
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
    corrected_fa, corrected_mask = apply_correction(
        correction=correction,
        frame_path=frame_path,
        model_pose=model_pose,
        track_state=state,
        all_detections=all_dets,
    )

    # Save annotated frame
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
        radius=req.propagation_radius,
        frames_dir=frames_dir,
        sot=sot,
        init_mask=corrected_mask,
        end_frame=req.propagation_end_frame,
    )

    # Re-annotate updated frames
    for fa in updated_fas:
        fp = str(out_dir / "frames" / f"frame_{fa.frame_idx:06d}.jpg")
        ap = str(out_dir / "annotated" / f"annotated_{fa.frame_idx:06d}.jpg")
        if Path(fp).exists():
            annotate_frame(fp, fa, ap)

    # Serialize to JSON-safe dicts
    def _fa_to_dict(fa: FrameAnalysis) -> dict:
        return {
            "frame_idx":           fa.frame_idx,
            "timestamp_s":         round(fa.timestamp_s, 3),
            "person_detected":     fa.person_detected,
            "track_id":            fa.track_id,
            "camera_angle":        fa.camera_angle.value,
            "shoulder_ratio":      round(fa.shoulder_ratio, 4),
            "keypoints_valid":     fa.keypoints_valid_count,
            "quality_score":       fa.quality_score,
            "usable_for_analysis": fa.usable_for_analysis,
            "manually_corrected":  fa.manually_corrected,
            "correction_source":   fa.correction_source,
            "tracking_source":     fa.tracking_source,
            "has_mask":            fa.has_mask,
            "mask_area_px":        fa.mask_area_px,
            "annotated_image":     f"/frame/{video_name}/{fa.frame_idx}/annotated",
        }

    corrected_dict = _fa_to_dict(corrected_fa)
    updated_dicts  = [_fa_to_dict(fa) for fa in updated_fas]

    # Patch analysis.json with updated frames
    analysis = _load_analysis(video_name)
    if analysis and "frames" in analysis:
        idx_map = {f["frame_idx"]: i for i, f in enumerate(analysis["frames"])}
        for d in [corrected_dict] + updated_dicts:
            i = idx_map.get(d["frame_idx"])
            if i is not None:
                analysis["frames"][i].update(d)
        _save_analysis(video_name, analysis["frames"])

    return CorrectionResponse(
        corrected_frame=corrected_dict,
        updated_frames=updated_dicts,
        total_affected=1 + len(updated_dicts),
    )


def _build_fa_from_json(frame_data: dict) -> "FrameAnalysis":
    """Reconstruct a FrameAnalysis from a stored analysis.json frame entry."""
    from src.pose_analyzer import (
        FrameAnalysis, CameraAngle, KeypointData, CONF_THRESHOLD, KP,
        _estimate_camera_angle, _compute_quality,
    )
    fa = FrameAnalysis(
        frame_idx=frame_data.get("frame_idx", 0),
        timestamp_s=frame_data.get("timestamp_s", 0.0),
    )
    fa.person_detected     = frame_data.get("person_detected", False)
    fa.track_id            = frame_data.get("track_id")
    fa.has_mask            = frame_data.get("has_mask", False)
    fa.mask_area_px        = frame_data.get("mask_area_px", 0)
    fa.quality_score       = frame_data.get("quality_score", 0.0)
    fa.usable_for_analysis = frame_data.get("usable_for_analysis", False)
    fa.manually_corrected  = frame_data.get("manually_corrected", False)
    fa.correction_source   = frame_data.get("correction_source", "auto")
    fa.shoulder_ratio      = frame_data.get("shoulder_ratio", 0.0)
    fa.angle_confidence    = frame_data.get("angle_confidence", 0.0)
    fa.torso_height_px     = frame_data.get("torso_height_px", 0.0)
    fa.shoulder_width_px   = frame_data.get("shoulder_width_px", 0.0)
    fa.body_height_px      = frame_data.get("body_height_px", 0.0)

    try:
        fa.camera_angle = CameraAngle(frame_data.get("camera_angle", "UNKNOWN"))
    except ValueError:
        fa.camera_angle = CameraAngle.UNKNOWN

    # bbox from stored data (not always present)
    if frame_data.get("person_bbox"):
        fa.person_bbox = tuple(frame_data["person_bbox"])

    fa.keypoints_valid_count = frame_data.get("keypoints_valid", 0)
    return fa


@app.get("/frame/{video_name}/{frame_idx}")
def get_frame(video_name: str, frame_idx: int, annotated: bool = True):
    """
    Serve a frame image. If annotated=True and the pre-generated annotated
    image doesn't exist, generate it on-demand from analysis.json data.
    """
    if annotated:
        ann_path = OUTPUT_ROOT / video_name / "annotated" / f"annotated_{frame_idx:06d}.jpg"
        if ann_path.exists():
            return FileResponse(str(ann_path), media_type="image/jpeg")

        # On-demand annotation: rebuild from analysis.json
        raw_path = OUTPUT_ROOT / video_name / "frames" / f"frame_{frame_idx:06d}.jpg"
        if raw_path.exists():
            analysis = _load_analysis(video_name)
            if analysis and "frames" in analysis:
                frame_data = next(
                    (f for f in analysis["frames"] if f.get("frame_idx") == frame_idx),
                    None,
                )
                if frame_data:
                    try:
                        fa = _build_fa_from_json(frame_data)
                        ann_path.parent.mkdir(parents=True, exist_ok=True)
                        annotate_frame(str(raw_path), fa, str(ann_path))
                        return FileResponse(str(ann_path), media_type="image/jpeg")
                    except Exception:
                        pass  # fall through to raw frame

        # Final fallback: raw frame
        if raw_path.exists():
            return FileResponse(str(raw_path), media_type="image/jpeg")
        raise HTTPException(404, f"Frame {frame_idx} not found for {video_name}")

    # raw
    raw_path = OUTPUT_ROOT / video_name / "frames" / f"frame_{frame_idx:06d}.jpg"
    if not raw_path.exists():
        raise HTTPException(404, f"Frame {frame_idx} not found for {video_name}")
    return FileResponse(str(raw_path), media_type="image/jpeg")


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
    frame_path = OUTPUT_ROOT / video_name / "frames" / f"frame_{frame_idx:06d}.jpg"
    if not frame_path.exists():
        raise HTTPException(404, f"Frame {frame_idx} not found")

    frame = cv2.imread(str(frame_path))
    dets  = detections_for_frame(frame, model_seg)

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
