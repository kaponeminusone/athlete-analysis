"""
Second-pass reanalysis: appearance-first athlete selection.

Problem the first pass has:
  ByteTrack assigns IDs frame-by-frame. When the athlete is far away,
  partially occluded, or the camera angle changes, ByteTrack can lock
  onto the wrong person and the AppearanceModel never corrects it in time.

What this module does differently:
  1. Seed phase — load the best N frames from the first-pass analysis.json
     (high quality, lateral/semi-back angle). Re-run YOLO seg on those
     frames to extract appearance features (no tracking needed yet).
     Build a calibrated AppearanceModel from these ground-truth samples.

  2. Second-pass — stream the full video again. For EVERY frame:
     a. Run YOLO seg (plain inference, no ByteTrack `.track()`)
     b. Among all detected persons, pick the one whose HSV histogram is
        most similar to the calibrated AppearanceModel.
     c. Run YOLO pose on a tight crop of that detection.
     d. Annotate and save.

  Output goes to  output/<video_name>_refined/
  so it never overwrites the first-pass results.

  If the refined output is better, you can use it; if not, the original
  output is untouched.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .athlete_tracker import AppearanceModel, _padded_crop, MIN_BBOX_AREA
from .frame_extractor import get_video_info
from .job_store import ProgressCallback, noop_progress
from .pose_analyzer import (
    CameraAngle,
    FrameAnalysis,
    analyze_frame_from_tracker,
)
from .visualizer import annotate_frame

SEED_MIN_QUALITY = 0.65
SEED_MAX_FRAMES  = 15       # how many best frames to use for seeding
MIN_SIMILARITY   = 0.30     # Bhattacharyya-based; 0 = identical, lower = better match
                             # we keep detections where similarity() >= MIN_SIMILARITY


@dataclass
class ReanalysisConfig:
    video_path:          str
    original_output_dir: str     # where first-pass analysis.json lives
    output_dir:          str     # where refined results go
    stride:              int   = 1
    start_sec:           float = 0.0
    end_sec:             Optional[float] = None
    seed_frames:         int   = SEED_MAX_FRAMES
    annotate_every:      int   = 1
    seed_start_frame:    Optional[int] = None   # restrict seed to this interval
    seed_end_frame:      Optional[int] = None   # (both must be set to take effect)


# ─── Seed phase ───────────────────────────────────────────────────────────────

def _seed_appearance(
    original_output_dir: str,
    model_seg,
    max_frames: int = SEED_MAX_FRAMES,
    on_progress: ProgressCallback = noop_progress,
    seed_start_frame: Optional[int] = None,
    seed_end_frame:   Optional[int] = None,
) -> Optional[AppearanceModel]:
    """
    Build an AppearanceModel from the N highest-quality frames of the
    first-pass analysis.  Re-runs YOLO seg (no tracking) to get bboxes.

    If seed_start_frame and seed_end_frame are both set, only frames
    within [start, end] are considered for seeding — useful when the
    user knows a specific interval where detection was clean.
    Frames outside that interval are still used as fallback if the
    interval yields fewer than max_frames candidates.
    """
    analysis_path = Path(original_output_dir) / "analysis.json"
    if not analysis_path.exists():
        return None

    with open(analysis_path) as f:
        data = json.load(f)

    frames_data = data.get("frames", [])
    use_interval = (seed_start_frame is not None and seed_end_frame is not None
                    and seed_end_frame > seed_start_frame)

    def _is_good(fr) -> bool:
        return (fr.get("person_detected")
                and fr.get("quality_score", 0) >= SEED_MIN_QUALITY
                and fr.get("camera_angle") in ("LATERAL", "SEMI_BACK"))

    def _in_interval(fr) -> bool:
        return seed_start_frame <= fr.get("frame_idx", -1) <= seed_end_frame

    if use_interval:
        # Priority 1: good frames inside the user-defined interval
        interval_good = sorted(
            [fr for fr in frames_data if _is_good(fr) and _in_interval(fr)],
            key=lambda x: x["quality_score"], reverse=True,
        )
        # Priority 2: any detected frame inside the interval
        interval_any = sorted(
            [fr for fr in frames_data if fr.get("person_detected") and _in_interval(fr)
             and fr not in interval_good],
            key=lambda x: x.get("quality_score", 0), reverse=True,
        )
        combined = (interval_good + interval_any)[:max_frames]
        # If the interval is thin, pad with the best frames from the whole video
        if len(combined) < max_frames:
            rest = sorted(
                [fr for fr in frames_data if _is_good(fr) and fr not in combined],
                key=lambda x: x["quality_score"], reverse=True,
            )
            combined = (combined + rest)[:max_frames]
        seed_candidates = combined
        print(f"  [Reanalyzer] Seeding from interval [{seed_start_frame}–{seed_end_frame}]: "
              f"{len(interval_good)} good + {len(interval_any)} any → {len(seed_candidates)} total")
    else:
        usable = sorted(
            [fr for fr in frames_data if _is_good(fr)],
            key=lambda x: x["quality_score"], reverse=True,
        )
        seed_candidates = usable[:max_frames]

    if not seed_candidates:
        # last-resort fallback
        seed_candidates = sorted(
            [fr for fr in frames_data if fr.get("person_detected")],
            key=lambda x: x.get("quality_score", 0),
            reverse=True,
        )[:max_frames]

    if not seed_candidates:
        return None

    frames_dir = Path(original_output_dir) / "frames"
    model = AppearanceModel()
    seeded = 0

    for i, fr in enumerate(seed_candidates):
        fidx  = fr["frame_idx"]
        fpath = frames_dir / f"frame_{fidx:06d}.jpg"
        if not fpath.exists():
            continue

        img = cv2.imread(str(fpath))
        if img is None:
            continue

        # Run seg (plain, no track) to get per-person bboxes
        results = model_seg(img, classes=[0], conf=0.35, verbose=False)
        if not results or results[0].boxes is None:
            continue

        res    = results[0]
        bboxes = res.boxes.xyxy.cpu().numpy()
        areas  = [(b[2]-b[0])*(b[3]-b[1]) for b in bboxes]
        valid  = [i for i, a in enumerate(areas) if a >= MIN_BBOX_AREA]
        if not valid:
            continue

        # pick largest detection (most likely the athlete in a seed frame)
        best = max(valid, key=lambda i: areas[i])
        bbox = tuple(int(v) for v in bboxes[best])

        seg_mask = None
        if res.masks is not None and best < len(res.masks.data):
            mt = res.masks.data[best].cpu().numpy()
            seg_mask = cv2.resize(
                mt, (img.shape[1], img.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

        model.update(img, mask=seg_mask, bbox=bbox)
        seeded += 1

        on_progress({
            "stage":   "seeding",
            "message": f"Semilla {seeded}/{len(seed_candidates)} — frame {fidx} Q={fr['quality_score']:.2f}",
            "percent": 10.0 + 10.0 * (seeded / len(seed_candidates)),
        })

    if seeded == 0:
        return None

    print(f"  [Reanalyzer] AppearanceModel seeded from {seeded} frames")
    return model


# ─── Second-pass frame analysis ───────────────────────────────────────────────

def _analyze_frame_refined(
    image: np.ndarray,
    frame_idx: int,
    timestamp_s: float,
    model_seg,
    model_pose,
    appearance: AppearanceModel,
) -> FrameAnalysis:
    """
    Run seg (no tracking) → pick best appearance match → pose on crop.
    Returns a FrameAnalysis with tracking_source = "refined".
    """
    empty = FrameAnalysis(frame_idx=frame_idx, timestamp_s=timestamp_s)
    empty.tracking_source = "refined"

    results = model_seg(image, classes=[0], conf=0.30, verbose=False)
    if not results or results[0].boxes is None:
        return empty

    res    = results[0]
    bboxes = res.boxes.xyxy.cpu().numpy()
    confs  = res.boxes.conf.cpu().numpy()
    areas  = [(b[2]-b[0])*(b[3]-b[1]) for b in bboxes]
    valid  = [i for i, a in enumerate(areas) if a >= MIN_BBOX_AREA]
    if not valid:
        return empty

    # Build per-detection masks
    h, w = image.shape[:2]
    det_masks = []
    for i in range(len(bboxes)):
        if res.masks is not None and i < len(res.masks.data):
            mt = res.masks.data[i].cpu().numpy()
            m  = cv2.resize(mt, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
            det_masks.append(m)
        else:
            det_masks.append(None)

    # Score each detection by appearance similarity
    scores = []
    for i in valid:
        bbox  = tuple(int(v) for v in bboxes[i])
        sim   = appearance.similarity(image, mask=det_masks[i], bbox=bbox)
        scores.append((i, sim))

    # Pick highest similarity
    best_i, best_sim = max(scores, key=lambda x: x[1])

    if best_sim < MIN_SIMILARITY:
        return empty

    x1, y1, x2, y2 = (int(v) for v in bboxes[best_i])
    seg_mask  = det_masks[best_i]
    mask_area = 0

    if seg_mask is not None:
        mask_area = int(seg_mask.sum())
        ys, xs = np.where(seg_mask)
        if len(xs) > 0:
            x1, x2 = int(xs.min()), int(xs.max())
            y1, y2 = int(ys.min()), int(ys.max())

    # Pose on tight crop
    kps_xy_full = kps_conf = None
    quality = 0.0

    crop, (ox, oy, _) = _padded_crop(image, x1, y1, x2, y2)
    if crop.size > 0:
        pose_results = model_pose(crop, verbose=False, conf=0.25)
        if (pose_results and pose_results[0].keypoints is not None
                and len(pose_results[0].keypoints.xy) > 0):
            kp_data = pose_results[0].keypoints

            # pick largest detection in crop
            crop_boxes = pose_results[0].boxes
            best_crop  = 0
            if crop_boxes is not None and len(crop_boxes) > 1:
                crop_areas = [(b[2]-b[0])*(b[3]-b[1])
                              for b in crop_boxes.xyxy.cpu().numpy()]
                best_crop  = int(np.argmax(crop_areas))

            kps_xy_crop = kp_data.xy.cpu().numpy()[best_crop]
            kps_conf    = kp_data.conf.cpu().numpy()[best_crop]
            kps_xy_full = kps_xy_crop.copy()
            kps_xy_full[:, 0] += ox
            kps_xy_full[:, 1] += oy

            n_valid  = int((kps_conf >= 0.45).sum())
            det_conf = float(confs[best_i])
            quality  = round(
                0.45 * n_valid / 17.0
                + 0.30 * det_conf
                + 0.15 * min(1.0, mask_area / 30000.0)
                + 0.10 * best_sim,
                3,
            )

    # Update appearance model from high-confidence frames
    det_conf = float(confs[best_i])
    if det_conf >= 0.60 and mask_area > 8000:
        appearance.update(image, mask=seg_mask, bbox=(x1, y1, x2, y2))

    tracker_result = {
        "found":          True,
        "track_id":       None,
        "bbox":           (x1, y1, x2, y2),
        "mask_area_px":   mask_area,
        "crop_offset":    (0, 0),
        "kps_xy":         kps_xy_full,
        "kps_conf":       kps_conf,
        "seg_mask":       seg_mask,
        "quality_score":  quality,
        "appearance_sim": round(float(best_sim), 3),
    }

    fa = analyze_frame_from_tracker(
        frame_idx=frame_idx,
        timestamp_s=timestamp_s,
        tracker_result=tracker_result,
    )
    fa.tracking_source = "refined"
    return fa


# ─── Full second-pass pipeline ────────────────────────────────────────────────

def run_reanalysis(
    config: ReanalysisConfig,
    on_progress: ProgressCallback = noop_progress,
) -> dict:
    """
    Full second-pass pipeline.  Returns summary dict.
    Never touches the original output directory.
    """
    out_dir       = Path(config.output_dir)
    out_frames    = out_dir / "frames"
    out_annotated = out_dir / "annotated"
    out_charts    = out_dir / "charts"
    for d in [out_frames, out_annotated, out_charts]:
        d.mkdir(parents=True, exist_ok=True)

    # ── Load models ───────────────────────────────────────────────────────────
    on_progress({"stage": "loading_models", "message": "Cargando modelos YOLO", "percent": 2.0})
    from ultralytics import YOLO
    t0         = time.time()
    model_pose = YOLO("yolo11s-pose.pt")
    model_seg  = YOLO("yolo11s-seg.pt")
    on_progress({"stage": "loading_models", "message": f"Modelos listos ({time.time()-t0:.1f}s)", "percent": 8.0})

    # ── Seed AppearanceModel from first-pass best frames ──────────────────────
    on_progress({"stage": "seeding", "message": "Sembrando modelo de apariencia...", "percent": 10.0})
    appearance = _seed_appearance(
        config.original_output_dir, model_seg,
        max_frames=config.seed_frames,
        on_progress=on_progress,
        seed_start_frame=config.seed_start_frame,
        seed_end_frame=config.seed_end_frame,
    )
    if appearance is None or not appearance.is_ready:
        return {"error": "No se pudo sembrar el modelo de apariencia. Ejecuta el pipeline original primero."}

    on_progress({"stage": "seeding", "message": "Modelo de apariencia listo", "percent": 20.0})

    # ── Video metadata ────────────────────────────────────────────────────────
    info     = get_video_info(config.video_path)
    fps      = info["fps"]
    start_f  = int(config.start_sec * fps)
    end_f    = int(config.end_sec * fps) if config.end_sec else info["total_frames"]
    total_stream = end_f - start_f

    on_progress({
        "stage":        "reading_video",
        "message":      f"{info['width']}x{info['height']} @ {fps:.1f}fps · {total_stream} frames",
        "total_frames": total_stream,
        "percent":      22.0,
    })

    # ── Second pass ───────────────────────────────────────────────────────────
    analyses: list[FrameAnalysis] = []
    analysis_count  = 0
    annotated_count = 0
    frame_abs       = start_f

    cap = cv2.VideoCapture(config.video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)

    while frame_abs < end_f:
        ret, frame = cap.read()
        if not ret:
            break

        is_analysis_frame = (frame_abs - start_f) % config.stride == 0

        if is_analysis_frame:
            ts = frame_abs / fps
            fa = _analyze_frame_refined(
                image=frame,
                frame_idx=frame_abs,
                timestamp_s=ts,
                model_seg=model_seg,
                model_pose=model_pose,
                appearance=appearance,
            )
            analyses.append(fa)
            analysis_count += 1

            fpath = out_frames / f"frame_{frame_abs:06d}.jpg"
            cv2.imwrite(str(fpath), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])

            if analysis_count % 5 == 0 or frame_abs >= end_f - config.stride:
                streamed = frame_abs - start_f
                pct = 22.0 + 70.0 * (streamed / max(total_stream, 1))
                on_progress({
                    "stage":           "reanalyzing",
                    "current_frame":   frame_abs,
                    "total_frames":    total_stream,
                    "analyzed_frames": analysis_count,
                    "percent":         round(pct, 1),
                    "message":         f"Refinando frame {frame_abs}/{end_f} · {analysis_count} analizados",
                    "last_log": (
                        f"t={ts:.2f}s | {fa.camera_angle.value} | "
                        f"Q={fa.quality_score:.2f} | kps={fa.keypoints_valid_count}/11"
                    ),
                })

            if analysis_count % config.annotate_every == 0:
                out_img = out_annotated / f"annotated_{frame_abs:06d}.jpg"
                annotate_frame(str(fpath), fa, str(out_img),
                               seg_mask=None, appearance_sim=0.0)
                annotated_count += 1

        frame_abs += 1

    cap.release()

    # ── Summary + JSON ────────────────────────────────────────────────────────
    on_progress({"stage": "writing_outputs", "message": "Escribiendo resultados refinados", "percent": 93.0})

    summary = _summarize_refined(analyses, analysis_count)
    result_data = {
        "video":      config.video_path,
        "video_info": info,
        "pass":       "refined",
        "config": {
            "stride":    config.stride,
            "start_sec": config.start_sec,
            "end_sec":   config.end_sec,
        },
        "summary": summary,
        "frames": [
            {
                "frame_idx":           a.frame_idx,
                "timestamp_s":         round(a.timestamp_s, 3),
                "person_detected":     a.person_detected,
                "track_id":            a.track_id,
                "camera_angle":        a.camera_angle.value,
                "shoulder_ratio":      round(a.shoulder_ratio, 4),
                "angle_confidence":    round(a.angle_confidence, 4),
                "keypoints_valid":     a.keypoints_valid_count,
                "quality_score":       a.quality_score,
                "usable_for_analysis": a.usable_for_analysis,
                "has_mask":            a.has_mask,
                "mask_area_px":        a.mask_area_px,
                "tracking_source":     "refined",
            }
            for a in analyses
        ],
    }

    json_path = out_dir / "analysis.json"
    with open(json_path, "w") as f:
        json.dump(result_data, f, indent=2)

    on_progress({
        "stage":           "done",
        "message":         f"Refinado completo · {analysis_count} frames · {annotated_count} anotados",
        "analyzed_frames": analysis_count,
        "percent":         100.0,
    })

    return summary


def _summarize_refined(analyses: list[FrameAnalysis], total: int) -> dict:
    detected = [a for a in analyses if a.person_detected]
    if not detected:
        return {"error": "No person detected in refined pass"}

    angle_counts: dict[str, int] = {}
    for a in detected:
        k = a.camera_angle.value
        angle_counts[k] = angle_counts.get(k, 0) + 1

    total_det = len(detected)
    angle_pct = {k: round(v / total_det * 100, 1) for k, v in angle_counts.items()}
    usable    = [a for a in detected if a.usable_for_analysis]
    q_scores  = [a.quality_score for a in detected]
    kp_counts = [a.keypoints_valid_count for a in detected]

    return {
        "total_frames_analyzed":      len(analyses),
        "frames_with_person":         total_det,
        "detection_rate_pct":         round(total_det / max(len(analyses), 1) * 100, 1),
        "frames_usable_for_analysis": len(usable),
        "usable_rate_pct":            round(len(usable) / max(len(analyses), 1) * 100, 1),
        "camera_angle_distribution":  angle_pct,
        "dominant_angle":             max(angle_counts, key=angle_counts.get) if angle_counts else "UNKNOWN",
        "quality_score": {
            "mean": round(float(np.mean(q_scores)), 3),
            "min":  round(float(np.min(q_scores)),  3),
            "max":  round(float(np.max(q_scores)),  3),
        },
        "keypoints_valid_avg": round(float(np.mean(kp_counts)), 2) if kp_counts else 0,
        "lateral_frames_pct":  angle_pct.get("LATERAL", 0),
        "pass": "refined",
    }
