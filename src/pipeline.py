"""
Core analysis pipeline — callable from CLI (run.py) or API (api_server.py).

All progress is emitted through the `on_progress` callback so callers
can route it to stdout, a job store, a websocket, etc.
"""

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .frame_extractor import get_video_info
from .pose_analyzer   import FrameAnalysis, CameraAngle, analyze_frame, analyze_frame_from_tracker
from .athlete_tracker import TrackState, run_tracked_frame
from .visualizer      import annotate_frame, generate_timeline_chart
from .job_store       import ProgressCallback, noop_progress
from .schemas         import (
    build_analysis_document,
    frame_analysis_to_dict,
    write_derived_stubs,
)
from .track_scorer      import TrackScorerContext


@dataclass
class PipelineConfig:
    video_path:     str
    output_dir:     str
    stride:         int   = 3
    start_sec:      float = 0.0
    end_sec:        Optional[float] = None
    max_frames:     Optional[int]   = None
    use_seg:        bool  = True
    annotate_every: int   = 1   # 1 = every analysis frame (recommended)


def run_pipeline(
    config: PipelineConfig,
    on_progress: ProgressCallback = noop_progress,
) -> dict:
    """
    Run the full analysis pipeline on one video.

    Emits structured progress events via `on_progress` at each stage.
    Returns the summary dict on success; raises on failure.
    """

    out_frames    = Path(config.output_dir) / "frames"
    out_annotated = Path(config.output_dir) / "annotated"
    out_charts    = Path(config.output_dir) / "charts"
    for d in [out_frames, out_annotated, out_charts]:
        d.mkdir(parents=True, exist_ok=True)

    # ── Stage: loading_models ─────────────────────────────────────────────────
    on_progress({
        "stage":   "loading_models",
        "message": "Cargando YOLO pose y segmentacion",
        "percent": 2.0,
    })

    try:
        from ultralytics import YOLO
    except ImportError:
        raise RuntimeError("ultralytics not installed")

    t0 = time.time()
    model_pose = YOLO("yolo11s-pose.pt")
    on_progress({
        "stage":   "loading_models",
        "message": f"Pose model listo ({time.time()-t0:.1f}s)",
        "percent": 5.0,
    })

    model_seg = None
    if config.use_seg:
        t0 = time.time()
        model_seg = YOLO("yolo11s-seg.pt")
        on_progress({
            "stage":   "loading_models",
            "message": f"Seg model listo ({time.time()-t0:.1f}s)",
            "percent": 8.0,
        })

    # ── Stage: reading_video ──────────────────────────────────────────────────
    on_progress({
        "stage":   "reading_video",
        "message": f"Leyendo metadatos de {Path(config.video_path).name}",
        "percent": 9.0,
    })

    info       = get_video_info(config.video_path)
    fps        = info["fps"]
    start_f    = int(config.start_sec * fps)
    end_f      = int(config.end_sec * fps) if config.end_sec else info["total_frames"]
    total_stream = end_f - start_f

    on_progress({
        "stage":        "reading_video",
        "message":      f"{info['width']}x{info['height']} @ {fps:.1f}fps · {info['duration_s']:.1f}s · {total_stream} frames a streamear",
        "total_frames": total_stream,
        "percent":      10.0,
    })

    # ── Stage: tracking / analyzing_pose ──────────────────────────────────────
    use_tracker     = model_seg is not None
    track_state     = TrackState()
    output_path     = Path(config.output_dir)
    track_scorer    = TrackScorerContext.from_output_dir(
        output_path, info["width"], info["height"],
    )
    analyses: list[FrameAnalysis] = []
    tracker_outputs: list[dict]   = []
    annotated_count = 0
    analysis_count  = 0
    frame_abs       = start_f

    cap = cv2.VideoCapture(config.video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)

    while frame_abs < end_f:
        ret, frame = cap.read()
        if not ret:
            break

        ts = frame_abs / fps
        is_analysis_frame = (frame_abs - start_f) % config.stride == 0

        tracker_out = {}

        if use_tracker:
            pose_model_this_frame = model_pose if is_analysis_frame else None
            tracker_out = run_tracked_frame(
                image=frame,
                model_seg=model_seg,
                state=track_state,
                frame_idx=frame_abs,
                model_pose=pose_model_this_frame,
                track_scorer=track_scorer,
            )

            if is_analysis_frame:
                fa = analyze_frame_from_tracker(
                    frame_idx=frame_abs,
                    timestamp_s=ts,
                    tracker_result=tracker_out,
                )
                analyses.append(fa)
                tracker_outputs.append(tracker_out)

                fpath = out_frames / f"frame_{frame_abs:06d}.jpg"
                cv2.imwrite(str(fpath), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])

                analysis_count += 1

                # Progress event every 5 analysis frames
                if analysis_count % 5 == 0 or frame_abs >= end_f - config.stride:
                    streamed_so_far = frame_abs - start_f
                    pct = 10.0 + 80.0 * (streamed_so_far / max(total_stream, 1))
                    on_progress({
                        "stage":          "tracking",
                        "current_frame":  frame_abs,
                        "total_frames":   total_stream,
                        "analyzed_frames": analysis_count,
                        "percent":        round(pct, 1),
                        "message":        f"Trackeando frame {frame_abs}/{end_f} · pose en {analysis_count} frames",
                        "last_log": (
                            f"t={ts:.2f}s | {fa.camera_angle.value} | "
                            f"Q={fa.quality_score:.2f} | "
                            f"kps={fa.keypoints_valid_count}/11 | "
                            f"ID={fa.track_id}"
                        ),
                    })

                should_annotate = (analysis_count % config.annotate_every == 0)
                if should_annotate:
                    out_img = out_annotated / f"annotated_{frame_abs:06d}.jpg"
                    annotate_frame(
                        str(fpath), fa, str(out_img),
                        seg_mask=tracker_out.get("seg_mask"),
                        appearance_sim=tracker_out.get("appearance_sim", 0.0),
                    )
                    annotated_count += 1

                if config.max_frames and analysis_count >= config.max_frames:
                    break
        else:
            if is_analysis_frame:
                fa = analyze_frame(
                    frame_idx=frame_abs,
                    timestamp_s=ts,
                    image=frame,
                    model_pose=model_pose,
                )
                analyses.append(fa)
                tracker_outputs.append({})

                fpath = out_frames / f"frame_{frame_abs:06d}.jpg"
                cv2.imwrite(str(fpath), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                analysis_count += 1

                if analysis_count % 5 == 0:
                    pct = 10.0 + 80.0 * ((frame_abs - start_f) / max(total_stream, 1))
                    on_progress({
                        "stage":           "analyzing_pose",
                        "current_frame":   frame_abs,
                        "total_frames":    total_stream,
                        "analyzed_frames": analysis_count,
                        "percent":         round(pct, 1),
                        "message":         f"Pose en frame {frame_abs}/{end_f}",
                    })

                if should_annotate := (analysis_count % config.annotate_every == 0):
                    out_img = out_annotated / f"annotated_{frame_abs:06d}.jpg"
                    annotate_frame(str(fpath), fa, str(out_img))
                    annotated_count += 1

                if config.max_frames and analysis_count >= config.max_frames:
                    break

        frame_abs += 1

    cap.release()

    # ── Stage: writing_outputs ────────────────────────────────────────────────
    on_progress({
        "stage":   "writing_outputs",
        "message": "Calculando resumen y escribiendo resultados",
        "percent": 92.0,
    })

    summary = _summarize(analyses, analysis_count)
    output_path = Path(config.output_dir)

    frames_data = [
        frame_analysis_to_dict(
            a,
            appearance_sim=(
                tracker_outputs[i].get("appearance_sim", 0.0)
                if i < len(tracker_outputs) else 0.0
            ),
            extra={
                k: tracker_outputs[i].get(k)
                for k in ("track_overlap", "athlete_state", "position_s", "predicted_bbox")
                if i < len(tracker_outputs) and tracker_outputs[i].get(k) is not None
            } if i < len(tracker_outputs) else None,
        )
        for i, a in enumerate(analyses)
    ]

    result_data = build_analysis_document(
        video_path=config.video_path,
        video_info=info,
        config={
            "stride":      config.stride,
            "start_sec":   config.start_sec,
            "end_sec":     config.end_sec,
            "use_tracker": use_tracker,
        },
        summary=summary,
        frames=frames_data,
        output_dir=output_path,
    )

    write_derived_stubs(output_path)

    json_path = output_path / "analysis.json"
    with open(json_path, "w") as f:
        json.dump(result_data, f, indent=2)

    on_progress({
        "stage":   "writing_outputs",
        "message": "Generando grafica de timeline",
        "percent": 96.0,
    })

    chart_path = out_charts / "camera_angle_timeline.png"
    generate_timeline_chart(analyses, str(chart_path), fps=fps)

    on_progress({
        "stage":           "done",
        "message":         f"Analisis completo · {analysis_count} frames · {annotated_count} anotados",
        "analyzed_frames": analysis_count,
        "percent":         100.0,
    })

    return summary


def _summarize(analyses: list[FrameAnalysis], total_streamed: int) -> dict:
    detected  = [a for a in analyses if a.person_detected]
    if not detected:
        return {"error": "No person detected in any frame"}

    angle_counts = {}
    for a in detected:
        k = a.camera_angle.value
        angle_counts[k] = angle_counts.get(k, 0) + 1

    total_detected = len(detected)
    angle_pct = {k: round(v / total_detected * 100, 1) for k, v in angle_counts.items()}

    usable   = [a for a in detected if a.usable_for_analysis]
    q_scores = [a.quality_score for a in detected]
    ratios   = [a.shoulder_ratio for a in detected if a.shoulder_ratio > 0]
    kp_counts= [a.keypoints_valid_count for a in detected]
    track_ids= list({a.track_id for a in detected if a.track_id is not None})

    return {
        "total_frames_analyzed":      len(analyses),
        "frames_with_person":         total_detected,
        "detection_rate_pct":         round(total_detected / max(len(analyses), 1) * 100, 1),
        "frames_usable_for_analysis": len(usable),
        "usable_rate_pct":            round(len(usable) / max(len(analyses), 1) * 100, 1),
        "athlete_track_ids":          track_ids,
        "camera_angle_distribution":  angle_pct,
        "dominant_angle":             max(angle_counts, key=angle_counts.get) if angle_counts else "UNKNOWN",
        "quality_score": {
            "mean": round(float(np.mean(q_scores)), 3),
            "min":  round(float(np.min(q_scores)),  3),
            "max":  round(float(np.max(q_scores)),  3),
        },
        "shoulder_ratio": {
            "mean": round(float(np.mean(ratios)), 3) if ratios else 0,
            "min":  round(float(np.min(ratios)),  3) if ratios else 0,
            "max":  round(float(np.max(ratios)),  3) if ratios else 0,
            "std":  round(float(np.std(ratios)),  3) if ratios else 0,
        },
        "keypoints_valid_avg": round(float(np.mean(kp_counts)), 2) if kp_counts else 0,
        "lateral_frames_pct":  angle_pct.get("LATERAL", 0),
        "note": (
            f"{len(usable)} frames usables ({round(len(usable)/max(len(analyses),1)*100,1)}%). "
            "Usable = quality >= 0.55 AND angle LATERAL o SEMI_BACK."
        ),
    }
