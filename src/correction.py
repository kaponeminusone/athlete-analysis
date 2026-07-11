"""
Human-in-the-loop correction module.

Supports three correction types from the UI:
  bbox_correction   — user drew a rectangle around the athlete
  click_selection   — user clicked on the athlete; we pick the nearest detection
  mask_correction   — user painted a pixel mask (advanced, optional)

Flow for each correction:
  1. Re-run pose on the corrected region (high-quality crop).
  2. Update AppearanceModel with this frame as ground truth (high weight).
  3. Re-analyze the N frames before and after (propagation radius) using
     the updated appearance model — this is the key benefit.
  4. Return all updated FrameAnalysis objects so the UI can refresh.

The propagation re-runs only the athlete-selection step (seg + appearance
matching) per frame, not a full model reload — so it is fast (~80-150ms
per adjacent frame on GTX 1050).
"""

from __future__ import annotations
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Literal
from pathlib import Path

from .pose_analyzer   import FrameAnalysis, analyze_frame_from_tracker
from .athlete_tracker import TrackState, run_tracked_frame, _padded_crop
from .sot             import SotBackend


CorrectionType = Literal["bbox_correction", "click_selection", "mask_correction"]


@dataclass
class Correction:
    """
    A single manual correction from the UI.

    For bbox_correction:
        data = {"x1": int, "y1": int, "x2": int, "y2": int}
    For click_selection:
        data = {"x": int, "y": int}   (pixel coordinates of click)
    For mask_correction:
        data = {"mask": list[list[int]]}  (binary 2D array, H x W)
    """
    frame_idx:       int
    correction_type: CorrectionType
    data:            dict
    propagation_radius: int = 15   # re-analyze this many frames before and after


def _bbox_from_correction(correction: Correction,
                           frame_shape: tuple) -> tuple[int, int, int, int]:
    """Return (x1, y1, x2, y2) regardless of correction type."""
    h, w = frame_shape[:2]
    d = correction.data

    if correction.correction_type == "bbox_correction":
        x1 = max(0, int(d["x1"]));  y1 = max(0, int(d["y1"]))
        x2 = min(w, int(d["x2"]));  y2 = min(h, int(d["y2"]))
        return x1, y1, x2, y2

    if correction.correction_type == "click_selection":
        # The caller must pass detections; this returns the whole frame as bbox
        # and let pose find the person near the click. Handled separately.
        cx, cy = int(d["x"]), int(d["y"])
        pad = 200
        return max(0, cx-pad), max(0, cy-pad), min(w, cx+pad), min(h, cy+pad)

    if correction.correction_type == "mask_correction":
        mask = np.array(d["mask"], dtype=np.uint8)
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return 0, 0, w, h
        return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

    return 0, 0, w, h


def _mask_from_correction(correction: Correction,
                           frame_shape: tuple) -> Optional[np.ndarray]:
    """Return boolean mask [H,W] if mask data is available, else None."""
    if correction.correction_type != "mask_correction":
        return None
    h, w = frame_shape[:2]
    raw = np.array(correction.data["mask"], dtype=np.uint8)
    if raw.shape != (h, w):
        raw = cv2.resize(raw, (w, h), interpolation=cv2.INTER_NEAREST)
    return raw.astype(bool)


def apply_correction(
    correction: Correction,
    frame_path: str,
    model_pose,
    track_state: TrackState,
    all_detections: Optional[list[dict]] = None,   # from seg model on same frame
) -> tuple[FrameAnalysis, Optional[np.ndarray]]:  # (fa, seg_mask)
    """
    Re-compute pose estimation for a single frame using the manual correction.

    For click_selection: if all_detections is provided (list of {bbox, track_id,
    mask}) from the seg model, picks the detection whose bbox center is nearest
    the click point. Otherwise falls back to a padded crop around the click.

    Returns an updated FrameAnalysis with manually_corrected=True.
    """
    img = cv2.imread(frame_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read frame: {frame_path}")

    h, w = img.shape[:2]
    fps_estimate = 30.0  # used for timestamp; will be overridden by caller
    ts = correction.frame_idx / fps_estimate

    seg_mask = _mask_from_correction(correction, img.shape)

    if correction.correction_type == "mask_correction":
        if seg_mask is None or not seg_mask.any():
            raise ValueError(
                f"La mascara esta vacia en el frame {correction.frame_idx}. "
                "Pinta sobre el atleta antes de aplicar."
            )

    # ── Click selection: pick nearest YOLO detection ──────────────────────────
    if correction.correction_type == "click_selection" and all_detections:
        cx, cy = int(correction.data["x"]), int(correction.data["y"])
        best = None
        best_dist = float("inf")
        for det in all_detections:
            bx1, by1, bx2, by2 = det["bbox"]
            det_cx = (bx1 + bx2) / 2
            det_cy = (by1 + by2) / 2
            dist = ((det_cx - cx) ** 2 + (det_cy - cy) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best = det

        if best:
            # Lock the track state to this detection's ID
            track_state.force_lock(best["track_id"])
            x1, y1, x2, y2 = [int(v) for v in best["bbox"]]
            seg_mask = best.get("mask")
            print(f"  [Correction] Click → nearest detection ID={best['track_id']} "
                  f"(dist={best_dist:.0f}px)")

    # ── Build bbox from correction ────────────────────────────────────────────
    x1, y1, x2, y2 = _bbox_from_correction(correction, img.shape)

    # ── Tighten bbox to mask if available ─────────────────────────────────────
    if seg_mask is not None:
        ys, xs = np.where(seg_mask)
        if len(xs) > 0:
            x1, x2 = int(xs.min()), int(xs.max())
            y1, y2 = int(ys.min()), int(ys.max())

    # ── Crop and run pose ─────────────────────────────────────────────────────
    crop, (ox, oy, _) = _padded_crop(img, x1, y1, x2, y2, pad=0.15)
    if crop.size == 0:
        raise ValueError(
            f"La region corregida esta vacia en el frame {correction.frame_idx}. "
            "Pinta un area mas grande con el mask brush."
        )

    kps_xy_full = None
    kps_conf    = None
    pose_detected = False

    for conf in (0.25, 0.1):
        pose_results = model_pose(crop, verbose=False, conf=conf)
        if not pose_results or pose_results[0].keypoints is None or \
           len(pose_results[0].keypoints.xy) == 0:
            continue

        crop_boxes = pose_results[0].boxes
        if crop_boxes is not None and len(crop_boxes) > 1:
            crop_areas = [(b[2]-b[0])*(b[3]-b[1])
                          for b in crop_boxes.xyxy.cpu().numpy()]
            best_idx = int(np.argmax(crop_areas))
        else:
            best_idx = 0

        kps_xy_crop = pose_results[0].keypoints.xy.cpu().numpy()[best_idx]
        kps_conf    = pose_results[0].keypoints.conf.cpu().numpy()[best_idx]
        kps_xy_full = kps_xy_crop.copy()
        kps_xy_full[:, 0] += ox
        kps_xy_full[:, 1] += oy
        pose_detected = True
        break

    mask_area = int(seg_mask.sum()) if seg_mask is not None else 0
    if not pose_detected:
        print(f"  [Correction] WARNING: no pose in crop for frame {correction.frame_idx} "
              f"— saving bbox/mask only (area={mask_area}px)")

    quality = 1.0 if pose_detected else round(
        max(0.2, min(0.45, 0.2 + mask_area / 60000.0)), 3
    )

    tracker_result = {
        "found":          True,
        "track_id":       track_state.athlete_track_id,
        "bbox":           (x1, y1, x2, y2),
        "mask_area_px":   mask_area,
        "crop_offset":    (ox, oy),
        "kps_xy":         kps_xy_full,
        "kps_conf":       kps_conf,
        "seg_mask":       seg_mask,
        "quality_score":  quality,
        "appearance_sim": 1.0,
    }

    fa = analyze_frame_from_tracker(
        frame_idx=correction.frame_idx,
        timestamp_s=ts,
        tracker_result=tracker_result,
    )
    fa.manually_corrected = True
    fa.correction_source  = correction.correction_type

    # ── Update appearance model with this as ground truth ────────────────────
    track_state.appearance.update(img, mask=seg_mask,
                                  bbox=(x1, y1, x2, y2))
    # Extra weight: update 3 times so it dominates the rolling average
    track_state.appearance.update(img, mask=seg_mask, bbox=(x1, y1, x2, y2))
    track_state.appearance.update(img, mask=seg_mask, bbox=(x1, y1, x2, y2))

    print(f"  [Correction] Frame {correction.frame_idx} corrected via "
          f"{correction.correction_type} — kps valid: "
          f"{fa.keypoints_valid_count}/11, Q={fa.quality_score:.2f}"
          f"{'' if fa.keypoints_valid_count else ' (bbox/mask only)'}")

    return fa, seg_mask


def propagate_correction(
    corrected_frame_idx: int,
    corrected_fa: FrameAnalysis,
    video_path: str,
    fps: float,
    model_seg,
    model_pose,
    track_state: TrackState,
    radius: int = 15,
    frames_dir: Optional[str] = None,
    sot: Optional[SotBackend] = None,
    init_mask: Optional[np.ndarray] = None,     # seg mask of the corrected frame
    end_frame: Optional[int] = None,            # override forward propagation end
) -> list[FrameAnalysis]:
    """
    Re-analyze frames [frame_idx - radius .. frame_idx + radius] using
    the updated appearance model (post-correction).

    Strategy:
      - Backward pass (frame_idx-1 .. frame_idx-radius): appearance model
        is already updated; re-run tracking selection on saved frames.
        Always uses ByteTrack — SOT is forward-only.
      - Forward pass (frame_idx+1 .. frame_idx+radius):
        * If sot is provided: initialize SOT with corrected frame/bbox,
          then call sot.update() per frame. Pose runs on SOT bbox.
        * If sot is None (default): re-run the full ByteTrack tracker.

    Returns a list of updated FrameAnalysis objects for affected frames,
    NOT including the corrected frame itself (it is already handled).
    """
    updated: list[FrameAnalysis] = []
    start_f = max(0, corrected_frame_idx - radius)
    end_f   = corrected_frame_idx + radius

    use_sot = sot is not None and corrected_fa.person_bbox is not None
    # Allow caller to set a specific end frame for the forward pass
    if end_frame is not None:
        end_f = min(end_frame, end_f)
    print(f"  [Propagation] Re-analyzing frames {start_f}–{end_f} "
          f"(radius={radius}, backend={'bytetrack' if not use_sot else sot.tracking_source}) ...")

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    end_f = min(end_f, total_frames - 1)

    # ── Backward pass: always ByteTrack + appearance ───────────────────────────
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    frame_idx = start_f
    while frame_idx < corrected_frame_idx:
        ret, frame = cap.read()
        if not ret:
            break
        ts = frame_idx / fps
        tracker_out = run_tracked_frame(
            image=frame, model_seg=model_seg,
            state=track_state, frame_idx=frame_idx, model_pose=model_pose,
        )
        fa = analyze_frame_from_tracker(frame_idx=frame_idx, timestamp_s=ts,
                                        tracker_result=tracker_out)
        if frames_dir:
            fpath = str(Path(frames_dir) / f"frame_{frame_idx:06d}.jpg")
            cv2.imwrite(fpath, frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        updated.append(fa)
        frame_idx += 1

    # ── Forward pass ───────────────────────────────────────────────────────────
    cap.set(cv2.CAP_PROP_POS_FRAMES, corrected_frame_idx)
    ret, init_frame = cap.read()   # read corrected frame (to init SOT)

    if use_sot and ret:
        sot.initialize(init_frame, corrected_fa.person_bbox, mask=init_mask)

    frame_idx = corrected_frame_idx + 1
    while frame_idx <= end_f:
        ret, frame = cap.read()
        if not ret:
            break

        ts = frame_idx / fps

        if use_sot:
            fa = _sot_frame(sot, frame, frame_idx, ts, model_pose, track_state)
        else:
            tracker_out = run_tracked_frame(
                image=frame, model_seg=model_seg,
                state=track_state, frame_idx=frame_idx, model_pose=model_pose,
            )
            fa = analyze_frame_from_tracker(frame_idx=frame_idx, timestamp_s=ts,
                                            tracker_result=tracker_out)

        if frames_dir:
            fpath = str(Path(frames_dir) / f"frame_{frame_idx:06d}.jpg")
            cv2.imwrite(fpath, frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        updated.append(fa)
        frame_idx += 1

    cap.release()
    print(f"  [Propagation] Done — {len(updated)} frames updated")
    return updated


def _sot_frame(
    sot: SotBackend,
    frame: np.ndarray,
    frame_idx: int,
    timestamp_s: float,
    model_pose,
    track_state: TrackState,
) -> FrameAnalysis:
    """
    Run one SOT update and pose estimation, returning a FrameAnalysis.
    Shared by propagate_correction (forward pass) and any direct SOT calls.
    """
    ok, bbox = sot.update(frame, frame_idx)
    seg_mask  = sot.get_mask(frame_idx)
    mask_area = int(seg_mask.sum()) if seg_mask is not None else 0

    if not ok:
        fa = FrameAnalysis(frame_idx=frame_idx, timestamp_s=timestamp_s)
        fa.tracking_source = sot.tracking_source
        return fa

    x1, y1, x2, y2 = bbox

    # tighten to mask if available
    if seg_mask is not None:
        ys, xs = np.where(seg_mask)
        if len(xs) > 0:
            x1, x2 = int(xs.min()), int(xs.max())
            y1, y2 = int(ys.min()), int(ys.max())

    kps_xy_full = kps_conf = None
    quality = 0.0

    if model_pose is not None:
        crop, (ox, oy, _) = _padded_crop(frame, x1, y1, x2, y2)
        if crop.size > 0:
            pose_results = model_pose(crop, verbose=False, conf=0.25)
            if (pose_results and pose_results[0].keypoints is not None
                    and len(pose_results[0].keypoints.xy) > 0):
                kp_data  = pose_results[0].keypoints
                crop_boxes = pose_results[0].boxes
                best = 0
                if crop_boxes is not None and len(crop_boxes) > 1:
                    crop_areas = [(b[2]-b[0])*(b[3]-b[1])
                                  for b in crop_boxes.xyxy.cpu().numpy()]
                    best = int(np.argmax(crop_areas))
                kps_xy_crop = kp_data.xy.cpu().numpy()[best]
                kps_conf    = kp_data.conf.cpu().numpy()[best]
                kps_xy_full = kps_xy_crop.copy()
                kps_xy_full[:, 0] += ox
                kps_xy_full[:, 1] += oy
                n_valid = int((kps_conf >= 0.45).sum())
                quality = round(0.45 * n_valid / 17.0 + 0.55 * min(1.0, mask_area / 30000.0), 3)

    tracker_result = {
        "found":          True,
        "track_id":       track_state.athlete_track_id,
        "bbox":           (x1, y1, x2, y2),
        "mask_area_px":   mask_area,
        "crop_offset":    (0, 0),
        "kps_xy":         kps_xy_full,
        "kps_conf":       kps_conf,
        "seg_mask":       seg_mask,
        "quality_score":  quality,
        "appearance_sim": 1.0,
    }
    fa = analyze_frame_from_tracker(
        frame_idx=frame_idx,
        timestamp_s=timestamp_s,
        tracker_result=tracker_result,
    )
    fa.tracking_source = sot.tracking_source
    return fa


def detections_for_frame(
    frame: np.ndarray,
    model_seg,
) -> list[dict]:
    """
    Run seg model on a single frame and return all person detections.
    Used by click_selection to let the user pick which person is the athlete.

    Returns list of:
      {"track_id": int, "bbox": (x1,y1,x2,y2), "conf": float,
       "mask": np.ndarray bool [H,W] or None}
    """
    results = model_seg(frame, classes=[0], conf=0.3, verbose=False)
    if not results or results[0].boxes is None:
        return []

    res    = results[0]
    boxes  = res.boxes
    bboxes = boxes.xyxy.cpu().numpy()
    confs  = boxes.conf.cpu().numpy()
    h, w   = frame.shape[:2]

    detections = []
    for i, bbox in enumerate(bboxes):
        mask = None
        if res.masks is not None and i < len(res.masks.data):
            mt   = res.masks.data[i].cpu().numpy()
            mask = cv2.resize(mt, (w, h),
                              interpolation=cv2.INTER_NEAREST).astype(bool)
        detections.append({
            "track_id": i,     # no ByteTrack here, just index
            "bbox":     tuple(int(v) for v in bbox),
            "conf":     float(confs[i]),
            "mask":     mask,
        })

    return detections
