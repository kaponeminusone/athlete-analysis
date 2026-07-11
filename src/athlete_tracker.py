"""
Athlete tracker: uses YOLO11 segmentation + ByteTrack + appearance model
to lock onto the correct athlete across all frames.

Identification pipeline (3 layers, in priority order):
  1. ByteTrack ID continuity — if the locked ID is still present, use it.
  2. Appearance similarity — HSV color histogram of the athlete's torso,
     built from the first high-confidence detections. When ByteTrack
     loses the ID, appearance picks the most visually similar person.
  3. Motion heuristic — during the initial calibration window, the person
     with the most horizontal displacement is the sprinting athlete.

Appearance model detail:
  - Extract the pixel region inside the segmentation mask, cropped to
    the torso zone (between shoulders-y and hips-y, or 30-70% of bbox height).
  - Compute a 3D HSV histogram (8x8x8 bins) normalized to [0,1].
  - Similarity = OpenCV Bhattacharyya distance (lower = more similar).
  - Reference histogram is updated as a running average from the 10 most
    confident frames (quality >= 0.7) to handle lighting changes.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from .track_scorer import TrackScorerContext


CALIBRATION_FRAMES   = 12    # frames to identify athlete by motion
PADDING_RATIO        = 0.20  # bbox padding for pose crop
MIN_BBOX_AREA        = 4000  # px² minimum detection size
APPEARANCE_THRESHOLD = 0.55  # Bhattacharyya distance above which we reject match
MAX_REF_SAMPLES      = 10    # rolling window for reference histogram


class AppearanceModel:
    """
    Builds and maintains a reference HSV color histogram for the athlete.
    Used to re-identify the athlete when ByteTrack loses the track ID.
    """

    def __init__(self):
        self._histograms: list[np.ndarray] = []
        self._reference: Optional[np.ndarray] = None
        self._bins = [8, 8, 8]
        self._ranges = [0, 180, 0, 256, 0, 256]

    def _compute_histogram(self, image_bgr: np.ndarray,
                            mask: Optional[np.ndarray] = None,
                            bbox: Optional[tuple] = None) -> Optional[np.ndarray]:
        """
        Compute normalized HSV histogram from the torso region of the athlete.
        Uses the segmentation mask if available, otherwise uses center 40-80%
        of the bounding box (rough torso zone).
        """
        h, w = image_bgr.shape[:2]

        if mask is not None:
            # use only pixels inside the mask
            region = image_bgr.copy()
            region[~mask] = 0
            pixel_mask = mask.astype(np.uint8) * 255
        elif bbox is not None:
            x1, y1, x2, y2 = bbox
            bh = y2 - y1
            # torso zone: 30% to 75% of bbox height
            ty1 = max(0, int(y1 + bh * 0.30))
            ty2 = min(h, int(y1 + bh * 0.75))
            region = np.zeros_like(image_bgr)
            region[ty1:ty2, x1:x2] = image_bgr[ty1:ty2, x1:x2]
            pixel_mask = np.zeros((h, w), dtype=np.uint8)
            pixel_mask[ty1:ty2, x1:x2] = 255
        else:
            return None

        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], pixel_mask,
                             self._bins, self._ranges)
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return hist.flatten()

    def update(self, image_bgr: np.ndarray,
               mask: Optional[np.ndarray] = None,
               bbox: Optional[tuple] = None) -> None:
        """Add a new reference sample from a high-quality frame."""
        hist = self._compute_histogram(image_bgr, mask, bbox)
        if hist is None:
            return
        self._histograms.append(hist)
        if len(self._histograms) > MAX_REF_SAMPLES:
            self._histograms.pop(0)
        self._reference = np.mean(self._histograms, axis=0)

    def similarity(self, image_bgr: np.ndarray,
                   mask: Optional[np.ndarray] = None,
                   bbox: Optional[tuple] = None) -> float:
        """
        Returns similarity score 0-1 (1 = identical, 0 = completely different).
        Returns 0.5 if no reference has been built yet.
        """
        if self._reference is None:
            return 0.5
        hist = self._compute_histogram(image_bgr, mask, bbox)
        if hist is None:
            return 0.5
        # Bhattacharyya: 0 = identical, 1 = completely different
        ref_hist = self._reference.reshape(self._bins).astype(np.float32)
        cmp_hist = hist.reshape(self._bins).astype(np.float32)
        dist = cv2.compareHist(ref_hist.flatten(), cmp_hist.flatten(),
                               cv2.HISTCMP_BHATTACHARYYA)
        return float(np.clip(1.0 - dist, 0.0, 1.0))

    @property
    def is_ready(self) -> bool:
        return self._reference is not None and len(self._histograms) >= 2


@dataclass
class TrackState:
    athlete_track_id: Optional[int] = None
    calibration_done: bool          = False
    appearance: AppearanceModel     = field(default_factory=AppearanceModel)

    _displacement: dict = field(default_factory=dict)
    _last_cx: dict      = field(default_factory=dict)
    _frame_count: int   = 0

    def update_calibration(self, detections: list[dict]) -> None:
        if self.calibration_done:
            return
        for det in detections:
            tid = det["track_id"]
            cx  = det["cx"]
            if tid in self._last_cx:
                self._displacement[tid] = (
                    self._displacement.get(tid, 0.0)
                    + abs(cx - self._last_cx[tid])
                )
            self._last_cx[tid] = cx
        self._frame_count += 1
        if self._frame_count >= CALIBRATION_FRAMES:
            self._lock_athlete()

    def _lock_athlete(self) -> None:
        if not self._displacement:
            return
        self.athlete_track_id = max(self._displacement, key=self._displacement.get)
        self.calibration_done = True
        disp = self._displacement[self.athlete_track_id]
        print(f"  [Tracker] Athlete locked → Track ID {self.athlete_track_id} "
              f"(motion: {disp:.0f}px over {CALIBRATION_FRAMES} frames)")

    def force_lock(self, track_id: int) -> None:
        self.athlete_track_id = track_id
        self.calibration_done = True


def _padded_crop(image: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                 pad: float = PADDING_RATIO) -> tuple[np.ndarray, tuple]:
    h, w = image.shape[:2]
    bw, bh = x2 - x1, y2 - y1
    px, py = int(bw * pad), int(bh * pad)
    cx1 = max(0, x1 - px);  cy1 = max(0, y1 - py)
    cx2 = min(w, x2 + px);  cy2 = min(h, y2 + py)
    return image[cy1:cy2, cx1:cx2], (cx1, cy1, 1.0)


def _select_athlete(
    valid: list[int],
    track_ids: np.ndarray,
    bboxes: np.ndarray,
    areas: list[float],
    masks: list[Optional[np.ndarray]],   # per-detection full-frame mask or None
    image: np.ndarray,
    state: TrackState,
    *,
    frame_idx: int = 0,
    track_scorer: Optional[TrackScorerContext] = None,
    det_motion: Optional[dict[int, float]] = None,
) -> int:
    """
    Choose which detected person index is the athlete using:
      Priority 1 — exact ByteTrack ID match
      Priority 2 — appearance similarity (if model is ready)
      Priority 3 — track scorer + largest bbox (fallback)
    Returns the index into the detection arrays.
    """
    # Priority 1: ByteTrack ID match
    if state.athlete_track_id is not None:
        exact = [i for i in valid if int(track_ids[i]) == state.athlete_track_id]
        if exact:
            return exact[0]

    # Priority 2: appearance similarity
    if state.appearance.is_ready:
        scores = []
        for i in valid:
            bbox = (int(bboxes[i][0]), int(bboxes[i][1]),
                    int(bboxes[i][2]), int(bboxes[i][3]))
            sim = state.appearance.similarity(image, mask=masks[i], bbox=bbox)
            track_bonus = 0.0
            if track_scorer is not None:
                motion = (det_motion or {}).get(int(track_ids[i]), 0.0)
                track_bonus = track_scorer.candidate_selection_score(
                    bbox, frame_idx, sim, motion_px=motion,
                )
            scores.append((i, sim + track_bonus * 0.15))
        best_i, best_sim = max(scores, key=lambda x: x[1])
        if best_sim >= (1.0 - APPEARANCE_THRESHOLD):
            new_tid = int(track_ids[best_i])
            if new_tid != state.athlete_track_id:
                print(f"  [Appearance] Re-ID: {state.athlete_track_id} → {new_tid} "
                      f"(similarity={best_sim:.2f})")
                state.athlete_track_id = new_tid
            return best_i

    # Priority 3: track scorer or largest bbox
    if track_scorer is not None:
        ranked = []
        for i in valid:
            bbox = (float(bboxes[i][0]), float(bboxes[i][1]),
                    float(bboxes[i][2]), float(bboxes[i][3]))
            motion = (det_motion or {}).get(int(track_ids[i]), 0.0)
            sim = state.appearance.similarity(image, mask=masks[i], bbox=(
                int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]),
            ))
            ts = track_scorer.candidate_selection_score(
                bbox, frame_idx, sim, motion_px=motion,
            )
            ranked.append((i, ts + areas[i] * 1e-6))
        return max(ranked, key=lambda x: x[1])[0]

    return max(valid, key=lambda i: areas[i])


def run_tracked_frame(
    image: np.ndarray,
    model_seg,
    state: TrackState,
    frame_idx: int,
    model_pose=None,   # None = tracking only (no pose estimation)
    track_scorer: Optional[TrackScorerContext] = None,
) -> dict:
    """
    Seg+track → appearance selection → (optional) pose on crop.

    Pass model_pose=None for tracking-only frames (cheap: ~8ms on GPU).
    Pass model_pose=<model> for full analysis frames (adds ~25ms for pose).

    Returns dict with keys:
      found, track_id, bbox, mask_area_px, crop_offset,
      kps_xy (None if no pose), kps_conf (None if no pose),
      seg_mask, quality_score, appearance_sim
    """
    empty = {
        "found": False, "track_id": None, "bbox": None,
        "mask_area_px": 0, "crop_offset": (0, 0),
        "kps_xy": None, "kps_conf": None,
        "seg_mask": None, "quality_score": 0.0,
        "appearance_sim": 0.0,
        "track_overlap": None, "athlete_state": None,
        "position_s": None, "predicted_bbox": None,
    }

    # ── Seg + ByteTrack ───────────────────────────────────────────────────────
    seg_results = model_seg.track(
        image,
        tracker="bytetrack.yaml",
        classes=[0],
        conf=0.35,
        iou=0.45,
        persist=True,
        verbose=False,
    )
    if not seg_results or seg_results[0].boxes is None:
        return empty

    res      = seg_results[0]
    boxes    = res.boxes
    track_ids = (boxes.id.cpu().numpy().astype(int)
                 if boxes.id is not None else np.arange(len(boxes)))
    bboxes   = boxes.xyxy.cpu().numpy()
    confs    = boxes.conf.cpu().numpy()
    areas    = [(b[2]-b[0])*(b[3]-b[1]) for b in bboxes]
    valid    = [i for i, a in enumerate(areas) if a >= MIN_BBOX_AREA]
    if not valid:
        if track_scorer is not None:
            lost = track_scorer.score_bbox(None, frame_idx)
            empty.update(lost)
        return empty

    # ── Build per-detection masks (full frame) ────────────────────────────────
    det_masks: list[Optional[np.ndarray]] = []
    for i in range(len(bboxes)):
        if res.masks is not None and i < len(res.masks.data):
            mt = res.masks.data[i].cpu().numpy()
            m  = cv2.resize(mt, (image.shape[1], image.shape[0]),
                            interpolation=cv2.INTER_NEAREST).astype(bool)
            det_masks.append(m)
        else:
            det_masks.append(None)

    # ── Calibration (motion-based) ────────────────────────────────────────────
    detections_for_cal = [
        {"track_id": int(track_ids[i]),
         "cx": float((bboxes[i][0]+bboxes[i][2])/2),
         "cy": float((bboxes[i][1]+bboxes[i][3])/2),
         "area": float(areas[i])}
        for i in valid
    ]
    state.update_calibration(detections_for_cal)

    det_motion = {
        int(track_ids[i]): float(state._displacement.get(int(track_ids[i]), 0.0))
        for i in valid
    }

    # ── Select athlete ────────────────────────────────────────────────────────
    athlete_idx = _select_athlete(
        valid, track_ids, bboxes, areas, det_masks, image, state,
        frame_idx=frame_idx,
        track_scorer=track_scorer,
        det_motion=det_motion,
    )

    bbox = bboxes[athlete_idx]
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    tid      = int(track_ids[athlete_idx])
    seg_mask = det_masks[athlete_idx]
    mask_area = 0

    # tighten bbox to mask bounds if mask available
    if seg_mask is not None:
        mask_area = int(seg_mask.sum())
        ys, xs = np.where(seg_mask)
        if len(xs) > 0:
            x1, x2 = int(xs.min()), int(xs.max())
            y1, y2 = int(ys.min()), int(ys.max())

    # ── Appearance similarity score ───────────────────────────────────────────
    bbox_tuple = (x1, y1, x2, y2)
    app_sim = state.appearance.similarity(image, mask=seg_mask, bbox=bbox_tuple)

    # ── Update appearance model from high-quality detections ──────────────────
    det_conf = float(confs[athlete_idx])
    if det_conf >= 0.60 and mask_area > 8000:
        state.appearance.update(image, mask=seg_mask, bbox=bbox_tuple)

    # ── Crop athlete + run pose (optional) ───────────────────────────────────
    kps_xy_full = None
    kps_conf    = None
    quality     = 0.0

    if model_pose is not None:
        crop, (ox, oy, _) = _padded_crop(image, x1, y1, x2, y2)
        if crop.size > 0:
            pose_results = model_pose(crop, verbose=False, conf=0.3)
            if pose_results and pose_results[0].keypoints is not None:
                kps_data = pose_results[0].keypoints
                if len(kps_data.xy) > 0:
                    crop_boxes = pose_results[0].boxes
                    if crop_boxes is not None and len(crop_boxes) > 1:
                        crop_areas = [(b[2]-b[0])*(b[3]-b[1])
                                      for b in crop_boxes.xyxy.cpu().numpy()]
                        best = int(np.argmax(crop_areas))
                    else:
                        best = 0

                    kps_xy_crop = kps_data.xy.cpu().numpy()[best]
                    kps_conf    = kps_data.conf.cpu().numpy()[best]
                    kps_xy_full = kps_xy_crop.copy()
                    kps_xy_full[:, 0] += ox
                    kps_xy_full[:, 1] += oy

                    n_valid = int((kps_conf >= 0.45).sum())
                    quality = (0.45 * n_valid / 17.0
                               + 0.30 * det_conf
                               + 0.15 * min(1.0, mask_area / 30000.0)
                               + 0.10 * app_sim)

    track_fields: dict = {}
    if track_scorer is not None:
        track_fields = track_scorer.score_bbox(
            (float(x1), float(y1), float(x2), float(y2)), frame_idx,
        )

    return {
        "found":          True,
        "track_id":       tid,
        "bbox":           (x1, y1, x2, y2),
        "mask_area_px":   mask_area,
        "crop_offset":    (0, 0),
        "kps_xy":         kps_xy_full,
        "kps_conf":       kps_conf,
        "seg_mask":       seg_mask,
        "quality_score":  round(float(quality), 3),
        "appearance_sim": round(float(app_sim), 3),
        **track_fields,
    }
