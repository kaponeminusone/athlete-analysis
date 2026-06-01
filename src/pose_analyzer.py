"""
Pose analyzer: runs YOLO11 pose + segmentation on a frame and
estimates the camera angle relative to the athlete.

Camera angle classification uses the ratio of the projected
shoulder width to the torso height — a well-known 2D heuristic:
  - FRONTAL:     shoulders wide, both clearly visible
  - SEMI_FRONT:  athlete turned ~30-60 degrees
  - LATERAL:     profile view, shoulder width minimal (best for biomechanics)
  - SEMI_BACK:   athlete mostly facing away
  - UNKNOWN:     not enough confident keypoints to decide
"""

import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

# COCO keypoint indices used in this project
KP = {
    "nose":        0,
    "l_shoulder":  5,
    "r_shoulder":  6,
    "l_elbow":     7,
    "r_elbow":     8,
    "l_hip":       11,
    "r_hip":       12,
    "l_knee":      13,
    "r_knee":      14,
    "l_ankle":     15,
    "r_ankle":     16,
}

CONF_THRESHOLD = 0.45  # minimum keypoint confidence to use


class CameraAngle(str, Enum):
    FRONTAL    = "FRONTAL"       # facing camera  ~0-30°
    SEMI_FRONT = "SEMI_FRONT"   # ~30-60°
    LATERAL    = "LATERAL"       # profile       ~60-90°  ← best for analysis
    SEMI_BACK  = "SEMI_BACK"    # ~90-150°
    UNKNOWN    = "UNKNOWN"


@dataclass
class KeypointData:
    name: str
    x: float
    y: float
    conf: float

    @property
    def valid(self) -> bool:
        return self.conf >= CONF_THRESHOLD


@dataclass
class FrameAnalysis:
    frame_idx: int
    timestamp_s: float

    # raw detection
    person_detected: bool = False
    person_bbox: Optional[tuple] = None          # (x1, y1, x2, y2)
    person_bbox_area: float = 0.0
    has_mask: bool = False
    mask_area_px: int = 0
    track_id: Optional[int] = None               # ByteTrack ID (None = no tracking)

    # keypoints
    keypoints: dict[str, KeypointData] = field(default_factory=dict)
    keypoints_valid_count: int = 0

    # camera angle
    camera_angle: CameraAngle = CameraAngle.UNKNOWN
    shoulder_ratio: float = 0.0   # shoulder_width / torso_height (0=lateral, 1=frontal)
    angle_confidence: float = 0.0 # 0-1, how sure we are about the angle estimate

    # derived geometry
    torso_height_px: float = 0.0
    shoulder_width_px: float = 0.0
    hip_width_px: float = 0.0
    body_height_px: float = 0.0   # ankle to nose approx

    # overall usability score for biomechanical analysis (0-1)
    quality_score: float = 0.0
    usable_for_analysis: bool = False

    # manual correction metadata
    manually_corrected: bool = False
    # "auto" | "bbox_correction" | "click_selection" | "mask_correction"
    correction_source: str = "auto"

    # tracking backend that produced this frame
    # "bytetrack" | "sot_csrt" | "sot_sam2"
    tracking_source: str = "bytetrack"


def _point(kps_xy, kps_conf, idx) -> Optional[KeypointData]:
    """Extract a single keypoint; return None if below threshold."""
    name = [k for k, v in KP.items() if v == idx][0]
    conf = float(kps_conf[idx])
    x    = float(kps_xy[idx][0])
    y    = float(kps_xy[idx][1])
    kp = KeypointData(name=name, x=x, y=y, conf=conf)
    return kp  # always return; caller checks .valid


def _dist(a: KeypointData, b: KeypointData) -> float:
    return float(np.hypot(a.x - b.x, a.y - b.y))


def _estimate_camera_angle(analysis: FrameAnalysis) -> None:
    """
    Heuristic: compare projected shoulder width to torso height.

    When the athlete faces the camera:
        shoulder_width is ~35-40% of body height  → ratio ~0.35-0.40
    When perfectly lateral (profile):
        shoulder_width drops to ~5-10% of body height → ratio ~0.05-0.10

    We normalise this ratio to [0, 1] where 1 = frontal, 0 = lateral,
    then map to the 5 angle categories.
    """
    kps = analysis.keypoints

    ls = kps.get("l_shoulder")
    rs = kps.get("r_shoulder")
    lh = kps.get("l_hip")
    rh = kps.get("r_hip")

    # need at least shoulders and one hip pair
    if not (ls and rs and ls.valid and rs.valid):
        analysis.camera_angle = CameraAngle.UNKNOWN
        analysis.angle_confidence = 0.0
        return

    shoulder_w = _dist(ls, rs)
    analysis.shoulder_width_px = shoulder_w

    # torso height: mid-shoulder to mid-hip
    mid_s_y = (ls.y + rs.y) / 2
    torso_h = 0.0
    if lh and rh and lh.valid and rh.valid:
        mid_h_y = (lh.y + rh.y) / 2
        torso_h = abs(mid_s_y - mid_h_y)
        analysis.hip_width_px = _dist(lh, rh)
    elif lh and lh.valid:
        torso_h = abs(mid_s_y - lh.y)
    elif rh and rh.valid:
        torso_h = abs(mid_s_y - rh.y)

    analysis.torso_height_px = torso_h

    if torso_h < 10:
        analysis.camera_angle = CameraAngle.UNKNOWN
        analysis.angle_confidence = 0.0
        return

    # ratio: shoulder_width / torso_height
    ratio = shoulder_w / torso_h
    analysis.shoulder_ratio = ratio

    # empirical mapping (calibrate with your specific camera / focal length)
    # ratio > 0.80 → FRONTAL
    # 0.50 - 0.80 → SEMI_FRONT
    # 0.20 - 0.50 → LATERAL
    # 0.05 - 0.20 → SEMI_BACK (one shoulder mostly hidden)
    # < 0.05      → UNKNOWN (likely bad detection)

    # confidence is based on keypoint confidence average
    avg_conf = np.mean([ls.conf, rs.conf])
    if lh and lh.valid and rh and rh.valid:
        avg_conf = np.mean([ls.conf, rs.conf, lh.conf, rh.conf])
    analysis.angle_confidence = float(avg_conf)

    if ratio > 0.80:
        analysis.camera_angle = CameraAngle.FRONTAL
    elif ratio > 0.50:
        analysis.camera_angle = CameraAngle.SEMI_FRONT
    elif ratio > 0.20:
        analysis.camera_angle = CameraAngle.LATERAL
    elif ratio > 0.05:
        analysis.camera_angle = CameraAngle.SEMI_BACK
    else:
        analysis.camera_angle = CameraAngle.UNKNOWN


def _compute_quality(result: FrameAnalysis) -> None:
    """
    Compute overall quality score and set usable_for_analysis flag.

    quality_score (0-1):
      40% — fraction of valid keypoints (out of 11 relevant ones)
      30% — camera angle suitability (LATERAL=1.0, SEMI_BACK=0.7, SEMI_FRONT=0.3, FRONTAL=0.1)
      20% — angle_confidence (keypoint conf of shoulders/hips)
      10% — mask quality (has_mask bonus)

    usable_for_analysis: quality >= 0.55 AND angle is LATERAL or SEMI_BACK
    """
    if not result.person_detected:
        result.quality_score = 0.0
        result.usable_for_analysis = False
        return

    kp_score = result.keypoints_valid_count / 11.0

    angle_scores = {
        CameraAngle.LATERAL:    1.0,
        CameraAngle.SEMI_BACK:  0.7,
        CameraAngle.SEMI_FRONT: 0.3,
        CameraAngle.FRONTAL:    0.1,
        CameraAngle.UNKNOWN:    0.0,
    }
    angle_score = angle_scores.get(result.camera_angle, 0.0)
    mask_score  = 0.5 + 0.5 * min(1.0, result.mask_area_px / 30000.0) if result.has_mask else 0.4

    result.quality_score = round(
        0.40 * kp_score
        + 0.30 * angle_score
        + 0.20 * result.angle_confidence
        + 0.10 * mask_score,
        3
    )
    result.usable_for_analysis = (
        result.quality_score >= 0.55
        and result.camera_angle in (CameraAngle.LATERAL, CameraAngle.SEMI_BACK)
    )


def analyze_frame_from_tracker(
    frame_idx: int,
    timestamp_s: float,
    tracker_result: dict,
) -> FrameAnalysis:
    """
    Build a FrameAnalysis from the output of athlete_tracker.run_tracked_frame().
    Keypoints are already in full-frame coordinates; no additional model calls needed.
    """
    result = FrameAnalysis(frame_idx=frame_idx, timestamp_s=timestamp_s)

    if not tracker_result.get("found"):
        return result

    result.person_detected = True
    result.track_id        = tracker_result["track_id"]
    result.has_mask        = tracker_result["seg_mask"] is not None
    result.mask_area_px    = tracker_result["mask_area_px"]
    result.quality_score   = tracker_result["quality_score"]  # tracker's own score

    bbox = tracker_result["bbox"]
    result.person_bbox      = bbox
    result.person_bbox_area = float((bbox[2]-bbox[0]) * (bbox[3]-bbox[1]))

    kps_xy   = tracker_result["kps_xy"]    # [17, 2]
    kps_conf = tracker_result["kps_conf"]  # [17]

    if kps_xy is None or kps_conf is None:
        return result

    for name, idx in KP.items():
        kp = _point(kps_xy, kps_conf, idx)
        result.keypoints[name] = kp

    result.keypoints_valid_count = sum(1 for kp in result.keypoints.values() if kp.valid)

    nose = result.keypoints.get("nose")
    la   = result.keypoints.get("l_ankle")
    ra   = result.keypoints.get("r_ankle")
    if nose and nose.valid:
        ankle_ys = [a.y for a in [la, ra] if a and a.valid]
        if ankle_ys:
            result.body_height_px = max(ankle_ys) - nose.y

    _estimate_camera_angle(result)
    _compute_quality(result)  # recompute with camera angle info

    return result


def analyze_frame(
    frame_idx: int,
    timestamp_s: float,
    image,          # numpy BGR image
    model_pose,     # ultralytics YOLO pose model
    model_seg=None, # ultralytics YOLO seg model (optional)
) -> FrameAnalysis:
    """
    Fallback: run pose on a single frame without tracking.
    Used when athlete_tracker is not available.
    """
    result = FrameAnalysis(frame_idx=frame_idx, timestamp_s=timestamp_s)

    pose_results = model_pose(image, verbose=False, conf=0.3)
    if not pose_results or len(pose_results[0].boxes) == 0:
        return result

    boxes = pose_results[0].boxes
    areas = [
        (b[2] - b[0]) * (b[3] - b[1])
        for b in boxes.xyxy.cpu().numpy()
    ]
    best_idx = int(np.argmax(areas))

    result.person_detected = True
    bbox = boxes.xyxy.cpu().numpy()[best_idx]
    result.person_bbox = tuple(bbox.tolist())
    result.person_bbox_area = float(areas[best_idx])

    kps_xy   = pose_results[0].keypoints.xy.cpu().numpy()[best_idx]
    kps_conf = pose_results[0].keypoints.conf.cpu().numpy()[best_idx]

    for name, idx in KP.items():
        kp = _point(kps_xy, kps_conf, idx)
        result.keypoints[name] = kp

    result.keypoints_valid_count = sum(1 for kp in result.keypoints.values() if kp.valid)

    nose = result.keypoints.get("nose")
    la   = result.keypoints.get("l_ankle")
    ra   = result.keypoints.get("r_ankle")
    if nose and nose.valid:
        ankle_ys = [a.y for a in [la, ra] if a and a.valid]
        if ankle_ys:
            result.body_height_px = max(ankle_ys) - nose.y

    _estimate_camera_angle(result)

    if model_seg is not None:
        seg_results = model_seg(image, verbose=False, conf=0.3, classes=[0])
        if seg_results and seg_results[0].masks is not None:
            result.has_mask = True

    _compute_quality(result)
    return result
