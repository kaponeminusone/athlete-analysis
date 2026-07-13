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
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .athlete_tracker import _padded_crop, MIN_BBOX_AREA
from .calibration import load_calibration
from .frame_extractor import get_video_info
from .job_store import ProgressCallback, noop_progress
from .mask_utils import athlete_mask_overlap, load_mask_png
from .pose_analyzer import (
    FrameAnalysis,
    analyze_frame_from_tracker,
)
from .visualizer import annotate_frame, annotate_frame_array
from . import opt_flags
from .schemas import (
    build_analysis_document,
    frame_analysis_to_dict,
    write_derived_stubs,
)

SEED_MIN_QUALITY = 0.65
SEED_MAX_FRAMES  = 15
MIN_SIMILARITY   = 0.25     # minimum to accept a match (appearance-only)
MIN_SIMILARITY_WITH_MASKS = 0.20  # slightly lower floor when venue masks bias selection
TRACK_OV_WEIGHT  = 0.30
SAND_OV_WEIGHT   = 0.20
MASK_CAL_MODES   = ("color_masks", "cnn_masks", "keyframe_masks")

# refine_v2-only knobs (classic path never reads these)
V2_TEMPORAL_IOU_WEIGHT = 0.12
V2_TEMPORAL_CLOSE_GAP = 0.08       # amplify temporal when top scores within this gap

# Mid-window clothing lock (progress through refine analysis frames)
V2_MID_WINDOW = (0.30, 0.90)       # learn / strengthen appearance here
V2_LANDING_TAIL = 0.90             # sand/landing zone starts here

# Mid-window: aggressive online appearance updates (clothing lock)
# Early + landing: online appearance is frozen (no contamination)
V2_MID_ONLINE_MIN_CONF = 0.70
V2_MID_ONLINE_MIN_SIM = 0.55
V2_MID_ONLINE_MIN_MASK_AREA = 8000
V2_MID_ONLINE_MIN_TRACK_OV = 0.25

# Landing / sand tail: permissive selection, never learn appearance
V2_LANDING_SAND_WEIGHT = 0.45      # vs SAND_OV_WEIGHT 0.20
V2_LANDING_MIN_SIM = 0.15          # with venue masks
V2_LANDING_MIN_SIM_NO_MASK = 0.18  # appearance-only
V2_TEMPORAL_BOOST_LANDING = 0.28   # stronger prev-track preference when blurred/close
V2_LANDING_TEMPORAL_HISTORY = 4    # IoU vs recent chosen bboxes (wider search)
V2_LANDING_SAND_PREFER = 0.10      # extra score boost for high sand_ov in landing

# Gap fill / early re-score (post mid-lock propagation into early + landing unknowns)
V2_GAP_MIN_BBOX_AREA = 600         # distant early athlete often << classic MIN_BBOX_AREA
V2_GAP_IOU_WEIGHT = 0.50           # strong temporal prior vs next-known / expected bbox
V2_GAP_CENTER_WEIGHT = 0.18        # soft center proximity vs expected
V2_GAP_MIN_SIM = 0.08              # appearance floor during gap fill
V2_GAP_MIN_SIM_RELAXED = 0.04      # when IoU / sand / track overlap is strong
V2_GAP_HIGH_IOU = 0.22             # IoU vs expected → allow relaxed appearance
V2_GAP_HIGH_SAND = 0.25
V2_GAP_HIGH_TRACK = 0.28
V2_GAP_ACCEPT_IOU = 0.20           # accept via IoU alone (locked identity + temporal)
V2_GAP_ACCEPT_TRACK = 0.28         # accept via track overlap alone
V2_GAP_DEBUG_EARLY = 10            # log top candidates for this many early fills
V2_QUALITY_FLOOR = 0.15            # never leave Q=0 when person_detected + bbox
V2_EARLY_RESCORE_MAX_Q = 0.35      # re-score early found frames below this quality
V2_TRACKING_SRC = "refined_v2"
V2_TRACKING_SRC_BBOX = "refined_v2_bbox_only"


@dataclass
class VenueMaskBias:
    """Load track/sand PNGs from original calibration for refine scoring."""

    output_dir: Path
    mask_frames: dict
    mode: str = ""
    _cache: dict = field(default_factory=dict, repr=False)

    @classmethod
    def from_output_dir(cls, output_dir: str | Path) -> Optional["VenueMaskBias"]:
        out = Path(output_dir)
        cal = load_calibration(out)
        if cal is None:
            print(f"  [Reanalyzer] use_cnn_masks: no calibration.json in {out}")
            return None
        mode = cal.get("mode") or ""
        mask_frames = cal.get("mask_frames") or {}
        if mode not in MASK_CAL_MODES or not mask_frames:
            print(
                f"  [Reanalyzer] use_cnn_masks: no usable mask_frames "
                f"(mode={mode!r}, frames={len(mask_frames)})"
            )
            return None
        print(
            f"  [Reanalyzer] Venue masks ON ({mode}): "
            f"{len(mask_frames)} mask_frames from {out}"
        )
        return cls(output_dir=out, mask_frames=mask_frames, mode=mode)

    def _nearest_mask_entry(self, frame_idx: int) -> Optional[dict]:
        if not self.mask_frames:
            return None
        key = str(frame_idx)
        if key in self.mask_frames:
            return self.mask_frames[key]
        keys = sorted(int(k) for k in self.mask_frames)
        nearest = min(keys, key=lambda k: abs(k - frame_idx))
        return self.mask_frames[str(nearest)]

    def load_masks(
        self, frame_idx: int,
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if frame_idx in self._cache:
            return self._cache[frame_idx]
        track_mask = sand_mask = None
        entry = self._nearest_mask_entry(frame_idx)
        if entry:
            track_rel = entry.get("track")
            sand_rel = entry.get("sand")
            if track_rel:
                track_mask = load_mask_png(self.output_dir / track_rel)
            if sand_rel:
                sand_mask = load_mask_png(self.output_dir / sand_rel)
        self._cache[frame_idx] = (track_mask, sand_mask)
        return track_mask, sand_mask

    def overlaps(
        self,
        frame_idx: int,
        bbox: tuple[float, float, float, float],
        seg_mask: Optional[np.ndarray],
        width: int,
        height: int,
    ) -> tuple[float, float]:
        track_mask, sand_mask = self.load_masks(frame_idx)
        track_ov = athlete_mask_overlap(bbox, seg_mask, track_mask, width, height)
        sand_ov = athlete_mask_overlap(bbox, seg_mask, sand_mask, width, height)
        return float(track_ov), float(sand_ov)


def _blend_mask_score(
    appearance_sim: float,
    track_ov: float,
    sand_ov: float,
    sand_weight: float = SAND_OV_WEIGHT,
) -> float:
    """Combine appearance with venue overlap. score = sim + 0.30*track + sand_w*sand."""
    return float(appearance_sim + TRACK_OV_WEIGHT * track_ov + sand_weight * sand_ov)


def _v2_zone(frame_progress: float) -> str:
    """Map refine-run progress [0,1] → early | mid | landing (refine_v2 only)."""
    if frame_progress >= V2_LANDING_TAIL:
        return "landing"
    if V2_MID_WINDOW[0] <= frame_progress < V2_MID_WINDOW[1]:
        return "mid"
    return "early"


def _bbox_iou(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def _bbox_lerp(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
    t: float,
) -> tuple[float, float, float, float]:
    """Linear interpolate bbox corners; t in [0, 1]."""
    t = max(0.0, min(1.0, float(t)))
    return tuple(a[i] + (b[i] - a[i]) * t for i in range(4))  # type: ignore[return-value]


def _v2_bbox_only_quality(
    det_conf: float,
    mask_area: int,
    appearance_sim: float,
) -> float:
    """Quality when identity is locked via bbox/mask but pose is missing."""
    return round(
        max(
            0.20,
            min(
                0.48,
                0.18
                + 0.22 * float(det_conf)
                + 0.12 * min(1.0, mask_area / 30000.0)
                + 0.12 * float(appearance_sim),
            ),
        ),
        3,
    )


def _v2_build_frame_analysis(
    image: np.ndarray,
    frame_idx: int,
    timestamp_s: float,
    model_pose,
    bbox: tuple[int, int, int, int],
    seg_mask: Optional[np.ndarray],
    det_conf: float,
    best_sim: float,
) -> FrameAnalysis:
    """
    Pose on crop → FrameAnalysis. Pose failure still yields person_detected
    with bbox/mask (tracking_source refined_v2_bbox_only).
    """
    x1, y1, x2, y2 = bbox
    mask_area = int(seg_mask.sum()) if seg_mask is not None else 0

    kps_xy_full = kps_conf = None
    quality = 0.0
    pose_ok = False

    crop, (ox, oy, _) = _padded_crop(image, x1, y1, x2, y2)
    if crop.size > 0:
        pose_results = model_pose(crop, verbose=False, conf=0.25)
        if (pose_results and pose_results[0].keypoints is not None
                and len(pose_results[0].keypoints.xy) > 0):
            kp_data = pose_results[0].keypoints
            crop_boxes = pose_results[0].boxes
            best_crop = 0
            if crop_boxes is not None and len(crop_boxes) > 1:
                crop_areas = [
                    (b[2] - b[0]) * (b[3] - b[1])
                    for b in crop_boxes.xyxy.cpu().numpy()
                ]
                best_crop = int(np.argmax(crop_areas))
            kps_xy_crop = kp_data.xy.cpu().numpy()[best_crop]
            kps_conf = kp_data.conf.cpu().numpy()[best_crop]
            kps_xy_full = kps_xy_crop.copy()
            kps_xy_full[:, 0] += ox
            kps_xy_full[:, 1] += oy
            n_valid = int((kps_conf >= 0.45).sum())
            quality = round(
                0.45 * n_valid / 17.0
                + 0.30 * float(det_conf)
                + 0.15 * min(1.0, mask_area / 30000.0)
                + 0.10 * float(best_sim),
                3,
            )
            pose_ok = True

    if not pose_ok:
        quality = _v2_bbox_only_quality(det_conf, mask_area, best_sim)

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
    fa.tracking_source = V2_TRACKING_SRC if pose_ok else V2_TRACKING_SRC_BBOX
    fa._appearance_sim = round(float(best_sim), 3)  # type: ignore[attr-defined]
    # analyze_frame_from_tracker may overwrite quality via angle (UNKNOWN→~0);
    # keep a floor whenever we locked a bbox identity.
    if fa.person_detected and fa.person_bbox is not None:
        if fa.quality_score < V2_QUALITY_FLOOR:
            fa.quality_score = max(
                V2_QUALITY_FLOOR,
                _v2_bbox_only_quality(det_conf, mask_area, best_sim),
            )
    return fa


def _v2_progress(i: int, n: int) -> float:
    if n <= 1:
        return 0.5
    return i / (n - 1)


def _v2_expected_bbox(
    analyses: list[FrameAnalysis],
    idx: int,
) -> Optional[tuple[float, float, float, float]]:
    """Interpolate (or copy) bbox from nearest known left/right neighbors."""
    left_i = right_i = None
    for j in range(idx - 1, -1, -1):
        if analyses[j].person_detected and analyses[j].person_bbox is not None:
            left_i = j
            break
    for j in range(idx + 1, len(analyses)):
        if analyses[j].person_detected and analyses[j].person_bbox is not None:
            right_i = j
            break
    if left_i is None and right_i is None:
        return None
    if left_i is None:
        return tuple(float(v) for v in analyses[right_i].person_bbox)  # type: ignore[index]
    if right_i is None:
        return tuple(float(v) for v in analyses[left_i].person_bbox)
    span = right_i - left_i
    t = (idx - left_i) / span if span > 0 else 0.0
    lb = tuple(float(v) for v in analyses[left_i].person_bbox)
    rb = tuple(float(v) for v in analyses[right_i].person_bbox)
    return _bbox_lerp(lb, rb, t)  # type: ignore[arg-type]


def _v2_expected_bbox_backward(
    analyses: list[FrameAnalysis],
    idx: int,
) -> Optional[tuple[float, float, float, float]]:
    """
    When walking early←mid, seed expected from the next-in-time known bbox
    (idx+1 already filled or mid anchor), not sparse left/right interpolate.
    """
    if idx + 1 < len(analyses):
        nxt = analyses[idx + 1]
        if nxt.person_detected and nxt.person_bbox is not None:
            return tuple(float(v) for v in nxt.person_bbox)
    return _v2_expected_bbox(analyses, idx)


def _v2_gap_select_candidate(
    image: np.ndarray,
    frame_idx: int,
    model_seg,
    appearance: RobustAppearanceModel,
    mask_bias: Optional[VenueMaskBias],
    expected_bbox: Optional[tuple[float, float, float, float]],
    zone: str,
    debug: bool = False,
) -> Optional[tuple[tuple[int, int, int, int], Optional[np.ndarray], float, float]]:
    """
    YOLO seg + locked appearance + mask + IoU vs expected bbox.
    Returns (bbox_int, seg_mask, det_conf, best_sim) or None.

    Accept when appearance clears the floor OR IoU vs expected is decent
    OR track overlap is strong — so distant early frames are not dropped
    solely for low clothing sim (classic MIN_BBOX_AREA also relaxed here).
    """
    results = model_seg(image, classes=[0], conf=0.25, verbose=False)
    if not results or results[0].boxes is None:
        if debug:
            print(f"  [v2 gap dbg] frame={frame_idx}: no YOLO boxes")
        return None

    res = results[0]
    bboxes = res.boxes.xyxy.cpu().numpy()
    confs = res.boxes.conf.cpu().numpy()
    areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in bboxes]
    valid = [i for i, a in enumerate(areas) if a >= V2_GAP_MIN_BBOX_AREA]
    if not valid:
        if debug:
            print(
                f"  [v2 gap dbg] frame={frame_idx}: all {len(bboxes)} dets "
                f"below V2_GAP_MIN_BBOX_AREA={V2_GAP_MIN_BBOX_AREA} "
                f"(areas={[int(a) for a in areas]})"
            )
        return None

    h, w = image.shape[:2]
    det_masks: list[Optional[np.ndarray]] = []
    for i in range(len(bboxes)):
        if res.masks is not None and i < len(res.masks.data):
            mt = res.masks.data[i].cpu().numpy()
            m = cv2.resize(mt, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
            det_masks.append(m)
        else:
            det_masks.append(None)

    in_landing = zone == "landing"
    sand_w = V2_LANDING_SAND_WEIGHT if in_landing else SAND_OV_WEIGHT

    scored: list[tuple] = []
    rejected: list[str] = []
    for i in valid:
        bbox = tuple(float(v) for v in bboxes[i])
        sim = appearance.similarity(
            image, mask=det_masks[i], bbox=tuple(int(v) for v in bbox),
        )
        track_ov = sand_ov = 0.0
        if mask_bias is not None:
            track_ov, sand_ov = mask_bias.overlaps(
                frame_idx, bbox, det_masks[i], w, h,
            )
        combined = (
            _blend_mask_score(sim, track_ov, sand_ov, sand_weight=sand_w)
            if mask_bias is not None else float(sim)
        )
        iou = _bbox_iou(bbox, expected_bbox) if expected_bbox is not None else 0.0
        combined += V2_GAP_IOU_WEIGHT * iou
        if in_landing and sand_ov > 0.0:
            combined += V2_LANDING_SAND_PREFER * sand_ov
        dist_n = 0.0
        # center proximity soft boost when expected exists
        if expected_bbox is not None:
            ecx = 0.5 * (expected_bbox[0] + expected_bbox[2])
            ecy = 0.5 * (expected_bbox[1] + expected_bbox[3])
            cx = 0.5 * (bbox[0] + bbox[2])
            cy = 0.5 * (bbox[1] + bbox[3])
            diag = max(1.0, float(np.hypot(w, h)))
            dist_n = float(np.hypot(cx - ecx, cy - ecy)) / diag
            combined += V2_GAP_CENTER_WEIGHT * max(0.0, 1.0 - dist_n * 4.0)
            # Penalize high-appearance spectators far from the temporal prior
            # so they cannot beat a weaker-sim athlete near the expected bbox.
            if iou < 0.05 and dist_n > 0.12:
                combined -= 0.20

        relax = (
            iou >= V2_GAP_HIGH_IOU
            or sand_ov >= V2_GAP_HIGH_SAND
            or track_ov >= V2_GAP_HIGH_TRACK
        )
        min_sim = V2_GAP_MIN_SIM_RELAXED if relax else V2_GAP_MIN_SIM
        accept = (
            sim >= min_sim
            or iou >= V2_GAP_ACCEPT_IOU
            or track_ov >= V2_GAP_ACCEPT_TRACK
            or (
                sim >= V2_GAP_MIN_SIM_RELAXED
                and (iou >= 0.12 or track_ov >= 0.18 or sand_ov >= V2_GAP_HIGH_SAND)
            )
        )
        # With a temporal prior, do not let a weak appearance-only match jump
        # to a distant spectator (poisons the backward chain). Prefer leaving
        # the frame empty over locking the wrong person.
        if (
            accept
            and expected_bbox is not None
            and iou < 0.08
            and track_ov < 0.15
            and dist_n > 0.08
            and sim < 0.22
        ):
            accept = False
        if not accept:
            rejected.append(
                f"i={i} area={areas[i]:.0f} sim={sim:.3f} "
                f"iou={iou:.2f} track={track_ov:.2f} dist={dist_n:.2f}"
            )
            continue
        scored.append((i, sim, combined, track_ov, sand_ov, iou, areas[i]))

    if debug:
        top = sorted(scored, key=lambda x: x[2], reverse=True)[:4]
        print(
            f"  [v2 gap dbg] frame={frame_idx} zone={zone} "
            f"ndet={len(bboxes)} valid={len(valid)} accepted={len(scored)} "
            f"expected={'yes' if expected_bbox is not None else 'no'}"
        )
        for i, sim, comb, tov, sov, iou, area in top:
            print(
                f"    cand i={i} area={area:.0f} sim={sim:.3f} "
                f"iou={iou:.2f} track={tov:.2f} sand={sov:.2f} score={comb:.3f}"
            )
        for line in rejected[:5]:
            print(f"    reject {line}")

    if not scored:
        if debug:
            print(f"  [v2 gap dbg] frame={frame_idx}: no accepted candidates — skip fill")
        return None

    # Prefer temporal continuity when expected exists: among near-tied scores,
    # pick the higher-IoU candidate so clothing-similar spectators lose.
    best = max(scored, key=lambda x: x[2])
    if expected_bbox is not None:
        high_iou = [s for s in scored if s[5] >= V2_GAP_ACCEPT_IOU]
        if high_iou:
            best_iou_cand = max(high_iou, key=lambda x: (x[5], x[2]))
            if best[5] < 0.08 and best_iou_cand[2] >= best[2] - 0.12:
                best = best_iou_cand

    best_i, best_sim, _, _, _, best_iou, _ = best
    x1, y1, x2, y2 = (int(v) for v in bboxes[best_i])
    seg_mask = det_masks[best_i]
    if seg_mask is not None:
        ys, xs = np.where(seg_mask)
        if len(xs) > 0:
            x1, x2 = int(xs.min()), int(xs.max())
            y1, y2 = int(ys.min()), int(ys.max())
    if debug:
        print(
            f"  [v2 gap dbg] frame={frame_idx}: FILL bbox=({x1},{y1},{x2},{y2}) "
            f"sim={best_sim:.3f} iou={best_iou:.2f}"
        )
    return (x1, y1, x2, y2), seg_mask, float(confs[best_i]), float(best_sim)


def _v2_gap_fill_pass(
    analyses: list[FrameAnalysis],
    frames_dir: Path,
    model_seg,
    model_pose,
    appearance: RobustAppearanceModel,
    mask_bias: Optional[VenueMaskBias],
    annotate_every: int,
    annotated_dir: Path,
) -> dict:
    """
    After primary refine_v2: propagate mid-window identity into early/landing gaps.

    Mid successes are anchors. Walk backward into early unknowns (expected bbox
    seeded from the next-in-time filled frame), then forward into landing
    unknowns. Empty frames (and weak early finds) are re-scored with the locked
    mid appearance model + temporal prior + venue masks. Pose failure still
    keeps bbox identity (refined_v2_bbox_only).
    """
    n = len(analyses)
    if n == 0:
        return {
            "mid_anchors": 0, "filled_early": 0, "filled_late": 0,
            "bbox_only": 0, "rescored_early": 0,
        }

    mid_idxs = [
        i for i in range(n)
        if _v2_zone(_v2_progress(i, n)) == "mid"
        and analyses[i].person_detected
        and analyses[i].person_bbox is not None
    ]
    mid_anchors = len(mid_idxs)
    if mid_anchors == 0:
        print("  [Reanalyzer v2] Gap fill: no mid anchors — skipped")
        return {
            "mid_anchors": 0, "filled_early": 0, "filled_late": 0,
            "bbox_only": 0, "rescored_early": 0,
        }

    first_mid = mid_idxs[0]
    last_mid = mid_idxs[-1]

    def _needs_fill(i: int) -> bool:
        fa = analyses[i]
        zone = _v2_zone(_v2_progress(i, n))
        if not fa.person_detected:
            return True
        # Light early re-score with locked mid clothing model
        if zone == "early" and fa.quality_score < V2_EARLY_RESCORE_MAX_Q:
            return True
        return False

    # Bidirectional order: early ← mid, then mid → landing
    early_order = list(range(first_mid - 1, -1, -1))
    late_order = list(range(last_mid + 1, n))

    filled_early = filled_late = rescored_early = bbox_only = 0
    early_debug_left = V2_GAP_DEBUG_EARLY

    def _process(i: int, side: str) -> None:
        nonlocal filled_early, filled_late, rescored_early, bbox_only, early_debug_left
        if not _needs_fill(i):
            return
        zone = _v2_zone(_v2_progress(i, n))
        fa_old = analyses[i]
        was_detected = fa_old.person_detected

        fpath = frames_dir / f"frame_{fa_old.frame_idx:06d}.jpg"
        if not fpath.exists():
            print(
                f"  [v2 gap dbg] frame={fa_old.frame_idx}: missing file {fpath.name} — skip"
            )
            return
        image = cv2.imread(str(fpath))
        if image is None:
            return

        # Backward walk: seed from next-known (forward neighbor already filled).
        # Forward/landing: interpolate / left-neighbor continuity.
        if side == "early":
            expected = _v2_expected_bbox_backward(analyses, i)
        else:
            expected = _v2_expected_bbox(analyses, i)
            if zone == "landing" and expected is None:
                for j in range(i - 1, -1, -1):
                    if analyses[j].person_detected and analyses[j].person_bbox is not None:
                        expected = tuple(float(v) for v in analyses[j].person_bbox)
                        break

        debug = side == "early" and early_debug_left > 0
        if debug:
            early_debug_left -= 1

        picked = _v2_gap_select_candidate(
            image, fa_old.frame_idx, model_seg, appearance, mask_bias,
            expected, zone, debug=debug,
        )
        if picked is None:
            if debug:
                print(f"  [v2 gap dbg] frame={fa_old.frame_idx}: fill NOT applied")
            return
        bbox, seg_mask, det_conf, best_sim = picked

        # If replacing a weak early find, require improved appearance or IoU
        if was_detected and zone == "early":
            old_sim = float(getattr(fa_old, "_appearance_sim", 0.0) or 0.0)
            iou_exp = (
                _bbox_iou(tuple(float(v) for v in bbox), expected)
                if expected is not None else 0.0
            )
            if best_sim + 0.05 < old_sim and iou_exp < V2_GAP_HIGH_IOU:
                if debug:
                    print(
                        f"  [v2 gap dbg] frame={fa_old.frame_idx}: keep old "
                        f"(sim {best_sim:.3f} vs old {old_sim:.3f}, iou={iou_exp:.2f})"
                    )
                return

        fa_new = _v2_build_frame_analysis(
            image, fa_old.frame_idx, fa_old.timestamp_s, model_pose,
            bbox, seg_mask, det_conf, best_sim,
        )
        analyses[i] = fa_new

        if fa_new.tracking_source == V2_TRACKING_SRC_BBOX:
            bbox_only += 1
        if side == "early":
            if was_detected:
                rescored_early += 1
            else:
                filled_early += 1
            if debug:
                print(
                    f"  [v2 gap dbg] frame={fa_new.frame_idx}: applied "
                    f"src={fa_new.tracking_source} Q={fa_new.quality_score:.2f} "
                    f"det={fa_new.person_detected}"
                )
        else:
            filled_late += 1

        if opt_flags.write_annotated() and annotate_every > 0 and (i % annotate_every == 0):
            out_img = annotated_dir / f"annotated_{fa_new.frame_idx:06d}.jpg"
            annotate_frame(
                str(fpath), fa_new, str(out_img),
                seg_mask=None,
                appearance_sim=float(getattr(fa_new, "_appearance_sim", 0.0) or 0.0),
            )

    print(
        f"  [Reanalyzer v2] Gap fill walk: early←mid "
        f"({len(early_order)} frames from idx {first_mid - 1}→0), "
        f"mid→landing ({len(late_order)} frames)"
    )
    for i in early_order:
        _process(i, "early")
    for i in late_order:
        _process(i, "late")

    # Count bbox-only across full run (gap fill + primary) for logging below
    bbox_only_total = sum(
        1 for a in analyses
        if a.person_detected and a.tracking_source == V2_TRACKING_SRC_BBOX
    )

    print(
        f"  [Reanalyzer v2] Gap fill: mid_anchors={mid_anchors}, "
        f"filled_early={filled_early}, filled_late={filled_late}, "
        f"rescored_early={rescored_early}, "
        f"bbox_only_gap={bbox_only}, bbox_only_total={bbox_only_total}"
    )
    return {
        "mid_anchors": mid_anchors,
        "filled_early": filled_early,
        "filled_late": filled_late,
        "bbox_only": bbox_only_total,
        "rescored_early": rescored_early,
    }


def _copy_venue_assets_for_sections(original_dir: Path, refined_dir: Path) -> None:
    """Copy calibration.json + venue_masks/ so section analysis can use track/sand."""
    src_cal = original_dir / "calibration.json"
    if src_cal.exists():
        shutil.copy2(src_cal, refined_dir / "calibration.json")
        print(f"  [Reanalyzer v2] Copied calibration.json → {refined_dir.name}/")
    src_masks = original_dir / "venue_masks"
    if src_masks.is_dir():
        dst_masks = refined_dir / "venue_masks"
        if dst_masks.exists():
            shutil.rmtree(dst_masks)
        shutil.copytree(src_masks, dst_masks)
        print(f"  [Reanalyzer v2] Copied venue_masks/ → {refined_dir.name}/")


def _reapply_venue_masks_for_refined(
    refined_dir: Path,
    video_path: Path,
) -> bool:
    """
    Re-run venue mask apply on the refined output so every refined analysis
    frame gets a mask PNG (first-pass copy is sparse when strides differ).

    Mirrors /api/venue/apply use_masks gating. Returns True if apply ran.
    Raises on hard failures — caller should soft-fail and keep copied masks.
    """
    from .venue_masks import should_use_keyframe_pipeline
    from .venue_profile import (
        DEFAULT_VENUE_ID,
        apply_masks_to_output,
        load_profile,
    )
    from .venue_seg_infer import has_trained_seg_model

    refined_dir = Path(refined_dir)
    video_path = Path(video_path)
    cal_existing = load_calibration(refined_dir)

    venue_id = DEFAULT_VENUE_ID
    if cal_existing:
        vp = cal_existing.get("venue_profile") or {}
        if vp.get("venue_id"):
            venue_id = str(vp["venue_id"])

    use_cnn = has_trained_seg_model(venue_id)
    use_keyframe_pipeline = (
        not use_cnn
        and cal_existing is not None
        and should_use_keyframe_pipeline(cal_existing, prefer_keyframes=True)
    )
    profile = load_profile(venue_id)
    use_masks = (
        use_cnn
        or use_keyframe_pipeline
        or (profile is not None and int(profile.get("version", 2)) >= 3)
    )
    if not use_masks:
        print(
            "  [Reanalyzer v2] venue_masks: skip re-apply "
            "(no CNN / keyframes / profile v3)"
        )
        return False

    apply_masks_to_output(
        refined_dir,
        video_path,
        profile=profile,
        venue_id=venue_id,
        prefer_keyframes=True,
    )
    return True


# ─── Robust appearance model ──────────────────────────────────────────────────

class RobustAppearanceModel:
    """
    Improved appearance model for second-pass reanalysis.

    Key improvements over the basic AppearanceModel:

    1. Zone-based histograms — the person bbox is split into three vertical
       bands (head 0-25%, torso 25-70%, legs 70-100%) and each gets its own
       histogram. Torso weight is 0.60 (most discriminative for clothing).

    2. Finer hue bins (32 instead of 8) — better separates e.g. white from
       cream, or the exact red hue from orange.

    3. Achromatic ratio — fraction of pixels with saturation < 40. A white
       sweater scores ~0.80; a red/blue sweater scores ~0.10. This single
       number is very discriminative and is compared before the histograms.

    4. Reference set instead of averaged model — keeps the last MAX_REFS
       individual zone-histogram tuples and uses the MEDIAN distance across
       all of them. Averaging destroys color distinctiveness.

    5. Negative examples — histograms of OTHER persons detected in the same
       seed frames. During matching, candidates that look like a known-wrong
       person are penalized.
    """

    ZONE_BOUNDS  = [(0.00, 0.25), (0.25, 0.70), (0.70, 1.00)]
    ZONE_WEIGHTS = [0.10,          0.60,          0.30]
    BINS_H, BINS_S, BINS_V = 32, 8, 8   # 2048 bins per zone — captures hue, saturation, brightness
    MAX_REFS = 20
    MAX_NEGS = 15

    def __init__(self) -> None:
        self._pos_refs:  list[list[np.ndarray]] = []   # zone histograms per positive reference
        self._neg_refs:  list[list[np.ndarray]] = []

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _zone_features(self, img: np.ndarray,
                        mask: Optional[np.ndarray],
                        bbox: tuple) -> list[np.ndarray]:
        """
        Compute per-zone HSV histograms.

        The 3-axis histogram (hue × saturation × value) already captures
        everything needed to distinguish any two clothing colors:
          - white vs yellow: separated by the saturation axis
          - yellow vs red:   separated by the hue axis
          - dark vs light:   separated by the value axis
        No separate achromatic ratio is needed — it would only work for
        white clothing and breaks for any other color pair.
        """
        x1, y1, x2, y2 = (int(v) for v in bbox)
        bh = max(y2 - y1, 1)
        h, w = img.shape[:2]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hists: list[np.ndarray] = []

        for (z0, z1) in self.ZONE_BOUNDS:
            zy1 = max(0, y1 + int(bh * z0))
            zy2 = min(h, y1 + int(bh * z1))
            zm = np.zeros((h, w), dtype=np.uint8)
            if mask is not None:
                zm[zy1:zy2, x1:x2] = mask[zy1:zy2, x1:x2].astype(np.uint8) * 255
            else:
                zm[zy1:zy2, x1:x2] = 255

            if int(zm.sum() / 255) < 20:
                hists.append(np.zeros(self.BINS_H * self.BINS_S * self.BINS_V, dtype=np.float32))
                continue

            hist = cv2.calcHist([hsv], [0, 1, 2], zm,
                                [self.BINS_H, self.BINS_S, self.BINS_V],
                                [0, 180, 0, 256, 0, 256])
            cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
            hists.append(hist.flatten().astype(np.float32))

        return hists

    def _zone_dist(self, hists_a: list[np.ndarray],
                   hists_b: list[np.ndarray]) -> float:
        """Weighted Bhattacharyya distance across zones."""
        total = 0.0
        for h_a, h_b, w in zip(hists_a, hists_b, self.ZONE_WEIGHTS):
            if h_a.sum() == 0 or h_b.sum() == 0:
                total += 0.5 * w
                continue
            d = cv2.compareHist(h_a, h_b, cv2.HISTCMP_BHATTACHARYYA)
            total += float(d) * w
        return total

    # ── Public API ────────────────────────────────────────────────────────────

    def add_positive(self, img: np.ndarray,
                     mask: Optional[np.ndarray], bbox: tuple) -> None:
        hists = self._zone_features(img, mask, bbox)
        self._pos_refs.append(hists)
        if len(self._pos_refs) > self.MAX_REFS:
            self._pos_refs.pop(0)

    def add_negative(self, img: np.ndarray,
                     mask: Optional[np.ndarray], bbox: tuple) -> None:
        hists = self._zone_features(img, mask, bbox)
        self._neg_refs.append(hists)
        if len(self._neg_refs) > self.MAX_NEGS:
            self._neg_refs.pop(0)

    def similarity(self, img: np.ndarray,
                   mask: Optional[np.ndarray], bbox: tuple) -> float:
        """
        Returns score 0–1 (1 = identical to references).

        Steps:
          a) Achromatic pre-filter: if the candidate's achromatic ratio
             differs too much from the reference average, penalise hard.
          b) Positive similarity: median Bhattacharyya distance across all
             positive references (median is robust to outlier seed frames).
          c) Negative penalty: if candidate looks like a known-wrong person,
             reduce the score proportionally.
        """
        if not self._pos_refs:
            return 0.5

        cand_hists = self._zone_features(img, mask, bbox)

        # (a) Positive similarity — median Bhattacharyya distance across all
        #     stored references, converted to similarity. Median is robust to
        #     a few bad seed frames; keeping individual refs (not averaging)
        #     preserves the exact color fingerprint learned from the seed interval.
        pos_dists = [self._zone_dist(ref, cand_hists) for ref in self._pos_refs]
        pos_dist  = float(np.median(pos_dists))
        pos_sim   = float(np.clip(1.0 - pos_dist, 0.0, 1.0))

        # (b) Negative penalty — if the candidate looks like a known-wrong
        #     person (collected from other detections in seed frames), reduce
        #     the score. Works for any color pair: yellow athlete vs yellow
        #     spectator, white vs white, etc. because the histograms encode
        #     the exact hue/saturation/value distribution of each person.
        neg_factor = 1.0
        if self._neg_refs:
            neg_dists    = [self._zone_dist(nh, cand_hists) for nh in self._neg_refs]
            min_neg_dist = float(np.min(neg_dists))
            neg_sim      = float(np.clip(1.0 - min_neg_dist, 0.0, 1.0))
            neg_factor   = float(np.clip(1.0 - neg_sim * 0.60, 0.40, 1.0))

        return float(np.clip(pos_sim * neg_factor, 0.0, 1.0))

    @property
    def is_ready(self) -> bool:
        return len(self._pos_refs) >= 2


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
    use_cnn_masks:       bool  = False          # bias selection with venue mask_frames
    refine_v2:           bool  = False          # experimental: temporal + safer seeds + sections


# ─── Seed phase ───────────────────────────────────────────────────────────────

def _seed_appearance(
    original_output_dir: str,
    model_seg,
    max_frames: int = SEED_MAX_FRAMES,
    on_progress: ProgressCallback = noop_progress,
    seed_start_frame: Optional[int] = None,
    seed_end_frame:   Optional[int] = None,
    mask_bias: Optional[VenueMaskBias] = None,
    refine_v2: bool = False,
) -> Optional[RobustAppearanceModel]:
    """
    Build an AppearanceModel from the N highest-quality frames of the
    first-pass analysis.  Re-runs YOLO seg (no tracking) to get bboxes.

    If seed_start_frame and seed_end_frame are both set, only frames
    within [start, end] are considered for seeding — useful when the
    user knows a specific interval where detection was clean.
    Frames outside that interval are still used as fallback if the
    interval yields fewer than max_frames candidates.

    When refine_v2: prefer manually_corrected frames as seed positives,
    prefer frames in the mid-window (30–90% of first-pass span) for clothing
    lock, and prefer high track-overlap detections when venue masks are available.
    Still requires ≥2 positive refs (RobustAppearanceModel.is_ready).
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

    # refine_v2 mid-window span over first-pass frame indices (clothing lock seed)
    mid_lo = mid_hi = None
    if refine_v2 and frames_data:
        idxs = [fr.get("frame_idx", 0) for fr in frames_data]
        span_lo, span_hi = min(idxs), max(idxs)
        span = max(span_hi - span_lo, 1)
        mid_lo = span_lo + int(span * V2_MID_WINDOW[0])
        mid_hi = span_lo + int(span * V2_MID_WINDOW[1])

    def _in_mid_window(fr) -> bool:
        if mid_lo is None:
            return False
        return mid_lo <= fr.get("frame_idx", -1) <= mid_hi

    def _quality_key(fr):
        # refine_v2: manually_corrected frames rank above quality alone;
        # mid-window frames preferred next (stable clothing view).
        manual_boost = 10.0 if (refine_v2 and fr.get("manually_corrected")) else 0.0
        mid_boost = 3.0 if (refine_v2 and _in_mid_window(fr)) else 0.0
        return (manual_boost, mid_boost, fr.get("quality_score", 0))

    if use_interval:
        # Priority 1: good frames inside the user-defined interval
        interval_good = sorted(
            [fr for fr in frames_data if _is_good(fr) and _in_interval(fr)],
            key=_quality_key, reverse=True,
        )
        # Priority 2: any detected frame inside the interval
        interval_any = sorted(
            [fr for fr in frames_data if fr.get("person_detected") and _in_interval(fr)
             and fr not in interval_good],
            key=_quality_key, reverse=True,
        )
        combined = (interval_good + interval_any)[:max_frames]
        # If the interval is thin, pad with the best frames from the whole video
        if len(combined) < max_frames:
            rest = sorted(
                [fr for fr in frames_data if _is_good(fr) and fr not in combined],
                key=_quality_key, reverse=True,
            )
            combined = (combined + rest)[:max_frames]
        seed_candidates = combined
        print(f"  [Reanalyzer] Seeding from interval [{seed_start_frame}–{seed_end_frame}]: "
              f"{len(interval_good)} good + {len(interval_any)} any → {len(seed_candidates)} total")
    else:
        usable = sorted(
            [fr for fr in frames_data if _is_good(fr)],
            key=_quality_key, reverse=True,
        )
        seed_candidates = usable[:max_frames]
        if refine_v2 and mid_lo is not None:
            n_mid = sum(1 for fr in seed_candidates if _in_mid_window(fr))
            print(
                f"  [Reanalyzer v2] Mid-window seed bias "
                f"[{mid_lo}–{mid_hi}]: {n_mid}/{len(seed_candidates)} seeds in mid-window"
            )

    # refine_v2: if manually_corrected frames exist, put them first (still need ≥2)
    if refine_v2:
        manual = [
            fr for fr in frames_data
            if fr.get("manually_corrected") and fr.get("person_detected")
        ]
        if manual:
            seen = {fr["frame_idx"] for fr in manual}
            rest = [fr for fr in seed_candidates if fr["frame_idx"] not in seen]
            seed_candidates = (manual + rest)[:max_frames]
            print(f"  [Reanalyzer v2] Preferring {len(manual)} manually_corrected seed frames")

    if not seed_candidates:
        # last-resort fallback
        seed_candidates = sorted(
            [fr for fr in frames_data if fr.get("person_detected")],
            key=_quality_key,
            reverse=True,
        )[:max_frames]

    if not seed_candidates:
        return None

    frames_dir = Path(original_output_dir) / "frames"
    model  = RobustAppearanceModel()
    seeded = 0
    negs   = 0

    for fr in seed_candidates:
        fidx  = fr["frame_idx"]
        fpath = frames_dir / f"frame_{fidx:06d}.jpg"
        if not fpath.exists():
            continue

        img = cv2.imread(str(fpath))
        if img is None:
            continue

        # Run seg (plain, no track)
        results = model_seg(img, classes=[0], conf=0.30, verbose=False)
        if not results or results[0].boxes is None:
            continue

        res    = results[0]
        bboxes = res.boxes.xyxy.cpu().numpy()
        areas  = [(b[2]-b[0])*(b[3]-b[1]) for b in bboxes]
        valid  = [i for i, a in enumerate(areas) if a >= MIN_BBOX_AREA]
        if not valid:
            continue

        h, w = img.shape[:2]

        def _get_mask(idx):
            if res.masks is not None and idx < len(res.masks.data):
                mt = res.masks.data[idx].cpu().numpy()
                return cv2.resize(mt, (w, h),
                                  interpolation=cv2.INTER_NEAREST).astype(bool)
            return None

        # Prefer largest detection; with masks, boost by track/sand overlap
        # refine_v2: stronger preference for high track overlap over pure area
        def _seed_rank(i):
            area = areas[i]
            if mask_bias is None:
                return area
            bbox = tuple(float(v) for v in bboxes[i])
            track_ov, sand_ov = mask_bias.overlaps(fidx, bbox, _get_mask(i), w, h)
            if refine_v2:
                return (
                    area * (1.0 + 1.5 * TRACK_OV_WEIGHT * track_ov
                            + SAND_OV_WEIGHT * sand_ov)
                    + 50_000.0 * track_ov
                )
            return area * (1.0 + TRACK_OV_WEIGHT * track_ov + SAND_OV_WEIGHT * sand_ov)

        best = max(valid, key=_seed_rank)
        bbox = tuple(int(v) for v in bboxes[best])
        model.add_positive(img, _get_mask(best), bbox)
        seeded += 1

        # All other detections in this frame = negative examples
        for j in valid:
            if j != best:
                model.add_negative(img, _get_mask(j), tuple(int(v) for v in bboxes[j]))
                negs += 1

        on_progress({
            "stage":   "seeding",
            "message": f"Semilla {seeded}/{len(seed_candidates)} — frame {fidx}  Q={fr.get('quality_score',0):.2f}",
            "percent": 10.0 + 10.0 * (seeded / len(seed_candidates)),
        })

    if seeded == 0:
        return None

    print(f"  [Reanalyzer] RobustAppearanceModel: {seeded} positivos + {negs} negativos")
    return model


# ─── Second-pass frame analysis ───────────────────────────────────────────────

def _analyze_frame_refined(
    image: np.ndarray,
    frame_idx: int,
    timestamp_s: float,
    model_seg,
    model_pose,
    appearance: RobustAppearanceModel,
    mask_bias: Optional[VenueMaskBias] = None,
) -> FrameAnalysis:
    """
    Run seg (no tracking) → pick best appearance (+ optional venue mask) match → pose.
    Returns a FrameAnalysis with tracking_source = "refined".

    When mask_bias is set:
      score = appearance_sim + 0.30 * track_ov + 0.20 * sand_ov
      still require appearance_sim >= MIN_SIMILARITY_WITH_MASKS (0.20)
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

    min_sim = MIN_SIMILARITY_WITH_MASKS if mask_bias is not None else MIN_SIMILARITY

    # Score each detection by appearance (+ optional venue overlap)
    scores = []
    for i in valid:
        bbox  = tuple(int(v) for v in bboxes[i])
        sim   = appearance.similarity(image, mask=det_masks[i], bbox=bbox)
        track_ov = sand_ov = 0.0
        if mask_bias is not None:
            track_ov, sand_ov = mask_bias.overlaps(
                frame_idx, bbox, det_masks[i], w, h,
            )
            combined = _blend_mask_score(sim, track_ov, sand_ov)
        else:
            combined = sim
        scores.append((i, sim, combined, track_ov, sand_ov))

    # Pick highest combined score among candidates that clear appearance floor
    eligible = [s for s in scores if s[1] >= min_sim]
    if not eligible:
        return empty

    best_i, best_sim, _, _, _ = max(eligible, key=lambda x: x[2])

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

    # Reinforce appearance model from high-confidence frames;
    # also add the rejected detections as new negatives
    det_conf = float(confs[best_i])
    if det_conf >= 0.65 and mask_area > 8000 and best_sim >= 0.50:
        appearance.add_positive(image, seg_mask, (x1, y1, x2, y2))
    for j, sim_j, _, _, _ in scores:
        if j != best_i and best_sim > 0.50:
            bj = tuple(int(v) for v in bboxes[j])
            appearance.add_negative(image, det_masks[j], bj)

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


def _analyze_frame_refined_v2(
    image: np.ndarray,
    frame_idx: int,
    timestamp_s: float,
    model_seg,
    model_pose,
    appearance: RobustAppearanceModel,
    mask_bias: Optional[VenueMaskBias] = None,
    prev_bbox: Optional[tuple[float, float, float, float]] = None,
    frame_progress: float = 0.5,
    bbox_history: Optional[list[tuple[float, float, float, float]]] = None,
) -> tuple[FrameAnalysis, Optional[tuple[float, float, float, float]], bool]:
    """
    Experimental refine_v2 frame pass.

    Same appearance (+ optional mask) scoring as classic, plus:
      - Soft temporal consistency vs prev_bbox (IoU boost; stronger when scores close)
      - Mid-window (30–90%): aggressive online appearance updates (clothing lock)
      - Early / landing: freeze online appearance (no contamination)
      - Landing (≥90%): higher sand weight, lower sim floor, stronger temporal
      - Pose-optional: missing keypoints still emit person_detected + bbox
        (tracking_source refined_v2 / refined_v2_bbox_only)

    Returns (FrameAnalysis, chosen_bbox_or_None, appearance_updated).
    Classic path is unchanged. Gap fill runs separately after the full pass.
    """
    empty = FrameAnalysis(frame_idx=frame_idx, timestamp_s=timestamp_s)
    empty.tracking_source = V2_TRACKING_SRC
    zone = _v2_zone(frame_progress)
    in_landing = zone == "landing"
    in_mid = zone == "mid"

    results = model_seg(image, classes=[0], conf=0.30, verbose=False)
    if not results or results[0].boxes is None:
        return empty, None, False

    res    = results[0]
    bboxes = res.boxes.xyxy.cpu().numpy()
    confs  = res.boxes.conf.cpu().numpy()
    areas  = [(b[2]-b[0])*(b[3]-b[1]) for b in bboxes]
    valid  = [i for i, a in enumerate(areas) if a >= MIN_BBOX_AREA]
    if not valid:
        return empty, None, False

    h, w = image.shape[:2]
    det_masks = []
    for i in range(len(bboxes)):
        if res.masks is not None and i < len(res.masks.data):
            mt = res.masks.data[i].cpu().numpy()
            m  = cv2.resize(mt, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
            det_masks.append(m)
        else:
            det_masks.append(None)

    if in_landing:
        min_sim = (
            V2_LANDING_MIN_SIM if mask_bias is not None
            else V2_LANDING_MIN_SIM_NO_MASK
        )
        sand_w = V2_LANDING_SAND_WEIGHT
    else:
        min_sim = MIN_SIMILARITY_WITH_MASKS if mask_bias is not None else MIN_SIMILARITY
        sand_w = SAND_OV_WEIGHT

    # (i, sim, combined, track_ov, sand_ov, bbox)
    scores = []
    for i in valid:
        bbox  = tuple(float(v) for v in bboxes[i])
        sim   = appearance.similarity(image, mask=det_masks[i], bbox=tuple(int(v) for v in bbox))
        track_ov = sand_ov = 0.0
        if mask_bias is not None:
            track_ov, sand_ov = mask_bias.overlaps(
                frame_idx, bbox, det_masks[i], w, h,
            )
            combined = _blend_mask_score(sim, track_ov, sand_ov, sand_weight=sand_w)
            if in_landing and sand_ov > 0.0:
                combined += V2_LANDING_SAND_PREFER * sand_ov
        else:
            combined = sim
        scores.append((i, sim, combined, track_ov, sand_ov, bbox))

    eligible = [s for s in scores if s[1] >= min_sim]
    if not eligible:
        return empty, None, False

    # Temporal soft blend: boost candidates overlapping previous chosen bbox(es).
    # Landing: wider history + stronger weight (athlete close/blurred in sand).
    history: list[tuple[float, float, float, float]] = []
    if bbox_history:
        history = list(bbox_history)
    elif prev_bbox is not None:
        history = [prev_bbox]

    if history:
        ranked = sorted(eligible, key=lambda x: x[2], reverse=True)
        top_gap = (
            ranked[0][2] - ranked[1][2] if len(ranked) >= 2 else 1.0
        )
        if in_landing:
            temporal_w = V2_TEMPORAL_BOOST_LANDING
            hist_refs = history[-V2_LANDING_TEMPORAL_HISTORY:]
        else:
            temporal_w = V2_TEMPORAL_IOU_WEIGHT
            hist_refs = history[-1:]
            if top_gap <= V2_TEMPORAL_CLOSE_GAP:
                temporal_w = V2_TEMPORAL_IOU_WEIGHT * 2.0

        def _final_score(s):
            iou = max((_bbox_iou(s[5], ref) for ref in hist_refs), default=0.0)
            return s[2] + temporal_w * iou

        best_i, best_sim, _, best_track_ov, best_sand_ov, _ = max(
            eligible, key=_final_score,
        )
    else:
        best_i, best_sim, _, best_track_ov, best_sand_ov, _ = max(
            eligible, key=lambda x: x[2],
        )

    x1, y1, x2, y2 = (int(v) for v in bboxes[best_i])
    seg_mask  = det_masks[best_i]
    mask_area = 0

    if seg_mask is not None:
        mask_area = int(seg_mask.sum())
        ys, xs = np.where(seg_mask)
        if len(xs) > 0:
            x1, x2 = int(xs.min()), int(xs.max())
            y1, y2 = int(ys.min()), int(ys.max())

    chosen_bbox: tuple[float, float, float, float] = (
        float(x1), float(y1), float(x2), float(y2),
    )

    # Online appearance update (before pose — identity lock does not need keypoints):
    #   mid-window — aggressive (clothing lock)
    #   early / landing — freeze (no contamination from start or sand close-ups)
    appearance_updated = False
    det_conf = float(confs[best_i])
    if in_mid:
        min_conf = V2_MID_ONLINE_MIN_CONF
        min_sim_online = V2_MID_ONLINE_MIN_SIM
        min_mask = V2_MID_ONLINE_MIN_MASK_AREA
        min_track = V2_MID_ONLINE_MIN_TRACK_OV
        venue_ok = True
        if mask_bias is not None:
            venue_ok = (
                best_track_ov >= min_track
                or best_sand_ov >= min_track
            )
        if (
            det_conf >= min_conf
            and mask_area > min_mask
            and best_sim >= min_sim_online
            and venue_ok
        ):
            appearance.add_positive(image, seg_mask, (x1, y1, x2, y2))
            appearance_updated = True
        for j, sim_j, _, _, _, _ in scores:
            if j != best_i and best_sim > min_sim_online:
                bj = tuple(int(v) for v in bboxes[j])
                appearance.add_negative(image, det_masks[j], bj)
    # early + landing: deliberately skip online updates

    # Pose-optional: keep person_detected + bbox even when keypoints missing
    fa = _v2_build_frame_analysis(
        image, frame_idx, timestamp_s, model_pose,
        (x1, y1, x2, y2), seg_mask, det_conf, float(best_sim),
    )
    return fa, chosen_bbox, appearance_updated


# ─── Full second-pass pipeline ────────────────────────────────────────────────

def run_reanalysis(
    config: ReanalysisConfig,
    on_progress: ProgressCallback = noop_progress,
) -> dict:
    """
    Full second-pass pipeline.  Returns summary dict.
    Never touches the original output directory.

    When config.refine_v2 is False, behavior matches the classic appearance-only
    refine (same thresholds, online update, no temporal smoothing, no section pass).
    """
    out_dir       = Path(config.output_dir)
    out_frames    = out_dir / "frames"
    out_annotated = out_dir / "annotated"
    out_charts    = out_dir / "charts"
    # Gate de escritura opt-in (defaults TRUE → mismo comportamiento que hoy)
    _persist_frames  = opt_flags.persist_frames()
    _write_annotated = opt_flags.write_annotated()
    _dirs = [out_charts]
    if _persist_frames:
        _dirs.append(out_frames)
    if _write_annotated:
        _dirs.append(out_annotated)
    for d in _dirs:
        d.mkdir(parents=True, exist_ok=True)

    # ── Load models ───────────────────────────────────────────────────────────
    on_progress({"stage": "loading_models", "message": "Cargando modelos YOLO", "percent": 2.0})
    from ultralytics import YOLO
    t0         = time.time()
    model_pose = YOLO("yolo11s-pose.pt")
    model_seg  = YOLO("yolo11s-seg.pt")
    on_progress({"stage": "loading_models", "message": f"Modelos listos ({time.time()-t0:.1f}s)", "percent": 8.0})

    # ── Optional venue masks from original calibration (not _refined) ─────────
    # Mask policy:
    #   classic: masks only when use_cnn_masks is True
    #   refine_v2: if masks exist on original, use them for seed bias, temporal
    #              consistency, AND selection blend (0.30/0.20) — even when
    #              use_cnn_masks is False. use_cnn_masks alone still enables
    #              masks on the classic path.
    mask_bias: Optional[VenueMaskBias] = None
    if config.use_cnn_masks:
        mask_bias = VenueMaskBias.from_output_dir(config.original_output_dir)
        if mask_bias is None:
            print("  [Reanalyzer] use_cnn_masks requested but masks unavailable — appearance-only")
    elif config.refine_v2:
        mask_bias = VenueMaskBias.from_output_dir(config.original_output_dir)
        if mask_bias is None:
            print("  [Reanalyzer v2] No venue masks on original — appearance + temporal only")
        else:
            print("  [Reanalyzer v2] Using existing venue masks (auto when present)")
    else:
        print("  [Reanalyzer] Venue masks OFF (appearance-only refine)")

    # ── Seed AppearanceModel from first-pass best frames ──────────────────────
    on_progress({"stage": "seeding", "message": "Sembrando modelo de apariencia...", "percent": 10.0})
    appearance = _seed_appearance(
        config.original_output_dir, model_seg,
        max_frames=config.seed_frames,
        on_progress=on_progress,
        seed_start_frame=config.seed_start_frame,
        seed_end_frame=config.seed_end_frame,
        mask_bias=mask_bias,
        refine_v2=config.refine_v2,
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

    # Precompute how many analysis frames this refine run will process
    # (progress = index / (n-1) over this list — mid-window / landing zones).
    analysis_frame_indices = list(range(start_f, end_f, config.stride))
    n_analysis_planned = len(analysis_frame_indices)

    # ── Second pass ───────────────────────────────────────────────────────────
    analyses: list[FrameAnalysis] = []
    analysis_count  = 0
    annotated_count = 0
    frame_abs       = start_f
    prev_bbox: Optional[tuple[float, float, float, float]] = None
    bbox_history: list[tuple[float, float, float, float]] = []

    # refine_v2 mid-window / landing stats
    mid_lock_updates = 0
    mid_lock_accepted = 0
    mid_lock_logged = False
    landing_mode_frames = 0
    refs_at_mid_start = 0
    entered_mid = False

    if config.refine_v2:
        print(
            f"  [Reanalyzer v2] Mid-window lock {V2_MID_WINDOW[0]:.0%}–{V2_MID_WINDOW[1]:.0%}; "
            f"landing tail ≥{V2_LANDING_TAIL:.0%} "
            f"({n_analysis_planned} analysis frames planned)"
        )

    cap = cv2.VideoCapture(config.video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)

    while frame_abs < end_f:
        ret, frame = cap.read()
        if not ret:
            break

        is_analysis_frame = (frame_abs - start_f) % config.stride == 0

        if is_analysis_frame:
            ts = frame_abs / fps
            if config.refine_v2:
                # Progress through refine analysis frames (not wall-clock)
                if n_analysis_planned <= 1:
                    frame_progress = 0.5
                else:
                    frame_progress = analysis_count / (n_analysis_planned - 1)
                zone = _v2_zone(frame_progress)

                fa, chosen_bbox, app_updated = _analyze_frame_refined_v2(
                    image=frame,
                    frame_idx=frame_abs,
                    timestamp_s=ts,
                    model_seg=model_seg,
                    model_pose=model_pose,
                    appearance=appearance,
                    mask_bias=mask_bias,
                    prev_bbox=prev_bbox,
                    frame_progress=frame_progress,
                    bbox_history=bbox_history,
                )
                if chosen_bbox is not None:
                    prev_bbox = chosen_bbox
                    bbox_history.append(chosen_bbox)
                    if len(bbox_history) > V2_LANDING_TEMPORAL_HISTORY:
                        bbox_history = bbox_history[-V2_LANDING_TEMPORAL_HISTORY:]

                if zone == "mid":
                    if not entered_mid:
                        entered_mid = True
                        refs_at_mid_start = len(appearance._pos_refs)
                    if fa.person_detected:
                        mid_lock_accepted += 1
                    if app_updated:
                        mid_lock_updates += 1
                elif zone == "landing":
                    landing_mode_frames += 1

                # Once mid-window ends, freeze model and log lock stats once
                if (
                    not mid_lock_logged
                    and frame_progress >= V2_LANDING_TAIL
                ):
                    mid_lock_logged = True
                    n_refs = len(appearance._pos_refs)
                    print(
                        f"  [Reanalyzer v2] Mid-window clothing lock: "
                        f"{mid_lock_accepted} accepted, {mid_lock_updates} appearance updates, "
                        f"refs {refs_at_mid_start}→{n_refs} (frozen for landing)"
                    )
            else:
                fa = _analyze_frame_refined(
                    image=frame,
                    frame_idx=frame_abs,
                    timestamp_s=ts,
                    model_seg=model_seg,
                    model_pose=model_pose,
                    appearance=appearance,
                    mask_bias=mask_bias,
                )
            analyses.append(fa)
            analysis_count += 1

            fpath = out_frames / f"frame_{frame_abs:06d}.jpg"
            if _persist_frames:
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

            if _write_annotated and analysis_count % config.annotate_every == 0:
                out_img = out_annotated / f"annotated_{frame_abs:06d}.jpg"
                if _persist_frames:
                    annotate_frame(str(fpath), fa, str(out_img),
                                   seg_mask=None, appearance_sim=0.0)
                else:
                    ann_img = annotate_frame_array(frame.copy(), fa,
                                                   seg_mask=None, appearance_sim=0.0)
                    out_img.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(out_img), ann_img)
                annotated_count += 1

        frame_abs += 1

    cap.release()

    if config.refine_v2:
        if not mid_lock_logged:
            n_refs = len(appearance._pos_refs)
            print(
                f"  [Reanalyzer v2] Mid-window clothing lock: "
                f"{mid_lock_accepted} accepted, {mid_lock_updates} appearance updates, "
                f"refs {refs_at_mid_start}→{n_refs}"
            )
        print(
            f"  [Reanalyzer v2] Landing-mode frames: {landing_mode_frames} "
            f"(sand_w={V2_LANDING_SAND_WEIGHT}, temporal={V2_TEMPORAL_BOOST_LANDING})"
        )

        # Post-pass: propagate mid identity into early/landing unknowns
        on_progress({
            "stage": "gap_fill",
            "message": "Propagando identidad mid → early/landing…",
            "percent": 90.0,
        })
        gap_stats = _v2_gap_fill_pass(
            analyses=analyses,
            frames_dir=out_frames,
            model_seg=model_seg,
            model_pose=model_pose,
            appearance=appearance,
            mask_bias=mask_bias,
            annotate_every=config.annotate_every,
            annotated_dir=out_annotated,
        )
        print(
            f"  [Reanalyzer v2] Identity summary: "
            f"bbox_only={gap_stats['bbox_only']} "
            f"(tracking_source={V2_TRACKING_SRC_BBOX!r})"
        )

    # ── Summary + JSON ────────────────────────────────────────────────────────
    on_progress({"stage": "writing_outputs", "message": "Escribiendo resultados refinados", "percent": 93.0})

    summary = _summarize_refined(analyses, analysis_count)

    if config.refine_v2:
        frames_data = [
            frame_analysis_to_dict(
                a,
                appearance_sim=float(getattr(a, "_appearance_sim", 0.0) or 0.0),
            )
            for a in analyses
        ]
    else:
        frames_data = [
            frame_analysis_to_dict(a, extra={"tracking_source": "refined"})
            for a in analyses
        ]

    result_data = build_analysis_document(
        video_path=config.video_path,
        video_info=info,
        config={
            "stride":         config.stride,
            "start_sec":      config.start_sec,
            "end_sec":        config.end_sec,
            "use_cnn_masks":  bool(mask_bias is not None),
            "refine_v2":      bool(config.refine_v2),
        },
        summary=summary,
        frames=frames_data,
        output_dir=out_dir,
        analysis_pass="refined",
    )

    write_derived_stubs(out_dir)

    json_path = out_dir / "analysis.json"
    with open(json_path, "w") as f:
        json.dump(result_data, f, indent=2)

    # refine_v2 only: copy venue calibration/masks + regenerate masks for
    # every refined analysis frame (stride may differ from first-pass) + sections
    if config.refine_v2:
        on_progress({
            "stage": "sections",
            "message": "Copiando calibración y analizando secciones…",
            "percent": 96.0,
        })
        try:
            _copy_venue_assets_for_sections(
                Path(config.original_output_dir), out_dir,
            )
        except Exception as exc:
            print(f"  [Reanalyzer v2] Warning: could not copy venue assets: {exc}")

        on_progress({
            "stage": "venue_masks",
            "message": "Regenerando máscaras de pista para frames refinados…",
            "percent": 97.0,
        })
        try:
            if _reapply_venue_masks_for_refined(out_dir, Path(config.video_path)):
                print(
                    f"  [Reanalyzer v2] venue_masks re-applied → {out_dir.name}/"
                )
        except Exception as exc:
            print(
                f"  [Reanalyzer v2] Warning: venue mask re-apply failed "
                f"(keeping copied masks): {exc}"
            )

        try:
            from .section_analyzer import run_section_analysis
            sections = run_section_analysis(out_dir, use_pose=True)
            n_hops = len((sections.get("hops") or []))
            print(f"  [Reanalyzer v2] Section analysis OK ({n_hops} hops)")
        except Exception as exc:
            print(f"  [Reanalyzer v2] Warning: section analysis failed (refine OK): {exc}")

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
