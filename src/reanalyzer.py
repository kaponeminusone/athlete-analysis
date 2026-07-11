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
from .visualizer import annotate_frame
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


def _blend_mask_score(appearance_sim: float, track_ov: float, sand_ov: float) -> float:
    """Combine appearance with venue overlap. score = sim + 0.30*track + 0.20*sand."""
    return float(appearance_sim + TRACK_OV_WEIGHT * track_ov + SAND_OV_WEIGHT * sand_ov)


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


# ─── Seed phase ───────────────────────────────────────────────────────────────

def _seed_appearance(
    original_output_dir: str,
    model_seg,
    max_frames: int = SEED_MAX_FRAMES,
    on_progress: ProgressCallback = noop_progress,
    seed_start_frame: Optional[int] = None,
    seed_end_frame:   Optional[int] = None,
    mask_bias: Optional[VenueMaskBias] = None,
) -> Optional[RobustAppearanceModel]:
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
        def _seed_rank(i):
            area = areas[i]
            if mask_bias is None:
                return area
            bbox = tuple(float(v) for v in bboxes[i])
            track_ov, sand_ov = mask_bias.overlaps(fidx, bbox, _get_mask(i), w, h)
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

    # ── Optional venue masks from original calibration (not _refined) ─────────
    mask_bias: Optional[VenueMaskBias] = None
    if config.use_cnn_masks:
        mask_bias = VenueMaskBias.from_output_dir(config.original_output_dir)
        if mask_bias is None:
            print("  [Reanalyzer] use_cnn_masks requested but masks unavailable — appearance-only")
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
                mask_bias=mask_bias,
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
