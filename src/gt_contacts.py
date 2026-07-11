"""
Durable ground-truth hop/landing contact store.

Survives video output deletion. Used to rebuild pose_tag prototypes and
improve automatic contact detection.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .pose_features import FEATURE_NAMES, extract_pose_features

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GT_PATH = PROJECT_ROOT / "data" / "gt_contacts.json"
PROTOTYPES_PATH = PROJECT_ROOT / "data" / "phase_prototypes.json"
OUTPUT_ROOT = PROJECT_ROOT / "output"

QUALITY_MIN = 0.35
USABLE_ANGLES = frozenset({"LATERAL", "SEMI_BACK"})

# Seed definitions: preferred contact frames per video (snap to nearest analyzed).
# Phases are hop_1..hop_4 or landing. Mid-of-range preferred when stride skips.
SEED_GT: dict[str, list[dict[str, Any]]] = {
    "VOD2": [
        {"frame_idx": 60, "phase": "hop_1", "surface": "track"},
        {"frame_idx": 75, "phase": "hop_2", "surface": "track"},
        {"frame_idx": 90, "phase": "hop_3", "surface": "track"},
        {"frame_idx": 108, "phase": "hop_4", "surface": "track"},
        {"frame_idx": 144, "phase": "landing", "surface": "sand"},
    ],
    "VOD3": [
        {"frame_idx": 379, "phase": "hop_1", "surface": "track"},
        {"frame_idx": 400, "phase": "hop_2", "surface": "track"},
        {"frame_idx": 409, "phase": "hop_3", "surface": "track"},
        {"frame_idx": 424, "phase": "hop_4", "surface": "track"},
        {"frame_idx": 454, "phase": "landing", "surface": "sand"},
    ],
    "VOD4": [
        {"frame_idx": 90, "phase": "hop_1", "surface": "track"},
        {"frame_idx": 106, "phase": "hop_2", "surface": "track"},  # mid 105-108
        {"frame_idx": 123, "phase": "hop_3", "surface": "track"},
        {"frame_idx": 139, "phase": "hop_4", "surface": "track"},  # mid 138-141
        {"frame_idx": 177, "phase": "landing", "surface": "sand"},
    ],
    "VOD5": [
        {"frame_idx": 1063, "phase": "hop_1", "surface": "track"},
        {"frame_idx": 1076, "phase": "hop_2", "surface": "track"},  # mid 1075-1078
        {"frame_idx": 1090, "phase": "hop_3", "surface": "track"},
        {"frame_idx": 1126, "phase": "landing", "surface": "sand"},  # only 3 hops known
    ],
    "VOD6": [
        {"frame_idx": 589, "phase": "hop_1", "surface": "track"},  # mid 588-591
        {"frame_idx": 604, "phase": "hop_2", "surface": "track"},
        {"frame_idx": 619, "phase": "hop_3", "surface": "track"},
        {"frame_idx": 634, "phase": "hop_4", "surface": "track"},
        {"frame_idx": 675, "phase": "landing", "surface": "sand"},
    ],
    "VOD7": [
        {"frame_idx": 641, "phase": "hop_1", "surface": "track"},
        {"frame_idx": 658, "phase": "hop_2", "surface": "track"},  # mid 657-659
        {"frame_idx": 671, "phase": "hop_3", "surface": "track"},
        {"frame_idx": 688, "phase": "hop_4", "surface": "track"},  # mid 687-689
        {"frame_idx": 725, "phase": "landing", "surface": "sand"},  # 727 unusable
    ],
    "VOD9": [
        # Time order: 320, 332, 348, 358 — hops only (final unclear)
        {"frame_idx": 320, "phase": "hop_1", "surface": "track"},
        {"frame_idx": 332, "phase": "hop_2", "surface": "track"},
        {"frame_idx": 348, "phase": "hop_3", "surface": "track"},
        {"frame_idx": 358, "phase": "hop_4", "surface": "track"},
    ],
}

# Prefer refined analysis when present and higher quality at GT frames.
VIDEO_ANALYSIS_CANDIDATES: dict[str, list[str]] = {
    "VOD2": ["VOD2"],
    "VOD3": ["VOD3"],
    "VOD4": ["VOD4"],
    "VOD5": ["VOD5"],
    "VOD6": ["VOD6"],
    "VOD7": ["VOD7"],
    "VOD9": ["VOD9_refined", "VOD9"],
}

ATHLETE_BY_VIDEO: dict[str, str] = {
    "VOD2": "Mateo",
    "VOD3": "Mateo",
    "VOD4": "Mateo",
    "VOD5": "Mateo",
    "VOD6": "Mateo",
    "VOD7": "Mateo",
    "VOD9": "Mateo",
}


def empty_gt_store() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "description": "Durable hop/landing contact ground truth for pose prototypes.",
        "samples": [],
    }


def load_gt_contacts(path: Optional[Path] = None) -> dict[str, Any]:
    p = path or GT_PATH
    if not p.exists():
        return empty_gt_store()
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    if "samples" not in data:
        data["samples"] = []
    return data


def save_gt_contacts(data: dict[str, Any], path: Optional[Path] = None) -> Path:
    p = path or GT_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return p


def _sample_key(video_id: str, frame_idx: int, phase: str) -> tuple[str, int, str]:
    return (video_id, int(frame_idx), phase)


def upsert_gt_sample(
    sample: dict[str, Any],
    *,
    path: Optional[Path] = None,
    rebuild_prototypes: bool = False,
    unique_phase_per_video: bool = False,
) -> dict[str, Any]:
    """Insert or replace sample keyed by video_id+frame_idx+phase."""
    data = load_gt_contacts(path)
    key = _sample_key(sample["video_id"], sample["frame_idx"], sample["phase"])
    samples = data["samples"]

    if unique_phase_per_video:
        # Keep one contact per (video, phase) — used for manual corrections
        samples = [
            s for s in samples
            if not (
                s.get("video_id") == sample["video_id"]
                and s.get("phase") == sample["phase"]
            )
        ]

    replaced = False
    for i, s in enumerate(samples):
        if _sample_key(s["video_id"], s["frame_idx"], s["phase"]) == key:
            samples[i] = sample
            replaced = True
            break
    if not replaced:
        samples.append(sample)
    samples.sort(key=lambda s: (s.get("video_id", ""), int(s.get("frame_idx", 0))))
    data["samples"] = samples
    save_gt_contacts(data, path)
    if rebuild_prototypes:
        rebuild_pose_tag_prototypes_from_gt(data)
    return data


def append_gt_from_marker(
    video_id: str,
    frame: dict,
    phase: str,
    *,
    pose_tag: Optional[str] = None,
    athlete_id: Optional[str] = None,
    surface: Optional[str] = None,
    source: str = "manual_mark",
    rebuild_prototypes: bool = True,
) -> Optional[dict[str, Any]]:
    """Persist a user-confirmed hop/landing marker into the GT store."""
    if phase not in ("hop_1", "hop_2", "hop_3", "hop_4", "landing"):
        return None
    if not frame.get("person_detected"):
        return None

    feat = extract_pose_features(frame)
    if not feat.valid or feat.quality < QUALITY_MIN:
        return None

    tag = pose_tag or ("landing" if phase == "landing" else "hop_contact")
    if phase == "landing" and tag == "hop_contact":
        tag = "landing"
    surf = surface or ("sand" if phase == "landing" else "track")

    sample = {
        "athlete_id": athlete_id or ATHLETE_BY_VIDEO.get(video_id),
        "video_id": video_id,
        "frame_idx": int(frame["frame_idx"]),
        "phase": phase,
        "pose_tag": tag,
        "surface": surf,
        "feature_vector": [round(float(v), 4) for v in feat.vector],
        "quality_score": round(float(feat.quality), 3),
        "camera_angle": frame.get("camera_angle"),
        "source": source,
    }
    upsert_gt_sample(
        sample,
        rebuild_prototypes=rebuild_prototypes,
        unique_phase_per_video=True,
    )
    return sample


def _snap_frame(frames: list[dict], target: int) -> Optional[dict]:
    if not frames:
        return None
    fmap = {int(f["frame_idx"]): f for f in frames if "frame_idx" in f}
    if target in fmap:
        return fmap[target]
    idxs = sorted(fmap)
    if not idxs:
        return None
    nearest = min(idxs, key=lambda x: abs(x - target))
    # Reject if snap drifted too far (e.g. sparse stride)
    if abs(nearest - target) > 6:
        return None
    return fmap[nearest]


def _frame_quality_score(frame: dict) -> float:
    if not frame.get("person_detected"):
        return 0.0
    score = 0.4
    angle = frame.get("camera_angle")
    if angle in USABLE_ANGLES:
        score += 0.35
    if frame.get("usable_for_analysis"):
        score += 0.15
    if angle == "LATERAL":
        score += 0.1
    feat = extract_pose_features(frame)
    if feat.valid:
        score += 0.2 * feat.quality
    return score


def _prefer_analysis_dir(video_id: str) -> Optional[Path]:
    """Pick best analysis among candidates (prefer refined / LATERAL quality)."""
    candidates = VIDEO_ANALYSIS_CANDIDATES.get(video_id, [video_id])
    seeds = SEED_GT.get(video_id) or []
    best_dir: Optional[Path] = None
    best_score = -1.0

    for name in candidates:
        d = OUTPUT_ROOT / name
        ap = d / "analysis.json"
        if not ap.exists():
            continue
        with open(ap, encoding="utf-8") as f:
            data = json.load(f)
        frames = data.get("frames") or []
        if not frames:
            continue
        q = 0.0
        n = 0
        for seed in seeds:
            fr = _snap_frame(frames, int(seed["frame_idx"]))
            if fr is None:
                continue
            q += _frame_quality_score(fr)
            n += 1
        # Prefer refined name slightly when quality ties
        score = (q / max(n, 1)) + (0.05 if "refined" in name else 0.0)
        if n == 0:
            score = 0.01
        if score > best_score:
            best_score = score
            best_dir = d

    return best_dir


def _pose_tag_for_phase(phase: str) -> str:
    if phase == "landing":
        return "landing"
    return "hop_contact"


def seed_gt_contacts_from_analysis(
    *,
    output_root: Optional[Path] = None,
    gt_path: Optional[Path] = None,
    rebuild: bool = True,
) -> dict[str, Any]:
    """
    Populate data/gt_contacts.json from SEED_GT + available analysis.json files.
    Idempotent upsert. Skips missing videos with a log message.
    """
    global OUTPUT_ROOT
    if output_root is not None:
        OUTPUT_ROOT = Path(output_root)

    data = load_gt_contacts(gt_path)
    counts: dict[str, int] = {}
    skipped: list[str] = []

    for video_id, seeds in SEED_GT.items():
        analysis_dir = _prefer_analysis_dir(video_id)
        if analysis_dir is None:
            msg = f"Skip {video_id}: no analysis.json found"
            logger.warning(msg)
            print(msg)
            skipped.append(video_id)
            counts[video_id] = 0
            continue

        # Drop previous seed_gt samples for this video (re-seed idempotent by phase)
        data = load_gt_contacts(gt_path)
        data["samples"] = [
            s for s in data["samples"]
            if not (s.get("video_id") == video_id and s.get("source") == "seed_gt")
        ]
        save_gt_contacts(data, gt_path)

        with open(analysis_dir / "analysis.json", encoding="utf-8") as f:
            analysis = json.load(f)
        frames = analysis.get("frames") or []
        athlete = ATHLETE_BY_VIDEO.get(video_id)

        # Merge manual markers from sections.json when present (VOD2 etc.)
        sections_path = analysis_dir / "sections.json"
        marker_overrides: dict[str, int] = {}
        if sections_path.exists():
            with open(sections_path, encoding="utf-8") as f:
                sections = json.load(f)
            for m in sections.get("phase_markers") or []:
                phase = m.get("phase")
                if phase in ("hop_1", "hop_2", "hop_3", "hop_4", "landing"):
                    if m.get("source") in ("manual", "propagated") or m.get("confidence", 0) >= 0.9:
                        marker_overrides[phase] = int(m["frame_idx"])
            if sections.get("athlete_id"):
                athlete = sections["athlete_id"]

        n_ok = 0
        for seed in seeds:
            phase = seed["phase"]
            candidates = []
            if phase in marker_overrides:
                candidates.append(marker_overrides[phase])
            candidates.append(int(seed["frame_idx"]))
            # Deduplicate while preserving order
            seen_t: set[int] = set()
            ordered_targets = []
            for t in candidates:
                if t not in seen_t:
                    seen_t.add(t)
                    ordered_targets.append(t)

            frame = None
            for target in ordered_targets:
                cand = _snap_frame(frames, target)
                if cand is None:
                    continue
                if not cand.get("person_detected"):
                    fmap = {int(f["frame_idx"]): f for f in frames}
                    for d in range(1, 4):
                        for alt_i in (target - d, target + d):
                            fr = fmap.get(alt_i)
                            if fr and fr.get("person_detected"):
                                cand = fr
                                break
                        else:
                            continue
                        break
                if not cand.get("person_detected"):
                    continue
                feat = extract_pose_features(cand)
                if feat.valid:
                    frame = cand
                    break
                # Marker unusable — try next candidate (seed frame)
                print(
                    f"  {video_id}: unusable at {cand['frame_idx']} "
                    f"({phase}), trying fallback"
                )

            if frame is None:
                print(f"  {video_id}: no usable frame for {phase} near {ordered_targets}")
                continue

            feat = extract_pose_features(frame)
            if not feat.valid:
                print(f"  {video_id}: unusable keypoints at {frame['frame_idx']} ({phase})")
                continue
            if feat.quality < QUALITY_MIN:
                print(
                    f"  {video_id}: low quality {feat.quality} at "
                    f"{frame['frame_idx']} ({phase}) — keeping anyway for seed"
                )

            sample = {
                "athlete_id": athlete,
                "video_id": video_id,
                "frame_idx": int(frame["frame_idx"]),
                "phase": phase,
                "pose_tag": _pose_tag_for_phase(phase),
                "surface": seed.get("surface", "unknown"),
                "feature_vector": [round(float(v), 4) for v in feat.vector],
                "quality_score": round(float(feat.quality), 3),
                "camera_angle": frame.get("camera_angle"),
                "source": "seed_gt",
                "analysis_dir": analysis_dir.name,
            }
            upsert_gt_sample(
                sample,
                path=gt_path,
                rebuild_prototypes=False,
                unique_phase_per_video=True,
            )
            n_ok += 1

        counts[video_id] = n_ok
        print(f"{video_id}: seeded {n_ok}/{len(seeds)} from {analysis_dir.name}")

    data = load_gt_contacts(gt_path)
    if rebuild:
        rebuild_pose_tag_prototypes_from_gt(data)

    return {"counts": counts, "skipped": skipped, "total": len(data.get("samples") or [])}


def rebuild_pose_tag_prototypes_from_gt(
    gt_data: Optional[dict[str, Any]] = None,
    *,
    quality_threshold: float = QUALITY_MIN,
) -> dict[str, Any]:
    """
    Update phase_prototypes.json pose_tags.hop_contact / landing / feet_together
    as mean of GT feature vectors. Keeps phase centroids unchanged.
    """
    gt = gt_data or load_gt_contacts()
    samples = [
        s for s in (gt.get("samples") or [])
        if float(s.get("quality_score", 0)) >= quality_threshold
        and s.get("feature_vector")
    ]

    buckets: dict[str, list[np.ndarray]] = {
        "hop_contact": [],
        "landing": [],
        "feet_together": [],
    }
    for s in samples:
        tag = s.get("pose_tag") or _pose_tag_for_phase(s.get("phase", ""))
        if tag == "landing":
            buckets["landing"].append(np.array(s["feature_vector"], dtype=float))
        elif tag in ("hop_contact", "feet_together"):
            buckets[tag].append(np.array(s["feature_vector"], dtype=float))
        elif s.get("phase") in ("hop_1", "hop_2", "hop_3", "hop_4"):
            buckets["hop_contact"].append(np.array(s["feature_vector"], dtype=float))

    prototypes = {}
    if PROTOTYPES_PATH.exists():
        with open(PROTOTYPES_PATH, encoding="utf-8") as f:
            prototypes = json.load(f)
    if not prototypes:
        prototypes = {"schema_version": 1, "phases": {}, "pose_tags": {}}
    tags = prototypes.setdefault("pose_tags", {})

    notes: list[str] = []
    for tag, vecs in buckets.items():
        if len(vecs) < 2 and tag != "hop_contact":
            continue
        if not vecs:
            continue
        mean = np.mean(np.stack(vecs, axis=0), axis=0)
        existing = tags.get(tag) or {}
        weights = existing.get("weights") or [1.0] * len(FEATURE_NAMES)
        tags[tag] = {
            "weights": weights,
            "centroid": [round(float(v), 4) for v in mean],
            "sample_count": len(vecs),
            "feature_names": list(FEATURE_NAMES),
            "updated_from": "gt_contacts",
        }
        notes.append(f"{tag}={len(vecs)}")

    # hop_1..4 share contact pose — optionally sync hop phase centroids lightly
    hop_vecs = buckets["hop_contact"]
    if hop_vecs:
        hop_mean = np.mean(np.stack(hop_vecs, axis=0), axis=0)
        phases = prototypes.setdefault("phases", {})
        for hop in ("hop_1", "hop_2", "hop_3", "hop_4"):
            entry = phases.get(hop) or {}
            # Keep existing weights; note shared contact pose
            phases[hop] = {
                **entry,
                "centroid": [round(float(v), 4) for v in hop_mean],
                "sample_count": len(hop_vecs),
                "note": "Shares hop_contact pose prototype from GT",
                "feature_names": list(FEATURE_NAMES),
            }

    prototypes["gt_rebuild_note"] = (
        f"pose_tags rebuilt from GT ({', '.join(notes)}). "
        "hop_1..4 share contact pose."
    )

    with open(PROTOTYPES_PATH, "w", encoding="utf-8") as f:
        json.dump(prototypes, f, indent=2)

    print(f"Rebuilt prototypes: {', '.join(notes) or 'none'}")
    return prototypes


def score_pose_tag(
    feature_vector: list[float],
    pose_tag: str,
) -> float:
    """Similarity 0–1 to a pose_tag prototype (hop_contact / landing)."""
    if not feature_vector:
        return 0.0
    if not PROTOTYPES_PATH.exists():
        return 0.0
    with open(PROTOTYPES_PATH, encoding="utf-8") as f:
        prototypes = json.load(f)
    entry = (prototypes.get("pose_tags") or {}).get(pose_tag)
    if not entry or not entry.get("centroid"):
        return 0.0
    vec = np.array(feature_vector, dtype=float)
    centroid = np.array(entry["centroid"], dtype=float)
    weights = np.array(entry.get("weights", [1.0] * len(centroid)), dtype=float)
    n = min(len(vec), len(centroid), len(weights))
    diff = vec[:n] - centroid[:n]
    dist = float(np.sqrt(np.sum(weights[:n] * diff * diff)))
    return float(min(1.0, np.exp(-dist * 3.5)))
