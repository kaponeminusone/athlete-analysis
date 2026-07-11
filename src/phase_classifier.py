"""
Generic + per-athlete pose phase classification.

Layer 1: biomechanical prototypes (data/phase_prototypes.json)
Layer 2: athlete-specific templates (athletes/<id>/pose_templates.json)

Supports backward propagation from manual anchors (e.g. final_jump → hops).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .pose_features import FEATURE_NAMES, PoseFeatures, extract_features_for_frames
from .schemas import SECTION_PHASES

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROTOTYPES_PATH = PROJECT_ROOT / "data" / "phase_prototypes.json"
ATHLETES_ROOT = PROJECT_ROOT / "athletes"

HOP_PHASES = ("hop_1", "hop_2", "hop_3", "hop_4")
CONTACT_PHASES = (*HOP_PHASES, "landing")
ANCHOR_SEARCH_WINDOW = 45
ANCHOR_MIN_GAP = 10
TEMPLATE_BLEND = 0.35  # weight for athlete template vs generic


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_prototypes() -> dict[str, Any]:
    return _load_json(PROTOTYPES_PATH)


def load_athlete_templates(athlete_id: Optional[str]) -> dict[str, Any]:
    if not athlete_id:
        return {}
    path = ATHLETES_ROOT / athlete_id / "pose_templates.json"
    return _load_json(path)


def _prototype_entry(prototypes: dict, key: str) -> Optional[tuple[np.ndarray, np.ndarray]]:
    phases = prototypes.get("phases") or {}
    tags = prototypes.get("pose_tags") or {}
    entry = phases.get(key) or tags.get(key)
    if not entry:
        return None
    centroid = np.array(entry["centroid"], dtype=float)
    weights = np.array(entry.get("weights", [1.0] * len(centroid)), dtype=float)
    return centroid, weights


def _weighted_distance(
    vec: np.ndarray,
    centroid: np.ndarray,
    weights: np.ndarray,
) -> float:
    diff = vec - centroid
    return float(np.sqrt(np.sum(weights * diff * diff)))


def score_phase(
    features: PoseFeatures,
    phase: str,
    *,
    athlete_id: Optional[str] = None,
) -> float:
    """Return similarity 0–1 (higher = better match)."""
    if not features.valid or not features.vector:
        return 0.0

    prototypes = load_prototypes()
    proto = _prototype_entry(prototypes, phase)
    if proto is None:
        return 0.0

    vec = np.array(features.vector, dtype=float)
    centroid, weights = proto
    dist = _weighted_distance(vec, centroid, weights)
    score = math.exp(-dist * 3.5)

    if athlete_id:
        athlete_data = load_athlete_templates(athlete_id)
        templates = athlete_data.get("phases") or {}
        at = templates.get(phase)
        if at and at.get("centroid"):
            a_centroid = np.array(at["centroid"], dtype=float)
            a_weights = np.array(at.get("weights", weights), dtype=float)
            a_dist = _weighted_distance(vec, a_centroid, a_weights)
            a_score = math.exp(-a_dist * 3.5)
            score = (1 - TEMPLATE_BLEND) * score + TEMPLATE_BLEND * a_score

    return round(min(1.0, score * features.quality), 4)


def classify_frame(
    features: PoseFeatures,
    *,
    athlete_id: Optional[str] = None,
) -> tuple[str, float, dict[str, float]]:
    """Best-matching phase among SECTION_PHASES."""
    scores = {
        phase: score_phase(features, phase, athlete_id=athlete_id)
        for phase in SECTION_PHASES
    }
    best = max(scores, key=scores.get)
    return best, scores[best], scores


def _frame_by_idx(frames: list[dict]) -> dict[int, dict]:
    return {int(f["frame_idx"]): f for f in frames if "frame_idx" in f}


def _ts(frame: dict) -> float:
    return float(frame.get("timestamp_s", 0))


def add_phase_marker(
    sections: dict[str, Any],
    frame_idx: int,
    phase: str,
    *,
    timestamp_s: Optional[float] = None,
    pose_tag: Optional[str] = None,
    source: str = "manual",
    confidence: float = 1.0,
) -> dict[str, Any]:
    if phase not in SECTION_PHASES:
        raise ValueError(f"Invalid phase '{phase}'. Must be one of {SECTION_PHASES}")

    markers = list(sections.get("phase_markers") or [])
    markers = [
        m for m in markers
        if int(m["frame_idx"]) != frame_idx and m.get("phase") != phase
    ]
    entry: dict[str, Any] = {
        "frame_idx": frame_idx,
        "phase": phase,
        "source": source,
        "confidence": confidence,
    }
    if timestamp_s is not None:
        entry["timestamp_s"] = round(timestamp_s, 3)
    if pose_tag:
        entry["pose_tag"] = pose_tag
    markers.append(entry)
    markers.sort(key=lambda m: m["frame_idx"])
    sections["phase_markers"] = markers
    return sections


def remove_phase_marker(sections: dict[str, Any], frame_idx: int) -> dict[str, Any]:
    markers = [m for m in (sections.get("phase_markers") or []) if int(m["frame_idx"]) != frame_idx]
    sections["phase_markers"] = markers
    return sections


def move_phase_marker(
    sections: dict[str, Any],
    from_frame_idx: int,
    to_frame_idx: int,
    frames: list[dict],
) -> dict[str, Any]:
    """Move a marker to another frame; swap if destination is occupied."""
    if from_frame_idx == to_frame_idx:
        return sections

    markers = list(sections.get("phase_markers") or [])
    from_m = next((m for m in markers if int(m["frame_idx"]) == from_frame_idx), None)
    if from_m is None:
        raise ValueError(f"No phase marker at frame {from_frame_idx}")

    to_m = next((m for m in markers if int(m["frame_idx"]) == to_frame_idx), None)
    fmap = {int(f["frame_idx"]): f for f in frames}

    if to_m is not None:
        to_m["frame_idx"] = from_frame_idx
        if from_frame_idx in fmap:
            to_m["timestamp_s"] = round(float(fmap[from_frame_idx].get("timestamp_s", 0)), 3)
        to_m["source"] = "manual"

    from_m["frame_idx"] = to_frame_idx
    if to_frame_idx in fmap:
        from_m["timestamp_s"] = round(float(fmap[to_frame_idx].get("timestamp_s", 0)), 3)
    from_m["source"] = "manual"

    markers.sort(key=lambda m: int(m["frame_idx"]))
    sections["phase_markers"] = markers
    return sections


def _find_anchor(markers: list[dict]) -> Optional[dict]:
    for phase in ("landing", "final_jump"):
        hits = [m for m in markers if m.get("phase") == phase]
        if hits:
            return max(hits, key=lambda m: m["frame_idx"])
    return None


def _search_hop_backward(
    start_frame: int,
    features_by_frame: dict[int, PoseFeatures],
    detected_frames: list[int],
    *,
    athlete_id: Optional[str],
    hop_label: str,
) -> Optional[tuple[int, float]]:
    """Find best hop_contact-like frame before start_frame."""
    candidates = [
        f for f in detected_frames
        if start_frame - ANCHOR_SEARCH_WINDOW <= f < start_frame - ANCHOR_MIN_GAP
    ]
    if not candidates:
        return None

    best_f, best_s = None, 0.0
    for fidx in reversed(candidates):
        feat = features_by_frame.get(fidx)
        if not feat or not feat.valid:
            continue
        hop_score = score_phase(feat, hop_label, athlete_id=athlete_id)
        contact_score = score_phase(feat, "hop_1", athlete_id=athlete_id)
        feat_hop = feat.hop_contact_score
        combined = 0.45 * hop_score + 0.35 * contact_score + 0.2 * feat_hop
        if combined > best_s:
            best_s = combined
            best_f = fidx

    if best_f is None or best_s < 0.25:
        return None
    return best_f, best_s


def propagate_from_anchors(
    frames: list[dict],
    sections: dict[str, Any],
    *,
    athlete_id: Optional[str] = None,
) -> dict[str, Any]:
    """
    Backward search from manual final_jump/landing anchors to suggest hop markers.
    Merges into phase_markers with source='propagated'.
    """
    markers = list(sections.get("phase_markers") or [])
    anchor = _find_anchor(markers)
    if anchor is None:
        sections["notes"] = (sections.get("notes") or "") + " Sin ancla final_jump/landing para propagar."
        return sections

    anchor_frame = int(anchor["frame_idx"])
    features_by_frame = extract_features_for_frames(frames)
    detected_frames = sorted(features_by_frame.keys())

    manual_hops = {
        m["phase"]: int(m["frame_idx"])
        for m in markers
        if m.get("phase") in HOP_PHASES and m.get("source") == "manual"
    }

    propagated: list[dict] = []
    search_from = anchor_frame
    if anchor.get("phase") == "landing":
        fj = next((m for m in markers if m.get("phase") == "final_jump"), None)
        if fj:
            search_from = int(fj["frame_idx"])
        else:
            search_from = anchor_frame - 5

    for hop_phase in reversed(HOP_PHASES):
        if hop_phase in manual_hops:
            search_from = manual_hops[hop_phase]
            continue

        result = _search_hop_backward(
            search_from, features_by_frame, detected_frames,
            athlete_id=athlete_id, hop_label=hop_phase,
        )
        if result is None:
            continue
        fidx, conf = result
        frame = _frame_by_idx(frames).get(fidx, {})
        propagated.append({
            "frame_idx": fidx,
            "phase": hop_phase,
            "source": "propagated",
            "confidence": round(conf, 3),
            "timestamp_s": round(_ts(frame), 3),
            "pose_tag": "hop_contact",
        })
        search_from = fidx

    existing_manual = {
        (int(m["frame_idx"]), m["phase"])
        for m in markers
        if m.get("source") == "manual"
    }
    markers = [
        m for m in markers
        if not (m.get("source") == "propagated" and m.get("phase") in HOP_PHASES)
    ]
    for p in propagated:
        key = (p["frame_idx"], p["phase"])
        if key not in existing_manual:
            markers = [m for m in markers if not (
                m.get("phase") == p["phase"] and m.get("source") == "propagated"
            )]
            markers.append(p)

    markers.sort(key=lambda m: m["frame_idx"])
    sections["phase_markers"] = markers
    n_prop = len(propagated)
    if n_prop:
        note = f" Propagados {n_prop} hops desde ancla frame {anchor_frame}."
        sections["notes"] = (sections.get("notes") or "").strip() + note
    return sections


def _phase_bounds_from_markers(
    markers: list[dict],
    last_frame_idx: int,
) -> dict[str, dict[str, Optional[int]]]:
    phases = {name: {"start_frame": None, "end_frame": None} for name in SECTION_PHASES}
    if not markers:
        return phases

    sorted_m = sorted(markers, key=lambda m: m["frame_idx"])
    hop_markers = [m for m in sorted_m if m.get("phase") in HOP_PHASES]
    landing_m = next((m for m in sorted_m if m.get("phase") == "landing"), None)
    final_m = next((m for m in sorted_m if m.get("phase") == "final_jump"), None)
    approach_m = next((m for m in sorted_m if m.get("phase") == "approach"), None)

    first_hop_frame = hop_markers[0]["frame_idx"] if hop_markers else None
    if first_hop_frame is not None:
        phases["approach"]["start_frame"] = 0
        phases["approach"]["end_frame"] = max(0, int(first_hop_frame) - 1)
    elif approach_m:
        phases["approach"]["start_frame"] = 0
        phases["approach"]["end_frame"] = int(approach_m["frame_idx"])

    for i, hm in enumerate(hop_markers):
        pname = hm["phase"]
        start = int(hm["frame_idx"])
        if i + 1 < len(hop_markers):
            end = int(hop_markers[i + 1]["frame_idx"]) - 1
        elif final_m:
            end = int(final_m["frame_idx"]) - 1
        elif landing_m:
            end = int(landing_m["frame_idx"]) - 1
        else:
            end = last_frame_idx
        phases[pname]["start_frame"] = start
        phases[pname]["end_frame"] = max(start, end)

    if final_m:
        fj_start = int(final_m["frame_idx"])
        fj_end = int(landing_m["frame_idx"]) - 1 if landing_m else last_frame_idx
        phases["final_jump"]["start_frame"] = fj_start
        phases["final_jump"]["end_frame"] = max(fj_start, fj_end)
    elif hop_markers and landing_m:
        phases["final_jump"]["start_frame"] = int(hop_markers[-1]["frame_idx"]) + 1
        phases["final_jump"]["end_frame"] = int(landing_m["frame_idx"]) - 1

    if landing_m:
        phases["landing"]["start_frame"] = int(landing_m["frame_idx"])
        phases["landing"]["end_frame"] = last_frame_idx

    return phases


def _contacts_from_markers(markers: list[dict], frames: list[dict]) -> list[dict]:
    fmap = _frame_by_idx(frames)
    hop_ms = sorted(
        [m for m in markers if m.get("phase") in HOP_PHASES],
        key=lambda m: m["frame_idx"],
    )
    landing_m = next((m for m in markers if m.get("phase") == "landing"), None)

    contacts: list[dict] = []
    for i, hm in enumerate(hop_ms[:4], start=1):
        fidx = int(hm["frame_idx"])
        frame = fmap.get(fidx, {})
        contacts.append({
            "index": i,
            "frame_idx": fidx,
            "timestamp_s": round(_ts(frame) if frame else hm.get("timestamp_s", 0), 3),
            "type": "hop",
            "surface": "track",
            "phase": hm["phase"],
            "confidence": hm.get("confidence", 1.0),
            "source": hm.get("source", "manual"),
        })

    if landing_m:
        fidx = int(landing_m["frame_idx"])
        frame = fmap.get(fidx, {})
        contacts.append({
            "index": len(contacts) + 1,
            "frame_idx": fidx,
            "timestamp_s": round(_ts(frame) if frame else landing_m.get("timestamp_s", 0), 3),
            "type": "landing",
            "surface": "sand",
            "phase": "landing",
            "confidence": landing_m.get("confidence", 1.0),
            "source": landing_m.get("source", "manual"),
        })

    return contacts


def rebuild_sections_from_markers(
    frames: list[dict],
    sections: dict[str, Any],
) -> dict[str, Any]:
    """Rebuild phases + contacts from phase_markers."""
    markers = sections.get("phase_markers") or []
    if not markers:
        return sections

    last_frame_idx = max(int(f.get("frame_idx", 0)) for f in frames) if frames else 0
    phases = _phase_bounds_from_markers(markers, last_frame_idx)
    contacts = _contacts_from_markers(markers, frames)

    sections["phases"] = phases
    sections["contacts"] = contacts
    n = len(contacts)
    sections["confidence"] = round(min(1.0, n / 5) * 0.7 + 0.3, 3)
    return sections


def update_athlete_template(
    athlete_id: str,
    phase: str,
    features: PoseFeatures,
    *,
    pose_tag: Optional[str] = None,
) -> None:
    """EMA-update athlete-specific prototype from a confirmed marker."""
    if not athlete_id or not features.valid:
        return

    path = ATHLETES_ROOT / athlete_id / "pose_templates.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _load_json(path)
    if not data:
        data = {"schema_version": 1, "athlete_id": athlete_id, "phases": {}, "pose_tags": {}}

    bucket = data["pose_tags" if pose_tag else "phases"]
    key = pose_tag or phase
    entry = bucket.get(key)
    vec = features.vector
    alpha = 0.25

    if entry and entry.get("centroid"):
        old = np.array(entry["centroid"], dtype=float)
        new = (1 - alpha) * old + alpha * np.array(vec, dtype=float)
        count = int(entry.get("sample_count", 1)) + 1
    else:
        new = np.array(vec, dtype=float)
        count = 1

    bucket[key] = {
        "centroid": [round(float(v), 4) for v in new],
        "weights": list(entry.get("weights", [1.0] * len(vec))) if entry else [1.0] * len(vec),
        "sample_count": count,
        "feature_names": list(FEATURE_NAMES),
    }
    data["updated_from"] = phase

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def pose_classify_frames(
    frames: list[dict],
    *,
    athlete_id: Optional[str] = None,
) -> dict[int, dict[str, Any]]:
    """Per-frame pose classification scores (for debugging / overlay)."""
    features_by_frame = extract_features_for_frames(frames)
    out: dict[int, dict[str, Any]] = {}
    for fidx, feat in features_by_frame.items():
        phase, score, all_scores = classify_frame(feat, athlete_id=athlete_id)
        out[fidx] = {
            "predicted_phase": phase,
            "confidence": score,
            "scores": all_scores,
            "features": feat.as_dict(),
        }
    return out


def enhance_auto_contacts_with_pose(
    frames: list[dict],
    contact_frames: list[int],
    *,
    athlete_id: Optional[str] = None,
    is_landing: Optional[list[bool]] = None,
) -> list[int]:
    """
    Refine auto-detected contact frames (±3) maximizing hop_contact / landing
    pose-tag similarity fused with biomechanical hop_contact_score.
    """
    if not contact_frames:
        return contact_frames

    from .gt_contacts import score_pose_tag

    features_by_frame = extract_features_for_frames(frames)
    fmap = _frame_by_idx(frames)
    refined: list[int] = []
    landing_flags = is_landing or [False] * len(contact_frames)

    for i, fidx in enumerate(contact_frames):
        want_landing = bool(landing_flags[i]) if i < len(landing_flags) else False
        radius = 6 if want_landing else 3
        window = range(max(0, fidx - radius), fidx + radius + 1)
        best_f, best_s = fidx, -1.0
        for w in window:
            feat = features_by_frame.get(w)
            if not feat or not feat.valid:
                continue
            if want_landing:
                tag_sim = score_pose_tag(feat.vector, "landing")
                s = (
                    0.45 * tag_sim
                    + 0.30 * feat.landing_score
                    + 0.25 * score_phase(feat, "landing", athlete_id=athlete_id)
                )
            else:
                tag_sim = score_pose_tag(feat.vector, "hop_contact")
                s = (
                    0.50 * tag_sim
                    + 0.30 * feat.hop_contact_score
                    + 0.20 * score_phase(feat, "hop_1", athlete_id=athlete_id)
                )
            if s > best_s:
                best_s = s
                best_f = w
        if best_f not in refined and fmap.get(best_f):
            refined.append(best_f)
        elif fmap.get(fidx) and fidx not in refined:
            refined.append(fidx)

    return sorted(refined)
