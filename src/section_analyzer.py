"""
Section analyzer — Phase 3: approach, hops, final jump, landing.

Detects foot contacts from ankle keypoints (not athlete_state), classifies
surface via venue masks, and writes sections.json.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
from scipy.signal import find_peaks, savgol_filter

from .calibration import load_calibration
from .mask_utils import athlete_mask_overlap, load_mask_png
from .phase_classifier import (
    enhance_auto_contacts_with_pose,
    propagate_from_anchors,
    rebuild_sections_from_markers,
    update_athlete_template,
)
from .pose_features import extract_pose_features
from .schemas import SECTION_PHASES, empty_sections
from .track_scorer import _interp_keyframe, _project_position_s

USABLE_ANGLES = frozenset({"LATERAL", "SEMI_BACK"})
ANKLE_CONF_MIN = 0.4
CONTACT_MIN_FRAME_GAP = 8
RUNNING_MAX_INTERVAL_S = 0.45
PEAK_PROMINENCE = 4.0
PATCH_RADIUS = 14
PATCH_THRESH = 0.12
EXPECTED_CONTACTS = 5


@dataclass
class DetectedContact:
    frame_idx: int
    timestamp_s: float
    foot_x: float
    foot_y: float
    on_track: bool
    on_sand: bool
    position_s: Optional[float]
    prominence: float
    surface: str = "unknown"
    contact_type: str = "hop"
    confidence: float = 0.5


@dataclass
class SectionContext:
    output_dir: Path
    width: int
    height: int
    keyframes: list[dict] = field(default_factory=list)
    mask_frames: dict[str, dict[str, Any]] = field(default_factory=dict)
    _mask_cache: dict[int, tuple[Optional[np.ndarray], Optional[np.ndarray]]] = field(
        default_factory=dict, repr=False,
    )

    @classmethod
    def from_output_dir(cls, output_dir: Path, width: int, height: int) -> "SectionContext":
        cal = load_calibration(output_dir) or {}
        kfs = sorted(cal.get("keyframes") or [], key=lambda k: k["frame_idx"])
        return cls(
            output_dir=output_dir,
            width=width,
            height=height,
            keyframes=kfs,
            mask_frames=cal.get("mask_frames") or {},
        )

    def _nearest_mask_entry(self, frame_idx: int) -> Optional[dict[str, Any]]:
        if not self.mask_frames:
            return None
        if str(frame_idx) in self.mask_frames:
            return self.mask_frames[str(frame_idx)]
        keys = sorted(int(k) for k in self.mask_frames)
        nearest = min(keys, key=lambda k: abs(k - frame_idx))
        return self.mask_frames[str(nearest)]

    def load_masks(self, frame_idx: int) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if frame_idx in self._mask_cache:
            return self._mask_cache[frame_idx]
        track_mask = sand_mask = None
        entry = self._nearest_mask_entry(frame_idx)
        if entry:
            track_rel = entry.get("track")
            sand_rel = entry.get("sand")
            if track_rel:
                track_mask = load_mask_png(self.output_dir / track_rel)
            if sand_rel:
                sand_mask = load_mask_png(self.output_dir / sand_rel)
        self._mask_cache[frame_idx] = (track_mask, sand_mask)
        return track_mask, sand_mask

    def axis_at(self, frame_idx: int) -> Optional[dict[str, Any]]:
        if not self.keyframes:
            return None
        kf = _interp_keyframe(self.keyframes, frame_idx)
        return kf.get("axis")


def _is_usable_frame(frame: dict) -> bool:
    if frame.get("usable_for_analysis"):
        return True
    return frame.get("camera_angle") in USABLE_ANGLES


def _foot_y(frame: dict) -> Optional[float]:
    kps = {k["name"]: k for k in frame.get("keypoints") or []}
    ys: list[float] = []
    for name in ("l_ankle", "r_ankle"):
        kp = kps.get(name)
        if kp and float(kp.get("conf", 0)) >= ANKLE_CONF_MIN:
            ys.append(float(kp["y"]))
    if not ys:
        return None
    return max(ys)


def _foot_point(frame: dict) -> Optional[tuple[float, float]]:
    kps = {k["name"]: k for k in frame.get("keypoints") or []}
    ankles: list[tuple[float, float]] = []
    for name in ("l_ankle", "r_ankle"):
        kp = kps.get(name)
        if kp and float(kp.get("conf", 0)) >= ANKLE_CONF_MIN:
            ankles.append((float(kp["x"]), float(kp["y"])))
    if ankles:
        return (
            sum(p[0] for p in ankles) / len(ankles),
            sum(p[1] for p in ankles) / len(ankles),
        )
    bbox = frame.get("person_bbox")
    if bbox and len(bbox) >= 4:
        return ((bbox[0] + bbox[2]) * 0.5, bbox[3])
    return None


def _mask_fraction_at_point(
    x: float, y: float, mask: Optional[np.ndarray], radius: int = PATCH_RADIUS,
) -> float:
    if mask is None:
        return 0.0
    h, w = mask.shape[:2]
    ix, iy = int(round(x)), int(round(y))
    x0, x1 = max(0, ix - radius), min(w, ix + radius + 1)
    y0, y1 = max(0, iy - radius), min(h, iy + radius + 1)
    patch = mask[y0:y1, x0:x1]
    if patch.size == 0:
        return 0.0
    return float((patch > 0).sum()) / float(patch.size)


def _classify_surface(
    frame: dict,
    fx: float,
    fy: float,
    track_mask: Optional[np.ndarray],
    sand_mask: Optional[np.ndarray],
    width: int,
    height: int,
) -> tuple[str, bool, bool]:
    track_frac = _mask_fraction_at_point(fx, fy, track_mask)
    sand_frac = _mask_fraction_at_point(fx, fy, sand_mask)

    if sand_frac >= PATCH_THRESH and sand_frac >= track_frac:
        return "sand", track_frac >= PATCH_THRESH, True
    if track_frac >= PATCH_THRESH:
        return "track", True, sand_frac >= PATCH_THRESH

    bbox = frame.get("person_bbox")
    if bbox and len(bbox) >= 4:
        bb = tuple(bbox[:4])
        t_ov = athlete_mask_overlap(bb, None, track_mask, width, height)
        s_ov = athlete_mask_overlap(bb, None, sand_mask, width, height)
        if s_ov > 0.08 and s_ov >= t_ov:
            return "sand", t_ov > 0.05, True
        if t_ov > 0.05:
            return "track", True, s_ov > 0.05

    return "unknown", False, False


def _infer_unknown_surfaces(contacts: list[DetectedContact], approach_end: int) -> None:
    post = [c for c in contacts if c.frame_idx > approach_end]
    sand_i = next((i for i, c in enumerate(post) if c.surface == "sand"), None)
    for i, c in enumerate(post):
        if c.surface != "unknown":
            continue
        if sand_i is not None:
            if i < sand_i:
                c.surface = "track"
                c.on_track = True
            elif i == sand_i:
                c.surface = "sand"
                c.on_sand = True
        elif i < 4:
            c.surface = "track"
            c.on_track = True


def _contact_signal(values: list[Optional[float]]) -> list[Optional[float]]:
    """High-pass residual of foot_y to expose hop contacts over camera drift."""
    valid_idx = [i for i, v in enumerate(values) if v is not None]
    if len(valid_idx) < 5:
        return values
    arr = np.array([values[i] for i in valid_idx], dtype=float)
    win_short = min(len(arr), 11)
    if win_short % 2 == 0:
        win_short -= 1
    win_long = min(len(arr), 21)
    if win_long % 2 == 0:
        win_long -= 1
    if win_long < win_short:
        win_long = win_short
    if win_short < 5:
        return values
    smooth = savgol_filter(arr, window_length=win_short, polyorder=2)
    baseline = (
        savgol_filter(arr, window_length=win_long, polyorder=3)
        if len(arr) >= win_long else smooth
    )
    residual = smooth - baseline
    out: list[Optional[float]] = [None] * len(values)
    for j, i in enumerate(valid_idx):
        out[i] = float(residual[j])
    return out


def _detect_contacts(
    frames: list[dict],
    ctx: SectionContext,
) -> list[DetectedContact]:
    detected = [f for f in frames if f.get("person_detected")]
    if not detected:
        return []

    foot_ys = [_foot_y(f) for f in detected]
    signal = _contact_signal(foot_ys)

    valid_pairs = [(i, v) for i, v in enumerate(signal) if v is not None]
    if len(valid_pairs) < 3:
        return []

    indices, series = zip(*valid_pairs)
    arr = np.array(series, dtype=float)
    prominence = max(PEAK_PROMINENCE, float(np.std(arr)) * 0.2)
    peak_idx, props = find_peaks(arr, prominence=prominence, distance=2)

    contacts: list[DetectedContact] = []
    last_frame_idx = -CONTACT_MIN_FRAME_GAP * 2

    for pi, prom in zip(peak_idx, props["prominences"]):
        frame = detected[indices[pi]]
        fidx = int(frame["frame_idx"])
        if fidx - last_frame_idx < CONTACT_MIN_FRAME_GAP:
            continue

        foot = _foot_point(frame)
        if foot is None:
            continue
        fx, fy = foot

        track_mask, sand_mask = ctx.load_masks(fidx)
        surface, on_track, on_sand = _classify_surface(
            frame, fx, fy, track_mask, sand_mask, ctx.width, ctx.height,
        )

        axis = ctx.axis_at(fidx)
        bbox = frame.get("person_bbox")
        pos_s = None
        if bbox and len(bbox) >= 4:
            pos_s = _project_position_s(
                tuple(bbox[:4]), axis, ctx.width, ctx.height,
            )

        conf = min(1.0, 0.35 + float(prom) / max(prominence * 2, 1.0))
        if surface != "unknown":
            conf = min(1.0, conf + 0.15)

        contacts.append(DetectedContact(
            frame_idx=fidx,
            timestamp_s=float(frame.get("timestamp_s", 0)),
            foot_x=fx,
            foot_y=fy,
            on_track=on_track,
            on_sand=on_sand,
            position_s=pos_s,
            prominence=float(prom),
            surface=surface,
            confidence=round(conf, 3),
        ))
        last_frame_idx = fidx

    if contacts:
        approach_end = _find_approach_end(contacts)
        _infer_unknown_surfaces(contacts, approach_end)

    return contacts


def _interval_s(c0: DetectedContact, c1: DetectedContact) -> float:
    return abs(c1.timestamp_s - c0.timestamp_s)


def _find_approach_end(contacts: list[DetectedContact]) -> int:
    """Return frame_idx where approach ends (frame before first hop contact)."""
    if not contacts:
        return 0

    track_contacts = [c for c in contacts if c.surface == "track"]
    if not track_contacts:
        return max(0, contacts[0].frame_idx - 1)

    running_end_idx = 0
    for i in range(len(track_contacts) - 1):
        dt = _interval_s(track_contacts[i], track_contacts[i + 1])
        if dt < RUNNING_MAX_INTERVAL_S:
            running_end_idx = i + 1
        else:
            break

    if running_end_idx + 1 < len(track_contacts):
        first_hop = track_contacts[running_end_idx + 1]
    else:
        first_hop = track_contacts[min(running_end_idx, len(track_contacts) - 1)]

    return max(0, first_hop.frame_idx - 1)


def _assign_phases(
    selected: list[DetectedContact],
    approach_end: int,
    last_frame_idx: int,
) -> dict[str, dict[str, Optional[int]]]:
    phases = {name: {"start_frame": None, "end_frame": None} for name in SECTION_PHASES}

    track_hops = [c for c in selected if c.surface != "sand" and c != selected[-1]][:4]
    if not track_hops and len(selected) > 1:
        track_hops = selected[:-1][:4]
    landing = selected[-1] if selected and (
        selected[-1].surface == "sand" or len(selected) >= 4
    ) else next((c for c in selected if c.surface == "sand"), None)

    phases["approach"]["start_frame"] = 0
    phases["approach"]["end_frame"] = approach_end

    for i, hc in enumerate(track_hops):
        phase_name = f"hop_{i + 1}"
        start = hc.frame_idx
        if i + 1 < len(track_hops):
            end = track_hops[i + 1].frame_idx - 1
        elif landing:
            end = landing.frame_idx - 1
        else:
            end = last_frame_idx
        phases[phase_name]["start_frame"] = start
        phases[phase_name]["end_frame"] = max(start, end)

    if landing:
        if track_hops:
            fj_start = track_hops[-1].frame_idx + 1
        else:
            fj_start = approach_end + 1
        fj_end = max(fj_start, landing.frame_idx - 1)
        if fj_end >= fj_start:
            phases["final_jump"]["start_frame"] = fj_start
            phases["final_jump"]["end_frame"] = fj_end
        phases["landing"]["start_frame"] = landing.frame_idx
        phases["landing"]["end_frame"] = last_frame_idx
    elif track_hops:
        phases["final_jump"]["start_frame"] = track_hops[-1].frame_idx
        phases["final_jump"]["end_frame"] = last_frame_idx

    return phases


def _select_jump_contacts(
    contacts: list[DetectedContact],
    approach_end: int,
) -> list[DetectedContact]:
    """Pick up to 4 track hops + 1 sand landing after approach."""
    post = sorted(
        [c for c in contacts if c.frame_idx > approach_end],
        key=lambda c: c.frame_idx,
    )
    if not post:
        return []

    sand = next((c for c in post if c.surface == "sand"), None)
    if sand is None:
        sand = post[-1]

    before_sand = [c for c in post if c.frame_idx < sand.frame_idx]
    tracks = before_sand[:4]

    selected = list(tracks)
    if sand not in selected:
        selected.append(sand)
    return selected[:EXPECTED_CONTACTS]


def _serialize_contacts(selected: list[DetectedContact]) -> list[dict]:
    out: list[dict] = []
    hop_num = 0
    for i, c in enumerate(selected[:EXPECTED_CONTACTS], start=1):
        is_landing = c.surface == "sand" or (i == len(selected) and len(selected) >= 4)
        if is_landing:
            ctype = "landing"
            phase = "landing"
        else:
            hop_num += 1
            ctype = "hop"
            phase = f"hop_{hop_num}"
        out.append({
            "index": i,
            "frame_idx": c.frame_idx,
            "timestamp_s": round(c.timestamp_s, 3),
            "type": ctype,
            "surface": c.surface,
            "phase": phase,
            "position_s": round(c.position_s, 2) if c.position_s is not None else None,
            "confidence": c.confidence,
        })
    return out


def _overall_confidence(contacts: list[dict], usable_ratio: float) -> float:
    n = len(contacts)
    count_score = min(1.0, n / EXPECTED_CONTACTS)
    avg_conf = sum(c.get("confidence", 0.5) for c in contacts) / max(n, 1)
    return round(0.5 * count_score + 0.3 * avg_conf + 0.2 * usable_ratio, 3)


def analyze_sections(
    frames: list[dict],
    output_dir: Path,
    *,
    width: int,
    height: int,
    derived_version: int = 1,
    existing: Optional[dict[str, Any]] = None,
    use_pose: bool = True,
) -> dict[str, Any]:
    """Run section analysis and return sections.json document."""
    ctx = SectionContext.from_output_dir(output_dir, width, height)
    athlete_id = (existing or {}).get("athlete_id")
    markers = list((existing or {}).get("phase_markers") or [])

    usable_count = sum(1 for f in frames if _is_usable_frame(f) and f.get("person_detected"))
    usable_ratio = usable_count / max(len(frames), 1)

    notes_parts: list[str] = []
    if usable_ratio < 0.3:
        notes_parts.append(
            "Pocos frames con angulo LATERAL/SEMI_BACK; resultados pueden ser imprecisos.",
        )

    last_frame_idx = max(int(f.get("frame_idx", 0)) for f in frames) if frames else 0

    manual_markers = [m for m in markers if m.get("source") in ("manual", "propagated")]
    if len(manual_markers) >= 2:
        doc = empty_sections()
        doc["derived_version"] = derived_version
        doc["phase_markers"] = markers
        if athlete_id:
            doc["athlete_id"] = athlete_id
        doc = rebuild_sections_from_markers(frames, doc)
        doc["confidence"] = _overall_confidence(doc.get("contacts") or [], usable_ratio)
        doc["notes"] = " ".join(notes_parts) + " Basado en marcadores manuales."
        return doc

    contacts_raw = _detect_contacts(frames, ctx)
    if not contacts_raw:
        notes_parts.append("No se detectaron contactos de pie.")
        doc = empty_sections()
        doc["derived_version"] = derived_version
        doc["phase_markers"] = markers
        if athlete_id:
            doc["athlete_id"] = athlete_id
        doc["confidence"] = 0.0
        doc["notes"] = " ".join(notes_parts)
        return doc

    approach_end = _find_approach_end(contacts_raw)
    selected = _select_jump_contacts(contacts_raw, approach_end)

    if use_pose and selected:
        contact_frames = [c.frame_idx for c in selected]
        refined_frames = enhance_auto_contacts_with_pose(
            frames, contact_frames, athlete_id=athlete_id,
        )
        if refined_frames and len(refined_frames) == len(selected):
            fmap = {int(f["frame_idx"]): f for f in frames}
            for i, rf in enumerate(refined_frames):
                if rf in fmap:
                    selected[i].frame_idx = rf
                    selected[i].timestamp_s = float(fmap[rf].get("timestamp_s", 0))
            notes_parts.append("Contactos refinados con pose.")

    phases = _assign_phases(selected, approach_end, last_frame_idx)
    contacts = _serialize_contacts(selected)

    auto_markers: list[dict] = []
    for c in contacts:
        auto_markers.append({
            "frame_idx": c["frame_idx"],
            "phase": c["phase"],
            "timestamp_s": c.get("timestamp_s"),
            "source": "auto",
            "confidence": c.get("confidence", 0.5),
        })
    merged_markers = list(markers)
    auto_frames = {m["frame_idx"] for m in merged_markers}
    for am in auto_markers:
        if am["frame_idx"] not in auto_frames:
            merged_markers.append(am)

    if len(contacts) != EXPECTED_CONTACTS:
        notes_parts.append(
            f"Se detectaron {len(contacts)}/{EXPECTED_CONTACTS} contactos esperados.",
        )

    doc: dict[str, Any] = {
        "schema_version": 2,
        "derived_version": derived_version,
        "phases": phases,
        "contacts": contacts,
        "phase_markers": merged_markers,
        "confidence": _overall_confidence(contacts, usable_ratio),
        "notes": " ".join(notes_parts),
    }
    if athlete_id:
        doc["athlete_id"] = athlete_id
    return doc


def phase_at_frame(sections: dict[str, Any], frame_idx: int) -> Optional[str]:
    phases = sections.get("phases") or {}
    for name in SECTION_PHASES:
        bounds = phases.get(name) or {}
        start = bounds.get("start_frame")
        end = bounds.get("end_frame")
        if start is None or end is None:
            continue
        if start <= frame_idx <= end:
            return name
    return None


def write_sections(output_dir: Path, sections: dict[str, Any]) -> Path:
    path = output_dir / "sections.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sections, f, indent=2)
    return path


def run_section_analysis(
    output_dir: Path,
    *,
    use_pose: bool = True,
) -> dict[str, Any]:
    """Load analysis.json, analyze, write sections.json, bump derived_version."""
    analysis_path = output_dir / "analysis.json"
    if not analysis_path.exists():
        raise FileNotFoundError(f"No analysis.json in {output_dir}")

    with open(analysis_path, encoding="utf-8") as f:
        data = json.load(f)

    frames = data.get("frames") or []
    if not frames:
        raise ValueError("analysis.json has no frames")

    vi = data.get("video_info") or {}
    width = int(vi.get("width", 1280))
    height = int(vi.get("height", 720))

    prev_derived = int(data.get("derived_version", 0))
    new_derived = prev_derived + 1

    existing = None
    sections_path = output_dir / "sections.json"
    if sections_path.exists():
        with open(sections_path, encoding="utf-8") as f:
            existing = json.load(f)

    sections = analyze_sections(
        frames, output_dir, width=width, height=height,
        derived_version=new_derived, existing=existing, use_pose=use_pose,
    )
    write_sections(output_dir, sections)

    data["derived_version"] = new_derived
    with open(analysis_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return sections


def load_sections(output_dir: Path) -> dict[str, Any]:
    path = output_dir / "sections.json"
    if not path.exists():
        return empty_sections()
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def mark_phase_on_frame(
    output_dir: Path,
    frame_idx: int,
    phase: str,
    *,
    pose_tag: Optional[str] = None,
    athlete_id: Optional[str] = None,
    update_template: bool = True,
) -> dict[str, Any]:
    """Assign a phase marker to a frame; rebuild phases/contacts."""
    from .phase_classifier import add_phase_marker

    analysis_path = output_dir / "analysis.json"
    if not analysis_path.exists():
        raise FileNotFoundError(f"No analysis.json in {output_dir}")

    with open(analysis_path, encoding="utf-8") as f:
        data = json.load(f)
    frames = data.get("frames") or []
    fmap = {int(f["frame_idx"]): f for f in frames}
    frame = fmap.get(frame_idx)
    if frame is None:
        raise ValueError(f"Frame {frame_idx} not found in analysis.json")

    sections = load_sections(output_dir)
    if athlete_id:
        sections["athlete_id"] = athlete_id
    elif sections.get("athlete_id"):
        athlete_id = sections["athlete_id"]

    ts = float(frame.get("timestamp_s", 0))
    sections = add_phase_marker(
        sections, frame_idx, phase,
        timestamp_s=ts, pose_tag=pose_tag, source="manual",
    )
    sections = rebuild_sections_from_markers(frames, sections)

    if update_template and athlete_id:
        feat = extract_pose_features(frame)
        if feat.valid:
            update_athlete_template(athlete_id, phase, feat, pose_tag=pose_tag)

    prev_derived = int(data.get("derived_version", 0))
    new_derived = prev_derived + 1
    sections["derived_version"] = new_derived
    write_sections(output_dir, sections)

    data["derived_version"] = new_derived
    with open(analysis_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return sections


def move_phase_marker_on_frame(
    output_dir: Path,
    from_frame_idx: int,
    to_frame_idx: int,
) -> dict[str, Any]:
    from .phase_classifier import move_phase_marker

    analysis_path = output_dir / "analysis.json"
    with open(analysis_path, encoding="utf-8") as f:
        data = json.load(f)
    frames = data.get("frames") or []
    fmap = {int(f["frame_idx"]): f for f in frames}
    if to_frame_idx not in fmap:
        raise ValueError(f"Frame {to_frame_idx} not found in analysis.json")

    sections = load_sections(output_dir)
    sections = move_phase_marker(sections, from_frame_idx, to_frame_idx, frames)
    sections = rebuild_sections_from_markers(frames, sections)

    prev_derived = int(data.get("derived_version", 0))
    new_derived = prev_derived + 1
    sections["derived_version"] = new_derived
    write_sections(output_dir, sections)

    data["derived_version"] = new_derived
    with open(analysis_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return sections


def unmark_phase_frame(output_dir: Path, frame_idx: int) -> dict[str, Any]:
    from .phase_classifier import remove_phase_marker

    analysis_path = output_dir / "analysis.json"
    with open(analysis_path, encoding="utf-8") as f:
        data = json.load(f)
    frames = data.get("frames") or []

    sections = load_sections(output_dir)
    sections = remove_phase_marker(sections, frame_idx)
    sections = rebuild_sections_from_markers(frames, sections)

    prev_derived = int(data.get("derived_version", 0))
    new_derived = prev_derived + 1
    sections["derived_version"] = new_derived
    write_sections(output_dir, sections)

    data["derived_version"] = new_derived
    with open(analysis_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return sections


def run_phase_propagation(output_dir: Path) -> dict[str, Any]:
    """Backward-propagate hops from final_jump/landing anchors."""
    analysis_path = output_dir / "analysis.json"
    with open(analysis_path, encoding="utf-8") as f:
        data = json.load(f)
    frames = data.get("frames") or []

    sections = load_sections(output_dir)
    athlete_id = sections.get("athlete_id")
    sections = propagate_from_anchors(frames, sections, athlete_id=athlete_id)
    sections = rebuild_sections_from_markers(frames, sections)

    prev_derived = int(data.get("derived_version", 0))
    new_derived = prev_derived + 1
    sections["derived_version"] = new_derived
    write_sections(output_dir, sections)

    data["derived_version"] = new_derived
    with open(analysis_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return sections
