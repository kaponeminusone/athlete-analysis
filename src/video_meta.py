"""Sidecar metadata next to uploaded videos (athlete assignment, note, date).

VIDEO_ROOT/foo.mp4
VIDEO_ROOT/foo.meta.json  →  { "athlete_id": "Mateo", "note"?: "...", "date"?: "YYYY-MM-DD" }
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional


def meta_path_for_video(video_path: Path) -> Path:
    """Same stem as the video + `.meta.json` (foo.mp4 → foo.meta.json)."""
    return video_path.with_name(f"{video_path.stem}.meta.json")


def sanitize_athlete_id(raw: str) -> str:
    """Strip; empty → unassigned. Allow alphanumeric, spaces, accents, hyphen/underscore."""
    s = (raw or "").strip()
    if not s:
        return ""
    # \w with UNICODE includes accented letters; keep spaces and hyphen.
    s = re.sub(r"[^\w\s\-]", "", s, flags=re.UNICODE)
    return s.strip()


def read_video_meta(video_path: Path) -> Optional[dict[str, Any]]:
    meta_path = meta_path_for_video(video_path)
    if not meta_path.is_file():
        return None
    try:
        with open(meta_path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def write_video_meta(
    video_path: Path,
    athlete_id: str,
    note: Optional[str] = None,
    date: Optional[str] = None,
) -> Path:
    payload: dict[str, Any] = {"athlete_id": athlete_id}
    note_s = (note or "").strip()
    date_s = (date or "").strip()
    if note_s:
        payload["note"] = note_s
    if date_s:
        payload["date"] = date_s
    dest = meta_path_for_video(video_path)
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return dest
