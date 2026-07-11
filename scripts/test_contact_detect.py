"""Quick auto-contact detection smoke test (ignores manual markers)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.section_analyzer import analyze_sections  # noqa: E402


def test_video(name: str) -> None:
    out = ROOT / "output" / name
    with open(out / "analysis.json", encoding="utf-8") as f:
        data = json.load(f)
    frames = data["frames"]
    vi = data.get("video_info") or {}
    # No existing markers → force auto path
    doc = analyze_sections(
        frames,
        out,
        width=int(vi.get("width", 1280)),
        height=int(vi.get("height", 720)),
        existing={"athlete_id": "Mateo", "phase_markers": []},
        use_pose=True,
    )
    contacts = doc.get("contacts") or []
    print(f"=== {name} auto contacts {len(contacts)}/5  conf={doc.get('confidence')}")
    for c in contacts:
        print(
            f"  {c.get('phase')}: frame={c['frame_idx']} "
            f"surf={c.get('surface')} type={c.get('type')} conf={c.get('confidence')}"
        )
    print(f"  notes: {doc.get('notes')}")


if __name__ == "__main__":
    for v in sys.argv[1:] or ["VOD2", "VOD4"]:
        test_video(v)
