"""Test venue profile learning and recompute on VOD2."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.calibration_propagator import target_frames_from_analysis
from src.venue_profile import (
    learn_from_calibration,
    load_profile,
    save_debug_frames,
)

API = "http://localhost:8000"
VIDEO_NAME = "VOD2"
VIDEO_PATH = ROOT / "VOD2.mp4"
OUT_DIR = ROOT / "output" / VIDEO_NAME
DEBUG_DIR = ROOT / "output" / "venue_test"


def main() -> None:
    print("=== 1. Learn venue profile from calibration ===")
    profile = learn_from_calibration(
        VIDEO_PATH,
        OUT_DIR / "calibration.json",
        output_dir=DEBUG_DIR,
    )
    print(f"  frames_used={profile['frames_used']} sand_frames={profile.get('sand_frames_used', 0)}")

    print("=== 2. Save debug detection images ===")
    frames = target_frames_from_analysis(OUT_DIR)
    sample = frames[:: max(1, len(frames) // 5)][:5]
    saved = save_debug_frames(VIDEO_PATH, profile, DEBUG_DIR, sample, max_frames=5)
    print(f"  saved {len(saved)} images to {DEBUG_DIR}")

    print("=== 3. POST /api/recompute-tracking/VOD2 ===")
    r = requests.post(f"{API}/api/recompute-tracking/{VIDEO_NAME}", timeout=120)
    print(f"  status={r.status_code}")
    if r.status_code != 200:
        print(f"  body={r.text}")
        sys.exit(1)
    data = r.json()
    print(f"  frames_updated={data.get('frames_updated')}")

    print("=== 4. GET /api/venue/profile ===")
    r = requests.get(f"{API}/api/venue/profile", timeout=30)
    print(f"  status={r.status_code} learned={r.json().get('learned')}")

    summary = {
        "profile_frames_used": profile.get("frames_used"),
        "recompute_status": 200,
        "debug_images": [p.name for p in saved],
    }
    with open(DEBUG_DIR / "test_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("Done.")


if __name__ == "__main__":
    main()
