"""Test fixed venue profile detection on VOD2."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.venue_profile import (
    detect_track_sand_frame,
    learn_from_calibration,
    load_profile,
    save_debug_frames,
)

VIDEO_PATH = ROOT / "VOD2.mp4"
OUT_DIR = ROOT / "output" / "VOD2"
DEBUG_DIR = ROOT / "output" / "venue_test_fixed"
TEST_FRAMES = [0, 21, 36]


def area_from_detect(frame, profile):
    _, _, conf, area = detect_track_sand_frame(frame, profile)
    return conf, area


def broken_detect(frame, profile):
    """Simulate old naive inRange + largest contour with absurd HSV range."""
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv_range = profile.get("track_hsv_range") or {"low": [0, 0, 37], "high": [179, 211, 255]}
    low = np.array(hsv_range["low"], dtype=np.uint8)
    high = np.array(hsv_range["high"], dtype=np.uint8)
    mask = cv2.inRange(hsv, low, high)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    best = max(contours, key=cv2.contourArea)
    return cv2.contourArea(best) / (w * h)


def main() -> None:
    # Snapshot old profile if present
    old_profile_path = ROOT / "venues" / "default" / "profile.json"
    backup = ROOT / "output" / "venue_test_fixed" / "profile_before.json"
    backup.parent.mkdir(parents=True, exist_ok=True)
    if old_profile_path.exists():
        shutil.copy(old_profile_path, backup)
        old_profile = json.loads(backup.read_text(encoding="utf-8"))
    else:
        old_profile = None

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    broken_before = {}
    absurd_profile = old_profile or {"track_hsv_range": {"low": [0, 0, 37], "high": [179, 211, 255]}}
    for fidx in TEST_FRAMES:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ok, frame = cap.read()
        if ok:
            broken_before[fidx] = broken_detect(frame, absurd_profile)
    cap.release()

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    before = {}
    if old_profile:
        for fidx in TEST_FRAMES:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
            ok, frame = cap.read()
            if ok:
                conf, area = area_from_detect(frame, old_profile)
                before[fidx] = {"confidence": conf, "area_ratio": area}
    cap.release()

    print("=== Re-learn from VOD2 calibration ===")
    profile = learn_from_calibration(
        VIDEO_PATH,
        OUT_DIR / "calibration.json",
        output_dir=DEBUG_DIR,
    )
    print(f"  track_hsv_range: {profile['track_hsv_range']}")
    if profile.get("sand_hsv_range"):
        print(f"  sand_hsv_range: {profile['sand_hsv_range']}")
    print(f"  expected_area_norm: {profile.get('expected_area_norm')}")

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    after = {}
    for fidx in TEST_FRAMES:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ok, frame = cap.read()
        if ok:
            conf, area = area_from_detect(frame, profile)
            after[fidx] = {"confidence": conf, "area_ratio": area}
            print(f"  frame {fidx}: area={area:.4f} conf={conf:.3f}")
    cap.release()

    save_debug_frames(VIDEO_PATH, profile, DEBUG_DIR, TEST_FRAMES, max_frames=len(TEST_FRAMES))

    summary = {
        "before_broken_range": {
            str(fidx): {"area_ratio": round(broken_before.get(fidx, 0), 4)}
            for fidx in TEST_FRAMES
        },
        "after_fixed": after,
        "track_hsv_range": profile.get("track_hsv_range"),
        "sand_hsv_range": profile.get("sand_hsv_range"),
        "expected_area_norm": profile.get("expected_area_norm"),
        "frames_used": profile.get("frames_used"),
    }
    with open(DEBUG_DIR / "test_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Summary ===")
    for fidx in TEST_FRAMES:
        b = broken_before.get(fidx, 0)
        a = after.get(fidx, {})
        print(
            f"  frame {fidx}: broken area={b:.4f} "
            f"-> fixed area={a.get('area_ratio', 'N/A')} conf={a.get('confidence', 'N/A')}"
        )


if __name__ == "__main__":
    main()
