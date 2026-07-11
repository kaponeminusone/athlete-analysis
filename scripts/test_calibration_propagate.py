#!/usr/bin/env python3
"""Test seed calibration propagation on a video clip."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.calibration_propagator import (  # noqa: E402
    propagate_calibration,
    target_frames_from_analysis,
    target_frames_from_video,
)


def _load_seeds(seeds_path: Path) -> list[dict]:
    with open(seeds_path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return data.get("seeds") or []


def _default_seeds() -> list[dict]:
    """Generic quadrilateral covering central track band (normalized coords)."""
    return [{
        "frame_idx": 0,
        "seed_points": [
            [0.12, 0.52],
            [0.88, 0.52],
            [0.92, 0.78],
            [0.08, 0.78],
        ],
        "labels": ["corner_tl", "corner_tr", "corner_br", "corner_bl"],
    }]


def _draw_overlay(frame: np.ndarray, polygon: list[list[float]]) -> np.ndarray:
    h, w = frame.shape[:2]
    out = frame.copy()
    if len(polygon) < 3:
        return out
    pts = np.array([[int(p[0] * w), int(p[1] * h)] for p in polygon], dtype=np.int32)
    overlay = out.copy()
    cv2.fillPoly(overlay, [pts], (52, 211, 153))
    cv2.addWeighted(overlay, 0.25, out, 0.75, 0, out)
    cv2.polylines(out, [pts], True, (52, 211, 153), 2, cv2.LINE_AA)
    for i, (nx, ny) in enumerate(polygon):
        cx, cy = int(nx * w), int(ny * h)
        cv2.circle(out, (cx, cy), 5, (34, 197, 94), -1)
        cv2.putText(out, str(i + 1), (cx + 6, cy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Test calibration optical-flow propagation")
    parser.add_argument("video", nargs="?", default=str(ROOT / "VOD2.mp4"))
    parser.add_argument("--frames", type=int, default=100, help="Max frames to propagate")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--seeds", type=Path, help="JSON file with seeds array")
    parser.add_argument("--snap", action="store_true", help="Enable line snapping")
    parser.add_argument("--output", type=Path, default=ROOT / "output" / "test_calibration")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Video not found: {video_path}")
        return 1

    seeds = _load_seeds(args.seeds) if args.seeds else _default_seeds()
    video_name = video_path.stem
    analysis_dir = ROOT / "output" / video_name

    targets = target_frames_from_analysis(analysis_dir)
    if not targets:
        targets = target_frames_from_video(video_path, stride=args.stride, max_frames=args.frames)
    else:
        targets = [f for f in targets if f < args.frames]

    if not targets:
        print("No target frames.")
        return 1

    print(f"Video: {video_path}")
    print(f"Seeds: {len(seeds)} keyframe(s), targets: {len(targets)} frames")

    result = propagate_calibration(
        video_path,
        seeds,
        targets,
        snap_to_lines=args.snap,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    meta_path = args.output / "propagation_result.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "video": str(video_path),
                "targets": len(targets),
                "keyframes": len(result["keyframes"]),
                "propagation": result["propagation"],
            },
            f,
            indent=2,
        )

    cap = cv2.VideoCapture(str(video_path))
    saved = 0
    per_frame = result["per_frame_polygons"]
    for fidx in targets:
        if fidx % max(1, len(targets) // 20) != 0 and fidx not in (targets[0], targets[-1]):
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ok, frame = cap.read()
        if not ok:
            continue
        poly = per_frame.get(str(fidx)) or per_frame.get(fidx) or []
        overlay = _draw_overlay(frame, poly)
        cv2.putText(
            overlay,
            f"frame {fidx}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        out_path = args.output / f"frame_{fidx:06d}.jpg"
        cv2.imwrite(str(out_path), overlay)
        saved += 1
    cap.release()

    print(f"Saved {saved} debug overlays to {args.output}")
    print(f"Metadata: {meta_path}")
    print(f"Propagated keyframes: {result['propagation']['frame_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
