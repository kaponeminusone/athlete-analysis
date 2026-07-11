#!/usr/bin/env python3
"""Verify multi-video dataset append does not wipe prior videos."""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import train_venue_seg as tvs


def _mock_calibration(video_name: str, frame_indices: list[int]) -> dict:
    keyframes = []
    for idx in frame_indices:
        keyframes.append({
            "frame_idx": idx,
            "source": "manual",
            "track_polygon": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.5], [0.1, 0.5]],
            "landing_zone": [[0.1, 0.55], [0.9, 0.55], [0.9, 0.9], [0.1, 0.9]],
        })
    return {"version": 2, "video": f"{video_name}.mp4", "keyframes": keyframes}


def _write_cal(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _fake_video(path: Path, num_frames: int = 10) -> None:
    import cv2
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        30.0,
        (320, 180),
    )
    for i in range(num_frames):
        frame = np.full((180, 320, 3), (i * 20) % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def main() -> int:
    venue_id = "test_append"
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        orig_root = tvs.ROOT
        tvs.ROOT = tmp_path

        try:
            out_a = tmp_path / "output" / "VIDEO_A"
            out_b = tmp_path / "output" / "VIDEO_B"
            cal_a = out_a / "calibration.json"
            cal_b = out_b / "calibration.json"
            vid_a = tmp_path / "VIDEO_A.mp4"
            vid_b = tmp_path / "VIDEO_B.mp4"

            _write_cal(cal_a, _mock_calibration("VIDEO_A", [0, 3, 6]))
            _write_cal(cal_b, _mock_calibration("VIDEO_B", [1, 4]))
            _fake_video(vid_a, 10)
            _fake_video(vid_b, 10)

            r1 = tvs.export_dataset_append("VIDEO_A", cal_a, video_path=vid_a, venue_id=venue_id)
            assert r1["frames_exported"] == 3
            assert r1["videos_in_dataset"] == 1
            assert r1["total_dataset_frames"] == 3

            r2 = tvs.export_dataset_append("VIDEO_B", cal_b, video_path=vid_b, venue_id=venue_id)
            assert r2["videos_in_dataset"] == 2
            assert r2["total_dataset_frames"] == 5

            manifest = tvs.load_dataset_manifest(venue_id)
            names = {v["video_name"] for v in manifest["videos"]}
            assert names == {"VIDEO_A", "VIDEO_B"}

            dataset = tvs.dataset_root(venue_id)
            train_images = list((dataset / "images" / "train").glob("*.jpg"))
            val_images = list((dataset / "images" / "val").glob("*.jpg"))
            all_stems = {p.stem for p in train_images + val_images}
            assert any(s.startswith("VIDEO_A_") for s in all_stems), all_stems
            assert any(s.startswith("VIDEO_B_") for s in all_stems), all_stems
            assert len(all_stems) == 5

            # Re-append VIDEO_A with more frames — VIDEO_B must remain
            _write_cal(cal_a, _mock_calibration("VIDEO_A", [0, 3, 6, 9]))
            r3 = tvs.export_dataset_append("VIDEO_A", cal_a, video_path=vid_a, venue_id=venue_id)
            assert r3["total_dataset_frames"] == 6
            manifest2 = tvs.load_dataset_manifest(venue_id)
            b_entry = next(v for v in manifest2["videos"] if v["video_name"] == "VIDEO_B")
            assert b_entry["frames_exported"] == 2

            print("OK: append preserves prior videos and updates manifest correctly")
            return 0
        finally:
            tvs.ROOT = orig_root
            shutil.rmtree(tmp_path / "venues" / venue_id, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
