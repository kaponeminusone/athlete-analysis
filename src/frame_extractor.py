"""
Frame extractor: sample frames from video for analysis.
Extracts evenly spaced frames and saves metadata about each one.
"""

import cv2
import os
import json
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class FrameInfo:
    frame_idx: int        # absolute frame index in video
    timestamp_ms: float   # timestamp in milliseconds
    timestamp_s: float    # timestamp in seconds
    file_path: str        # saved image path
    width: int
    height: int


def extract_frames(
    video_path: str,
    output_dir: str,
    sample_every_n: int = 5,       # take 1 frame every N frames
    max_frames: Optional[int] = None,
    start_sec: float = 0.0,
    end_sec: Optional[float] = None,
) -> list[FrameInfo]:
    """
    Extract sampled frames from a video file.

    Args:
        video_path:     path to input video
        output_dir:     directory to save extracted frames
        sample_every_n: stride — 1 = every frame, 5 = every 5th frame, etc.
        max_frames:     hard cap on total frames extracted
        start_sec:      start time in seconds (0 = beginning)
        end_sec:        end time in seconds (None = end of video)

    Returns:
        List of FrameInfo with metadata for each saved frame.
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_s = total_frames / fps if fps > 0 else 0

    start_frame = int(start_sec * fps)
    end_frame   = int(end_sec * fps) if end_sec is not None else total_frames

    print(f"\n[Extractor] Video info:")
    print(f"  Resolution : {width}x{height}")
    print(f"  FPS        : {fps:.2f}")
    print(f"  Duration   : {duration_s:.2f}s  ({total_frames} frames)")
    print(f"  Range      : frame {start_frame} → {end_frame}")
    print(f"  Stride     : every {sample_every_n} frames")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    extracted: list[FrameInfo] = []
    absolute_idx = start_frame

    while absolute_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        if (absolute_idx - start_frame) % sample_every_n == 0:
            ts_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            fname = f"frame_{absolute_idx:06d}.jpg"
            fpath = os.path.join(output_dir, fname)
            cv2.imwrite(fpath, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

            extracted.append(FrameInfo(
                frame_idx=absolute_idx,
                timestamp_ms=ts_ms,
                timestamp_s=ts_ms / 1000.0,
                file_path=fpath,
                width=width,
                height=height,
            ))

            if max_frames and len(extracted) >= max_frames:
                break

        absolute_idx += 1

    cap.release()

    # save metadata
    meta_path = os.path.join(output_dir, "frames_meta.json")
    with open(meta_path, "w") as f:
        json.dump([asdict(fi) for fi in extracted], f, indent=2)

    print(f"  Extracted  : {len(extracted)} frames → {output_dir}")
    return extracted


def get_video_info(video_path: str) -> dict:
    """Return basic video metadata without extracting frames."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    info = {
        "fps":           cap.get(cv2.CAP_PROP_FPS),
        "total_frames":  int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width":         int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height":        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "codec":         int(cap.get(cv2.CAP_PROP_FOURCC)),
    }
    info["duration_s"] = info["total_frames"] / info["fps"] if info["fps"] > 0 else 0
    cap.release()
    return info
