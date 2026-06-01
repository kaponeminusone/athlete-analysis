"""
CSRT-family Single Object Tracking backend.

Tries trackers in this order (best accuracy → always available):
  1. cv2.TrackerCSRT_create()    — requires opencv-contrib-python
  2. cv2.TrackerMIL_create()     — available in base opencv-python
  3. cv2.TrackerKCF_create()     — available in opencv-contrib-python

Install contrib for CSRT (recommended):
  pip uninstall opencv-python -y && pip install opencv-contrib-python==4.9.0.80

Workflow:
  1. User corrects a frame via the UI → call initialize(frame, bbox).
  2. For each subsequent frame call update(frame) → new bbox for pose crop.
  3. No segmentation mask — get_mask() always returns None; the pipeline
     uses a padded-bbox crop for pose estimation.

Limitations:
  - One-directional (forward only). Backward frames still use ByteTrack.
  - Fails on drastic appearance changes (fast spin, full occlusion).
  - No pixel mask — pose quality depends entirely on bbox accuracy.
"""

import cv2
import numpy as np
from typing import Optional

from .base import SotBackend


def _make_opencv_tracker():
    """Create the best available OpenCV correlation tracker."""
    for factory, name in [
        (getattr(cv2, "TrackerCSRT_create", None),  "CSRT"),
        (getattr(cv2, "TrackerKCF_create",  None),  "KCF"),
        (getattr(cv2, "TrackerMIL_create",  None),  "MIL"),
    ]:
        if factory is not None:
            try:
                tracker = factory()
                return tracker, name
            except Exception:
                continue
    raise RuntimeError(
        "No OpenCV tracker available. Install opencv-contrib-python:\n"
        "  pip uninstall opencv-python -y\n"
        "  pip install opencv-contrib-python==4.9.0.80"
    )


class CsrtSotTracker(SotBackend):
    """
    OpenCV correlation-filter tracker (CSRT preferred, MIL fallback).
    Activated by the UI after a manual correction; tracks forward only.
    """

    def __init__(self) -> None:
        self._tracker   = None
        self._algo_name = "unknown"
        self._active    = False
        self._last_bbox: Optional[tuple] = None

    # ── Interface ──────────────────────────────────────────────────────────────

    @property
    def tracking_source(self) -> str:
        return "sot_csrt"

    @property
    def is_active(self) -> bool:
        return self._active

    def initialize(self, frame: np.ndarray, bbox: tuple) -> None:
        x1, y1, x2, y2 = (int(v) for v in bbox)
        self._tracker, self._algo_name = _make_opencv_tracker()
        # OpenCV trackers expect (x, y, width, height)
        self._tracker.init(frame, (x1, y1, x2 - x1, y2 - y1))
        self._last_bbox = (x1, y1, x2, y2)
        self._active    = True
        print(f"  [CSRT/{self._algo_name}] Initialized  bbox=({x1},{y1},{x2},{y2})")

    def update(self, frame: np.ndarray, frame_idx: int = 0) -> tuple[bool, tuple]:
        if not self._active or self._tracker is None:
            return False, self._last_bbox or (0, 0, 0, 0)

        ok, rect = self._tracker.update(frame)
        if ok:
            x, y, w, h = (int(v) for v in rect)
            self._last_bbox = (x, y, x + w, y + h)
            return True, self._last_bbox

        self._active = False
        print(f"  [CSRT/{self._algo_name}] Tracking lost at frame {frame_idx}")
        return False, self._last_bbox or (0, 0, 0, 0)

    def get_mask(self, frame_idx: int = 0) -> Optional[np.ndarray]:
        return None  # correlation trackers produce no pixel mask

    def reset(self) -> None:
        self._tracker   = None
        self._active    = False
        self._last_bbox = None
