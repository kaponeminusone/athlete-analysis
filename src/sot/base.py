"""
Abstract base class for Single Object Tracking backends.

Both CSRT and SAM 2 implement this interface so pipeline.py and
correction.py never need to know which backend is active.
"""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class SotBackend(ABC):

    @property
    @abstractmethod
    def tracking_source(self) -> str:
        """Identifier stored in analysis.json per frame."""
        ...

    @property
    @abstractmethod
    def is_active(self) -> bool:
        """True after initialize(), False after reset() or tracking loss."""
        ...

    @abstractmethod
    def initialize(self, frame: np.ndarray, bbox: tuple) -> None:
        """
        Seed the tracker with the corrected frame and bounding box.

        bbox: (x1, y1, x2, y2) in pixel coordinates, full-frame space.
        Must be called once before any update() calls.
        """
        ...

    @abstractmethod
    def update(self, frame: np.ndarray, frame_idx: int = 0) -> tuple[bool, tuple]:
        """
        Advance tracker by one frame.

        Returns (success, bbox) where bbox is (x1, y1, x2, y2).
        If success is False, bbox is the last known good bbox.
        """
        ...

    @abstractmethod
    def get_mask(self, frame_idx: int = 0) -> Optional[np.ndarray]:
        """
        Return the segmentation mask for the most recently updated frame.

        Returns bool ndarray [H, W] or None if the backend has no mask.
        Must be called after update() for the same frame.
        """
        ...

    def reset(self) -> None:
        """Release resources and go back to uninitialized state."""
