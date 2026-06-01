"""
Single Object Tracking (SOT) — interchangeable backends.

Usage:
    from src.sot import create_sot

    sot = create_sot("csrt")   # or "sam2"
    sot.initialize(init_frame, bbox)
    for frame in frames:
        ok, bbox = sot.update(frame, frame_idx)
        mask = sot.get_mask(frame_idx)   # None for CSRT, np.ndarray for SAM 2
"""

from .base         import SotBackend
from .csrt_tracker import CsrtSotTracker


def create_sot(backend: str) -> SotBackend:
    """
    Factory function.  Import SAM 2 lazily so CSRT works without sam2 installed.

    backend: "csrt" | "sam2"
    """
    if backend == "csrt":
        return CsrtSotTracker()
    if backend == "sam2":
        from .sam2_tracker import Sam2SotTracker   # lazy — heavy import
        return Sam2SotTracker()
    raise ValueError(f"Unknown SOT backend: {backend!r}  (choose 'csrt' or 'sam2')")


__all__ = ["SotBackend", "CsrtSotTracker", "create_sot"]
