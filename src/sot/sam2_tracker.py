"""
SAM 2 Single Object Tracking backend.

Uses Meta's Segment Anything Model 2 (SAM 2) image predictor chained
frame-by-frame: the bounding box of the previous frame's mask is used as
the prompt for the next frame, creating temporal tracking with pixel-accurate
masks.

Installation:
    pip install git+https://github.com/facebookresearch/sam2.git

Model checkpoint (download once):
    https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt

Set env variable SAM2_CHECKPOINT to the checkpoint path, or pass it to
Sam2SotTracker(checkpoint=...). Default: "sam2.1_hiera_small.pt" (local).

Workflow:
  1. initialize(frame, bbox) — runs SAM 2 on the first frame, produces a mask.
  2. update(frame) per subsequent frame — expands previous bbox by 10%,
     re-runs SAM 2 to refine the mask, returns new bbox.
  3. get_mask() — returns the pixel mask from the most recent update().

Upgrade path: replace per-frame image predictor with SAM2VideoPredictor
for full bidirectional propagation across the entire video at once.
"""

import os
import cv2
import numpy as np
from typing import Optional

from .base import SotBackend

_DEFAULT_CFG   = "configs/sam2.1/sam2.1_hiera_s.yaml"
_DEFAULT_CKPT  = "sam2.1_hiera_small.pt"
_EXPAND_RATIO  = 0.10   # bbox expansion ratio between frames


class Sam2SotTracker(SotBackend):

    def __init__(
        self,
        checkpoint: Optional[str] = None,
        model_cfg:  Optional[str] = None,
    ) -> None:
        self._ckpt      = checkpoint or os.environ.get("SAM2_CHECKPOINT", _DEFAULT_CKPT)
        self._cfg       = model_cfg  or _DEFAULT_CFG
        self._predictor = None
        self._active    = False
        self._last_mask: Optional[np.ndarray] = None
        self._last_bbox: Optional[tuple]       = None
        self._frame_hw:  Optional[tuple]       = None

    # ── Lazy load ──────────────────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        if self._predictor is not None:
            return
        try:
            import torch
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError as exc:
            raise RuntimeError(
                "SAM 2 is not installed.\n"
                "  pip install git+https://github.com/facebookresearch/sam2.git\n"
                "Then download the checkpoint:\n"
                "  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt"
            ) from exc

        if not os.path.isfile(self._ckpt):
            raise FileNotFoundError(
                f"SAM 2 checkpoint not found: {self._ckpt}\n"
                "Set env SAM2_CHECKPOINT or pass checkpoint= to Sam2SotTracker."
            )

        model = build_sam2(self._cfg, self._ckpt)
        self._predictor = SAM2ImagePredictor(model)
        print(f"  [SAM2] Loaded {self._ckpt}")

    # ── Prediction helper ─────────────────────────────────────────────────────

    def _predict(self, frame_bgr: np.ndarray, prompt_box: np.ndarray):
        """Run SAM 2 image predictor; returns (mask_bool_HW | None)."""
        import torch

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        device    = next(iter(self._predictor.model.parameters())).device

        with torch.inference_mode(), torch.autocast(str(device).split(":")[0], dtype=torch.bfloat16):
            self._predictor.set_image(frame_rgb)
            masks, scores, _ = self._predictor.predict(
                point_coords=None,
                point_labels=None,
                box=prompt_box,
                multimask_output=False,
            )

        if masks is None or len(masks) == 0:
            return None
        return masks[0].astype(bool)

    def _bbox_from_mask(self, mask: np.ndarray) -> Optional[tuple]:
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return None
        return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))

    def _expand_bbox(self, bbox: tuple, frame_hw: tuple) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        h, w = frame_hw
        pad_x = int((x2 - x1) * _EXPAND_RATIO)
        pad_y = int((y2 - y1) * _EXPAND_RATIO)
        return np.array([[
            max(0, x1 - pad_x), max(0, y1 - pad_y),
            min(w, x2 + pad_x), min(h, y2 + pad_y),
        ]], dtype=float)

    # ── Interface ──────────────────────────────────────────────────────────────

    @property
    def tracking_source(self) -> str:
        return "sot_sam2"

    @property
    def is_active(self) -> bool:
        return self._active

    def initialize(self, frame: np.ndarray, bbox: tuple) -> None:
        self._ensure_loaded()
        self._frame_hw = frame.shape[:2]
        x1, y1, x2, y2 = (float(v) for v in bbox)
        prompt = np.array([[x1, y1, x2, y2]])

        mask = self._predict(frame, prompt)
        if mask is not None:
            self._last_mask = mask
            self._last_bbox = self._bbox_from_mask(mask) or tuple(int(v) for v in bbox)
        else:
            self._last_mask = None
            self._last_bbox = tuple(int(v) for v in bbox)

        self._active = True
        print(f"  [SAM2] Initialized  bbox={self._last_bbox}  mask={'yes' if mask is not None else 'no'}")

    def update(self, frame: np.ndarray, frame_idx: int = 0) -> tuple[bool, tuple]:
        if not self._active or self._last_bbox is None:
            return False, self._last_bbox or (0, 0, 0, 0)

        hw = frame.shape[:2]
        prompt = self._expand_bbox(self._last_bbox, hw)

        try:
            mask = self._predict(frame, prompt)
        except Exception as exc:
            print(f"  [SAM2] Error at frame {frame_idx}: {exc}")
            return False, self._last_bbox

        if mask is not None:
            bbox = self._bbox_from_mask(mask)
            if bbox is not None:
                self._last_mask = mask
                self._last_bbox = bbox
                return True, bbox

        # mask empty / prediction failed — keep last bbox, clear mask
        self._last_mask = None
        return False, self._last_bbox

    def get_mask(self, frame_idx: int = 0) -> Optional[np.ndarray]:
        return self._last_mask

    def reset(self) -> None:
        self._active    = False
        self._last_mask = None
        self._last_bbox = None
        self._frame_hw  = None
