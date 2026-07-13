"""
Visualizer: draws segmentation mask, skeleton, keypoints, bounding box,
camera angle label, tracking info and quality score on a frame.
Also generates a summary timeline chart.
"""

import cv2
import numpy as np
import os
from typing import Optional

from .pose_analyzer import FrameAnalysis, CameraAngle, KP, CONF_THRESHOLD


# ─── Colors (BGR) ────────────────────────────────────────────────────────────
COLORS = {
    CameraAngle.FRONTAL:    (0,   200, 255),   # yellow
    CameraAngle.SEMI_FRONT: (0,   180, 80),    # green
    CameraAngle.LATERAL:    (255, 120, 0),     # blue  ← best
    CameraAngle.SEMI_BACK:  (0,   100, 220),   # orange
    CameraAngle.UNKNOWN:    (100, 100, 100),   # grey
}

# Mask overlay color per angle (BGR, semi-transparent)
MASK_COLORS = {
    CameraAngle.LATERAL:    (255, 180, 60),
    CameraAngle.SEMI_BACK:  (60,  180, 255),
    CameraAngle.SEMI_FRONT: (60,  220, 100),
    CameraAngle.FRONTAL:    (60,  220, 255),
    CameraAngle.UNKNOWN:    (160, 160, 160),
}

SKELETON = [
    ("l_shoulder", "r_shoulder"),
    ("l_shoulder", "l_hip"),
    ("r_shoulder", "r_hip"),
    ("l_hip",      "r_hip"),
    ("l_shoulder", "l_elbow"),
    ("r_shoulder", "r_elbow"),
    ("l_hip",      "l_knee"),
    ("r_hip",      "r_knee"),
    ("l_knee",     "l_ankle"),
    ("r_knee",     "r_ankle"),
]

KP_COLORS = {
    "nose":       (255, 255, 255),
    "l_shoulder": (100, 255, 150),
    "r_shoulder": (100, 150, 255),
    "l_elbow":    (60,  200, 100),
    "r_elbow":    (60,  100, 200),
    "l_hip":      (255, 210, 60),
    "r_hip":      (210, 255, 60),
    "l_knee":     (255, 120, 60),
    "r_knee":     (200, 80,  60),
    "l_ankle":    (255, 60,  80),
    "r_ankle":    (200, 60,  220),
}


def _text(img, text: str, pos: tuple, color: tuple,
          scale: float = 0.55, thickness: int = 1, bg: bool = True):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), bl = cv2.getTextSize(text, font, scale, thickness)
    x, y = int(pos[0]), int(pos[1])
    if bg:
        cv2.rectangle(img, (x-2, y-th-3), (x+tw+2, y+bl+1), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)
    return th + bl + 4   # height consumed


def _draw_seg_mask(img: np.ndarray, mask: np.ndarray,
                   color_bgr: tuple, alpha: float = 0.38) -> np.ndarray:
    """Blend a boolean mask as a colored transparent overlay onto img."""
    overlay = img.copy()
    overlay[mask] = (
        np.array(overlay[mask], dtype=np.float32) * (1 - alpha)
        + np.array(color_bgr, dtype=np.float32) * alpha
    ).astype(np.uint8)

    # draw mask contour
    mask_u8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, color_bgr, 2)
    return overlay


def annotate_frame_array(
    img: np.ndarray,
    analysis: FrameAnalysis,
    seg_mask: Optional[np.ndarray] = None,   # bool [H,W] from tracker
    appearance_sim: float = 0.0,
    draw_skeleton: bool = True,
    draw_bbox: bool = True,
) -> np.ndarray:
    """
    Dibuja todas las anotaciones sobre una imagen BGR EN MEMORIA (sin IO a disco).
    Devuelve la imagen anotada (numpy BGR). Esta es la única lógica de dibujo;
    annotate_frame() sólo añade lectura/escritura de archivos por encima.

    Nota: dibuja parcialmente in-place. Pasa una copia si necesitas conservar el
    original intacto.
    """
    angle_color = COLORS.get(analysis.camera_angle, (128, 128, 128))
    mask_color  = MASK_COLORS.get(analysis.camera_angle, (160, 160, 160))

    # ── Segmentation mask overlay ─────────────────────────────────────────────
    if seg_mask is not None and seg_mask.any():
        img = _draw_seg_mask(img, seg_mask, mask_color, alpha=0.35)

    # ── Bounding box (tight, from mask) ───────────────────────────────────────
    if draw_bbox and analysis.person_bbox:
        x1, y1, x2, y2 = [int(v) for v in analysis.person_bbox]
        cv2.rectangle(img, (x1, y1), (x2, y2), angle_color, 2)
        # Track ID label on top of bbox
        if analysis.track_id is not None:
            tid_label = f"ID:{analysis.track_id}"
            (tw, th), _ = cv2.getTextSize(tid_label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(img, (x1, y1-th-6), (x1+tw+4, y1), angle_color, -1)
            cv2.putText(img, tid_label, (x1+2, y1-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)

    # ── Skeleton lines ────────────────────────────────────────────────────────
    if draw_skeleton and analysis.person_detected:
        kps = analysis.keypoints
        for (a_name, b_name) in SKELETON:
            a = kps.get(a_name)
            b = kps.get(b_name)
            if a and b and a.valid and b.valid:
                cv2.line(img, (int(a.x), int(a.y)),
                         (int(b.x), int(b.y)), (230, 230, 230), 2, cv2.LINE_AA)

    # ── Keypoints ─────────────────────────────────────────────────────────────
    if analysis.person_detected:
        for name, kp in analysis.keypoints.items():
            pt = (int(kp.x), int(kp.y))
            if not kp.valid:
                cv2.circle(img, pt, 3, (50, 50, 50), -1)
                continue
            color = KP_COLORS.get(name, (200, 200, 200))
            cv2.circle(img, pt, 6, color, -1)
            cv2.circle(img, pt, 7, (0, 0, 0), 1)
            # small confidence label
            cv2.putText(img, f"{kp.conf:.0%}", (pt[0]+7, pt[1]+4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, color, 1, cv2.LINE_AA)

    # ── Info panel (top-left) ─────────────────────────────────────────────────
    panel_w, panel_h = 340, 185
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)

    y = 20

    # ── Row 1 — detection type (prominent, replaces the old "Angle" header) ────
    src = getattr(analysis, "tracking_source", "bytetrack")
    src_labels = {
        "bytetrack": "ByteTrack",
        "sot_csrt":  "SOT:CSRT",
        "sot_sam2":  "SOT:SAM2",
    }
    src_colors = {
        "bytetrack": (140, 140, 140),
        "sot_csrt":  (60,  220, 255),
        "sot_sam2":  (80,  255, 160),
    }
    src_label = src_labels.get(src, src)
    src_color = src_colors.get(src, (140, 140, 140))

    if analysis.manually_corrected:
        ct = analysis.correction_source or "manual"
        ct_display = {
            "bbox_correction":  "BBOX",
            "click_selection":  "CLICK",
            "mask_correction":  "MASK",
        }.get(ct, ct.upper())
        det_label = f"CORREGIDO [{ct_display}]  {src_label}"
        det_color = (60, 80, 255)
        # solid filled bar so it stands out
        (tw, th), _ = cv2.getTextSize(det_label, cv2.FONT_HERSHEY_SIMPLEX, 0.60, 2)
        cv2.rectangle(img, (6, y - th - 4), (panel_w - 6, y + 4), det_color, -1)
        cv2.putText(img, det_label, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2, cv2.LINE_AA)
        y += th + 10
    else:
        y += _text(img, src_label, (8, y), src_color, 0.60, 2)

    # ── Row 2 — camera angle ──────────────────────────────────────────────────
    angle_label = analysis.camera_angle.value
    usable_tag  = "  [USABLE]" if analysis.usable_for_analysis else "  [skip]"
    y += _text(img, f"Angle: {angle_label}{usable_tag}", (8, y), angle_color, 0.50, 1)

    # ── Row 3 — track ID + appearance ────────────────────────────────────────
    tid_str = f"Track ID: {analysis.track_id}" if analysis.track_id is not None else "Track ID: --"
    app_str = f"  Appear: {appearance_sim:.0%}" if appearance_sim > 0 else ""
    y += _text(img, tid_str + app_str, (8, y), (200, 200, 200), 0.50, 1)

    # ── Row 4 — quality score with colored bar ────────────────────────────────
    q = analysis.quality_score
    q_color = (60, 220, 60) if q >= 0.70 else (60, 180, 255) if q >= 0.50 else (60, 60, 220)
    y += _text(img, f"Quality: {q:.2f}/1.00", (8, y), q_color, 0.50, 1)

    bar_x, bar_y, bar_w, bar_h = 8, y, 180, 7
    cv2.rectangle(img, (bar_x, bar_y), (bar_x+bar_w, bar_y+bar_h), (50,50,50), -1)
    fill = int(np.clip(q, 0, 1) * bar_w)
    cv2.rectangle(img, (bar_x, bar_y), (bar_x+fill, bar_y+bar_h), q_color, -1)
    y += bar_h + 6

    # ── Row 5 — keypoints ─────────────────────────────────────────────────────
    kp_str = f"Keypoints: {analysis.keypoints_valid_count}/11 valid"
    y += _text(img, kp_str, (8, y), (200, 200, 200), 0.50, 1)

    # ── Row 6 — shoulder ratio + mask ─────────────────────────────────────────
    ratio_str = f"Shoulder ratio: {analysis.shoulder_ratio:.2f}"
    mask_str  = f"  Mask: {analysis.mask_area_px:,}px" if analysis.mask_area_px > 0 else ""
    y += _text(img, ratio_str + mask_str, (8, y), (180, 180, 180), 0.45, 1)

    # ── Row 7 — frame / timestamp ─────────────────────────────────────────────
    _text(img, f"Frame {analysis.frame_idx:06d}  t={analysis.timestamp_s:.2f}s",
          (8, y), (140, 140, 140), 0.45, 1)

    # ── Shoulder ratio bar (lateral indicator) ────────────────────────────────
    bx, by = 8, panel_h - 16
    cv2.rectangle(img, (bx, by), (bx+300, by+7), (40,40,40), -1)
    fill2 = int(np.clip(analysis.shoulder_ratio/1.5, 0, 1) * 300)
    cv2.rectangle(img, (bx, by), (bx+fill2, by+7), angle_color, -1)
    cv2.putText(img, "LAT", (bx, by+20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (160,160,160), 1)
    cv2.putText(img, "FRONT", (bx+265, by+20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (160,160,160), 1)

    # ── Legend (bottom-right corner) ──────────────────────────────────────────
    lx, ly = img.shape[1] - 160, img.shape[0] - 100
    for i, (angle, col) in enumerate(COLORS.items()):
        cv2.rectangle(img, (lx, ly + i*18), (lx+12, ly + i*18 + 12), col, -1)
        cv2.putText(img, angle.value, (lx+16, ly + i*18 + 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200,200,200), 1)

    return img


def annotate_frame(
    image_path: str,
    analysis: FrameAnalysis,
    output_path: str,
    seg_mask: Optional[np.ndarray] = None,   # bool [H,W] from tracker
    appearance_sim: float = 0.0,
    draw_skeleton: bool = True,
    draw_bbox: bool = True,
) -> np.ndarray:
    """
    Load a frame, draw all annotations, save to output_path.
    Returns the annotated image (numpy BGR).
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    img = annotate_frame_array(
        img, analysis,
        seg_mask=seg_mask,
        appearance_sim=appearance_sim,
        draw_skeleton=draw_skeleton,
        draw_bbox=draw_bbox,
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, img)
    return img


def generate_timeline_chart(
    analyses: list[FrameAnalysis],
    output_path: str,
    fps: float = 30.0,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("  [Visualizer] matplotlib not installed — skipping timeline chart.")
        return

    timestamps  = [a.timestamp_s for a in analyses]
    ratios      = [a.shoulder_ratio for a in analyses]
    valid_kps   = [a.keypoints_valid_count for a in analyses]
    quality     = [a.quality_score for a in analyses]
    angles      = [a.camera_angle for a in analyses]
    usable      = [1.0 if a.usable_for_analysis else 0.0 for a in analyses]

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    fig.patch.set_facecolor("#1a1a2e")
    for ax in axes:
        ax.set_facecolor("#16213e")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for sp in ax.spines.values():
            sp.set_edgecolor("#444")

    color_map = {
        CameraAngle.LATERAL:    "#ff7800",
        CameraAngle.SEMI_BACK:  "#1e90ff",
        CameraAngle.SEMI_FRONT: "#2e7d32",
        CameraAngle.FRONTAL:    "#f9a825",
        CameraAngle.UNKNOWN:    "#424242",
    }

    # ── Plot 1: angle bands ───────────────────────────────────────────────────
    ax1 = axes[0]
    for i in range(len(timestamps) - 1):
        ax1.axvspan(timestamps[i], timestamps[i+1],
                    color=color_map[angles[i]], alpha=0.85)
    patches = [mpatches.Patch(color=color_map[a], label=a.value)
               for a in CameraAngle]
    ax1.legend(handles=patches, fontsize=7, facecolor="#0f3460",
               labelcolor="white", loc="upper right", ncol=5)
    ax1.set_ylabel("Camera angle", color="white")
    ax1.set_title("Triple Jump Video — Analysis Pipeline", color="white", fontsize=11)
    ax1.set_yticks([])

    # ── Plot 2: shoulder ratio ────────────────────────────────────────────────
    ax2 = axes[1]
    ax2.plot(timestamps, ratios, color="#00bcd4", linewidth=1.5)
    for thr, lbl, col in [(0.80, "frontal", "#f9a825"),
                           (0.50, "semi-front", "#4caf50"),
                           (0.20, "lateral", "#ff7800")]:
        ax2.axhline(thr, color=col, linestyle="--", linewidth=0.8, alpha=0.7, label=lbl)
    ax2.set_ylabel("Shoulder ratio", color="white")
    ax2.set_ylim(0, 1.6)
    ax2.legend(fontsize=7, facecolor="#0f3460", labelcolor="white", loc="upper right")

    # ── Plot 3: quality score + usable flag ───────────────────────────────────
    ax3 = axes[2]
    ax3.plot(timestamps, quality, color="#7c4dff", linewidth=1.5, label="quality score")
    ax3.fill_between(timestamps, quality, alpha=0.25, color="#7c4dff")
    # usable frames as green dots
    usable_ts = [timestamps[i] for i, u in enumerate(usable) if u]
    usable_q  = [quality[i]    for i, u in enumerate(usable) if u]
    ax3.scatter(usable_ts, usable_q, color="#00e676", s=12, zorder=5,
                label="usable for analysis")
    ax3.axhline(0.55, color="#ef5350", linestyle=":", linewidth=0.8,
                label="usable threshold (0.55)")
    ax3.set_ylim(0, 1.05)
    ax3.set_ylabel("Quality score", color="white")
    ax3.legend(fontsize=7, facecolor="#0f3460", labelcolor="white", loc="lower right")

    # ── Plot 4: valid keypoints ───────────────────────────────────────────────
    ax4 = axes[3]
    ax4.fill_between(timestamps, valid_kps, color="#b39ddb", alpha=0.5, step="mid")
    ax4.plot(timestamps, valid_kps, color="#b39ddb", linewidth=1, drawstyle="steps-mid")
    ax4.axhline(8, color="#ef5350", linestyle=":", linewidth=0.8,
                label="min reliable (8/11)")
    ax4.set_ylim(0, 12)
    ax4.set_ylabel("Valid keypoints", color="white")
    ax4.set_xlabel("Time (seconds)", color="white")
    ax4.legend(fontsize=7, facecolor="#0f3460", labelcolor="white")

    plt.tight_layout(pad=1.5)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print(f"  [Visualizer] Timeline → {output_path}")
