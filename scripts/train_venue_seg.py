#!/usr/bin/env python3
"""
Export venue calibration keyframes to YOLO11-seg format and fine-tune.

  python scripts/train_venue_seg.py export --video VOD2 --calibration output/VOD2/calibration.json
  python scripts/train_venue_seg.py train --venue-id default --epochs 40 --imgsz 640

Classes: 0=track, 1=sand
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.venue_masks import polygon_keyframes
from src.venue_seg_infer import clear_model_cache

CLASS_TRACK = 0
CLASS_SAND = 1
DEFAULT_VENUE_ID = "default"
ProgressCallback = Callable[[dict], None]


def _resolve_video(video_name: str, video_path: str | Path | None) -> Path:
    if video_path:
        p = Path(video_path)
        if p.exists():
            return p
    for ext in (".mp4", ".mov", ".MP4", ".MOV"):
        candidate = ROOT / f"{video_name}{ext}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Video not found for {video_name}")


def _polygon_to_yolo_line(cls: int, points: list[list[float]]) -> str:
    coords = " ".join(f"{p[0]:.6f} {p[1]:.6f}" for p in points)
    return f"{cls} {coords}"


def manifest_path(venue_id: str = DEFAULT_VENUE_ID) -> Path:
    return ROOT / "venues" / venue_id / "dataset_manifest.json"


def dataset_root(venue_id: str = DEFAULT_VENUE_ID) -> Path:
    return ROOT / "venues" / venue_id / "dataset"


def load_dataset_manifest(venue_id: str = DEFAULT_VENUE_ID) -> dict[str, Any]:
    path = manifest_path(venue_id)
    if not path.exists():
        return {"videos": [], "total_frames": 0}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_dataset_manifest(manifest: dict[str, Any], venue_id: str = DEFAULT_VENUE_ID) -> Path:
    path = manifest_path(venue_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return path


def _ensure_dataset_dirs(out_root: Path) -> tuple[Path, Path, Path, Path]:
    images_train = out_root / "images" / "train"
    images_val = out_root / "images" / "val"
    labels_train = out_root / "labels" / "train"
    labels_val = out_root / "labels" / "val"
    for d in (images_train, images_val, labels_train, labels_val):
        d.mkdir(parents=True, exist_ok=True)
    return images_train, images_val, labels_train, labels_val


def _write_dataset_yaml(out_root: Path, venue_id: str) -> Path:
    yaml_path = ROOT / "venues" / venue_id / "dataset.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(
        f"path: {out_root.as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n"
        f"  0: track\n"
        f"  1: sand\n",
        encoding="utf-8",
    )
    return yaml_path


def _remove_video_from_dataset(out_root: Path, video_name: str) -> None:
    prefix = f"{video_name}_"
    for split in ("train", "val"):
        for kind in ("images", "labels"):
            directory = out_root / kind / split
            if not directory.exists():
                continue
            for path in directory.iterdir():
                if path.name.startswith(prefix):
                    path.unlink()


def _export_video_keyframes(
    video_name: str,
    calibration: dict[str, Any],
    video_file: Path,
    out_root: Path,
    *,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> int:
    """Export one video's polygon keyframes; returns number of frames written."""
    images_train, images_val, labels_train, labels_val = _ensure_dataset_dirs(out_root)
    keyframes = polygon_keyframes(calibration)
    if not keyframes:
        return 0

    rng = random.Random(seed)
    indices = list(range(len(keyframes)))
    rng.shuffle(indices)
    val_count = max(1, int(len(keyframes) * val_ratio)) if len(keyframes) > 1 else 0
    val_set = set(indices[:val_count])

    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_file}")

    exported = 0
    try:
        for i, kf in enumerate(keyframes):
            frame_idx = int(kf["frame_idx"])
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            stem = f"{video_name}_{frame_idx:06d}"
            is_val = i in val_set
            img_dir = images_val if is_val else images_train
            lbl_dir = labels_val if is_val else labels_train

            cv2.imwrite(str(img_dir / f"{stem}.jpg"), frame)
            lines: list[str] = []
            track = kf.get("track_polygon") or []
            landing = kf.get("landing_zone") or []
            if len(track) >= 3:
                lines.append(_polygon_to_yolo_line(CLASS_TRACK, track))
            if len(landing) >= 3:
                lines.append(_polygon_to_yolo_line(CLASS_SAND, landing))
            (lbl_dir / f"{stem}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""),
                encoding="utf-8",
            )
            exported += 1
    finally:
        cap.release()

    return exported


def _update_manifest_entry(
    manifest: dict[str, Any],
    video_name: str,
    frames_exported: int,
) -> dict[str, Any]:
    videos = [v for v in manifest.get("videos", []) if v.get("video_name") != video_name]
    videos.append({
        "video_name": video_name,
        "frames_exported": frames_exported,
        "exported_at": datetime.now(timezone.utc).isoformat(),
    })
    total_frames = sum(int(v.get("frames_exported", 0)) for v in videos)
    return {"videos": videos, "total_frames": total_frames}


def export_dataset_append(
    video_name: str,
    calibration_path: Path,
    *,
    video_path: Path | None = None,
    venue_id: str = DEFAULT_VENUE_ID,
    val_ratio: float = 0.2,
    seed: int = 42,
    on_progress: Optional[ProgressCallback] = None,
) -> dict[str, Any]:
    """Append one video's keyframes to the shared dataset without wiping other videos."""
    if on_progress:
        on_progress({"stage": "export", "message": f"Añadiendo {video_name} al dataset CNN..."})

    with open(calibration_path, encoding="utf-8") as f:
        calibration = json.load(f)

    out_root = dataset_root(venue_id)
    _remove_video_from_dataset(out_root, video_name)

    video_file = video_path or _resolve_video(video_name, None)
    exported = _export_video_keyframes(
        video_name,
        calibration,
        video_file,
        out_root,
        val_ratio=val_ratio,
        seed=seed,
    )
    if exported == 0:
        raise ValueError(f"No keyframes with track_polygon in calibration for {video_name}")

    manifest = load_dataset_manifest(venue_id)
    manifest = _update_manifest_entry(manifest, video_name, exported)
    save_dataset_manifest(manifest, venue_id)
    yaml_path = _write_dataset_yaml(out_root, venue_id)

    if on_progress:
        on_progress({
            "stage": "export",
            "message": f"{video_name}: {exported} frames · dataset total {manifest['total_frames']}",
            "percent": 10.0,
        })

    print(f"Appended {exported} keyframes from {video_name} to {out_root}")
    print(f"Dataset manifest: {manifest_path(venue_id)} ({manifest['total_frames']} total frames)")
    print(f"Dataset config: {yaml_path}")
    return {
        "frames_exported": exported,
        "videos_in_dataset": len(manifest["videos"]),
        "total_dataset_frames": manifest["total_frames"],
        "dataset_manifest": manifest,
        "dataset_root": str(out_root),
    }


def export_dataset_rebuild(
    venue_id: str = DEFAULT_VENUE_ID,
    *,
    manifest: dict[str, Any] | None = None,
    val_ratio: float = 0.2,
    seed: int = 42,
    on_progress: Optional[ProgressCallback] = None,
) -> Path:
    """Clear dataset and re-export all videos listed in the manifest."""
    if on_progress:
        on_progress({"stage": "export", "message": "Reconstruyendo dataset CNN desde manifiesto..."})

    manifest = manifest or load_dataset_manifest(venue_id)
    out_root = dataset_root(venue_id)
    if out_root.exists():
        shutil.rmtree(out_root)
    _ensure_dataset_dirs(out_root)

    rebuilt_videos: list[dict[str, Any]] = []
    total_exported = 0
    for entry in manifest.get("videos", []):
        video_name = entry.get("video_name")
        if not video_name:
            continue
        cal_path = ROOT / "output" / video_name / "calibration.json"
        if not cal_path.exists():
            print(f"[export_dataset_rebuild] Skipping {video_name}: no calibration.json")
            continue
        try:
            with open(cal_path, encoding="utf-8") as f:
                calibration = json.load(f)
            video_file = _resolve_video(video_name, None)
            exported = _export_video_keyframes(
                video_name,
                calibration,
                video_file,
                out_root,
                val_ratio=val_ratio,
                seed=seed,
            )
        except Exception as exc:
            print(f"[export_dataset_rebuild] Skipping {video_name}: {exc}")
            continue
        if exported <= 0:
            continue
        rebuilt_videos.append({
            "video_name": video_name,
            "frames_exported": exported,
            "exported_at": datetime.now(timezone.utc).isoformat(),
        })
        total_exported += exported

    new_manifest = {"videos": rebuilt_videos, "total_frames": total_exported}
    save_dataset_manifest(new_manifest, venue_id)
    _write_dataset_yaml(out_root, venue_id)

    if on_progress:
        on_progress({
            "stage": "export",
            "message": f"Dataset reconstruido: {total_exported} frames de {len(rebuilt_videos)} videos",
            "percent": 10.0,
        })

    print(f"Rebuilt dataset: {total_exported} frames from {len(rebuilt_videos)} videos at {out_root}")
    return out_root


def export_dataset(
    video_name: str,
    calibration_path: Path,
    *,
    video_path: Path | None = None,
    venue_id: str | None = None,
    val_ratio: float = 0.2,
    seed: int = 42,
    on_progress: Optional[ProgressCallback] = None,
    append: bool = True,
) -> Path:
    """Export keyframe frames + polygon labels to venues/<id>/dataset/."""
    venue_id = venue_id or DEFAULT_VENUE_ID
    if append:
        result = export_dataset_append(
            video_name,
            calibration_path,
            video_path=video_path,
            venue_id=venue_id,
            val_ratio=val_ratio,
            seed=seed,
            on_progress=on_progress,
        )
        return Path(result["dataset_root"])

    if on_progress:
        on_progress({"stage": "export", "message": "Exportando keyframes a dataset YOLO (solo este video)..."})

    with open(calibration_path, encoding="utf-8") as f:
        calibration = json.load(f)

    out_root = dataset_root(venue_id)
    if out_root.exists():
        shutil.rmtree(out_root)
    _ensure_dataset_dirs(out_root)

    video_file = video_path or _resolve_video(video_name, None)
    exported = _export_video_keyframes(
        video_name,
        calibration,
        video_file,
        out_root,
        val_ratio=val_ratio,
        seed=seed,
    )
    if exported == 0:
        raise ValueError("No keyframes with track_polygon in calibration")

    manifest = _update_manifest_entry({"videos": [], "total_frames": 0}, video_name, exported)
    save_dataset_manifest(manifest, venue_id)
    yaml_path = _write_dataset_yaml(out_root, venue_id)

    if on_progress:
        on_progress({
            "stage": "export",
            "message": f"Exportados {exported} keyframes",
            "percent": 10.0,
        })

    print(f"Exported {exported} keyframes to {out_root}")
    print(f"Dataset config: {yaml_path}")
    return out_root


def get_dataset_info(venue_id: str = DEFAULT_VENUE_ID) -> dict[str, Any]:
    manifest = load_dataset_manifest(venue_id)
    total_frames = int(manifest.get("total_frames", 0))
    video_count = len(manifest.get("videos", []))
    return {
        "venue_id": venue_id,
        "videos": manifest.get("videos", []),
        "total_frames": total_frames,
        "video_count": video_count,
        "ready_to_train": total_frames >= 10 and video_count >= 1,
        "can_train": total_frames >= 5,
        "manifest": manifest,
    }


def train_venue_model(
    venue_id: str,
    *,
    epochs: int = 40,
    imgsz: int = 640,
    model_name: str = "yolo11n-seg.pt",
    on_progress: Optional[ProgressCallback] = None,
) -> dict[str, Any]:
    """Fine-tune YOLO11-seg and write venues/<id>/model.json."""
    yaml_path = ROOT / "venues" / venue_id / "dataset.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"Dataset not found: {yaml_path}. Run export first.")

    if on_progress:
        on_progress({
            "stage": "train",
            "message": f"Entrenando YOLO11-seg ({epochs} epochs)...",
            "percent": 15.0,
        })

    from ultralytics import YOLO

    project_dir = ROOT / "venues" / venue_id / "runs"
    model = YOLO(model_name)
    model.train(
        data=str(yaml_path),
        epochs=epochs,
        imgsz=imgsz,
        project=str(project_dir),
        name="seg",
        exist_ok=True,
        verbose=True,
    )

    best_pt = project_dir / "seg" / "weights" / "best.pt"
    if not best_pt.exists():
        raise FileNotFoundError(f"Training finished but weights not found: {best_pt}")

    weights_rel = best_pt.relative_to(ROOT).as_posix()
    meta: dict[str, Any] = {
        "weights": weights_rel,
        "classes": ["track", "sand"],
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "epochs": epochs,
        "imgsz": imgsz,
        "base_model": model_name,
    }

    metrics: dict[str, Any] = {}
    try:
        if hasattr(model, "trainer") and model.trainer is not None:
            metrics = dict(model.trainer.metrics or {})
        elif hasattr(model, "metrics") and model.metrics:
            metrics = dict(model.metrics)
    except Exception:
        pass
    if metrics:
        meta["metrics"] = metrics

    model_json = ROOT / "venues" / venue_id / "model.json"
    model_json.parent.mkdir(parents=True, exist_ok=True)
    with open(model_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    clear_model_cache(venue_id)

    if on_progress:
        on_progress({
            "stage": "done",
            "message": "Entrenamiento CNN completado",
            "percent": 100.0,
        })

    print(f"Best weights: {best_pt}")
    print(f"Model metadata: {model_json}")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Venue segmentation export + YOLO11-seg training")
    sub = parser.add_subparsers(dest="command", required=True)

    export_p = sub.add_parser("export", help="Export calibration keyframes to YOLO-seg dataset")
    export_p.add_argument("--video", required=True, help="Video name (e.g. VOD2)")
    export_p.add_argument("--calibration", type=Path, required=True, help="Path to calibration.json")
    export_p.add_argument("--video-path", type=Path, default=None)
    export_p.add_argument("--venue-id", default=DEFAULT_VENUE_ID)
    export_p.add_argument("--val-ratio", type=float, default=0.2)
    export_p.add_argument("--no-append", action="store_true", help="Replace entire dataset with this video only")

    train_p = sub.add_parser("train", help="Fine-tune YOLO11-seg on exported dataset")
    train_p.add_argument("--venue-id", default=DEFAULT_VENUE_ID)
    train_p.add_argument("--epochs", type=int, default=40)
    train_p.add_argument("--imgsz", type=int, default=640)
    train_p.add_argument("--model", default="yolo11n-seg.pt")
    train_p.add_argument("--rebuild", action="store_true", help="Rebuild dataset from manifest before training")

    args = parser.parse_args()
    if args.command == "export":
        export_dataset(
            args.video,
            args.calibration,
            video_path=args.video_path,
            venue_id=args.venue_id,
            val_ratio=args.val_ratio,
            append=not args.no_append,
        )
    elif args.command == "train":
        if args.rebuild:
            export_dataset_rebuild(args.venue_id)
        train_venue_model(
            args.venue_id,
            epochs=args.epochs,
            imgsz=args.imgsz,
            model_name=args.model,
        )


if __name__ == "__main__":
    main()
