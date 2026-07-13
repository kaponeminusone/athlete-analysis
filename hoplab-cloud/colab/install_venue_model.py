"""
Instalar / verificar modelo CNN de venue bajo DATA_ROOT/venues/default.

Uso en Colab (tras celda de env, con DATA_ROOT y ENGINE_DIR definidos)::

    %run /content/.../hoplab-cloud/colab/install_venue_model.py
    # o:
    import install_venue_model
    install_venue_model.main()

Prioridad de origen (copia hacia DATA_ROOT/venues/default):
  1) DATA_ROOT/venues-upload/  (subí ahí el zip descomprimido o los archivos)
  2) ENGINE_DIR/venues/default si best.pt existe (repo local con pesos)
  3) Ya presentes en DATA_ROOT/venues/default → solo verifica

Archivos esperados:
  model.json, profile.json, scale.json (opcional),
  runs/seg/weights/best.pt
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

VENUE_ID = "default"
WEIGHTS_REL = Path("runs") / "seg" / "weights" / "best.pt"
META_FILES = ("model.json", "profile.json", "scale.json")


def _data_root() -> Path:
    venue = os.environ.get("HOPLAB_VENUE_ROOT")
    if venue:
        return Path(venue).resolve().parent
    out = os.environ.get("HOPLAB_OUTPUT_ROOT")
    if out:
        return Path(out).resolve().parent
    return Path("/content/drive/MyDrive/hoplab-data")


def _engine_dir() -> Path | None:
    if "ENGINE_DIR" in globals():
        return Path(globals()["ENGINE_DIR"]).resolve()  # type: ignore[name-defined]
    cwd = Path.cwd()
    if (cwd / "api_server.py").exists():
        return cwd
    return None


def _ensure_layout(dest: Path) -> None:
    (dest / "runs" / "seg" / "weights").mkdir(parents=True, exist_ok=True)


def _copy_tree_files(src_root: Path, dest: Path) -> list[str]:
    copied: list[str] = []
    _ensure_layout(dest)
    for name in META_FILES:
        s = src_root / name
        if s.is_file():
            shutil.copy2(s, dest / name)
            copied.append(name)
    # También aceptar layout plano en venues-upload
    flat_pt = src_root / "best.pt"
    nested_pt = src_root / WEIGHTS_REL
    dest_pt = dest / WEIGHTS_REL
    if nested_pt.is_file():
        dest_pt.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(nested_pt, dest_pt)
        copied.append(str(WEIGHTS_REL))
    elif flat_pt.is_file():
        dest_pt.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(flat_pt, dest_pt)
        copied.append("best.pt → " + str(WEIGHTS_REL))
    return copied


def _normalize_model_json(dest: Path) -> None:
    """Asegura weights relativo venues/default/runs/seg/weights/best.pt."""
    meta_path = dest / "model.json"
    weights_rel = f"venues/{VENUE_ID}/runs/seg/weights/best.pt"
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            meta = {}
    else:
        meta = {
            "classes": ["track", "sand"],
            "trained_at": None,
            "epochs": 0,
            "imgsz": 640,
            "base_model": "yolo11n-seg.pt",
        }
    meta["weights"] = weights_rel
    meta.setdefault("classes", ["track", "sand"])
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def verify(dest: Path) -> dict:
    pt = dest / WEIGHTS_REL
    meta = dest / "model.json"
    ok = pt.is_file() and meta.is_file()
    size_mb = round(pt.stat().st_size / (1024 * 1024), 2) if pt.is_file() else 0
    return {
        "trained_ready": ok,
        "weights": str(pt),
        "weights_mb": size_mb,
        "model_json": meta.is_file(),
        "profile_json": (dest / "profile.json").is_file(),
        "scale_json": (dest / "scale.json").is_file(),
    }


def install(
    data_root: Path | None = None,
    engine_dir: Path | None = None,
    venue_id: str = VENUE_ID,
) -> dict:
    root = Path(data_root) if data_root else _data_root()
    engine = Path(engine_dir) if engine_dir else _engine_dir()
    dest = root / "venues" / venue_id
    upload = root / "venues-upload"

    print(f"DATA_ROOT: {root}")
    print(f"Destino:   {dest}")
    _ensure_layout(dest)

    copied: list[str] = []
    source_used = None

    if upload.is_dir() and (
        (upload / "best.pt").is_file()
        or (upload / WEIGHTS_REL).is_file()
        or (upload / venue_id / WEIGHTS_REL).is_file()
    ):
        src = upload / venue_id if (upload / venue_id).is_dir() else upload
        copied = _copy_tree_files(src, dest)
        source_used = str(src)
    elif engine is not None:
        eng_venue = engine / "venues" / venue_id
        if (eng_venue / WEIGHTS_REL).is_file():
            # Si venues del engine es symlink a dest, no hace falta copiar
            try:
                same = eng_venue.resolve() == dest.resolve()
            except OSError:
                same = False
            if same:
                source_used = f"{eng_venue} (mismo path / symlink)"
            else:
                copied = _copy_tree_files(eng_venue, dest)
                source_used = str(eng_venue)

    if (dest / WEIGHTS_REL).is_file():
        _normalize_model_json(dest)
    else:
        print()
        print("❌ No se encontró best.pt.")
        print("   Subí el zip (pack_venue_model) a Drive y descomprimí en:")
        print(f"     {root / 'venues' / venue_id}/")
        print("   o deja los archivos en:")
        print(f"     {upload}/")
        print("   Layout mínimo:")
        print(f"     model.json, profile.json, scale.json,")
        print(f"     runs/seg/weights/best.pt")

    status = verify(dest)
    print()
    if source_used:
        print(f"Origen: {source_used}")
    if copied:
        print(f"Copiados: {', '.join(copied)}")
    print(f"trained_ready: {status['trained_ready']}  ({status['weights_mb']} MB)")
    for k in ("model_json", "profile_json", "scale_json"):
        print(f"  {k}: {status[k]}")
    if status["trained_ready"]:
        print("✅ Listo. Tras arrancar la API: GET /api/venue/model → trained:true")
    return status


def main() -> int:
    data = None
    engine = None
    # Preferir globals del notebook si existen
    try:
        import __main__

        if hasattr(__main__, "DATA_ROOT"):
            data = Path(__main__.DATA_ROOT)
        if hasattr(__main__, "ENGINE_DIR"):
            engine = Path(__main__.ENGINE_DIR)
    except Exception:
        pass
    status = install(data_root=data, engine_dir=engine)
    return 0 if status["trained_ready"] else 1


if __name__ == "__main__":
    sys.exit(main())
