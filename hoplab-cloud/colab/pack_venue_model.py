"""
Empaqueta pesos CNN de venue (default) en venue-default-weights.zip.

Incluye solo:
  model.json, profile.json, scale.json, runs/seg/weights/best.pt

Uso (Windows / cualquier OS)::

    python hoplab-cloud/colab/pack_venue_model.py
    python hoplab-cloud/colab/pack_venue_model.py --src venues/default --out venue-default-weights.zip

Luego sube el zip a Drive y descomprime en:
  hoplab-data/venues/default/
o en:
  hoplab-data/venues-upload/
y ejecuta la celda install_venue_model en Colab.
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SRC = ROOT / "venues" / "default"
DEFAULT_OUT = ROOT / "venue-default-weights.zip"

MEMBERS = (
    "model.json",
    "profile.json",
    "scale.json",
    "runs/seg/weights/best.pt",
)


def pack(src: Path, out: Path) -> Path:
    src = src.resolve()
    out = out.resolve()
    missing = [m for m in MEMBERS if not (src / m).is_file()]
    # scale.json es opcional
    missing = [m for m in missing if m != "scale.json"]
    if missing:
        raise SystemExit(f"Faltan archivos en {src}: {', '.join(missing)}")

    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rel in MEMBERS:
            path = src / rel
            if path.is_file():
                zf.write(path, arcname=rel)
                print(f"  + {rel} ({path.stat().st_size} bytes)")
    print(f"OK -> {out} ({out.stat().st_size} bytes)")
    print("Sube a Drive y descomprime en hoplab-data/venues/default/")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Zip venue CNN weights for Drive upload")
    p.add_argument("--src", type=Path, default=DEFAULT_SRC)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()
    pack(args.src, args.out)


if __name__ == "__main__":
    main()
