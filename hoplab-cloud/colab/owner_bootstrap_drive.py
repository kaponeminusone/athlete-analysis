"""
SOLO OWNER — crear estructura Drive (ejecutar una vez en Colab).

Monta MyDrive, crea `hoplab-data` + subcarpetas, y imprime el folder ID
para compartir con invitados (`HOPLAB_DATA_FOLDER_ID`).

Uso en Colab: pega este archivo en una celda y ejecútala (o usa la celda
marcada en HopLab_Server.ipynb). Los invitados NO deben ejecutar esto.
"""

from __future__ import annotations

import pathlib

from google.colab import auth, drive
from googleapiclient.discovery import build

FOLDER_NAME = "hoplab-data"
DRIVE_ROOT = pathlib.Path("/content/drive/MyDrive") / FOLDER_NAME
SUBDIRS = ("videos", "output", "venues", "models", "logs")
_FOLDER_MIME = "application/vnd.google-apps.folder"


def _ensure_local_tree() -> None:
    DRIVE_ROOT.mkdir(parents=True, exist_ok=True)
    for name in SUBDIRS:
        (DRIVE_ROOT / name).mkdir(parents=True, exist_ok=True)
    print(f"✅ Estructura local lista: {DRIVE_ROOT}")
    for name in SUBDIRS:
        print(f"   · {name}/")


def _find_or_create_folder_id(service) -> str:
    """Busca `hoplab-data` bajo My Drive (root); la crea si no existe."""
    query = (
        f"name = '{FOLDER_NAME}' and mimeType = '{_FOLDER_MIME}' "
        "and 'root' in parents and trashed = false"
    )
    resp = (
        service.files()
        .list(q=query, spaces="drive", fields="files(id, name)", pageSize=10)
        .execute()
    )
    files = resp.get("files", [])
    if files:
        return files[0]["id"]

    meta = {
        "name": FOLDER_NAME,
        "mimeType": _FOLDER_MIME,
        "parents": ["root"],
    }
    created = service.files().create(body=meta, fields="id").execute()
    print(f"📁 Carpeta '{FOLDER_NAME}' creada en My Drive vía API.")
    return created["id"]


def _find_child_folder(service, parent_id: str, name: str) -> str | None:
    query = (
        f"name = '{name}' and mimeType = '{_FOLDER_MIME}' "
        f"and '{parent_id}' in parents and trashed = false"
    )
    resp = (
        service.files()
        .list(q=query, spaces="drive", fields="files(id, name)", pageSize=10)
        .execute()
    )
    files = resp.get("files", [])
    return files[0]["id"] if files else None


def _ensure_subdirs_via_api(service, parent_id: str) -> None:
    """Crea subcarpetas vía API (útil si FUSE mkdir falló o para alinear con guests)."""
    for name in SUBDIRS:
        existing = _find_child_folder(service, parent_id, name)
        if existing:
            print(f"   · API: '{name}/' ya existe (id={existing})")
            continue
        meta = {
            "name": name,
            "mimeType": _FOLDER_MIME,
            "parents": [parent_id],
        }
        created = service.files().create(body=meta, fields="id").execute()
        print(f"   · API: creada '{name}/' (id={created['id']})")


def main() -> None:
    print("🔐 Montando Google Drive (MyDrive)…")
    drive.mount("/content/drive")

    _ensure_local_tree()

    print("🔑 Autenticando para Drive API v3…")
    auth.authenticate_user()
    service = build("drive", "v3")

    folder_id = _find_or_create_folder_id(service)
    print(f"📁 Asegurando subcarpetas vía API bajo {folder_id}…")
    _ensure_subdirs_via_api(service, folder_id)

    share_url = f"https://drive.google.com/drive/folders/{folder_id}"

    print()
    print("─── RESULTADO (owner) ───────────────────────────────────────────")
    print(f"Ruta:       {DRIVE_ROOT}")
    print(f"Folder ID:  {folder_id}")
    print(f"URL share:  {share_url}")
    print()
    print("Comparte esa carpeta con los invitados (Editor) y dales:")
    print(f'HOPLAB_DATA_FOLDER_ID = "{folder_id}"')
    print("─────────────────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
