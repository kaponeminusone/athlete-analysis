# Colab — motor HopLab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TU-USUARIO/TU-REPO/blob/main/hoplab-cloud/colab/HopLab_Server.ipynb)

> **Cambiar `TU-USUARIO/TU-REPO`** por el repo real de GitHub antes de publicar el badge.

Notebook **Open in Colab** estilo ComfyUI.

## Estructura del notebook

| Sección | Contenido |
|---------|-----------|
| **0. Configuración** | Repo, Drive, `HOPLAB_DATA_FOLDER_ID` (única celda a editar) |
| **1. Solo owner** | Bootstrap Drive una vez (colapsar / omitir después) |
| **2. Setup** | GPU → Drive → clone → install → env/symlinks |
| **3. Arrancar API** | uvicorn + logs en `DATA_ROOT/logs/hoplab-api.log` |
| **4. Túnel + URL** | cloudflared + URL para Vercel (fallback localtunnel colapsable) |
| **5. Terminal** | Panel HTML live (▶ debe quedar corriendo) — si no ves texto, la celda no se ejecutó |

## Uso previsto (owner)

1. Abrir `HopLab_Server.ipynb` con el badge (cuando exista).
2. Runtime → GPU.
3. **Una vez:** ejecutar la sección «1. Solo owner» (o `owner_bootstrap_drive.py`) antes de invitar invitados; anota el folder ID.
4. Dejar `HOPLAB_DATA_FOLDER_ID = ""` (usa `MyDrive/hoplab-data`).
5. Runtime → Run all (o ejecutar secciones 0–5).
6. Copiar la URL `*.trycloudflare.com` de la sección **4**.
7. Pegarla en la UI desplegada (Conectar motor).
8. Ejecutá la celda **Terminal** (sección 5, play ▶) y dejala corriendo; debe mostrar el panel HTML + logs. Si no ves texto, la celda no se ejecutó. Stop = interrupt.

## Carpeta compartida (owner + invitado)

Varias personas pueden **Run All** en Colab con su propia cuenta, pero leer/escribir la misma carpeta `hoplab-data` del owner.

### Owner

1. Ejecuta una vez el bootstrap («SOLO OWNER…» en el notebook o `owner_bootstrap_drive.py`): crea `hoplab-data` + subcarpetas e imprime el folder ID.
2. Comparte esa carpeta con los emails de los invitados como **Editor**.
3. Envía el ID impreso a los invitados (es el valor de `HOPLAB_DATA_FOLDER_ID`).

### Invitado (tercero)

1. Acepta la invitación de Drive (aparecerá en “Compartido conmigo”).
2. Abre el notebook, Runtime → GPU.
3. En la celda de configuración, pega el ID del owner:
   ```python
   HOPLAB_DATA_FOLDER_ID = "FOLDER_ID_DEL_OWNER"
   ```
4. Runtime → Run all (autoriza **tu** cuenta de Google para montar Drive; es obligatorio).
5. Colab resuelve los datos en:
   `/content/drive/.shortcut-targets-by-id/<FOLDER_ID>/`
   (no en tu `MyDrive/hoplab-data` vacío).
6. Pega la URL del túnel en la UI de Vercel (Conectar motor).
7. Ejecutá la **Terminal** (sección 5, play ▶) y dejala corriendo; debe mostrar el panel HTML.

Si la ruta por ID no existe tras el mount: acepta el share, espera unos segundos y re-ejecuta la celda de Drive.

### errno 95 / `mkdir` en carpetas compartidas

En `.shortcut-targets-by-id`, Drive FUSE a menudo **no permite crear carpetas** (`OSError: [Errno 95] Operation not supported`). El notebook intenta `mkdir` local y, si falla con `HOPLAB_DATA_FOLDER_ID` definido, **crea las subcarpetas vía Drive API v3** (`auth.authenticate_user()` — puede pedir consent la primera vez). Los invitados ya no necesitan que el owner pre-cree `videos/output/venues/models/logs` si tienen rol **Editor** en la carpeta.

Si FUSE tarda en ver las carpetas nuevas, re-ejecuta la celda de Drive unos segundos después.

**Alternativa (escrituras más fiables):** Organizar → «Añadir acceso directo a Mi unidad» y en la config:
```python
HOPLAB_DATA_ROOT = "/content/drive/MyDrive/hoplab-data"
```
Orden de resolución: `HOPLAB_DATA_ROOT` → `HOPLAB_DATA_FOLDER_ID` → `MyDrive/hoplab-data`.

## Archivos

| Archivo | Rol |
|---------|-----|
| `HopLab_Server.ipynb` | Setup → API → túnel/URL → Terminal (live logs) |
| `owner_bootstrap_drive.py` | Owner: crear `hoplab-data` + folder ID (una vez) |
| `requirements-colab.txt` | Deps pinneadas para Colab |

Ver plan completo: [`../docs/COLAB_PORT_PLAN.md`](../docs/COLAB_PORT_PLAN.md).
