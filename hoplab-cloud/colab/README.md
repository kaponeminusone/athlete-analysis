# Colab — motor HopLab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TU-USUARIO/TU-REPO/blob/main/hoplab-cloud/colab/HopLab_Server.ipynb)

> **Cambiar `TU-USUARIO/TU-REPO`** por el repo real de GitHub antes de publicar el badge.

Notebook **Open in Colab** estilo ComfyUI.

## Uso previsto

1. Abrir `HopLab_Server.ipynb` con el badge (cuando exista).
2. Runtime → GPU.
3. Runtime → Run all.
4. Copiar la URL `*.trycloudflare.com` impresa.
5. Pegarla en la UI desplegada (Conectar motor).

## Archivos (por implementar)

| Archivo | Rol |
|---------|-----|
| `HopLab_Server.ipynb` | Celdas mount → install → API → tunnel |
| `start_api.py` | Arranca uvicorn + healthcheck |
| `requirements-colab.txt` | Deps pinneadas para Colab |
| `setup_drive_layout.py` | Crea `hoplab-data/{videos,output,venues}` |

Ver plan completo: [`../docs/COLAB_PORT_PLAN.md`](../docs/COLAB_PORT_PLAN.md).
