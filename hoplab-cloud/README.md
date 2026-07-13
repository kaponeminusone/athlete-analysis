# HopLab Cloud — porteo Colab + Drive + frontend estático

Carpeta **semi-aislada** del monorepo `prototype/`.  
El laboratorio local (`ui-family/`, `api_server.py`) sigue siendo la fuente de verdad de features; este árbol es el **adaptador de despliegue bajo demanda**.

```
hoplab-cloud/
  README.md                 ← este archivo
  docs/COLAB_PORT_PLAN.md   ← plan de acción completo
  colab/                    ← notebook + scripts que corren EN Google Colab
  web/                      ← frontend HopLab listo para Vercel/Render (build)
  shared/                   ← contratos (env, API base, health) sin lógica ML
```

## Arquitectura (on-demand, no 24/7)

```
┌─────────────────────────────┐
│  Vercel / Render (gratis)   │  HopLab UI estática
│  https://hoplab.vercel.app  │  VITE_API_BASE = URL del túnel
└──────────────┬──────────────┘
               │ HTTPS (CORS abierto en API)
               ▼
┌─────────────────────────────┐
│  Cloudflare Tunnel (gratis) │  URL tipo *.trycloudflare.com
│  (o ngrok / localtunnel)    │  cambia cada sesión Colab
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  Google Colab (GPU T4)      │  FastAPI + YOLO + refine
│  Open in Colab → Run All    │
└──────────────┬──────────────┘
               │ monta Drive
               ▼
┌─────────────────────────────┐
│  Google Drive               │  videos/, output/, venues/, models/
│  MyDrive/hoplab-data/       │  persiste entre sesiones
│  (o carpeta compartida      │  invitados: HOPLAB_DATA_FOLDER_ID
│   vía .shortcut-targets…)   │  → misma data del owner
└─────────────────────────────┘
```

## Flujo de uso (humano)

1. Abrir el notebook con el badge **Open in Colab**.
2. Runtime → GPU.
3. **Runtime → Run all** (monta Drive, instala deps, arranca API, imprime URL del túnel).
4. Copiar la URL `https://….trycloudflare.com` en la UI (o pegarla en Vercel env y redeploy rápido / pantalla “conectar motor”).
5. Usar HopLab desde el móvil/PC.
6. Al terminar: Runtime → Disconnect (Drive ya guardó lo importante).

**Invitado con Drive del owner:** el owner comparte `hoplab-data` (Editor) y pasa el ID de carpeta; el invitado pega `HOPLAB_DATA_FOLDER_ID` en la celda de config y hace Run All. Detalle en [`colab/README.md`](colab/README.md).

## Relación con el repo local

| Local (`prototype/`) | Cloud (`hoplab-cloud/`) |
|----------------------|-------------------------|
| Desarrollo diario, Cursor, GPU local | Demo / uso bajo demanda |
| `api_server.py` + `src/` | Colab clona o sincroniza ese código |
| `ui-family/` | Build/deploy en `web/` (fork adaptado) |

No mezclar paths de Windows ni dependencias de Vite proxy en el build de producción.
