# HopLab — UI familiar (landscape)

Prototipo **landscape-first**, sensación de app de teléfono, con atmósfera de video
**semi-maximizado** tipo streaming. Pensado para revisar carrera + 5 hops rápido.

Integra el **FastAPI** real (`api_server.py` en la raíz del repo) vía proxy Vite.
Si la API no responde, cae a fixtures mock + mounts locales de `output/` para no
bloquear la UI. **No toca** `ui/` (experta).

## Cómo correr (API + ui-family)

En **dos terminales**, desde la raíz del repo:

```bash
# Terminal 1 — backend
python api_server.py
# → http://127.0.0.1:8000
```

```bash
# Terminal 2 — UI familiar
cd ui-family
npm install
npm run dev
# → http://127.0.0.1:5174
```

Abrir **http://127.0.0.1:5174/**

El proxy en `vite.config.js` reenvía `/api`, `/media`, `/frame`, `/correct`, `/mask`,
`/analysis`, `/status` a `http://127.0.0.1:8000`. Los mounts `/media-vod2`,
`/media-vod9`, `/media-overlays-*` siguen disponibles como fallback offline.

### Solo demo (sin API)

```bash
cd ui-family && npm run dev
```

Verás un toast *«API no disponible — usando datos de demostración»* y la
biblioteca mock (Sofía / Mateo + VOD9/VOD2).

## Qué habla con la API

| Área | Endpoints |
|------|-----------|
| Biblioteca | `GET /api/videos`, `GET /api/project`, métricas/sections |
| Watch | proyecto, frames `/frame/{video}/{idx}`, sections, metrics |
| Overlays de pose | `GET /api/metrics/.../pose-overlay/{phase}` |
| Máscaras pista | calibration + `/media?path=` (track/sand en `venue_masks/`) |
| Analizar | `POST /api/analyze` (+ `POST /api/reanalyze` si «Mejorar seguimiento») + poll `/api/jobs/{id}` |
| Escala | `POST /api/metrics/{video}/scale` con `{ hops_corridor_m }` |
| Marcadores | `POST .../sections/mark` y `.../mark/.../move` (best-effort) |

## Biblioteca

- Lista videos reales; agrupa por `athlete_id` de metrics/sections, o **Sin asignar**.
- Thumb: primer frame anotado vía `/frame/...`.
- Estado de análisis + **% éxito** desde `consistency.overall` o `pose_quality`.
- Badge **API** / **Demo** en el header.

### % de éxito

`consistency.overall` → si no, `comparison.pose_quality.overall` → si no, promedio de hops.
Color: verde ≥80, ámbar ≥65, rojo abajo.

## Watch

- Frames desde `analysis.frames`; ojo **Seguimiento** = annotated ↔ raw (`/frame?...`).
- Ojo **Pista** si hay `calibration.mask_frames`.
- Contactos desde `sections.contacts` / `phase_markers`.
- Estadísticas mapeadas a hops + overlays de pose reales.
- **Configuración → Analizar** lanza job real; **Avanzado → Aplicar escala** actualiza métricas.

## Qué sigue mock

- Modal **Ingresar video** (asignación atleta/fecha).
- Corregir atleta (pincel) y editar pista (polígono) — solo toast local.
- Aprender / entrenar mapa de venue.
- Fixtures VOD9/VOD2 si la API está caída.

## Estructura

```
src/
  App.jsx
  api/
    client.js                 # fetchJson + endpoints
    mapSession.js             # videos → atletas, project → WatchPage
  mock/data.js                # fixtures offline + helpers de UI
  components/
    LibraryHome.jsx
    WatchPage.jsx
    AthleteRail.jsx
    Timeline.jsx
    SidePanel.jsx
    BrushLayer.jsx / PistaDraw.jsx / AnnotationHUD.jsx
```

## Stack

React 19 + Vite 8 + Tailwind 4 · puerto **5174** · Outfit + Roboto Condensed ·
acento `#ff0033` sobre `#0f0f0f`.

Ver interacción completa en [`../FAMILY_UI_PLAN.md`](../FAMILY_UI_PLAN.md).
