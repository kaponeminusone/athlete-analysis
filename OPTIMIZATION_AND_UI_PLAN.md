# Plan: optimización de almacenamiento + preparación UI familiar

**Estado:** plan (sin implementación aún)  
**Fecha:** 2026-07-11  
**Principio rector:** cero pérdida de funcionalidad en la UI de revisión actual (`ui/`). Toda migración va detrás de feature flags cuando haga falta. Optimización de API y nueva UI pueden avanzar **en paralelo e independientes**.

---

## Resumen ejecutivo

| Parte | Objetivo |
|-------|----------|
| **A** | Reducir disco de `output/<video>/` (sobre todo `frames/` + `annotated/` ~2×) manteniendo analizar, refinar, corregir, máscaras, venue, fases, métricas, overlays y scrubbing. |
| **B** | Preparar arquitectura dual: `ui/` = expert/review; nueva app familiar desacoplada que solo consume REST `:8000`. |
| **C** | Orden de trabajo: este plan → revisión usuario → fases 1–2 con flags → UX familiar (cuando se describa) → resto. |

**Fuera de alcance ahora:** LMDB, reescritura de modelos ML, romper contratos API usados por la UI actual.

---

## Contexto de arquitectura (baseline)

```
output/<video>/
  frames/*.jpg          # ~1× disco (JPEG crudos)
  annotated/*.jpg       # ~1× disco (duplicado visual; regenerable)
  venue_masks/          # más pequeño
  overlays/             # pose overlays cache
  charts/               # gráficos
  analysis.json         # fuente de verdad de pose/ángulo
  …                     # calibration, sections, metrics, corrections
```

- El video original suele seguir en disco.
- `GET /frame/{video}/{idx}` ya puede anotar on-demand desde `analysis.json` si falta el JPEG anotado.
- Corrección/propagación densifica y escribe JPEGs en `frames/` (`src/correction.py`).
- Reanálisis escribe árbol `_refined` con frames + annotated (`src/reanalyzer.py`).
- UI actual: `getFrameAsset` → rutas `/media?path=...` indexadas en `GET /api/project`.
- Backend: FastAPI `api_server.py`; pipeline compartido `src/pipeline.py`.

---

# Parte A — Plan de optimización (mantener todas las features)

## Features que NUNCA deben romperse

Checklist de aceptación permanente (toda fase):

- [ ] Analizar video (`POST /api/analyze` / jobs)
- [ ] Reanalizar / refine v2 (`POST /api/reanalyze`, árbol `_refined`)
- [ ] Correcciones + propagación (`POST /correct`)
- [ ] Brush de máscara venue + apply/correct
- [ ] Venue CNN (learn / train / apply / masks)
- [ ] Fases / hops / sections (mark, move, propagate, pose-scores)
- [ ] Análisis / métricas (compute, overrides, scale)
- [ ] Pose overlays por fase (`/api/metrics/.../pose-overlay/{phase}`)
- [ ] Timeline scrubbing y navegación frame a frame
- [ ] Calibración de pista (seeds, propagate, keyframes)
- [ ] Selección de persona en frame (`GET /mask/{video}/{idx}`)
- [ ] Charts y assets secundarios visibles en UI

---

## Feature flags (migración segura)

| Flag | Default inicial | Efecto |
|------|-----------------|--------|
| `write_annotated` | `true` (luego `false`) | Si `false`, pipeline/reanalyzer/correct no escriben `annotated/*.jpg`. |
| `persist_frames` | `true` (fase 4 → opcional `false`) | Si `false`, no se persisten JPEG crudos; decode desde video. |
| `annotated_cache` | off / memory / short-ttl | Cache opcional de anotados on-demand (fase 1). |
| `correction_write_frames` | `true` → `false` | Si `false`, propagate no densifica JPEGs (fase 2). |
| `prune_refined` | `false` | Política opcional de limpieza de `_refined` antiguos (fase 3). |

Flags vía config de pipeline / env / query en jobs. La UI actual no necesita conocerlos si el API sigue sirviendo las mismas URLs.

---

## Fase 1 — Dejar de persistir `annotated/`; servir on-demand

### Objetivo
Eliminar ~50% del volumen típico de imágenes (duplicado annotated ≈ frames).

### Ahorro estimado
**~40–55%** del disco de imágenes por video (depende de stride y ratio annotate-every).  
Ej.: si hoy `frames` + `annotated` ≈ 2×, pasar a solo `frames` ≈ **½**.

### Archivos / funciones a tocar

| Archivo | Cambio |
|---------|--------|
| `src/pipeline.py` | Condicionar `cv2.imwrite` de `annotated_*` con `write_annotated`; dejar de crear dir si flag off. |
| `src/reanalyzer.py` | Igual en refine / seed-gap / `_refined`. |
| `src/correction.py` + handlers en `api_server.py` (`POST /correct`) | No escribir annotated post-corrección; o escribir solo si flag. |
| `src/visualizer.py` (`annotate_frame`) | Sin cambio de API; usado on-demand. |
| `api_server.py` → `get_frame` / `get_annotated_frame` | Preferir anotación en memoria o cache corto; **no** forzar write a disco (hoy escribe al regenerar). |
| `api_server.py` → `_build_project` / `_indexed_images` | Indexar annotated vacío OK; UI debe caer a `/frame/...`. |
| `ui/src/App.jsx` → `getFrameAsset` | Preferir `/frame/{video}/{idx}` (o `/annotated`) sobre `/media` de annotated indexados. Cambio mínimo, compatible. |

### Qué ve el usuario (igual)
Todas las features de la checklist; overlays de pose/ángulo en el visor; scrubbing; correcciones visibles al instante.

### Cómo se preserva
- `analysis.json` sigue siendo la verdad.
- `GET /frame/{video}/{idx}?annotated=true` regenera con `annotate_frame` + FA desde JSON.
- Cache opcional: dict en memoria `(video, idx, analysis_mtime) → bytes` o TTL 1–5 min / LRU.

### Riesgos y mitigaciones

| Riesgo | Mitigación |
|--------|------------|
| Latencia al scrub rápido | Cache LRU; prefetch vecinos; JPEG quality bajo en cache. |
| UI solo mira `project.assets.annotated` | Actualizar `getFrameAsset` a URL canónica `/frame/...`. |
| Jobs antiguos con annotated en disco | Compat: si existe archivo, servirlo; no borrar en caliente salvo prune explícito. |

### Rollback
`write_annotated=true`; re-análisis opcional regenera annotated. No migrar datos destructivos en fase 1.

### Test checklist
- [ ] Analizar con flag off → no hay (o casi no hay) `annotated/`
- [ ] Abrir UI, scrub 50 frames → imágenes anotadas correctas
- [ ] Corregir un frame → overlay actualizado vía `/frame`
- [ ] Reanalyze `_refined` → UI de versión refinada OK
- [ ] Pose overlay y charts intactos
- [ ] Con flag on, comportamiento idéntico al actual

---

## Fase 2 — Corrección/propagación sin densificar todos los JPEG

### Objetivo
Evitar explosión de disco cuando propagate reescribe `frame_XXXXXX.jpg` en un radio amplio (a menudo densificando respecto al stride del análisis).

### Ahorro estimado
**Alto en sesiones de corrección:** decenas–cientos de MB por sesión de propagate (depende de radius/end_frame).  
No reduce el análisis inicial; evita crecimiento post-corrección.

### Archivos / funciones a tocar

| Archivo | Cambio |
|---------|--------|
| `src/correction.py` → `propagate_correction` (bloques `cv2.imwrite` ~L297–330) | Si `correction_write_frames=false`, solo actualizar `FrameAnalysis` en memoria/JSON; **no** escribir frames. |
| `api_server.py` → `POST /correct` | Pasar flag; no guardar annotated densos. |
| `api_server.py` → `get_frame`, `GET /mask/...` | Si falta JPEG, decode desde `video_path` (seek + read). |
| Consumidores lean-off | Si más adelante `persist_frames=false`, misma ruta de decode (ver fase 4). |

### Qué ve el usuario (igual)
Corregir bbox/persona, propagación forward/back, SOT/ByteTrack, scrubbing de frames afectados, re-anotación visual.

### Cómo se preserva
- Propagación ya lee el video con `VideoCapture` para ML; el JPEG era solo cache en disco.
- UI pide `/frame/...` → decode video + annotate desde JSON actualizado.
- Modo lean-off (`persist_frames=true`): se puede seguir escribiendo solo el frame corregido (o radio mínimo) si se necesita debug offline sin video.

### Riesgos y mitigaciones

| Riesgo | Mitigación |
|--------|------------|
| Seek lento / impreciso en algunos codecs | Buffer de frames vecinos; preferir lectura secuencial en propagate; documentar codecs amigos (mp4 h264). |
| `/mask` y pose asumen JPEG en disco | Helper único `load_frame_bgr(video_name, idx)` → file else video. |
| Diff visual vs JPEG Q=92 | Aceptable; mismo frame del video fuente. |

### Rollback
`correction_write_frames=true` (comportamiento actual de densificado).

### Test checklist
- [ ] Correct + propagate radius grande → conteo de nuevos JPEG ≈ 0 (flag off)
- [ ] Frames propagados muestran pose/track actualizados
- [ ] `/mask/{video}/{idx}` funciona en idx densificado sin archivo
- [ ] Reanalyze posterior sigue OK
- [ ] Flag on = parity con hoy

---

## Fase 3 — Alinear strides por defecto + prune opcional de `_refined`

### Objetivo
Menos frames analizados/escritos por default; opcionalmente no acumular árboles refine eternos.

### Ahorro estimado
- Stride UI/API alineado (p.ej. default 3 en analyze; refine documentado): **~proporcional a 1/stride** vs stride=1.
- Prune `_refined`: libera **100%** de ese árbol cuando el usuario confirma / política TTL.

### Archivos / funciones a tocar

| Archivo | Cambio |
|---------|--------|
| `src/pipeline.py` (`stride` default) | Confirmar default 3; documentar. |
| `src/reanalyzer.py` (`stride` default hoy 1) | Alinear documentación/UI; no forzar 1 si el usuario no lo pide. |
| `ui/src/App.jsx` (`useState(3)` stride) | Ya 3; asegurar que reanalyze envía stride explícito. |
| `api_server.py` jobs analyze/reanalyze | Persistir stride en config del job/output. |
| Nuevo helper o endpoint opcional | `prune_refined(video, keep_latest=N)` detrás de `prune_refined` flag / acción manual UI. |

### Qué ve el usuario (igual)
Mismos controles de stride; refine v2; posibilidad de borrar versiones viejas solo si activa prune (nunca automático destructivo sin confirmación en v1).

### Cómo se preserva
Defaults más conservadores = menos I/O, mismas APIs. Prune solo borra outputs no referenciados por la sesión activa.

### Riesgos y mitigaciones

| Riesgo | Mitigación |
|--------|------------|
| Usuario espera densificado refine | UI muestra stride; default documentado. |
| Borrar `_refined` en uso | Prune manual + confirmación; never auto en fase 3 v1. |

### Rollback
Restaurar defaults previos; no prune automático.

### Test checklist
- [ ] Analyze default stride=3
- [ ] Reanalyze con stride elegido por usuario
- [ ] Prune (si implementado) borra solo árboles marcados
- [ ] Proyecto activo no rompe tras prune de versión no seleccionada

---

## Fase 4 — Lean mode: `persist_frames=False` + decode de video en todos los consumidores

### Objetivo
Modo opcional sin carpeta `frames/` (o mínima). Máximo ahorro de disco.

### Ahorro estimado
**~90–98%** del disco de imágenes del video (queda analysis.json + masks/overlays/charts pequeños).  
CPU/latencia sube (decode).

### Consumidores que DEBEN tener fallback a video decode

Lista exhaustiva a cablear con helper compartido (p.ej. `src/frame_io.py` → `read_frame_bgr(...)`):

| Consumidor | Ubicación aproximada |
|------------|----------------------|
| `get_frame` (raw + annotated on-demand) | `api_server.py` |
| `GET /mask/{video}/{frame_idx}` | `api_server.py` |
| Pose overlay render (fondo de frame) | `src/pose_overlay.py` + endpoint metrics |
| Reanalyzer seed / gap fill | `src/reanalyzer.py` |
| Correction + propagate | `src/correction.py` (ya usa video para ML; unificar lectura) |
| Venue apply / correct / mask endpoints | `api_server.py` + `src/venue_*.py` si leen JPEG |
| Calibración / propagate visual debug | `src/calibration_propagator.py` (ya video) — verificar paths de UI |
| Charts que re-leen frames | `src/` chart helpers si aplica |
| Pipeline analyze inicial | `src/pipeline.py` — con flag, no `imwrite` frames |

### Qué ve el usuario (igual)
Misma UI; scrubbing puede ser algo más lento; requiere que `video_path` siga resoluble.

### Cómo se preserva
- Un solo `read_frame_bgr`: (1) JPEG si existe (2) else `VideoCapture` seek (3) else 404 claro.
- Project payload: `assets.frames` puede estar vacío; UI usa `/frame/...` siempre.

### Riesgos y mitigaciones

| Riesgo | Mitigación |
|--------|------------|
| Video borrado / movido | Error explícito en API; lean mode documentado como “video must remain”. |
| Seek aleatorio lento | Cache LRU de frames decodificados; prefetch timeline. |
| Offline sin video | No soportar lean; `persist_frames=true`. |

### Rollback
`persist_frames=true` + re-análisis o re-extracción de frames.

### Test checklist
- [ ] Analyze lean → `frames/` vacío o ausente
- [ ] Toda la checklist de features con video presente
- [ ] Desconectar/renombrar video → error claro, no crash silencioso
- [ ] Mezcla: algunos JPEG presentes (híbrido) funciona
- [ ] `_refined` lean + correct + mask + pose-overlay

---

## Fase 5 — Compresión (JPEG Q / WebP) y máscaras venue compactas

### Objetivo
Reducir tamaño por archivo sin cambiar el modelo de persistencia.

### Ahorro estimado
- JPEG Q 92 → 80–85: **~20–40%** por frame.
- WebP (si se adopta): adicional variable; requiere `media_type` y UI OK.
- Venue masks compact (PNG optimizado / RLE ya parcial en JSON): **menor** pero acumulativo.

### Archivos / funciones a tocar

| Archivo | Cambio |
|---------|--------|
| `src/pipeline.py`, `reanalyzer.py`, `correction.py`, `frame_extractor.py` | Constante `JPEG_QUALITY` centralizada. |
| `api_server.py` responses | Si WebP: `image/webp` + negociación opcional. |
| `src/mask_utils.py` / `venue_masks` | Compresión PNG o formato compacto documentado. |

### Qué ve el usuario (igual)
Misma nitidez suficiente para review; overlays legibles.

### Cómo se preserva
Calidad visual validada en checklist; no cambiar geometría de coords.

### Riesgos y mitigaciones

| Riesgo | Mitigación |
|--------|------------|
| Artefactos en keypoints finos | No bajar de Q~80 sin A/B visual. |
| WebP en edge cases | Empezar solo JPEG Q; WebP opt-in. |

### Rollback
Restaurar Q=92; desactivar WebP.

### Test checklist
- [ ] Diff visual en hops/contactos
- [ ] Tamaño medio de `frame_*.jpg` bajó
- [ ] Máscaras venue round-trip OK

---

## Fuera de alcance (explícito)

- LMDB / bases de frames embebidas  
- Reescribir modelos YOLO / CNN venue  
- Romper contratos REST que usa `ui/src/App.jsx`  
- Borrar videos fuente automáticamente  
- Nueva UI familiar (solo preparación en Parte B)

---

# Parte B — Preparación dual UI

## Arquitectura objetivo

```
                    ┌─────────────────────────┐
                    │   FastAPI :8000         │
                    │   api_server.py         │
                    │   (fuente de verdad)    │
                    └───────────┬─────────────┘
                                │ REST JSON + imágenes
              ┌─────────────────┴─────────────────┐
              │                                   │
   ┌──────────▼──────────┐             ┌──────────▼──────────┐
   │  ui/  (Vite)        │             │  ui-family/ o       │
   │  YOLO Review        │             │  apps/coach/        │
   │  expert / corrección│             │  familiar (nuevo)   │
   │  App.jsx monolítico │             │  SIN importar       │
   └─────────────────────┘             │  App.jsx            │
                                       └─────────────────────┘
```

### Reglas

1. **`ui/`** permanece como UI expert/review (espíritu YOLO Review). Cambios solo los mínimos para URLs de frames (`getFrameAsset`).
2. **Nueva UI** = app Vite separada (`ui-family/` o `apps/coach/`), mismo proxy a `:8000`.
3. **Contrato:** backend = source of truth; nueva UI solo consume REST; **cero acoplamiento** a `ui/src/App.jsx`.
4. Optimización de storage/API **primero o en paralelo**: ambas UIs se benefician.
5. No compartir componentes React entre apps salvo futura librería deliberada (`packages/api-client` opcional más adelante).

## Endpoints a estabilizar / documentar (ya existen)

Documentar en OpenAPI/`docs` o hoja de contrato cuando se implemente la UI familiar:

| Área | Endpoints |
|------|-----------|
| Proyecto / media | `GET /api/project`, `GET /api/videos`, `GET /api/demo`, `GET /media` |
| Análisis | `POST /analyze`, `POST /api/analyze`, `GET /analysis/{video}`, `GET /status` |
| Jobs | `GET /api/jobs`, `GET /api/jobs/{job_id}` |
| Reanálisis | `POST /api/reanalyze` |
| Frames | `GET /frame/{video}/{idx}`, `GET /frame/.../annotated`, `GET /mask/{video}/{idx}` |
| Corrección | `POST /correct` |
| Secciones / fases | `POST /api/sections/analyze|mark|move|propagate`, `DELETE .../mark/...`, `GET /api/sections/...`, `GET .../pose-scores` |
| Métricas | `GET/POST /api/metrics/...`, overrides, scale, `GET .../pose-overlay/{phase}` |
| Calibración | `GET/POST /api/calibration/...`, seeds, propagate |
| Tracking | `POST /api/recompute-tracking/{video}` |
| Venue | `GET /api/venue/profile`, learn, model, dataset, train, `POST .../apply|correct`, `GET .../masks/...` |
| Debug | `GET /api/debug_coords` |

**Recomendación:** tras Fase 1, tratar `/frame/{video}/{idx}` como URL canónica de imagen (no depender de `/media` indexado).

## Gaps TBD (después de que el usuario describa UX familiar)

Marcar como **TBD — esperar descripción de UX**:

- [ ] Auth / roles (familia vs coach vs admin)
- [ ] Flujos simplificados / wizard de “sube video → ver resumen”
- [ ] Qué métricas mostrar vs ocultar
- [ ] Edición permitida vs solo lectura
- [ ] Mobile-first / compartición de resultados
- [ ] Branding, copy, idioma
- [ ] Si necesita WebSockets para progreso de jobs o basta polling actual
- [ ] Packaging (mismo monorepo npm workspaces vs carpeta suelta)

## Independencia

| Trabajo | Bloquea al otro? |
|---------|------------------|
| Fases optimización 1–5 | No bloquea diseño de `ui-family` |
| Diseño/implementación `ui-family` | No bloquea flags de storage |
| Ideal | API canónica de frames lista (Fase 1) antes de invertir mucho en UI nueva |

---

# Parte C — Orden de trabajo recomendado

1. **Escribir este plan** ← hecho (`OPTIMIZATION_AND_UI_PLAN.md`)
2. **Usuario revisa** el plan (ahorro, flags, riesgos, fuera de alcance)
3. **Implementar optimización Fase 1–2** detrás de flags (`write_annotated`, `correction_write_frames`); ajuste mínimo `getFrameAsset`
4. **Usuario describe UX familiar** → diseño de `ui-family/` / `apps/coach/` (solo contrato REST)
5. **Fase 3** (strides + prune opcional) cuando convenga
6. **Fase 4** lean mode solo si el ahorro lo justifica y el video siempre está disponible
7. **Fase 5** compresión / WebP / masks compactas
8. (Opcional) Cliente API tipado compartido para ambas UIs — sin acoplar React

---

## Criterios de éxito globales

- Checklist de features en verde con flags de optimización activos.
- Disco por video claramente menor (medir antes/después en un VOD de referencia, p.ej. VOD7).
- UI review actual usable sin regresiones.
- Nueva UI puede empezar sin tocar `App.jsx` más allá de lo ya hecho para frames canónicos.

## Medición sugerida (al implementar)

Para un video de referencia:

| Métrica | Antes | Después Fase 1 | Después Fase 2 | Lean (4) |
|---------|-------|----------------|----------------|----------|
| Tamaño `frames/` | | | | |
| Tamaño `annotated/` | | | | |
| Total `output/<video>/` | | | | |
| Latencia p95 `/frame` scrub | | | | |

---

## Apéndice — mapa rápido “quién escribe imágenes hoy”

| Writer | Qué escribe |
|--------|-------------|
| `src/pipeline.py` | `frames/`, `annotated/` |
| `src/reanalyzer.py` | `frames/`, `annotated/` en `_refined` |
| `src/correction.py` | densifica `frames/` en propagate; correct path anota |
| `api_server.get_frame` | puede crear `annotated/` on-demand (hoy) |
| `src/pose_overlay.py` | cache en `overlays/` |
| Venue / masks | `venue_masks/` (menor) |

Fin del plan.
