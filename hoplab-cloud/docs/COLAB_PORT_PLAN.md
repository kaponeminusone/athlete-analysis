# Plan de acción — HopLab en Colab (estilo ComfyUI)

**Estado:** plan (sin implementación completa)  
**Fecha:** 2026-07-13  
**Principio:** uso **bajo demanda** (no 24/7), **sin AWS de pago**, frontend estático gratis, GPU de Colab + persistencia en Drive.

Referencias de patrón (ComfyUI / similares):
- Notebook con celdas: Mount Drive → Install → Start server → **cloudflared** imprime `*.trycloudflare.com`
- Workspace en Drive para sobrevivir reinicios
- Túnel gratis sin cuenta (quick tunnel); ngrok como fallback con token

---

## 1. Objetivos

| Objetivo | Criterio de éxito |
|----------|-------------------|
| Abrir un **único link** al notebook y poder arrancar sin errores | Badge Open in Colab + Run All idempotente |
| API con YOLO/refine en Colab GPU | `GET /status` OK vía túnel |
| HopLab en Vercel/Render | UI carga; pide/guarda URL del motor |
| Persistencia | Videos + `output/` + `venues/` en Drive |
| Aislamiento | Cambios de cloud viven en `hoplab-cloud/`; el lab local sigue usable |

**Fuera de alcance v1:** Colab Enterprise API, arranque 100 % automático sin abrir Colab, auth, Capacitor store.

---

## 2. Arquitectura y comunicación

### 2.1 Piezas

| Pieza | Dónde vive | Rol |
|-------|------------|-----|
| **Motor ML** | Colab + código clonado del repo | `api_server.py`, jobs, frames |
| **Datos** | Google Drive `MyDrive/hoplab-data/` | videos, output, venues, pesos |
| **Túnel** | cloudflared en el notebook | URL pública → `:8000` |
| **UI** | Vercel/Render (`hoplab-cloud/web`) | Solo HTTP al túnel |

### 2.2 Contrato de sesión (cada vez que abres Colab)

```
1. Usuario abre notebook (link fijo de GitHub/Drive)
2. Colab monta Drive → HOPLAB_ROOT / DATA_ROOT
3. pip install + (opcional) sync git del motor
4. uvicorn api_server :8000
5. cloudflared tunnel --url http://127.0.0.1:8000
6. Celda imprime API_PUBLIC_URL
7. Usuario pega URL en HopLab ("Conectar motor") o la guarda en localStorage
8. UI llama a https://API_PUBLIC_URL/api/...  (CORS *)
```

La URL del túnel **cambia cada sesión**. Por eso la UI **no** debe hardcodear el túnel en el build de Vercel como única opción: necesita un campo **“URL del motor”** (localStorage) + default opcional `VITE_API_BASE`.

### 2.3 Por qué no Drive como CDN de frames

Drive guarda archivos; el scrub necesita `GET /frame/...` rápido desde el disco de Colab (o symlink a Drive).  
Al terminar el job, `output/` ya está en Drive → persiste. Mientras la sesión vive, servir desde filesystem montado.

---

## 3. Layout en Google Drive

```
MyDrive/
  hoplab-data/                 ← DATA_ROOT (persistente)
    videos/                    ← MP4 a analizar
    output/                    ← analysis, frames, _refined
    venues/                    ← CNN pista, profiles
    models/                    ← yolo *.pt cache (opcional; Ultralytics también cachea)
    logs/
  Colab Notebooks/
    HopLab_Server.ipynb        ← copia del notebook (o abrir desde GitHub)
```

Código del motor (opción A recomendada):
- Clonar en `/content/hoplab-engine` (rápido, efímero) cada sesión, **datos** solo en Drive.
- Opción B: repo completo en Drive (más lento al I/O).

---

## 4. Separación de proyectos (semi-aislado)

```
prototype/                      ← lab local (sigue igual)
  api_server.py
  src/
  ui-family/

prototype/hoplab-cloud/         ← SOLO despliegue cloud
  docs/COLAB_PORT_PLAN.md
  colab/
    HopLab_Server.ipynb         ← Open in Colab
    start_api.sh / start_api.py
    requirements-colab.txt      ← pin torch+cu + ultralytics…
    README.md                   ← “cómo Run All”
  web/
    (fork mínimo de ui-family)
    - VITE_API_BASE / motor URL UI
    - sin proxy Vite en prod
    - pantalla Conectar motor
  shared/
    env.example
    health_contract.md          ← /status shape
```

**Regla:** no importar `ui/App.jsx` experto. El web cloud solo habla REST.

---

## 5. Adaptaciones técnicas necesarias (del repo actual)

Inventario verificado:

| Gap | Hoy | Qué hacer en el porteo |
|-----|-----|------------------------|
| `OUTPUT_ROOT` | hardcode `Path("output")` | Env `HOPLAB_OUTPUT_ROOT` / chdir a DATA_ROOT |
| `VENUE_ROOT` | `Path("venues")` | Env o symlink desde Drive |
| Lista videos | `rglob` desde CWD | `VIDEO_ROOT` + scan; o endpoint upload |
| Upload | **no existe** | v1: “sube a Drive/videos”; v1.1: `POST /api/upload` |
| CORS | ya `*` | OK para Vercel ↔ túnel |
| ui-family | URLs relativas + proxy Vite | Prod: `apiUrl = VITE_API_BASE \|\| localStorage.motorUrl` |
| `dataset.yaml` | path Windows absoluto | Regenerar path Linux en Drive |
| Annotated disk | optimización `TJ_WRITE_ANNOTATED` | En Colab: `0` (menos I/O a Drive) |

---

## 6. Notebook Colab (celdas, estilo ComfyUI)

Orden fijo (idempotente):

| # | Celda | Acción |
|---|-------|--------|
| 0 | Título + checklist | GPU? Drive? |
| 1 | **Config** | `USE_DRIVE=True`, `DATA_ROOT=...`, `REPO_URL=...`, `BRANCH=...` |
| 2 | **Mount Drive** | `drive.mount` |
| 3 | **Prepare dirs** | mkdir videos/output/venues |
| 4 | **Fetch engine** | `git clone` o `git pull` a `/content/hoplab-engine` |
| 5 | **Install** | pip torch cu121 + `-r requirements-colab.txt` (cada sesión) |
| 6 | **Env** | `os.environ["TJ_WRITE_ANNOTATED"]="0"`, `chdir`, symlinks output→Drive |
| 7 | **Start API** | thread/background `uvicorn api_server:app --host 0.0.0.0 --port 8000` |
| 8 | **Wait health** | poll `http://127.0.0.1:8000/status` |
| 9 | **Tunnel** | cloudflared → parse `trycloudflare.com` → **print grande** |
| 10 | **Instrucciones** | pegar URL en HopLab; link a Vercel |

Badge en README:
```markdown
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<ORG>/<REPO>/blob/main/hoplab-cloud/colab/HopLab_Server.ipynb)
```

Fallback túnel: localtunnel o ngrok (token en Colab Secrets) si cloudflared se cuelga.

---

## 7. Frontend en Vercel/Render

### 7.1 Cambios mínimos vs `ui-family`

1. `client.js`: base URL configurable  
   `getApiBase() → localStorage.getItem('hoplab_motor_url') || import.meta.env.VITE_API_BASE || ''`
2. Pantalla/modal **Conectar motor**: input URL + “Probar `/status`” + Guardar.
3. Banner si motor offline: “Abre el notebook Colab → Run all → pega la URL”.
4. Build: `npm run build` → estáticos; **sin** depender del proxy Vite.
5. Quitar mounts mock `/media-vod*` del build de producción.

### 7.2 Deploy

- Conectar repo (solo carpeta `hoplab-cloud/web` o root con `vercel.json`).
- Env opcional: `VITE_API_BASE=` vacío (el usuario pega el túnel).
- Dominio fijo de la **UI**; el **API** es dinámico.

---

## 8. Flujo de datos (sesión típica)

```
Usuario sube MP4 a Drive/hoplab-data/videos/   (o upload API v1.1)
       ↓
HopLab → POST /api/analyze  (túnel → Colab)
       ↓
Jobs en Colab GPU; escribe output/ en Drive
       ↓
UI scrub GET /frame/... (sirve Colab; archivos en Drive montado)
       ↓
Al cortar Colab: output ya en Drive; UI deja de funcionar hasta nuevo túnel
```

---

## 9. Fases de implementación

### Fase C0 — Esqueleto (esta carpeta) ✅
- [x] `hoplab-cloud/` + plan
- [ ] `shared/env.example`
- [ ] Stub `colab/README.md`

### Fase C1 — Motor Colab usable
- [ ] Notebook `HopLab_Server.ipynb` (celdas 1–10)
- [ ] `requirements-colab.txt`
- [ ] Script start + health wait + cloudflared
- [ ] Symlinks / env para `output` y `venues` en Drive
- [ ] Probar analyze de un VOD corto vía URL pública

### Fase C2 — API portable (parche mínimo al engine)
- [ ] `OUTPUT_ROOT` / `VIDEO_ROOT` / `VENUE_ROOT` por env (parche en `api_server.py` o wrapper en `hoplab-cloud` que setea cwd)
- [ ] Preferir wrapper primero para **no ensuciar** el lab local; promover a env en `prototype/` si conviene
- [ ] Fix `dataset.yaml` paths Linux
- [ ] `TJ_WRITE_ANNOTATED=0` por defecto en Colab

### Fase C3 — Web cloud
- [ ] Copiar/adaptar `ui-family` → `hoplab-cloud/web`
- [ ] Conectar motor (localStorage + status check)
- [ ] Deploy Vercel
- [ ] Probar CORS + scrub + Analizar end-to-end

### Fase C4 — UX on-demand
- [ ] Página “Cómo encender” con link Colab embebido
- [ ] Copiar URL con un clic desde notebook (celda HTML)
- [ ] Upload a Drive documentado; opcional `POST /api/upload`
- [ ] Limpieza: no acumular `_refined` eternos en Drive (script prune)

### Fase C5 — Mobile light (después)
- [ ] PWA del web cloud
- [ ] Capacitor cuando el flujo Colab+pegar-URL esté estable

---

## 10. Checklist “sin errores” al abrir el notebook

Antes de publicar el link:

- [ ] Primera celda verifica GPU (`torch.cuda.is_available()`) y avisa si es CPU
- [ ] Mount Drive con mensaje claro si el usuario cancela
- [ ] `pip` no falla si ya instalado (reinstalar OK)
- [ ] Puerto 8000 libre / matar proceso previo
- [ ] Esperar `/status` antes de cloudflared
- [ ] Si cloudflared no imprime URL en 60s → celda fallback localtunnel
- [ ] Imprimir: URL API, link UI Vercel, path Drive
- [ ] No depender de paths `C:\Users\...`

---

## 11. Costes y límites (honestos)

| Recurso | Coste | Nota |
|---------|-------|------|
| Colab free GPU | $0 | Sesiones limitadas; cola; no 24/7 |
| Colab Pro | opcional | Más estable para demos |
| Drive | $0 (15 GB) | Videos 1080p se comen cuota → usar `TJ_*` y borrar `_refined` viejos |
| cloudflared quick tunnel | $0 | URL nueva cada vez |
| Vercel/Render frontend | $0 | Solo estáticos |
| AWS | **no** | Cumple requisito |

---

## 12. Riesgos

| Riesgo | Mitigación |
|--------|------------|
| URL túnel cambia | UI “Conectar motor” + localStorage |
| Drive lento escribiendo JPEG | `TJ_WRITE_ANNOTATED=0`; stride ≥ 2; opcional lean frames |
| Colab desconecta a mitad de analyze | Jobs no sobreviven; re-lanzar; avisar en UI |
| YOLO download cada sesión | Cache pesos en `hoplab-data/models` o Drive ultralytics |
| ToS Colab (servidor largo) | Uso bajo demanda; no scrapear; no abusar |

---

## 13. Orden de trabajo recomendado (siguiente sprint)

1. Completar C0 stubs (`env.example`, `colab/README`).
2. Implementar C1 notebook + tunnel (prioridad: link Open in Colab funcional).
3. Wrapper cwd/env sin romper lab local (C2 ligero).
4. Fork web + Conectar motor + Vercel (C3).
5. Probar con un video real desde el móvil (misma WiFi no requerida: todo por HTTPS público).

---

## 14. Definición de “listo para demo”

- Un enlace al notebook → Run All → aparece URL túnel.
- HopLab en Vercel → pegar URL → library lista videos de Drive.
- Analizar un clip corto → ver frames/fases/métricas.
- Cerrar Colab; al reabrir otro día, los `output/` siguen en Drive.

Fin del plan.
