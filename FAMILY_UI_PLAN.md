# Plan: UI familiar (web → app)

**Estado:** mock **landscape cinematográfico** con frames reales (ui-family/ · marca **HopLab**) — el mock portrait/phone fue descartado
**Fecha:** 2026-07-11  
**Relación con otros planes:** la optimización de frames/almacenamiento y la preparación dual de UI están en [`OPTIMIZATION_AND_UI_PLAN.md`](OPTIMIZATION_AND_UI_PLAN.md) y **avanzan de forma independiente**. Este documento define el producto, el modelo de interacción y el proceso de mock visual de la **nueva** interfaz familiar.

---

## 1. Intención de producto

Herramienta web **landscape-first** (tablet/desktop) para familias, coaches y atletas que quieren revisar carrera + hops sin el flujo experto de YOLO Review. Sensación de producto de streaming (YouTube watch maximizado + Netflix browse), con gestos adaptados a horizontal (drawers, timeline inferior, gutters laterales).

| Dimensión | Decisión |
|-----------|----------|
| **Sensación** | Reproductor maximizado estilo YouTube + browse estilo Netflix + editor de video ligero |
| **UI experta** | Se mantiene `ui/` (revisión YOLO, correcciones densas, flujos ML) |
| **UI familiar** | Nueva app `ui-family/` (nombre TBD) — acoplamiento **solo por API** (`api_server.py`) |
| **Automatización por defecto** | Detectar fases automáticamente + aplicar máscaras de venue (CNN) en videos nuevos |
| **Avanzado** | Aprender / entrenar CNN solo en configuración avanzada, con advertencias |

**Flujo de trabajo acordado:** primero mock visual interactivo (iteración con el usuario), luego integración con la API real.

---

## 2. Relación con optimización de frames

| Plan | Alcance | Dependencia |
|------|---------|-------------|
| [`OPTIMIZATION_AND_UI_PLAN.md`](OPTIMIZATION_AND_UI_PLAN.md) | Disco (`frames/` / `annotated/`), flags, preparación dual UI | Independiente |
| **Este plan** | UX familiar, gestos, mock → API | Independiente |

La UI familiar consumirá frames vía API en la fase de integración. El mock actual sirve JPEGs reales desde `output/VOD*` con mounts Vite (`/media-vod2`, etc.), sin cablear `api_server.py`.

---

## 3. Modelo de pantallas e interacción

### 3.1 Núcleo: reproductor maximizado (estilo YouTube)

El video (secuencia de frames del análisis) es la **superficie hero a pantalla completa** (full-bleed).

| Gesto / control | Comportamiento |
|-----------------|----------------|
| **Swipe hacia arriba desde el borde inferior** | Abre el **drawer de timeline** (ver §3.2) |
| **Tap en pausa / freeze frame** | Congela el frame y abre una **action sheet** contextual encima (ver §3.3) |
| **Botones ← / → de frame** | En **esquinas o gutters laterales** — nunca centrados (no tapar la vista ni la selección). Análogo a ±5s de YouTube, pero **frame analizado anterior/siguiente** |
| **Scrub horizontal** sobre el player | Avanza/retrocede frames |
| **Swipe desde el borde derecho extremo** | Abre el **drawer de pestañas** (config + estadísticas) |

### 3.2 Timeline drawer (swipe up)

Timeline mejorada, pensada para dedo:

- Scroll horizontal fluido
- **Preview del thumbnail** al scrubber antes de confirmar el seek
- **Marcadores** de hops / aterrizajes visibles en la pista
- Tap en marcador → seek a ese frame
- Tap en zona vacía o **long-press** → crear / etiquetar marcador (hop o landing si falta)

### 3.3 Action sheet al congelar (freeze)

Sobre el frame congelado:

1. **Overlay de anotación simplificado** (HUD, §3.7)
2. **Corrección de máscara con brush** — modo pincel como el actual, adaptado a móvil (pintar mientras se mantiene el dedo)

### 3.4 Drawers izquierdo / derecho (sin análisis o siempre disponibles)

Pestañas del drawer (p. ej. desde el borde derecho):

| Pestaña | Contenido |
|---------|-----------|
| **1. Generación / tracking** | Ajustes del análisis inicial + toggle de refine en el mismo sitio. Nombres amigables (sin jerga ML) |
| **2. Estadísticas** | Al abrir: el video **se encoge** para dejar espacio legible a hops/métricas. Tap en un hop → seek a ese frame |

Si el video **ya tiene análisis**, los drawers siguen disponibles; el foco principal sigue siendo player + timeline + acciones de freeze.

### 3.5 Biblioteca de atletas (metáfora Netflix)

Al ingerir / abrir un video:

- Asignar **quién** (atleta), **fecha**, **nota** opcional
- Navegar las sesiones de ese atleta como **temporadas / episodios** con thumbnails

### 3.6 Pista / venue (secundario)

No es primario. Vive bajo **Configuración avanzada**.

| Acción | Nivel | Notas |
|--------|-------|-------|
| Aplicar CNN de zonas (venue masks) | **Normal** — fácil de encontrar | Nombre amigable (ver §4) |
| “Aprender de este video” | Avanzado | Advertir: mala config puede dañar el modelo |
| “Entrenar modelo” | Avanzado | Idem; no es acción cotidiana |

### 3.7 Annotation HUD (simplificado, relativo al viewport)

Siempre el **mismo layout y tamaño respecto a la pantalla** (CSS / viewport), **no** escalado a la resolución del video:

- Calidad del frame
- Número de frame
- Indicador skip / no-skip (frames omitidos a menudo excluidos de algunos análisis)

### 3.8 Brush vs dibujo de pista (móvil)

| Modo | Gestos |
|------|--------|
| **Brush** (corregir máscara del atleta) | Pintar mientras se mantiene el dedo |
| **Polígono de pista** (avanzado) | **Tap** = colocar punto · **Long-press + drag** = colocar punto y seguir trazando polilínea mientras el dedo se mueve (puntos a lo largo del camino, visualización en vivo) — solo mientras se mantiene |

### 3.9 Automatización por defecto

Para **videos nuevos**:

1. Detección automática de fases / hops  
2. Aplicar máscaras de venue (CNN)  

Aprender / entrenar CNN queda fuera del camino feliz.

---

## 4. Nombres amigables: aplicar CNN de pista

Objetivo: que “aplicar máscaras de venue” suene a estadio, no a ML.

| Opción | Texto propuesto |
|--------|-----------------|
| A | Usar mapa de pista y arena |
| B | Detectar superficie del estadio |
| C | Aplicar zonas del campo |

**Recomendación:** **A — “Usar mapa de pista y arena”**  
Es concreto (pista + arena del triple jump), evita “detectar/aplicar” genéricos y no suena a modelo neuronal. En config avanzada se puede aclarar entre paréntesis o tooltip: *usa el mapa aprendido del estadio*.

---

## 5. Arquitectura prevista

```
ui-family/          ← app Vite + React + Tailwind (landscape cinematográfico)
  └─ mock fixtures  ← data.js + mounts /media-vod* → output/VOD*
ui/                 ← expert YOLO Review (sin compartir App.jsx)
api_server.py       ← único acoplamiento en fase de integración
```

| Tema | Decisión |
|------|----------|
| Stack | Vite + React + Tailwind, gestos pointer (mouse + touch) |
| Fase mock | Fixtures + frames reales de `output/`; puerto **5174**; sin API |
| Layout mock | Home Netflix (filas) · Watch YouTube (player + related) · modo cine full-bleed |
| Fase integración | Sustituir fixtures por `fetch` a `api_server.py` |
| Diseño | Tokens streaming (`#0f0f0f`, acento `#ff0033`); Outfit + Roboto Condensed; **sin** importar desde `ui/src/App.jsx` |
| PWA | Preparable después; **no** requerido en el mock |

---

## 6. Proceso de iteración del mock

1. **Plan** — este documento  
2. **Mock visual interactivo** — gestos, drawers, datos fake; bucles de feedback con el usuario  
3. **Pulir** hasta aprobación explícita  
4. **Cablear API real** — jobs de análisis, correcciones, sections, metrics, overlays  
5. **Opcional más adelante** — envoltorio Capacitor / React Native  

No se implementa producto “de verdad” hasta cerrar el look & feel del mock.

---

## 7. Skills y herramientas recomendadas

| Recurso | ¿Usar? | Motivo |
|---------|--------|--------|
| Reglas de diseño frontend del usuario | **Sí** | Tipografía expresiva, sin púrpura genérico AI, motion intencional, mobile-first, composición clara |
| Mock React en el repo | **Preferido** | Mejor fidelidad de gestos (swipe, long-press, drawers) que un Figma estático |
| Figma + skill `figma-generate-design` | **Opcional** | Solo si se quieren pantallas estáticas primero; si no, omitir |
| Canvas skill | **No** | No es un artefacto de datos / canvas analítico |
| Automate / babysit skills | **No** | No aplican a este trabajo de diseño/mock |

---

## 8. Checklist por fases

### Fase M0 — Plan
- [x] Documento `FAMILY_UI_PLAN.md` (este archivo)
- [ ] Revisar y responder preguntas abiertas (§9)
- [x] Confirmar nombre de carpeta (`ui-family/`) + marca **HopLab**

### Fase M1 — Shell + chrome del player + drawers (mock)
- [x] Scaffold Vite + React + Tailwind en `ui-family/` *(mock started)*
- [x] Player full-bleed + botones de frame en esquinas/gutters
- [x] Scrub horizontal de frames
- [x] Swipe desde borde derecho → drawer de pestañas (config + stats)
- [x] Fixtures mínimos (proyecto sin/con análisis)

### Fase M2 — Timeline + freeze sheet + HUD de anotaciones
- [x] Swipe up → timeline drawer (scroll, scrub preview, marcadores)
- [x] Freeze → action sheet (anotación + brush)
- [x] HUD viewport-relative (calidad, nº frame, skip)
- [x] Crear/etiquetar marcadores (tap vacío / long-press) *(menú mock)*

### Fase M3 — Biblioteca de atletas (estilo Netflix)
- [x] Flujo ingest: atleta + fecha + nota *(chips editables en player; browse en home)*
- [x] Browse por atleta / sesiones (thumbnails, temporadas-episodios)

### Fase M4 — Pista / CNN con nombres amigables
- [x] “Usar mapa de pista y arena” en config normal
- [x] Sección avanzada: aprender / entrenar con warnings
- [ ] Gestos de polígono de pista (tap / long-press-drag) *(pendiente; fuera del mock M1–M4 core)*

### Fase M5 — Integración API
- [ ] Sustituir fixtures por endpoints de `api_server.py`
- [ ] Jobs de análisis, correcciones, sections, metrics, overlays
- [ ] Defaults: auto fases + aplicar mapa de pista/arena

### Fase M6 — (futuro) Envoltorio nativo
- [ ] PWA / Capacitor / React Native según necesidad
- [ ] Safe-areas, install prompt, gestos nativos si aplica

---

## 9. Preguntas abiertas

1. **Nombre de la app / carpeta:** ¿`ui-family/`, `ui-coach/`, `app/` u otro?  
2. **Nombre del “corredor” por defecto** en la biblioteca (etiqueta genérica si aún no hay atleta): ¿“Atleta”, “Corredor”, “Sesión sin asignar”?  
3. **¿Atleta obligatorio** al abrir/ingerir, o se puede posponer?  
4. **¿Drawers siempre visibles** (aunque no haya análisis) o solo el de config hasta el primer resultado?  
5. **Idioma de UI:** ¿solo español, o i18n desde el mock?  
6. **¿Dark / light / ambient?** Preferencia de atmósfera (evitar look genérico AI).  
7. **¿Mock Figma primero** o ir directo a mock React gestual? (recomendación: React)  
8. **¿Reutilizar frames reales** de algún `output/<video>/` en el mock, o solo placeholders?  
9. **Nombre final del toggle CNN** — ¿aceptamos “Usar mapa de pista y arena” o preferís B/C?  
10. **¿Refine** se muestra como “Mejorar seguimiento” / “Reanalizar con más detalle” u otra etiqueta?

---

## 10. Fuera de alcance (por ahora)

- Reemplazar o fusionar la UI experta `ui/`
- Entrenar/aprender CNN desde el camino feliz familiar
- PWA / app store / Capacitor (fase M6)
- Cambios de pipeline o optimización de disco (ver `OPTIMIZATION_AND_UI_PLAN.md`)
- Importar componentes desde `App.jsx` actual

---

## 11. Criterio de éxito del mock

El mock se considera listo para integración cuando el usuario pueda, en viewport **landscape** (desktop/tablet):

1. Ver el player 16:9 con **fotos reales** de triple jump (no siluetas fake)  
2. Biblioteca tipo Netflix + watch tipo YouTube (related column)  
3. Modo cine full-bleed + timeline (swipe-up / control) con preview JPEG y marcadores  
4. Freeze → sheet con HUD + brush  
5. Panel derecho config/stats (stats encoge el player) y seek desde un hop  
6. Encontrar “Usar mapa de pista y arena” sin jerga  

…y dé el visto bueno visual antes de cablear la API.
