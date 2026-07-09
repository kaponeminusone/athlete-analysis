# Plan de análisis — Carrera + 5 hops (single-video)

Punto de referencia antes de implementar capas de pista, fases y métricas.

## Evento

```text
[CARRERA] → [hop 1] → [hop 2] → [hop 3] → [hop 4] → [SALTO FINAL → arena]
```

- 4 hops con contacto en pista/corredor
- 5.º contacto = caída en arena
- Atleta asignado manualmente por sesión (ej. Mateo)
- Repetibilidad multi-video: fase posterior; ahora solo visualización por video

## Arquitectura de capas

1. **Calibración** — pista, corredor, landing_zone, eje 1D `s`
2. **Track scorer** — colisión pista↔atleta (lock fuerte) + predicción en vuelo
3. **Section analyzer** — 5 contactos, fases auto
4. **Métricas** — longitudes por hop, salto final (derivado, recalculable)
5. **UI** — timeline multicanal, overlays por capas, progreso hasta playhead

## Datos por video

```text
output/<video>/
├── analysis.json      # frames + kp + bbox + track_overlap + athlete_state
├── calibration.json   # geometría pista
├── sections.json      # fases, contactos
├── metrics.json       # longitudes (derivado)
└── corrections.json   # correcciones manuales
```

`derived_version` invalida métricas/secciones tras correcciones.

## Fases de implementación

| Fase | Entregable |
|------|------------|
| 0 | Schema extendido + persistir kp/bbox en pipeline |
| 1 | Calibración pista en UI + API |
| 2 | track_scorer.py + state machine |
| 3 | section_analyzer.py (5 contactos) |
| 4 | metrics.json |
| 5 | Multi-sesión atleta (futuro) |
| 6 | Recálculo post-corrección |

## UI (single-video)

- Modos: Revisar | Calibrar | Fases
- Timeline: detección, ángulo, fase, contactos ①–⑤, colisión
- Overlays: pista, colisión, predicción, badge de fase
- Inspector: progreso N/5 contactos, métricas del tramo actual
