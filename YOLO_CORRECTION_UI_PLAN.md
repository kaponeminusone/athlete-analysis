# Plan de mejora: correccion asistida de YOLO por frame

## Objetivo

Convertir el revisor en una herramienta para detectar y corregir errores de YOLO por frame:
persona equivocada, pose mal asignada, keypoints incorrectos, frames sin deteccion o segmentos
donde la camara no sirve para analisis biomecanico confiable.

## Acople con la UI actual

La UI ya queda organizada alrededor de un video:

- El usuario indica la ruta del video.
- El servidor local busca `output/<nombre_video>/analysis.json`.
- Si existe, carga video, frames anotados, frames crudos y grafica.
- Si no existe, puede lanzar `run.py` y generar el output con la misma estructura.

Sobre ese flujo se agregaria una capa de correcciones, sin modificar el resultado original de YOLO.

## Nuevo archivo de correcciones

Crear por video:

```text
output/<video>/corrections.json
```

Estructura propuesta:

```json
{
  "version": 1,
  "video": "IMG_2048.mp4",
  "frames": {
    "12": {
      "status": "reviewed",
      "usable_for_analysis": true,
      "person_override": "primary",
      "camera_angle_override": "SEMI_BACK",
      "notes": "Pose correcta, frame no lateral optimo."
    },
    "84": {
      "status": "needs_fix",
      "usable_for_analysis": false,
      "error_type": "wrong_person",
      "notes": "YOLO eligio una persona del fondo."
    }
  }
}
```

## Interacciones por frame

Primera fase:

- Marcar frame como `usable_for_analysis` o descartado.
- Etiquetar error: `wrong_person`, `bad_pose`, `missing_detection`, `occlusion`, `bad_camera_angle`.
- Agregar nota corta.
- Sobrescribir clase de angulo si la heuristica falla.

Segunda fase:

- Si YOLO detecta varias personas, mostrar miniaturas de candidatos y permitir elegir la persona correcta.
- Guardar `person_override` con el indice de deteccion.
- Regenerar imagen anotada del frame usando la persona elegida.

Tercera fase:

- Permitir mover keypoints manualmente sobre el frame.
- Guardar `keypoint_overrides` por punto.
- Recalcular metricas derivadas del frame corregido.

## Cambios necesarios en el backend

- Agregar endpoints en `ui_server.py`:
  - `GET /api/corrections?video_path=...`
  - `POST /api/corrections`
  - `POST /api/frame/reannotate`
- Separar en el pipeline los datos crudos de deteccion de los datos resumidos.
- Exportar candidatos de persona por frame cuando existan varias detecciones.
- Guardar keypoints completos en `analysis.json`, no solo conteos y ratios.

## Cambios necesarios en `analysis.json`

Para correccion real se necesita ampliar cada frame con:

- `person_candidates`: bbox, score, keypoints, mask opcional.
- `selected_person_idx`: persona elegida automaticamente.
- `keypoints`: coordenadas y confianza por punto.
- `raw_frame_path` y `annotated_frame_path`.

Con eso la UI puede mostrar exactamente que decidio YOLO y permitir corregirlo sin adivinar.

## Resultado esperado

El documento de analisis se refinaria usando dos capas:

- Resultado automatico: `analysis.json`.
- Criterio humano de revision: `corrections.json`.

La grafica y el timeline deberian poder filtrar por "frames confiables", no solo por angulo
lateral automatico. Eso permite elegir mejor los frames para calcular angulos articulares.
