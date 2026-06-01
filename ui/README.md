# Interfaz visual YOLO

Prototipo estatico para revisar varios `analysis.json`, sincronizar un video local y explorar
frames, metricas, timeline y graficas del analisis.

## Uso rapido

Desde la carpeta `prototype`:

```bash
python ui_server.py
```

Para desarrollo con React/Vite, deja ese servidor corriendo y en otra terminal ejecuta:

```bash
npm run dev
```

Luego abre:

```text
http://127.0.0.1:5173/
```

## Flujo recomendado

1. Clic en **Cargar demo** para leer `output/IMG_2048/analysis.json`.
2. Para un video real, pega la ruta del `.mp4` y presiona **Abrir**.
3. Si ya existe `output/<video>/analysis.json`, se carga automaticamente.
4. Si no existe, usa **Generar** para ejecutar `run.py` desde el servidor local.
5. Navega frames con el timeline o flechas y alterna entre **Video** y **Frame YOLO**.

## Salidas que consume

La UI espera el contrato actual del prototipo:

- `output/<video>/analysis.json`: datos por frame y resumen.
- `output/<video>/annotated/*.jpg`: frames con keypoints, bounding box, panel de angulo y barra de ratio.
- `output/<video>/frames/*.jpg`: frames crudos extraidos.
- `output/<video>/charts/camera_angle_timeline.png`: grafica principal del pipeline.

La grafica `camera_angle_timeline.png` se muestra como referencia del analisis y el timeline
interactivo permite saltar al frame correspondiente. La UI ya soporta `frame.crops` o
`frame.crop_paths` si mas adelante el pipeline exporta recortes por keypoint.
