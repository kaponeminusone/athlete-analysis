# Triple Jump — Frame Extraction & Camera Angle Prototype

Prototipo diagnostico: extrae frames de un video, corre YOLO11 pose,
y clasifica el angulo de la camara en cada frame.

## Setup rapido

```bash
# 1. Crear entorno virtual
python -m venv venv
venv\Scripts\activate

# 2. Instalar PyTorch con CUDA (elegir segun tu GPU)

# GTX 1050 → CUDA 11.8
pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu118

# RTX 3050 → CUDA 12.1
pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121

# 3. Instalar el resto
pip install -r requirements.txt
```

## Uso

```bash
# Basico (extrae 1 de cada 5 frames, anota cada 10)
python run.py mi_video.mp4

# Solo un segmento del video (segundos 3 a 18), mas denso
python run.py mi_video.mp4 --stride 2 --start 3 --end 18

# Sin modelo de segmentacion (mas rapido, solo pose)
python run.py mi_video.mp4 --no-seg

# Anotar todos los frames (genera muchas imagenes)
python run.py mi_video.mp4 --annotate-all

# Carpeta de salida personalizada
python run.py mi_video.mp4 --output resultados/atleta_01
```

## Interfaz visual

El prototipo incluye una UI tipo editor para revisar varios `analysis.json`, asociar un
video local, navegar frames, ver metricas, timeline y graficas:

```bash
python ui_server.py
```

Luego abre `http://localhost:8000/ui/`.

Para desarrollo de la interfaz React:

```bash
npm run dev
```

Luego abre `http://127.0.0.1:5173/` y deja `python ui_server.py` corriendo para la API.

La interfaz esta en `ui/` con React + Vite + Tailwind. El servidor local recibe una ruta de video, busca
automaticamente `output/<nombre_video>/analysis.json` y, si falta, puede lanzar `run.py`
para generar el analisis.

Mapeo en la UI:

- `output/<video>/annotated/*.jpg`: visor de frame anotado, con keypoints, bounding box,
  panel de angulo y barra de ratio.
- `output/<video>/frames/*.jpg`: fallback para ver el frame crudo extraido.
- `output/<video>/charts/camera_angle_timeline.png`: grafica principal del pipeline,
  util para decidir los tramos laterales confiables.
- `output/<video>/analysis.json`: timeline interactivo, metricas por frame y resumen.

## Salidas generadas

```
output/
├── frames/
│   ├── frame_000000.jpg      # frames crudos extraidos
│   ├── frame_000005.jpg
│   └── frames_meta.json      # metadata de cada frame
│
├── annotated/
│   ├── annotated_000000.jpg  # frames con keypoints + angulo dibujado
│   └── annotated_000050.jpg
│
├── charts/
│   └── camera_angle_timeline.png  # grafica de angulo a lo largo del tiempo
│
└── analysis.json             # resultados completos por frame (JSON)
```

## Clasificacion de angulo de camara

| Clase | Descripcion | Shoulder ratio | Uso en analisis |
|---|---|---|---|
| `LATERAL` | Perfil lateral | < 0.20 | **Optimo** para angulos articulares |
| `SEMI_BACK` | Semi-espalda | 0.05 - 0.20 | Util con precaucion |
| `SEMI_FRONT` | Semi-frontal | 0.50 - 0.80 | Angulos distorsionados |
| `FRONTAL` | De frente | > 0.80 | Evitar para biomecánica |
| `UNKNOWN` | Sin deteccion | — | Descartar |

El **shoulder ratio** = ancho hombros (px) / altura torso (px).
Cuando el atleta está de perfil, un hombro queda detras del otro
y el ratio cae. Cuando está de frente, ambos hombros son visibles
y el ratio sube.

## Estructura del JSON de salida

```json
{
  "video": "mi_video.mp4",
  "summary": {
    "camera_angle_distribution": { "LATERAL": 45.2, "SEMI_FRONT": 30.1, ... },
    "dominant_angle": "LATERAL",
    "lateral_frames_pct": 45.2
  },
  "frames": [
    {
      "frame_idx": 0,
      "timestamp_s": 0.0,
      "camera_angle": "FRONTAL",
      "shoulder_ratio": 0.91,
      "keypoints_valid": 9
    },
    ...
  ]
}
```
