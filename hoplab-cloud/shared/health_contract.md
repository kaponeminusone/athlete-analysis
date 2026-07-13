# Shared — contrato UI ↔ motor Colab

## Health

`GET {API_PUBLIC_URL}/status` debe responder JSON usable (cualquier 200 con cuerpo es suficiente para “motor online”).

## Base URL

- Producción web: `localStorage.hoplab_motor_url` o `VITE_API_BASE`
- Sin trailing slash
- Todas las rutas relativas del cliente: `/api/...`, `/frame/...`, `/correct`, `/media?...`

## CORS

El motor Colab mantiene `allow_origins=["*"]` en v1.

## Persistencia

Datos en Drive; la URL del túnel **no** se persiste en el build de Vercel (cambia por sesión).
