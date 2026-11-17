# Frontend

Interfaz de operador en HTML y JavaScript, sin paso de compilación. `query_service` la sirve desde el
mismo origen que la API, así que no hace falta CORS ni un servidor aparte:

```powershell
docker compose up -d postgres minio mediamtx   # MediaMTX publica también HLS en :8888
python -m monitoreo_urbano.query_service.main
# abrir http://127.0.0.1:8000/
```

Es una sola página (`index.html`) con dos vistas que comparten la barra superior y las conexiones en
tiempo real, así que cambiar de vista no corta una consulta en curso ni los canales:

- **Tablero:** cuadrícula de cámaras en vivo, panel de alertas y chat de consulta retrospectiva.
  «Revisar» en una alerta abre el panel de revisión sobre el chat, sin dejar de ver las cámaras.
- **Revisión** (`index.html#revision`; `revision.html` redirige ahí): cola de eventos sin confirmar,
  alertas y posibles eventos.

Detalles:

- El navegador no reproduce RTSP: el video en vivo va por HLS desde MediaMTX
  (`http://<host>:8888/<camera_id>/index.m3u8`) y se reconecta solo si se cae o se congela. `hls.js` se
  carga desde un CDN.
- **Una sola selección de cámaras (RF15).** Las que se eligen con «Elegir cámaras» se muestran en la
  cuadrícula y son las únicas de las que llegan alertas (`WS /alerts?cameras=cam-01,cam-02`). La
  consulta retrospectiva ve todas las cámaras que permite la clave.
- **Salud de cámara (RF10)** por su propio canal, `WS /camera-status`: la barra superior muestra
  `Status cámaras: OK` o cuántas fallan, con el detalle al tocarla, y la celda de la cámara marca
  «Sin señal», «Tapada» o «Sin análisis».
- Si el servidor tiene `MU_API_KEY`, la interfaz pide la clave del operador al abrir y la manda en
  `X-API-Key`; en los WebSocket va como primer mensaje y en el video HLS como autenticación básica
  (usuario `operador`).
- Todo texto que viene del servidor, en especial las descripciones que escribe un modelo, se inserta
  con `textContent`, nunca como HTML.
- Las marcas de tiempo se guardan en UTC y se muestran en hora local.

## Contrato de la API

- `POST /chat`: consulta en lenguaje natural, con cámara y período opcionales. Devuelve el texto y los
  eventos citados, solo entre eventos confirmados.
- `WS /alerts`: alertas en tiempo real, un mensaje por evento, solo de incendios y choques. Llega
  apenas se dispara el evento, casi siempre sin descripción: la interfaz la pide después con
  `GET /events/{event_id}`.
- `WS /camera-status` y `GET /cameras/health`: estado de salud de cada cámara.
- `GET /cameras`: catálogo de cámaras que la clave permite ver.
- `GET /events/pending-review`: eventos descritos sin confirmar.
- `GET /events/{event_id}`: un evento puntual.
- `POST /events/{id}/details`: revisión completa y **única forma de confirmar**:
  `{description_nl, trigger_reason?, vehicles: [{vehicle_type, color}]}`. `vehicle_type` es `auto`,
  `moto`, `camion`, `camioneta` o `pickup`. En `accident` y `posible_choque` exige al menos un
  vehículo (422 si falta). Responde 409 si el evento fue descartado o todavía no tiene descripción.
- `POST /events/{id}/review`: descarta (`dismissed`). Confirmar por acá responde 400.
- `POST /events/{id}/category`: cambia la categoría (`posible_altercado`, `posible_choque`, `fight`,
  `accident` o `fire`) y conserva la original.
- `POST /events/{id}/description`: corrige el texto de la descripción y reindexa.
- `GET /clip-url/{clip_path}`: URL prefirmada del clip en MinIO (`GET /clips/{clip_path}` redirige a
  la misma URL).
- `GET`, `POST` y `DELETE /config/runtime`: ajustes en vivo, como el umbral de confianza del detector.
- `GET /auth/check`: valida la clave y devuelve el operador.
