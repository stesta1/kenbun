# Frontend

Interfaz de operador en HTML + JavaScript sin paso de compilación. `query_service` la sirve desde
el mismo origen que la API, así que no hace falta CORS ni un servidor aparte:

```powershell
docker compose up -d postgres minio mediamtx   # mediamtx ahora también publica HLS en :8888
python -m monitoreo_urbano.query_service.main
# abrir http://127.0.0.1:8000/  (redirige a /ui/)
```

- `index.html` — **tablero**: cuadrícula de cámaras, panel de alertas y consulta (CU1–CU5).
- `revision.html` — **revisión** de eventos sin confirmar, con corrección de texto (CU6, CU7).
- El navegador no reproduce RTSP: el video en vivo va por HLS desde MediaMTX
  (`http://<host>:8888/<path>/index.m3u8`, mismo path que el `source` del catálogo). `hls.js` se
  carga desde un CDN, así que la primera carga necesita internet (o se puede copiar a `js/`).
- La salud de cámara (RF10) tiene su propio canal, `WS /camera-status`, y no entra al panel de
  alertas: la barra superior muestra `Status cámaras: OK` o `Status cámaras: N con falla`
  que al tocarlo abre el listado de las que fallan, con el problema y desde cuándo, y la celda de
  esa cámara se marca en rojo. Al conectarse el canal manda el estado actual
  de cada cámara, así que recargar no pierde nada, y avisa también cuando una cámara se recupera.
  `GET /cameras/health` devuelve el mismo estado por HTTP.
- Si el servidor tiene `MU_API_KEY`, la interfaz pide la clave al abrir (se guarda solo en esa
  pestaña) y la manda en el encabezado `X-API-Key`; en el WebSocket va como primer mensaje; en el
  video HLS, como autenticación básica (usuario `operador`). Los clips se piden con
  `GET /clip-url/<path>`, que devuelve la URL prefirmada como JSON.
- **Una sola selección de cámaras (RF15).** Las que se eligen con «Elegir cámaras» se muestran en la
  cuadrícula y son las únicas de las que se reciben alertas. La interfaz abre
  `WS /alerts?cameras=cam-01,cam-02` y `query_service` solo manda a esa conexión las alertas de
  esas cámaras, incluidas las de falla de cámara (sin el parámetro llegan todas; con `?cameras=`
  vacío, ninguna). La consulta retrospectiva no se acota: ve todas las que la clave permite.
- Cada operador entra con su propia clave, que define qué cámaras puede ver (RNF5, ver el README
  del proyecto).

## Contrato de la API

`query_service` (Módulo 3b) expone todo lo que la interfaz necesita:

- `POST /chat` — consulta en lenguaje natural, devuelve texto + eventos citados
  (cada uno con `clip_path`, servido en `GET /clips/<clip_path>`). Solo cita
  eventos `indexed` — ver más abajo qué significa eso exactamente.
- `WS /alerts` — alertas en tiempo real, un único mensaje por evento. Sólo se
  alertan incidentes reales (`fire`/`accident`/`fight`) — no cualquier evento
  detectado. La salud de cámara viaja por otro canal (`WS /camera-status`). Como la alerta llega apenas se crea el evento y el VLM tarda en
  describirlo, casi siempre va sin `description_nl` todavía: hay que pedirla
  aparte.
- `GET /events/{event_id}` — un evento puntual, para completar la descripción
  de una alerta que llegó sin ella, o para armar una vista de detalle.
- `GET /cameras` — catálogo de cámaras autoregistradas (para poblar un
  selector, por ejemplo).
- `POST /events/{id}/review` — descarta un evento (`dismissed`): lo excluye del
  historial consultable y no se indexa. Confirmar (`confirmed`) se rechaza con 400:
  un evento solo se confirma revisándolo con `POST /events/{id}/details`.
- `POST /events/{id}/details` — revisión completa de un evento, sea una alerta o un
  posible evento: `{description_nl, trigger_reason?, vehicles: [{vehicle_type, color}]}`.
  `vehicle_type` es `auto`, `moto`, `camion`, `camioneta` o `pickup`; `color` es texto libre.
  En `accident` y `posible_choque` exige al menos un vehículo (422 si falta); en las demás
  categorías ignora la lista. Confirma el evento, reemplaza sus vehículos y lo reindexa.
  Responde 409 si el evento fue descartado o todavía no tiene descripción.
- `POST /events/{id}/category` — cambia la categoría de un evento
  (`posible_altercado`, `posible_choque`, `fight`, `accident` o `fire`). Queda confirmado, con la categoría
  original conservada, y sale de la cola de revisión. Se usa desde la pantalla de
  revisión y desde la ventana de corrección de una alerta.
- `POST /events/{id}/description` — corrige el texto de la descripción de un
  evento. Reindexa solo, no hace falta ningún paso adicional del lado del
  frontend.
- `GET /events/pending-review` — eventos que nadie vio todavía y necesitan
  confirmación humana antes de ser buscables (ver "Eventos vs. alertas" abajo).

## Eventos vs. alertas — por qué importa para la UI

No todo lo que el sistema detecta es una alerta, y no todo lo que el sistema
detecta es consultable por el chat todavía:

- Un incidente confirmado (`fight`, `fire`, `accident`) **alerta por WebSocket y queda buscable automáticamente**.
- `posible_altercado` —una heurística de movimiento (dos personas cerca, una
  moviéndose erráticamente), no una clasificación visual directa— **no alerta
  y no es buscable hasta que una operadora lo confirme**. Antes de confirmarlo
  no aparece en ninguna respuesta de `/chat`: el sistema no quiere afirmar
  algo que la IA nunca confirmó con certeza.

Esto significa que la interfaz necesita **cuatro** vistas, no dos:

1. **Monitoreo en tiempo real**: feed de alertas entrantes por WebSocket, con
   prioridad en que se note de inmediato (color/sonido) por sobre mostrar
   mucha información de una. Cada alerta con acceso directo a "ver clip".
2. **Consulta retrospectiva (chat)**: input de texto libre, sin filtros
   armados a mano, respuesta con las citas de eventos y su clip reproducible
   en la misma vista.
3. **Detalle / revisión de un evento** *(en `revision.html`; desde una alerta del tablero solo se
   puede marcar confirmado/descartado, no corregir el texto)*: descripción
   + clip de un evento puntual (`GET /events/{id}` + `GET /clips/...`), con
   la acción de marcarlo confirmado/descartado (`POST /events/{id}/review`) o
   corregir su texto (`POST /events/{id}/description`). La abren tanto desde
   una alerta como desde un resultado del chat — es la misma pantalla en los
   dos casos.
4. **Eventos sin confirmar** *(cola de `revision.html`)*: lista de
   `GET /events/pending-review` —cosas como `posible_altercado` que nadie
   revisó— para que la operadora las mire cuando tenga tiempo, no algo que la
   interrumpa. Confirmar o descartar ahí usa la misma acción que la vista 3.
