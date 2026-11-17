# Monitoreo Urbano

Sistema de monitoreo urbano por video: detecta eventos de interés en tiempo
real, los describe automáticamente en lenguaje natural y permite consultar el
historial con preguntas en lenguaje natural (por ejemplo, "¿hubo un choque en
Independencia y Sarandí esta tarde?").

Arquitectura de cuatro módulos, cada uno un proceso Python independiente,
comunicados a través de una única base Postgres y los clips en MinIO — sin
broker de mensajes (ver "Decisiones de infraestructura" en
[`docs/architecture.md`](docs/architecture.md)). Las cámaras se conectan vía
MediaMTX, que las republica para que Módulo 1 nunca dependa de la cámara
física directo. Los componentes GPU corren nativos, sin Docker; Postgres,
MinIO y MediaMTX sí corren en Docker (`docker-compose.yml`), es lo único que
hace falta para correr los módulos en más de una máquina más adelante:

| Módulo | Directorio | GPU | Rol |
|---|---|---|---|
| 1 — Visión | `src/monitoreo_urbano/vision_pipeline` | Sí (YOLO + ByteTrack) | Captura, detecta, trackea, dispara y recorta clips |
| 2 — Descripción | `src/monitoreo_urbano/vlm_describer` | Sí (VLM cuantizado) | Describe cada clip en lenguaje natural |
| 3a — Almacén | `src/monitoreo_urbano/event_store` | No | Embeddings (pgvector) + búsqueda híbrida |
| 3b — Consulta | `src/monitoreo_urbano/query_service` | No (habla con un LLM servido aparte) | API de chat RAG + alertas en tiempo real |

Ver [`docs/architecture.md`](docs/architecture.md) para los diagramas de
componentes, secuencia y modelo de datos.

## Instalación

Requiere **Python 3.10+**, **ffmpeg** en el PATH (lo usa `clip_writer.py` para
NVENC y el simulador de cámara de `scripts/`) y, para los módulos 1 y 2, una
GPU NVIDIA con CUDA instalado.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1

# Instalar torch con CUDA ANTES que el resto — usar el comando específico de
# https://pytorch.org/get-started/locally/ según la GPU y versión de CUDA instalada.
# GPUs Blackwell (RTX 50xx) necesitan PyTorch 2.7+ con CUDA 12.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Instalar según qué módulo vayas a correr:
pip install -e .
pip install -r requirements/vision.txt
pip install -r requirements/vlm.txt
pip install -r requirements/store.txt
pip install -r requirements/query.txt
pip install -r requirements/dev.txt   # sólo para correr tests/lint
```

Un único entorno virtual para todo el proyecto alcanza — evita duplicar
instalaciones pesadas de torch entre módulos.

## Configuración

```powershell
Copy-Item .env.example .env
Copy-Item config\cameras.example.yaml config\cameras.yaml
docker compose up -d postgres minio mediamtx
```

Editar `config/cameras.yaml` con las cámaras a usar y `.env` con los
paths/modelos que correspondan (ver comentarios en
`src/monitoreo_urbano/common/config.py` para el detalle de cada variable).
`MU_DATABASE_URL` y `MU_S3_*` ya traen defaults que coinciden con
`docker-compose.yml`, no hace falta tocarlos en desarrollo local.

Las cámaras se configuran como URLs RTSP republicadas por MediaMTX, no la
cámara física directo (ver `docs/architecture.md`). Sin una cámara IP real
todavía, `cam-demo` se puede simular con:

```powershell
python scripts/simulate_ip_camera.py --camera-id cam-demo
```

Republica `data/raw_videos/sample_bus.mp4` en loop hacia
`rtsp://localhost:8554/cam-demo` — dejarlo corriendo en su propia terminal
mientras se prueba Módulo 1 contra `cam-demo`. Para una cámara real, el
`source` de `cameras.yaml` pasa a ser la URL RTSP de esa cámara.

El Módulo 3b requiere un **servidor de LLM local** ya corriendo, expuesto como
API compatible con OpenAI (llama.cpp server, Ollama o vLLM) en
`MU_LLM_SERVER_BASE_URL` (default `http://127.0.0.1:8080/v1`). Probado con
Ollama: `ollama pull qwen2.5:7b`; `.env.example` ya trae `MU_LLM_SERVER_BASE_URL=http://127.0.0.1:11434/v1`
y `MU_LLM_MODEL_NAME=qwen2.5:7b` (Ollama expone su API OpenAI-compatible
en el puerto 11434, no 8080).

`.env.example` trae la configuración con la que se probó el sistema: detector
`data/models/yolo/baseline_v6.pt` a 960 px, VLM `Qwen/Qwen2-VL-2B-Instruct` y LLM
`qwen2.5:7b`. El VLM y el modelo de embeddings se descargan de HuggingFace la primera vez
que arranca cada módulo.

### Pesos entrenados

`data/models/yolo/` reúne los pesos de cada entrenamiento del detector (`baseline_v1.pt` a
`baseline_v6.pt` y `baseline_v10.pt`; v6hn, v7, v8 y v9 en `weights/best.pt` de su carpeta) y del
clasificador de choques (`accident_cls_*`), cada uno con su `args.yaml`, `results.csv` y curvas.
`data/models/trt/` tiene v6 exportado a ONNX. Los datasets no se incluyen: los scripts
`scripts/prepare_training_dataset*.py` arman cada conjunto a partir de las fuentes públicas, y
`scripts/reentreno_v10/` contiene la preparación, el entrenamiento y la evaluación de v10.

## Correr el sistema

Con Docker Desktop y el servidor de LLM (Ollama) corriendo:

```powershell
.\scripts\start_system.ps1                      # todas las cámaras de config\cameras.yaml
.\scripts\start_system.ps1 -Cameras cam-v4 -ShortClips   # una cámara, para videos de prueba cortos
.\scripts\stop_system.ps1                       # corta los procesos (los contenedores siguen)
```

El lanzador levanta Postgres, MinIO y MediaMTX, y después `query_service`, `vlm_describer`,
`event_store` y un `vision_pipeline` por cámara, esperando a que cada uno termine de cargar sus
modelos. Los logs quedan en `logs\<módulo>.log`. La interfaz queda en `http://127.0.0.1:8000/`.

Para correr un módulo a mano (una terminal cada uno):

```powershell
python -m monitoreo_urbano.query_service.main                   # Módulo 3b (API + interfaz)
python -m monitoreo_urbano.vlm_describer.main                   # Módulo 2
python -m monitoreo_urbano.event_store.main                     # Módulo 3a
python -m monitoreo_urbano.vision_pipeline.main --camera-id cam-01   # Módulo 1, uno por cámara
```

`http://127.0.0.1:8000/docs` expone la API interactiva (Swagger). Cada Módulo 1 se autoregistra en
la tabla `cameras` al arrancar; `GET /cameras` devuelve el catálogo de cámaras que efectivamente
están (o estuvieron) corriendo.

## Cómo responde el chat

El chat es una conversación con memoria, y el modelo de lenguaje razona en tres pasos:

1. **Interpretar.** El modelo recibe el mensaje, la conversación, los eventos que mostró antes, las cámaras
   y la fecha y hora actuales (con fechas de referencia ya calculadas). Devuelve un plan: una a tres
   búsquedas con texto, categorías (incendio, choque, pelea, etc.), rango de tiempo y cámara; o decide que
   es charla («hola», «gracias») y contesta directo. También entiende seguimientos («ese no es, el que busco
   tenía una lanza»). El sistema solo acepta un rango de tiempo o un rechazo si el modelo cita las palabras
   exactas que el operador escribió, para que no los invente.
2. **Buscar.** El sistema ejecuta las búsquedas (híbrida texto + vectores, o listado por fecha si no se
   pide ningún hecho concreto, como «qué pasó hoy»), respetando las cámaras del operador, los filtros de la
   interfaz y los eventos ya rechazados.
3. **Responder.** El modelo lee las descripciones de los candidatos, se queda con los que realmente
   responden (entiende sinónimos y detalles) y redacta una respuesta breve en lenguaje natural. Si nada
   responde, replanifica una vez con otra formulación antes de decir que no hay nada.

La interfaz muestra solo los clips que el modelo eligió, y guarda la conversación en el navegador de cada
operador; «Limpiar chat» la borra junto con los rechazos acumulados. El modelo configurado es `qwen2.5:7b`
(`MU_LLM_MODEL_NAME`); con el de 3B la interpretación era inestable.

Límite conocido: el chat solo ve los candidatos más cercanos a la pregunta, así que no puede contar
cuántos eventos hubo en total, y por eso la respuesta evita dar cantidades exactas.

## Descripciones sin rasgos de apariencia (RF4)

La regla vale por categoría de evento, y cada categoría tiene su prompt en `config/vlm_prompts/`.
Para pelea, posible altercado, fuego y genérico, `_appearance_rules.txt` pide que la descripción
arranque por la acción y nombre a las personas de forma neutra, sin género, edad, raza ni rasgos
físicos, y el módulo de descripción revisa cada texto (`vlm_describer/appearance.py`): si encuentra
esos términos, le pide al modelo que lo reescriba una vez. La ropa sigue permitida como dato
secundario. **El choque está exento**: su prompt no incluye esas reglas ni se reescribe, porque ahí
conviene poder describir a quienes se bajan de los vehículos. Las categorías exentas se configuran
con `MU_VLM_APPEARANCE_EXEMPT_REASONS` (por defecto `["accident"]`). Para medir cuánto cumple un
conjunto de prompts:

```powershell
python scripts/evaluate_descriptions.py --label actual --videos rutaideo.mp4 --reasons posible_altercado default
```

Usa la GPU: conviene correrlo con `scripts\stop_system.ps1` aplicado.

## Acceso y credenciales (RNF5)

Una clave compartida por el grupo de operadores protege la API, el canal de alertas y el video.
Sin `MU_API_KEY` el sistema corre abierto (solo para desarrollo local). Para desplegar:

```powershell
python scripts/generate_secrets.py --access-only   # clave de servicio + autenticación de MediaMTX (no toca volúmenes existentes)
python scripts/generate_secrets.py                 # además, credenciales propias de Postgres y MinIO (solo en un despliegue nuevo)
docker compose up -d --force-recreate mediamtx
```

- **Operadores:** cada uno tiene su propia clave en `config/operators.yaml` (ver
  `config/operators.example.yaml`; se agregan con `python scripts/add_operator.py NOMBRE
  --cameras cam-01,cam-02`). La clave lo identifica —queda registrado como autor de cada revisión—
  y define qué cámaras ve; sin lista de cámaras, ve todas. `MU_API_KEY` es la clave de servicio de
  Módulo 1 y los simuladores.
- **API:** todo endpoint exige el encabezado `X-API-Key`, salvo `/health`, la interfaz (`/ui`) y
  `/auth/mediamtx`. `WS /alerts` acepta la clave en el encabezado o como primer mensaje
  `{"api_key": "..."}` (un navegador no puede mandar encabezados en un WebSocket).
- **Video:** MediaMTX valida cada conexión contra `POST /auth/mediamtx` de `query_service`, con la
  misma clave como contraseña (usuario `operador`). Módulo 1 y `simulate_ip_camera.py` la agregan
  solos a la URL RTSP; no se guarda en el catálogo de cámaras. Con la autenticación activa,
  `query_service` tiene que estar corriendo para que MediaMTX deje conectar a cualquiera.
- **Postgres y MinIO:** las contraseñas salen de `.env` en vez de las que traen las imágenes.
  Solo se aplican al crear el volumen: con datos existentes hay que recrearlos (`docker compose
  down -v`, borra los datos) o cambiarlas adentro.
- **API en toda la red:** con la autenticación de MediaMTX activa, `MU_API_HOST` tiene que ser `0.0.0.0` para que el contenedor alcance a la API.
- **No incluido:** la red privada (VPN) y cerrar los puertos de Postgres, MinIO y la API al exterior.

## Tests

```powershell
docker compose up -d postgres minio   # los tests necesitan los dos corriendo
pytest
```

`tests/test_db.py` y `tests/test_event_trigger.py` no requieren GPU ni modelos
descargados: cubren la cola de eventos sobre Postgres y el motor de reglas de
disparo. Cada test aísla sus datos en un schema de Postgres propio, creado y
descartado automáticamente.
