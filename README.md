# Kenbun

Sistema de monitoreo urbano por video: detecta choques e incendios en tiempo real, alerta al
operador, describe cada evento en lenguaje natural y permite consultar el historial con preguntas
en lenguaje natural (por ejemplo, «¿hubo un choque en Independencia y Sarandí esta tarde?»).

Son cuatro módulos, cada uno un proceso Python independiente, comunicados a través de una única
base Postgres y con los clips en MinIO, sin broker de mensajes. Las cámaras llegan a través de
MediaMTX, que republica cada flujo por RTSP para el análisis y por HLS para la interfaz. Los
componentes con GPU corren nativos; Postgres, MinIO y MediaMTX corren en Docker
(`docker-compose.yml`).

| Módulo | Directorio | GPU | Rol |
|---|---|---|---|
| 1 — Visión | `src/monitoreo_urbano/vision_pipeline` | Sí (YOLO + ByteTrack) | Captura varias cámaras por proceso, detecta, sigue, dispara eventos y graba clips |
| 2 — Descripción | `src/monitoreo_urbano/vlm_describer` | Sí (Qwen2-VL) | Describe cada clip en lenguaje natural |
| 3a — Almacén | `src/monitoreo_urbano/event_store` | No | Indexa los eventos confirmados (pgvector + texto) y resuelve la búsqueda híbrida |
| 3b — Consulta | `src/monitoreo_urbano/query_service` | No (usa un LLM servido aparte) | API, interfaz de operador, chat, alertas y salud de cámaras en tiempo real |

[`docs/architecture.md`](docs/architecture.md) tiene las decisiones de diseño, los diagramas y el
modelo de datos; [`frontend/README.md`](frontend/README.md), la interfaz y el contrato de la API.

## Instalación

Requiere **Python 3.10+**, **ffmpeg** en el PATH (lo usa el escritor de clips, con NVENC si está
disponible), **Docker Desktop**, **Ollama** y una GPU NVIDIA con CUDA para los módulos 1 y 2.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1

# Instalar torch con CUDA antes que el resto, con el comando de https://pytorch.org/get-started/locally/
# según la GPU. Las GPU Blackwell (RTX 50xx) necesitan PyTorch 2.7+ con CUDA 12.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

pip install -e .
pip install -r requirements/vision.txt -r requirements/vlm.txt -r requirements/store.txt -r requirements/query.txt
pip install -r requirements/dev.txt   # solo para tests
```

## Configuración

```powershell
Copy-Item .env.example .env
Copy-Item config\cameras.example.yaml config\cameras.yaml
Copy-Item config\operators.example.yaml config\operators.yaml   # opcional: claves por operador
Copy-Item config\instances.example.yaml config\instances.yaml   # opcional: reparto de cámaras por GPU
python scripts/generate_secrets.py --access-only                # clave de servicio y autenticación de MediaMTX
ollama pull qwen2.5:7b
```

`.env.example` trae la configuración con la que se probó el sistema: detector
`data/models/yolo/baseline_v6.pt` a 960 px, VLM `Qwen/Qwen2-VL-2B-Instruct` y LLM `qwen2.5:7b`
en Ollama (`http://127.0.0.1:11434/v1`). El VLM y el modelo de embeddings
(`intfloat/multilingual-e5-large`) se descargan de HuggingFace la primera vez que arranca cada
módulo. `MU_DATABASE_URL` y `MU_S3_*` ya coinciden con `docker-compose.yml`. Cada variable está
explicada en `src/monitoreo_urbano/common/config.py`.

Las cámaras se declaran en `config/cameras.yaml` como URLs RTSP republicadas por MediaMTX
(`rtsp://localhost:8554/<camera_id>`), con su nombre, su ubicación (calles) y su comuna. Sin una
cámara real, `scripts/simulate_ip_camera.py --camera-id cam-demo --source-file <archivo.mp4>` publica
un video en bucle en ese path.

`config/instances.yaml` reparte las cámaras entre instancias de visión: cada instancia es un
proceso con un solo detector, en una GPU, que atiende todas sus cámaras. Sin ese archivo se levanta
una única instancia con todas las cámaras del catálogo.

### Pesos entrenados

`data/models/yolo/` reúne los pesos de cada entrenamiento del detector (`baseline_v1.pt` a
`baseline_v6.pt` y `baseline_v10.pt`; v6hn, v7, v8 y v9 en `weights/best.pt` de su carpeta) y del
clasificador de choques (`accident_cls_*`), cada uno con su `args.yaml`, `results.csv` y curvas.
El sistema usa v6 (clases `person`, `car`, `motorcycle`, `fire`, `smoke` y `accident`).
`data/models/trt/` tiene v6 exportado a ONNX. Los datasets no se incluyen: los scripts
`scripts/prepare_training_dataset*.py` arman cada conjunto a partir de las fuentes públicas, y
`scripts/reentreno_v10/` contiene la preparación, el entrenamiento y la evaluación de v10.

## Correr el sistema

Con Docker Desktop y Ollama corriendo:

```powershell
.\scripts\start_system.ps1                              # todas las instancias de config\instances.yaml
.\scripts\start_system.ps1 -Cameras cam-01,cam-02       # una sola instancia con esas cámaras
.\scripts\start_system.ps1 -Cameras cam-v4 -ShortClips  # videos de prueba cortos: clips y deduplicación más cortos
.\scripts\stop_system.ps1                               # corta los procesos (los contenedores siguen)
```

El lanzador levanta Postgres, MinIO y MediaMTX y después `query_service`, `vlm_describer`,
`event_store` y un `vision_pipeline` por instancia, esperando a que cada uno cargue sus modelos.
Los logs quedan en `logs\<módulo>.log` y la interfaz en `http://127.0.0.1:8000/`.

Para correr un módulo a mano (una terminal cada uno):

```powershell
python -m monitoreo_urbano.query_service.main                            # Módulo 3b (API + interfaz)
python -m monitoreo_urbano.vlm_describer.main                            # Módulo 2
python -m monitoreo_urbano.event_store.main                              # Módulo 3a
python -m monitoreo_urbano.vision_pipeline.main --instance vision-1      # Módulo 1, una instancia
python -m monitoreo_urbano.vision_pipeline.main --camera-id cam-01,cam-02
```

`http://127.0.0.1:8000/docs` expone la API (Swagger).

## Alertas, revisión y consulta

- **Alerta.** Un choque o un incendio que el detector sostiene en el tiempo dispara un evento y una
  alerta inmediata en la interfaz, antes de que termine de grabarse el clip. Un choque con evidencia
  débil (`posible_choque`) y dos personas cercanas y agitadas (`posible_altercado`) no alertan:
  quedan en la cola de revisión.
- **Descripción.** Al cerrar el clip (30 s después del disparo), el VLM lo describe a partir de los
  fotogramas alrededor del disparo, con un prompt por categoría (`config/vlm_prompts/`).
- **Revisión.** Ningún evento, ni siquiera una alerta, queda consultable hasta que un operador lo
  confirma: revisa la descripción, puede corregirla o cambiar la categoría y, en un choque, carga los
  vehículos (tipo y color). También puede descartarlo.
- **Consulta.** El chat busca solo entre eventos confirmados.

## Cómo responde el chat

El chat es una conversación con memoria, y el modelo de lenguaje razona en tres pasos:

1. **Interpretar.** El modelo recibe el mensaje, la conversación, los eventos que mostró antes, las
   cámaras y la fecha y hora actuales. Devuelve un plan: una a tres búsquedas con texto, categorías,
   rango de tiempo, cámara y vehículo (tipo y color). Entiende seguimientos («ese no es, el que busco
   era de noche»). El sistema solo acepta un rango de tiempo, una cámara o un rechazo si el operador
   efectivamente los mencionó, para que el modelo no los invente. La respuesta de charla (saludos,
   agradecimientos, qué puede hacer el asistente) solo se admite contra un vocabulario cerrado.
2. **Buscar.** El sistema ejecuta las búsquedas (híbrida texto + vectores, o listado por fecha si no
   se pide un hecho concreto), respetando las cámaras del operador, los filtros de la interfaz y los
   eventos ya rechazados.
3. **Responder.** El modelo lee las descripciones de los candidatos, se queda con los que responden y
   redacta una respuesta breve. Si nada responde, replanifica una vez antes de decir que no hay nada.

Todo lo que no escribe el sistema (mensaje del operador, historial y descripciones) viaja entre
etiquetas y se declara como dato, no como instrucción; la respuesta se muestra sin enlaces ni marcas
y con largo acotado. Límite conocido: el chat solo ve los candidatos más cercanos a la pregunta, así
que no puede contar cuántos eventos hubo en total.

## Descripciones (RF4)

Cada categoría tiene su prompt en `config/vlm_prompts/`. Para posible altercado, fuego y genérico,
`_appearance_rules.txt` pide que la descripción empiece por la acción y nombre a las personas de
forma neutra, sin género, edad ni rasgos físicos; el módulo revisa cada texto
(`vlm_describer/appearance.py`) y, si encuentra esos términos, pide reescribirlo una vez. Los
choques (`accident`, `posible_choque`) están exentos (`MU_VLM_APPEARANCE_EXEMPT_REASONS`).

Además, el texto de carteles que el modelo copie se reemplaza por una marca, se quitan las oraciones
que hablan de un hecho de otra categoría y, si la descripción es demasiado corta o niega el hecho
detectado, se guarda una descripción neutra armada con los datos del sistema.

## Acceso y credenciales (RNF5)

Sin `MU_API_KEY` el sistema corre abierto (solo para desarrollo local). Para desplegar:

```powershell
python scripts/generate_secrets.py --access-only   # clave de servicio + autenticación de MediaMTX (no toca volúmenes existentes)
python scripts/generate_secrets.py                 # además, credenciales propias de Postgres y MinIO (solo en un despliegue nuevo)
docker compose up -d --force-recreate mediamtx
```

- **Operadores:** cada uno tiene su clave en `config/operators.yaml`; se agregan con
  `python scripts/add_operator.py NOMBRE --cameras cam-01,cam-02`. La clave identifica al operador
  (queda registrado como autor de cada revisión) y define qué cámaras ve; sin lista, ve todas.
  `MU_API_KEY` es la clave de servicio de los módulos y los simuladores.
- **API:** todo endpoint exige el encabezado `X-API-Key`, salvo `/health`, la interfaz (`/ui`) y
  `/auth/mediamtx`. Los WebSocket aceptan la clave en el encabezado o como primer mensaje
  `{"api_key": "..."}`.
- **Video:** MediaMTX valida cada conexión contra `POST /auth/mediamtx`, con la misma clave como
  contraseña (usuario `operador`). Con la autenticación activa, `query_service` tiene que estar
  corriendo y `MU_API_HOST` tiene que ser `0.0.0.0` para que el contenedor alcance la API.
- **Postgres y MinIO:** las contraseñas salen de `.env` y solo se aplican al crear el volumen.
- **No incluido:** la red privada (VPN) y el cierre de puertos hacia el exterior.

## Tests

```powershell
docker compose up -d postgres minio   # los tests de base y de clips los necesitan
pytest
```

Cada test aísla sus datos en un schema de Postgres propio, creado y descartado automáticamente.
