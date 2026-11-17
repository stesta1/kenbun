from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="MU_", extra="ignore")

    data_dir: Path = PROJECT_ROOT / "data"
    database_url: str = "postgresql://monitoreo:monitoreo@localhost:5432/monitoreo_urbano"
    raw_videos_dir: Path = PROJECT_ROOT / "data" / "raw_videos"
    models_dir: Path = PROJECT_ROOT / "data" / "models"
    live_preview_dir: Path = PROJECT_ROOT / "data" / "live"
    cameras_config_path: Path = PROJECT_ROOT / "config" / "cameras.yaml"
    instances_config_path: Path = PROJECT_ROOT / "config" / "instances.yaml"
    operators_config_path: Path = PROJECT_ROOT / "config" / "operators.yaml"

    yolo_weights_path: Path = PROJECT_ROOT / "data" / "models" / "yolo" / "baseline_v6.pt"
    yolo_device: str = "cuda:0"
    yolo_confidence_threshold: float = 0.35
    # Resolución de inferencia. baseline_v6 se entrenó y evaluó a 960 px; los pesos anteriores (v1 a v5) usan 640.
    yolo_imgsz: int = 960
    # Inferencia en FP16: unas 1,5 veces más rápida en la GPU y la mitad de VRAM, con diferencias numéricas
    # mínimas en las cajas. Apagada por defecto para conservar los resultados históricos.
    yolo_half: bool = False
    # Motor de lotes: máximo de fotogramas (uno por cámara) por llamada al detector, y cuánto se espera,
    # una vez que hay al menos uno, para completar el lote antes de inferir.
    detector_max_batch: int = 16
    detector_batch_fill_wait_seconds: float = 0.005
    detection_classes: list[str] = Field(
        default_factory=lambda: ["person", "car", "motorcycle", "fire", "smoke", "accident"]
    )
    # Confianza mínima por clase. Las que no figuran usan `yolo_confidence_threshold` (ajustable en vivo).
    # Choque e incendio alertan solos, así que exigen más; los valores se calibran con `scripts/evaluate_on_videos.py`.
    yolo_confidence_by_class: dict[str, float] = Field(default_factory=lambda: {"fire": 0.403, "accident": 0.25})
    frame_buffer_seconds: float = 30.0
    clip_post_event_seconds: float = 30.0
    track_buffer_seconds: float = 3.0
    # Reglas de disparo expresadas en tiempo, para que no dependan de los fps de análisis. Los valores
    # reproducen el comportamiento histórico a 30 fps (8 fotogramas, 150 de historial, 60 de ausencia).
    # Decodificación de video de las cámaras: "none" (CPU), "any" (hardware si está disponible), "d3d11",
    # "vaapi" o "mfx". Con hardware baja la CPU de forma drástica, a cambio de VRAM por cada flujo abierto.
    # Instancia de visión (varias cámaras en un proceso). `analysis_fps` en 0 analiza a los fps nativos de cada
    # cámara; con un valor, se analiza 1 de cada N fotogramas para acercarse a esa frecuencia.
    analysis_fps: float = 0.0
    camera_queue_size: int = 4
    # Calidad JPEG del buffer previo y de los clips abiertos (0 = guardar sin comprimir, como antes).
    clip_buffer_jpeg_quality: int = 85
    clip_writer_workers: int = 2
    # Vista previa con cajas dibujadas en data/live/<cámara>.jpg, solo para depuración (la interfaz usa HLS).
    preview_enabled: bool = False
    preview_interval_seconds: float = 1.0
    stats_log_interval_seconds: float = 30.0
    capture_hw_acceleration: str = "none"
    # Hilos que usa el decodificador de cada cámara. Por defecto OpenCV abre tantos como núcleos (16 en una
    # CPU de 16 hilos) por cada flujo: con decenas de cámaras son cientos de hilos que le quitan CPU al detector.
    # 1 reduce el costo por fotograma y los hilos; 0 deja el comportamiento automático de OpenCV.
    capture_decoder_threads: int = 1
    trigger_min_seconds: float = 0.27
    trigger_min_samples: int = 3
    trigger_history_seconds: float = 5.0
    trigger_stale_after_seconds: float = 2.0
    # Persistencia mínima (segundos) de un incidente directo antes de dispararlo, por clase. Un choque o un
    # incendio reales duran más que un parpadeo del detector; las demás clases usan `trigger_min_seconds`.
    incident_min_seconds: dict[str, float] = Field(default_factory=lambda: {"fire": 0.8, "accident": 0.65})
    # Ventana (segundos) en la que se acumulan las detecciones de un incidente. Un choque se detecta en ráfagas de pocos
    # fotogramas que el seguimiento fragmenta en varios identificadores: se cuentan las detecciones cercanas, no las de un
    # mismo identificador.
    incident_window_seconds: float = 2.0
    incident_cluster_radius: float = 3.0  # distancia máxima entre detecciones de una ráfaga, en tamaños de caja
    # Los choques entran a las reglas sin pasar por el seguimiento. ByteTrack solo confirma pistas con varios fotogramas
    # seguidos y el detector da el choque en ráfagas de 1 a 3 cuadros, que así nunca llegaban a la regla de ventana.
    accident_direct_trigger: bool = True
    # Niveles de certeza del choque. El detector deja pasar detecciones desde `yolo_confidence_by_class["accident"]` (0,25).
    #   Alerta (`accident`): detecciones con confianza >= `incident_alert_confidence` repetidas durante `incident_min_seconds`
    #       dentro de la ventana, o una sola con confianza >= `incident_strong_confidence`.
    #   Revisión (`posible_choque`): evidencia más débil, `incident_review_min_hits` detecciones >= `incident_review_confidence`
    #       en la ventana. No alerta: queda en la cola de eventos sin confirmar, donde el operador la confirma, la reclasifica
    #       o la descarta; un evento sin confirmar o descartado no se indexa, así que no aparece en las búsquedas.
    incident_alert_confidence: dict[str, float] = Field(default_factory=lambda: {"accident": 0.5})
    incident_strong_confidence: dict[str, float] = Field(default_factory=lambda: {"accident": 0.75})
    accident_review_enabled: bool = True
    incident_review_confidence: float = 0.25
    incident_review_min_hits: int = 2
    # Espera antes de abrir un pendiente de revisión: si en ese tiempo el incidente sale como alerta, no se abre.
    incident_review_hold_seconds: float = 1.5
    # Deduplicación de incidentes (fuego, choque) dentro de `event_dedup_window_seconds`:
    #   "global": cualquier incidente de la clase silencia a los siguientes de la cámara, vean donde lo vean (comportamiento histórico);
    #   "renovar": como "global", pero cada detección suprimida renueva la espera, así un choque que se sigue viendo con los
    #       vehículos detenidos da un solo evento y solo vuelve a disparar cuando deja de verse durante la ventana;
    #   "ubicacion": renueva la espera solo con detecciones cerca del evento (`incident_cluster_radius` tamaños de caja), así un
    #       incidente en otro lugar de la imagen sí dispara y un falso persistente no tapa un choque real.
    incident_dedup_mode: str = "renovar"
    # Posible altercado (solo revisión): dos personas cercanas y agitadas durante varios segundos seguidos.
    altercation_proximity_heights: float = 1.2  # distancia máxima entre centros, en alturas de persona
    altercation_min_speed: float = 0.8  # alturas de persona recorridas por segundo para considerar agitada a una persona
    altercation_tortuosity: float = 2.0  # cambios de dirección: camino recorrido sobre desplazamiento neto
    altercation_min_seconds: float = 3.0  # tiempo seguido que debe sostenerse la pareja agitada y cercana
    event_dedup_radius_px: float = 250.0
    event_dedup_window_seconds: float = 90.0
    clip_use_nvenc: bool = True

    s3_endpoint_url: str = "localhost:9000"
    s3_access_key: str = "monitoreo"
    s3_secret_key: str = "monitoreo123"
    s3_bucket_clips: str = "clips"
    s3_secure: bool = False
    s3_presigned_url_expiry_minutes: int = 60

    camera_blank_std_threshold: float = 6.0
    camera_covered_sharpness_threshold: float = 15.0
    camera_health_sustained_seconds: float = 5.0
    camera_signal_loss_sustained_seconds: float = 9.0

    vlm_backend: str = "qwen2-vl"
    vlm_model_name: str = "Qwen/Qwen2-VL-7B-Instruct-AWQ"
    vlm_device: str = "cuda:0"
    # "bnb4" carga el modelo sin cuantizar en 4 bits al arrancar (bitsandbytes). Sirve para el 7B donde AWQ no carga
    # (transformers 5 exige gptqmodel, que en Windows requiere compilar). Vacío: se carga el modelo tal cual.
    vlm_quantization: str = ""
    # "video" entrega los fotogramas al modelo como un solo video; "images" como imágenes sueltas.
    vlm_input_mode: str = "images"
    # Píxeles máximos por imagen o cuadro (en bloques de 28×28). Más píxeles distinguen mejor colores y vehículos lejanos, y cuestan memoria.
    vlm_max_pixels: int = 320 * 28 * 28
    vlm_max_frames: int = 8
    vlm_prompts_dir: Path = PROJECT_ROOT / "config" / "vlm_prompts"
    # El texto de carteles y pantallas que el modelo copie entre comillas o en mayúsculas sostenidas no se guarda en la
    # descripción: se reemplaza por una marca. Evita que una orden escrita en la escena llegue al registro.
    vlm_mask_scene_text: bool = True
    # Una descripción demasiado corta, o que califique o niegue el hecho que el sistema detectó, se reemplaza por una
    # descripción neutra armada con datos del propio sistema, en lugar de guardar lo que el modelo escribió.
    vlm_validate_output: bool = True
    vlm_appearance_exempt_reasons: list[str] = Field(default_factory=lambda: ["accident", "posible_choque"])
    vlm_worker_poll_interval_seconds: float = 1.0
    vlm_worker_stale_processing_minutes: float = 10.0
    stale_check_interval_seconds: float = 60.0  # cada cuánto un worker busca eventos trabados en PROCESSING
    heartbeat_interval_seconds: float = 5.0  # cada cuánto una instancia de visión renueva su latido
    heartbeat_stale_seconds: float = 30.0  # sin latido por este tiempo, la cámara figura como sin_analisis

    embedding_model_name: str = "intfloat/multilingual-e5-large"
    embedding_device: str = "cuda:0"
    hybrid_search_top_k_vector: int = 20
    hybrid_search_top_k_fts: int = 20
    hybrid_search_final_k: int = 8

    llm_server_base_url: str = "http://127.0.0.1:8080/v1"
    llm_model_name: str = "local-model"
    llm_temperature: float = 0.2
    chat_history_messages: int = 12
    api_host: str = "127.0.0.1"
    api_port: int = 8000

    api_key: str = ""
    stream_user: str = "operador"
    alert_poll_interval_seconds: float = 1.0


settings = Settings()
