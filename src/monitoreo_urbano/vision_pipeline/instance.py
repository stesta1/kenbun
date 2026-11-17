from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from monitoreo_urbano.common.config import settings
from monitoreo_urbano.common.logging import get_logger
from monitoreo_urbano.common.runtime_config import RuntimeThresholds
from monitoreo_urbano.common.schemas import ALERTABLE_REASONS, HEALTH_OK, CameraInfo, EventRecord, TrackedObject
from monitoreo_urbano.vision_pipeline.batch_detector import BatchDetector, DetectionResult, EngineStats
from monitoreo_urbano.vision_pipeline.camera_pipeline import CameraPipeline, OpenClip
from monitoreo_urbano.vision_pipeline.camera_source import Frame
from monitoreo_urbano.vision_pipeline.camera_stream import CameraStream
from monitoreo_urbano.vision_pipeline.clip_sink import ClipSink, ClipSinkStats
from monitoreo_urbano.vision_pipeline.clip_writer import write_clip
from monitoreo_urbano.vision_pipeline.detector import Detector
from monitoreo_urbano.vision_pipeline.frame_encoding import make_capture_hook
from monitoreo_urbano.vision_pipeline.instance_store import InstanceStore
from monitoreo_urbano.vision_pipeline.visualizer import draw_overlay, publish_preview

logger = get_logger(__name__)

_CONTROL_TICK_SECONDS = 0.5
_THRESHOLDS_REFRESH_SECONDS = 2.0
_QUEUE_DEPTH_REFRESH_SECONDS = 2.0

ClipWriter = Callable[[str, str, list, float], str]


@dataclass
class _Camera:
    info: CameraInfo
    stream: CameraStream
    pipeline: CameraPipeline | None = None
    fps: float | None = None
    last_preview: float = 0.0
    closed: bool = False


@dataclass(frozen=True)
class CameraStatus:
    camera_id: str
    connected: bool
    closed: bool
    effective_fps: float | None
    frames_delivered: int
    frames_dropped: int
    signal_losses: int


@dataclass(frozen=True)
class InstanceStats:
    engine: EngineStats
    clips: ClipSinkStats
    cameras: list[CameraStatus]


class VisionInstance:
    """Un proceso de visión: M cámaras, un detector compartido y un grupo de hilos para escribir clips.

    Armado:  captura (un hilo por cámara) → motor de lotes (un hilo, un modelo) → una `CameraPipeline` por
    cámara (seguimiento, reglas, buffer, salud) → escritura de clips en hilos propios.

    - El arranque no se bloquea por una cámara caída: cada cámara se incorpora al motor cuando se conecta por
      primera vez, y hasta entonces figura como `corte_de_señal`.
    - Una cámara cuyo flujo termina por un error se marca `corte_de_señal` y el resto sigue; la instancia se
      detiene sola cuando ya no queda ninguna.
    - La escritura de clips no frena el análisis: se hace aparte.
    """

    def __init__(
        self,
        cameras: list[CameraInfo],
        *,
        detector: Detector,
        store: InstanceStore,
        thresholds: RuntimeThresholds | None = None,
        analysis_fps: float | None = None,
        clip_writer: ClipWriter = write_clip,
        stream_factory: Callable[[CameraInfo], CameraStream] | None = None,
        clip_workers: int | None = None,
        name: str = "vision",
    ) -> None:
        if not cameras:
            raise ValueError("Una instancia necesita al menos una cámara")
        ids = [camera.camera_id for camera in cameras]
        if len(set(ids)) != len(ids):
            raise ValueError(f"Hay cámaras repetidas: {ids}")

        self.name = name
        self._store = store
        self._thresholds = thresholds or RuntimeThresholds()
        self._clip_writer = clip_writer

        effective_analysis_fps = analysis_fps if analysis_fps else (settings.analysis_fps or None)
        hook = make_capture_hook(settings.clip_buffer_jpeg_quality)
        factory = stream_factory or (
            lambda camera: CameraStream.from_camera(
                camera,
                analysis_fps=effective_analysis_fps,
                queue_size=settings.camera_queue_size,
                frame_hook=hook,
            )
        )
        self._cameras: dict[str, _Camera] = {camera.camera_id: _Camera(camera, factory(camera)) for camera in cameras}

        self._sink = ClipSink(clip_workers if clip_workers is not None else settings.clip_writer_workers)
        self._hub = BatchDetector(
            detector,
            self._on_result,
            on_signal_loss=self._on_signal_loss,
            on_stream_closed=self._on_stream_closed,
        )
        self._stop = threading.Event()
        self._control = threading.Thread(target=self._control_loop, name="instance-control", daemon=True)
        self._queue_depth = 0


    def start(self) -> VisionInstance:
        for camera in self._cameras.values():
            self._store.register_camera(camera.info)
        self._store.refresh_thresholds(self._thresholds)
        self._beat()

        self._sink.start()
        for camera in self._cameras.values():
            camera.stream.start()
        self._hub.start()
        self._control.start()
        logger.info("Instancia de visión con %d cámaras: %s", len(self._cameras), ", ".join(self._cameras))
        return self

    def request_stop(self) -> None:
        self._stop.set()

    def wait(self, timeout: float | None = None) -> bool:
        return self._stop.wait(timeout)

    def stop(self, clip_timeout: float = 30.0) -> None:
        self._stop.set()
        if self._control.ident is not None:
            self._control.join(5)
        for camera in self._cameras.values():
            camera.stream.request_stop()
        for camera in self._cameras.values():
            camera.stream.stop(timeout=4)
        self._hub.stop()
        if not self._sink.stop(timeout=clip_timeout):
            logger.warning("Se detuvo la instancia con clips sin terminar de escribir")
        self._store.close()


    def _bootstrap_ready_cameras(self) -> None:
        for camera in self._cameras.values():
            if camera.pipeline is not None or camera.closed:
                continue
            if camera.stream.wait_ready(0):
                try:
                    self._bootstrap(camera)
                except Exception:  # noqa: BLE001
                    logger.exception("No se pudo incorporar la cámara %s", camera.info.camera_id)
                    self._mark_closed(camera)
            elif not camera.stream.is_alive and camera.stream.error is not None:
                # La captura murió antes de conectarse: el motor nunca la vio, así que se la da de baja acá.
                logger.error("La captura de %s terminó antes de conectarse: %r", camera.info.camera_id, camera.stream.error)
                self._mark_closed(camera)

    def _mark_closed(self, camera: _Camera) -> None:
        camera.closed = True
        self._report_health(camera.info.camera_id, "corte_de_señal")
        if all(c.closed for c in self._cameras.values()):
            logger.error("Todas las cámaras de la instancia terminaron; se detiene")
            self._stop.set()

    def _bootstrap(self, camera: _Camera) -> None:
        camera_id = camera.info.camera_id
        fps = camera.stream.effective_fps
        if fps is None:
            return
        camera.fps = fps
        camera.pipeline = CameraPipeline.build(
            camera_id,
            fps=fps,
            reconnect_delay_seconds=camera.stream.reconnect_delay_seconds,
            thresholds=self._thresholds,
            on_clip=lambda clip, camera=camera: self._submit_clip(camera, clip),
            on_open=lambda clip, camera=camera: self._submit_open(camera, clip),
            on_health=lambda status, camera_id=camera_id: self._report_health(camera_id, status),
        )
        self._report_health(camera_id, HEALTH_OK)
        logger.info("Cámara %s conectada (%.1f fps de análisis)", camera_id, fps)
        self._hub.register(camera_id, camera.stream)


    def _on_result(self, result: DetectionResult) -> None:
        camera = self._cameras[result.camera_id]
        pipeline = camera.pipeline
        if pipeline is None:
            return
        pipeline.observe(result.frame)
        tracked = pipeline.update(result.frame, result.detections)
        if settings.preview_enabled:
            self._maybe_publish_preview(camera, result.frame, tracked)

    def _on_signal_loss(self, camera_id: str) -> None:
        pipeline = self._cameras[camera_id].pipeline
        if pipeline is not None:
            pipeline.on_signal_loss()

    def _on_stream_closed(self, camera_id: str) -> None:
        camera = self._cameras[camera_id]
        error = camera.stream.error
        if error is not None:
            logger.error("La captura de %s terminó por un error: %r", camera_id, error)
        self._mark_closed(camera)


    def _report_health(self, camera_id: str, status: str) -> None:
        try:
            self._store.set_health(camera_id, status)
        except Exception:  # noqa: BLE001
            logger.exception("No se pudo registrar la salud de %s (%s)", camera_id, status)

    def _submit_clip(self, camera: _Camera, clip: OpenClip) -> None:
        self._sink.submit(lambda: self._finalize_clip(camera, clip))

    def _submit_open(self, camera: _Camera, clip: OpenClip) -> None:
        """Registra el evento al dispararse: es lo que hace llegar el aviso al operador sin esperar al cierre del clip."""
        self._sink.submit(lambda: self._open_event(camera, clip))

    def _open_event(self, camera: _Camera, clip: OpenClip) -> None:
        camera_id = camera.info.camera_id
        try:
            self._store.open_event(self._event_record(camera, clip, f"{camera_id}/{clip.event_id}.mp4"))
            logger.info("Evento registrado al dispararse: %s", clip.event_id)
        except Exception:  # noqa: BLE001
            logger.exception("No se pudo registrar el evento %s al dispararse; se registrará al cerrar el clip", clip.event_id)
            return
        if clip.trigger_reason == "accident" and clip.trigger_reason in ALERTABLE_REASONS:
            # Si antes hubo un pendiente de revisión del mismo choque, la alerta lo absorbe: su clip ya incluye esos segundos.
            try:
                absorbed = self._store.absorb_review_events(camera_id, clip.event_id, settings.frame_buffer_seconds)
                if absorbed:
                    logger.info("La alerta %s absorbió %d pendiente(s) de revisión de %s", clip.event_id, absorbed, camera_id)
            except Exception:  # noqa: BLE001
                logger.exception("No se pudieron absorber los pendientes de revisión de %s", camera_id)

    @staticmethod
    def _event_record(camera: _Camera, clip: OpenClip, clip_path: str) -> EventRecord:
        return EventRecord(
            event_id=clip.event_id,
            camera_id=camera.info.camera_id,
            camera_label=camera.info.label,
            track_id=clip.track_id,
            object_classes=sorted(clip.object_classes),
            trigger_reason=clip.trigger_reason,
            start_ts=clip.frames[0].timestamp,
            end_ts=clip.frames[-1].timestamp,
            clip_path=clip_path,
            confidence=sum(clip.confidences) / len(clip.confidences) if clip.confidences else None,
            created_at=datetime.utcnow(),
        )

    def _finalize_clip(self, camera: _Camera, clip: OpenClip) -> None:
        camera_id = camera.info.camera_id
        event_id = clip.event_id
        clip_path = self._clip_writer(camera_id, event_id, clip.frames, camera.fps or 0.0)
        event = self._event_record(camera, clip, clip_path)
        if not self._store.complete_event(event):
            self._store.insert_event(event)
        logger.info("Clip escrito y encolado: %s (%s)", clip_path, event_id)


    def _maybe_publish_preview(self, camera: _Camera, frame: Frame, tracked: list[TrackedObject]) -> None:
        now = time.monotonic()
        if now - camera.last_preview < settings.preview_interval_seconds or camera.pipeline is None:
            return
        camera.last_preview = now
        annotated = draw_overlay(
            frame.image, tracked, camera.pipeline.recording_track_ids, camera.stream.stats().capture_fps, self._queue_depth
        )
        publish_preview(camera.info.camera_id, annotated)


    def _beat(self) -> None:
        try:
            self._store.heartbeat(list(self._cameras), self.name)
        except Exception:  # noqa: BLE001
            logger.exception("No se pudo renovar el latido de la instancia")

    def _control_loop(self) -> None:
        last_beat = time.monotonic()
        last_thresholds = time.monotonic()
        last_depth = 0.0
        last_stats = time.monotonic()
        while not self._stop.wait(_CONTROL_TICK_SECONDS):
            self._bootstrap_ready_cameras()
            now = time.monotonic()

            if now - last_beat >= settings.heartbeat_interval_seconds:
                last_beat = now
                self._beat()

            if now - last_thresholds >= _THRESHOLDS_REFRESH_SECONDS:
                last_thresholds = now
                try:
                    self._store.refresh_thresholds(self._thresholds)
                except Exception:  # noqa: BLE001
                    logger.exception("No se pudieron actualizar los umbrales")

            if settings.preview_enabled and now - last_depth >= _QUEUE_DEPTH_REFRESH_SECONDS:
                last_depth = now
                try:
                    self._queue_depth = self._store.pending_description_count()
                except Exception:  # noqa: BLE001
                    logger.exception("No se pudo consultar la cola de descripciones")

            if now - last_stats >= settings.stats_log_interval_seconds:
                last_stats = now
                self._log_stats()


    def stats(self) -> InstanceStats:
        cameras = []
        for camera in self._cameras.values():
            stream_stats = camera.stream.stats()
            cameras.append(
                CameraStatus(
                    camera_id=camera.info.camera_id,
                    connected=camera.pipeline is not None and not camera.closed,
                    closed=camera.closed,
                    effective_fps=stream_stats.effective_fps,
                    frames_delivered=stream_stats.frames_delivered,
                    frames_dropped=stream_stats.frames_dropped,
                    signal_losses=stream_stats.signal_losses,
                )
            )
        return InstanceStats(engine=self._hub.stats(), clips=self._sink.stats(), cameras=cameras)

    def _log_stats(self) -> None:
        stats = self.stats()
        engine = stats.engine
        connected = sum(1 for c in stats.cameras if c.connected)
        dropped = sum(c.frames_dropped for c in stats.cameras)
        logger.info(
            "Instancia: %d/%d cámaras conectadas | %.0f img/s | lote medio %.1f | inferencia %s ms (p95) | "
            "demora %s ms (p95) | descartados %d | clips pendientes %d, fallidos %d",
            connected, len(stats.cameras), engine.images_per_second, engine.mean_batch_size,
            engine.inference_ms_p95, engine.latency_ms_p95, dropped, stats.clips.pending, stats.clips.failed,
        )
