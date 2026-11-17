from __future__ import annotations

import queue
import statistics
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from monitoreo_urbano.common.config import settings
from monitoreo_urbano.common.logging import get_logger
from monitoreo_urbano.common.schemas import Detection
from monitoreo_urbano.vision_pipeline.camera_source import Frame
from monitoreo_urbano.vision_pipeline.camera_stream import StreamClosed
from monitoreo_urbano.vision_pipeline.detector import Detector

logger = get_logger(__name__)

_RATE_WINDOW_SECONDS = 5.0
_IDLE_WAIT_SECONDS = 0.05
_ERROR_BACKOFF_SECONDS = 0.5


class FrameStream(Protocol):

    def get(self, timeout: float | None = None) -> Frame | None: ...

    def set_wake_event(self, event: threading.Event | None) -> None: ...


@dataclass(frozen=True)
class DetectionResult:
    camera_id: str
    frame: Frame
    detections: list[Detection]


@dataclass(frozen=True)
class EngineStats:
    running: bool
    cameras: int
    batches: int
    images: int
    errors: int
    callback_errors: int
    mean_batch_size: float
    max_batch_size: int
    images_per_second: float
    inference_ms_p50: float | None
    inference_ms_p95: float | None
    latency_ms_p50: float | None
    latency_ms_p95: float | None


class BatchDetector:
    """Un hilo, un modelo, varias cámaras.

    Recoge fotogramas de las colas de las cámaras registradas, **uno por cámara en cada lote** y en turnos
    rotativos para que ninguna se quede sin servicio, llama al detector una sola vez por lote y entrega cada
    resultado a su cámara por `on_result`. Es el único hilo que toca el modelo (el predictor de Ultralytics
    no admite uso concurrente).

    - No espera a llenar el lote: toma lo que hay y, si hay al menos un fotograma, espera hasta
      `fill_wait_seconds` por más. Bajo carga el lote crece solo, porque mientras el detector trabaja las
      cámaras siguen dejando fotogramas en sus colas.
    - Los cortes de señal (`None`) no pasan por el detector: se avisan al instante por `on_signal_loss`.
    - Si el detector falla, el motor lo registra y reintenta; tras `max_consecutive_errors` fallos seguidos
      se detiene y deja el motivo en `error`, para que un supervisor lo reinicie.
    """

    def __init__(
        self,
        detector: Detector,
        on_result: Callable[[DetectionResult], None],
        *,
        on_signal_loss: Callable[[str], None] | None = None,
        on_stream_closed: Callable[[str], None] | None = None,
        max_batch: int | None = None,
        fill_wait_seconds: float | None = None,
        max_consecutive_errors: int = 5,
    ) -> None:
        self._detector = detector
        self._on_result = on_result
        self._on_signal_loss = on_signal_loss
        self._on_stream_closed = on_stream_closed
        self._max_batch = max(1, max_batch if max_batch is not None else settings.detector_max_batch)
        self._fill_wait = (
            fill_wait_seconds if fill_wait_seconds is not None else settings.detector_batch_fill_wait_seconds
        )
        self._max_consecutive_errors = max_consecutive_errors

        self._streams: dict[str, FrameStream] = {}
        self._order: list[str] = []
        self._cursor = 0
        self._streams_lock = threading.Lock()

        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="batch-detector", daemon=True)
        self.error: BaseException | None = None

        self._stats_lock = threading.Lock()
        self._batches = 0
        self._images = 0
        self._errors = 0
        self._callback_errors = 0
        self._max_batch_seen = 0
        self._recent: deque[tuple[float, int, float]] = deque()  # (instante, tamaño del lote, ms de inferencia)
        self._inference_ms: deque[float] = deque(maxlen=1000)
        self._latency_ms: deque[float] = deque(maxlen=4000)
        self._batch_sizes: deque[int] = deque(maxlen=1000)


    def register(self, camera_id: str, stream: FrameStream) -> None:
        with self._streams_lock:
            if camera_id in self._streams:
                raise ValueError(f"La cámara {camera_id} ya está registrada")
            self._streams[camera_id] = stream
            self._order.append(camera_id)
        stream.set_wake_event(self._wake)
        self._wake.set()

    def unregister(self, camera_id: str) -> None:
        with self._streams_lock:
            stream = self._streams.pop(camera_id, None)
            if camera_id in self._order:
                self._order.remove(camera_id)
        if stream is not None:
            stream.set_wake_event(None)


    def start(self) -> BatchDetector:
        self._thread.start()
        return self

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread.ident is not None:
            self._thread.join(timeout)

    @property
    def is_alive(self) -> bool:
        return self._thread.is_alive()


    def _run(self) -> None:
        consecutive_errors = 0
        try:
            while not self._stop.is_set():
                batch = self._collect()
                if not batch:
                    self._wake.wait(_IDLE_WAIT_SECONDS)
                    continue

                started = time.perf_counter()
                try:
                    results = self._detector.detect_batch([frame.image for _, frame in batch])
                    if len(results) != len(batch):
                        raise RuntimeError(f"El detector devolvió {len(results)} resultados para {len(batch)} imágenes")
                except Exception as error:  # noqa: BLE001
                    consecutive_errors += 1
                    with self._stats_lock:
                        self._errors += 1
                    logger.exception("Falló la detección de un lote de %d fotogramas", len(batch))
                    if consecutive_errors >= self._max_consecutive_errors:
                        self.error = error
                        logger.error("El motor de lotes se detiene tras %d fallos seguidos", consecutive_errors)
                        return
                    time.sleep(_ERROR_BACKOFF_SECONDS)
                    continue

                consecutive_errors = 0
                inference_ms = (time.perf_counter() - started) * 1000
                self._record_batch(len(batch), inference_ms)
                self._deliver(batch, results)
        finally:
            self._stop.set()

    def _collect(self) -> list[tuple[str, Frame]]:
        batch: list[tuple[str, Frame]] = []
        taken: set[str] = set()
        deadline: float | None = None
        while True:
            self._wake.clear()
            self._sweep(batch, taken)
            if len(batch) >= self._max_batch or not batch or self._stop.is_set():
                return batch
            now = time.monotonic()
            if deadline is None:
                deadline = now + self._fill_wait
            remaining = deadline - now
            if remaining <= 0:
                return batch
            self._wake.wait(remaining)

    def _sweep(self, batch: list[tuple[str, Frame]], taken: set[str]) -> None:
        with self._streams_lock:
            order = list(self._order)
            streams = dict(self._streams)
        if not order:
            return

        total = len(order)
        start = self._cursor % total
        served_until = start
        for offset in range(total):
            if len(batch) >= self._max_batch:
                break
            camera_id = order[(start + offset) % total]
            served_until = (start + offset + 1) % total
            if camera_id in taken:
                continue
            try:
                item = streams[camera_id].get(timeout=0)
            except queue.Empty:
                continue
            except StreamClosed:
                self._handle_closed(camera_id)
                continue

            if item is None:
                self._call(self._on_signal_loss, camera_id)
                continue
            batch.append((camera_id, item))
            taken.add(camera_id)
        self._cursor = served_until

    def _handle_closed(self, camera_id: str) -> None:
        logger.warning("El flujo de %s terminó; se deja de atender", camera_id)
        self.unregister(camera_id)
        self._call(self._on_stream_closed, camera_id)

    def _call(self, callback: Callable | None, *args) -> None:
        if callback is None:
            return
        try:
            callback(*args)
        except Exception:  # noqa: BLE001
            with self._stats_lock:
                self._callback_errors += 1
            logger.exception("Falló un callback del motor de lotes")

    def _deliver(self, batch: list[tuple[str, Frame]], results: list[list[Detection]]) -> None:
        now = datetime.utcnow()
        latencies = []
        for (camera_id, frame), detections in zip(batch, results):
            latencies.append((now - frame.timestamp).total_seconds() * 1000)
            self._call(self._on_result, DetectionResult(camera_id=camera_id, frame=frame, detections=detections))
        with self._stats_lock:
            self._latency_ms.extend(latencies)


    def _record_batch(self, size: int, inference_ms: float) -> None:
        now = time.monotonic()
        with self._stats_lock:
            self._batches += 1
            self._images += size
            self._max_batch_seen = max(self._max_batch_seen, size)
            self._recent.append((now, size, inference_ms))
            while self._recent and now - self._recent[0][0] > _RATE_WINDOW_SECONDS:
                self._recent.popleft()
            self._inference_ms.append(inference_ms)
            self._batch_sizes.append(size)

    def stats(self) -> EngineStats:
        now = time.monotonic()
        with self._stats_lock:
            while self._recent and now - self._recent[0][0] > _RATE_WINDOW_SECONDS:
                self._recent.popleft()
            window_images = sum(size for _, size, _ in self._recent)
            inference = sorted(self._inference_ms)
            latency = sorted(self._latency_ms)
            return EngineStats(
                running=self._thread.is_alive() and not self._stop.is_set(),
                cameras=len(self._order),
                batches=self._batches,
                images=self._images,
                errors=self._errors,
                callback_errors=self._callback_errors,
                mean_batch_size=round(statistics.mean(self._batch_sizes), 2) if self._batch_sizes else 0.0,
                max_batch_size=self._max_batch_seen,
                images_per_second=window_images / _RATE_WINDOW_SECONDS,
                inference_ms_p50=_percentile(inference, 0.50),
                inference_ms_p95=_percentile(inference, 0.95),
                latency_ms_p50=_percentile(latency, 0.50),
                latency_ms_p95=_percentile(latency, 0.95),
            )


def _percentile(sorted_values: list[float], fraction: float) -> float | None:
    if not sorted_values:
        return None
    index = min(len(sorted_values) - 1, int(round(fraction * (len(sorted_values) - 1))))
    return round(sorted_values[index], 1)
