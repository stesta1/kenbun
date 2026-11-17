from __future__ import annotations

import queue
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from monitoreo_urbano.common.logging import get_logger
from monitoreo_urbano.common.schemas import CameraInfo
from monitoreo_urbano.vision_pipeline.camera_source import CameraSource, CaptureStopped, Frame

logger = get_logger(__name__)

_RATE_WINDOW_SECONDS = 2.0


class StreamClosed(Exception):
    pass


class FrameSource(Protocol):
    @property
    def fps(self) -> float: ...

    def frames(self, stride: int = 1): ...

    def request_stop(self) -> None: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class StreamStats:
    camera_id: str
    running: bool
    native_fps: float | None
    effective_fps: float | None
    stride: int | None
    frames_delivered: int
    frames_dropped: int
    signal_losses: int
    queue_depth: int
    capture_fps: float
    last_frame_age_seconds: float | None


class _Mailbox:
    """Cola FIFO acotada. Si se llena, descarta el fotograma más viejo; los `None` (corte de señal) no se descartan."""

    def __init__(self, maxlen: int) -> None:
        self._maxlen = max(1, maxlen)
        self._items: deque[Frame | None] = deque()
        self._condition = threading.Condition()
        self._closed = False

    def put(self, item: Frame | None) -> int:
        with self._condition:
            dropped = 0
            if len(self._items) >= self._maxlen:
                for index, existing in enumerate(self._items):
                    if existing is not None:
                        del self._items[index]
                        dropped = 1
                        break
            self._items.append(item)
            self._condition.notify()
            return dropped

    def get(self, timeout: float | None) -> Frame | None:
        with self._condition:
            self._condition.wait_for(lambda: self._items or self._closed, timeout=timeout)
            if self._items:
                return self._items.popleft()
            if self._closed:
                raise StreamClosed
            raise queue.Empty

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    def __len__(self) -> int:
        with self._condition:
            return len(self._items)


class CameraStream:
    """Captura una cámara en un hilo propio y deja los fotogramas en una cola acotada.

    - El arranque no se bloquea: `start()` vuelve enseguida aunque la cámara esté caída. La espera por la
      primera conexión ocurre dentro del hilo; `wait_ready()` indica cuándo se conoce la frecuencia.
    - Entrega 1 de cada `stride` fotogramas, con `stride = round(fps_nativos / analysis_fps)`. La frecuencia
      efectiva (`effective_fps`) es la que debe usar el seguimiento y las reglas.
    - Si el consumidor se atrasa, la cola descarta el fotograma más viejo y lo cuenta en `frames_dropped`.
    - Los cortes de señal llegan como `None`, igual que en `CameraSource.frames()`.
    - `frame_hook` se ejecuta sobre cada fotograma en el hilo de captura, antes de dejarlo en la cola: permite
      hacer ahí el trabajo que no depende del estado de la cámara (comprimir, calcular señales de salud) en
      paralelo entre cámaras, y no en el hilo del detector.
    """

    def __init__(
        self,
        camera_id: str,
        source: FrameSource,
        *,
        analysis_fps: float | None = None,
        queue_size: int = 4,
        frame_hook: Callable[[Frame], None] | None = None,
    ) -> None:
        self.camera_id = camera_id
        self._source = source
        self._frame_hook = frame_hook
        self._analysis_fps = analysis_fps
        self._mailbox = _Mailbox(queue_size)

        self._native_fps: float | None = None
        self._stride: int | None = None
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"capture-{camera_id}", daemon=True)
        self.error: BaseException | None = None
        self._wake: threading.Event | None = None

        self._lock = threading.Lock()
        self._delivered = 0
        self._dropped = 0
        self._signal_losses = 0
        self._recent: deque[float] = deque()
        self._last_frame_at: float | None = None

    @classmethod
    def from_camera(
        cls,
        camera: CameraInfo,
        *,
        analysis_fps: float | None = None,
        queue_size: int = 4,
        frame_hook: Callable[[Frame], None] | None = None,
    ) -> CameraStream:
        return cls(
            camera.camera_id,
            CameraSource(camera.camera_id, camera.source),
            analysis_fps=analysis_fps,
            queue_size=queue_size,
            frame_hook=frame_hook,
        )

    def set_wake_event(self, event: threading.Event | None) -> None:
        self._wake = event


    def start(self) -> CameraStream:
        self._thread.start()
        return self

    def request_stop(self) -> None:
        self._stop.set()
        self._source.request_stop()

    def stop(self, timeout: float = 5.0) -> None:
        self.request_stop()
        if self._thread.ident is not None:
            self._thread.join(timeout)
        self._mailbox.close()

    @property
    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def wait_ready(self, timeout: float | None = None) -> bool:
        return self._ready.wait(timeout)


    @property
    def reconnect_delay_seconds(self) -> float:
        return getattr(self._source, "reconnect_delay_seconds", 3.0)

    @property
    def native_fps(self) -> float | None:
        return self._native_fps

    @property
    def stride(self) -> int | None:
        return self._stride

    @property
    def effective_fps(self) -> float | None:
        if self._native_fps is None or self._stride is None:
            return None
        return self._native_fps / self._stride

    def get(self, timeout: float | None = None) -> Frame | None:
        return self._mailbox.get(timeout)

    def stats(self) -> StreamStats:
        now = time.monotonic()
        with self._lock:
            while self._recent and now - self._recent[0] > _RATE_WINDOW_SECONDS:
                self._recent.popleft()
            capture_fps = len(self._recent) / _RATE_WINDOW_SECONDS
            age = None if self._last_frame_at is None else now - self._last_frame_at
            return StreamStats(
                camera_id=self.camera_id,
                running=self._thread.is_alive(),
                native_fps=self._native_fps,
                effective_fps=self.effective_fps,
                stride=self._stride,
                frames_delivered=self._delivered,
                frames_dropped=self._dropped,
                signal_losses=self._signal_losses,
                queue_depth=len(self._mailbox),
                capture_fps=capture_fps,
                last_frame_age_seconds=age,
            )


    def _run(self) -> None:
        try:
            native_fps = self._source.fps
            if self._analysis_fps:
                stride = max(1, round(native_fps / self._analysis_fps))
            else:
                stride = 1
            self._native_fps = native_fps
            self._stride = stride
            self._ready.set()
            logger.info(
                "Captura de %s: %.1f fps nativos, 1 de cada %d (%.1f fps de análisis)",
                self.camera_id, native_fps, stride, native_fps / stride,
            )

            for item in self._source.frames(stride):
                if self._stop.is_set():
                    break
                self._deliver(item)
        except CaptureStopped:
            pass
        except Exception as error:  # noqa: BLE001
            self.error = error
            logger.exception("La captura de %s terminó por un error", self.camera_id)
        finally:
            self._source.close()
            self._mailbox.close()
            self._notify()

    def _notify(self) -> None:
        wake = self._wake
        if wake is not None:
            wake.set()

    def _deliver(self, item: Frame | None) -> None:
        if item is not None and self._frame_hook is not None:
            try:
                self._frame_hook(item)
            except Exception:  # noqa: BLE001
                logger.exception("Falló el trabajo por fotograma de la captura de %s", self.camera_id)
        dropped = self._mailbox.put(item)
        self._notify()
        now = time.monotonic()
        with self._lock:
            self._dropped += dropped
            if item is None:
                self._signal_losses += 1
            else:
                self._delivered += 1
                self._recent.append(now)
                self._last_frame_at = now
