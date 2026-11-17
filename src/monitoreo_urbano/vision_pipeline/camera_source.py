from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Iterator

import cv2
import numpy as np

from monitoreo_urbano.common.auth import with_stream_credentials
from monitoreo_urbano.common.config import settings
from monitoreo_urbano.common.logging import get_logger

logger = get_logger(__name__)


def _hw_acceleration(name: str) -> int:
    options = {
        "none": cv2.VIDEO_ACCELERATION_NONE,
        "any": cv2.VIDEO_ACCELERATION_ANY,
        "d3d11": cv2.VIDEO_ACCELERATION_D3D11,
        "vaapi": cv2.VIDEO_ACCELERATION_VAAPI,
        "mfx": cv2.VIDEO_ACCELERATION_MFX,
    }
    try:
        return options[name.strip().lower()]
    except KeyError:
        raise ValueError(f"capture_hw_acceleration inválido: {name!r}. Valores: {', '.join(options)}") from None


@dataclass
class Frame:
    image: np.ndarray
    timestamp: datetime
    frame_index: int
    # Trabajo ya hecho en el hilo de captura (ver frame_encoding.make_capture_hook): la imagen comprimida en
    # JPEG y las señales de salud (desvío estándar, nitidez). Son opcionales: sin ellas todo se calcula donde se usa.
    jpeg: bytes | None = None
    health: tuple[float, float] | None = None


class CaptureStopped(Exception):
    pass


class CameraSource:
    def __init__(self, camera_id: str, source: str, reconnect_delay_seconds: float = 3.0) -> None:
        self.camera_id = camera_id
        self.source = source
        self.reconnect_delay_seconds = reconnect_delay_seconds
        self._cap: cv2.VideoCapture | None = None
        self._stop_requested = threading.Event()

    def _video_capture(self) -> cv2.VideoCapture:
        url = with_stream_credentials(self.source)
        params: list[int] = []
        acceleration = _hw_acceleration(settings.capture_hw_acceleration)
        if acceleration != cv2.VIDEO_ACCELERATION_NONE:
            params += [cv2.CAP_PROP_HW_ACCELERATION, acceleration]
        if settings.capture_decoder_threads > 0:
            params += [cv2.CAP_PROP_N_THREADS, settings.capture_decoder_threads]
        if not params:
            return cv2.VideoCapture(url)
        return cv2.VideoCapture(url, cv2.CAP_FFMPEG, params)

    def _connect(self) -> cv2.VideoCapture | None:
        if self.source.isdigit():
            cap = cv2.VideoCapture(int(self.source), cv2.CAP_DSHOW)
        else:
            cap = self._video_capture()
        if cap.isOpened():
            return cap
        cap.release()
        return None

    def _open(self) -> cv2.VideoCapture:
        while True:
            if self._stop_requested.is_set():
                raise CaptureStopped(self.camera_id)
            cap = self._connect()
            if cap is not None:
                return cap
            logger.warning(
                "No se pudo abrir la fuente de video de %s (%s), reintentando en %.1fs",
                self.camera_id, self.source, self.reconnect_delay_seconds,
            )
            time.sleep(self.reconnect_delay_seconds)

    def request_stop(self) -> None:
        self._stop_requested.set()

    @property
    def fps(self) -> float:
        if self._cap is None:
            self._cap = self._open()
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        return fps if fps and fps > 0 else 25.0

    def frames(self, stride: int = 1) -> Iterator[Frame | None]:
        """Entrega fotogramas y `None` por cada intento fallido de reconexión.

        Con `stride` mayor a 1 solo se entrega 1 de cada `stride` fotogramas de la cámara. Los demás se
        decodifican con `grab()` y se descartan sin convertirlos, que es bastante más barato que `read()`.
        `frame_index` cuenta los fotogramas entregados, no los de la cámara. Tras una reconexión, el primer
        fotograma disponible se entrega siempre.
        """
        stride = max(1, int(stride))
        frame_index = 0
        camera_index = 0
        while True:
            if self._stop_requested.is_set():
                return

            if self._cap is None:
                self._cap = self._connect()
                if self._cap is None:
                    logger.warning(
                        "No se pudo reconectar %s, reintentando en %.1fs", self.camera_id, self.reconnect_delay_seconds
                    )
                    time.sleep(self.reconnect_delay_seconds)
                    yield None
                    continue
                camera_index = 0

            if camera_index % stride == 0:
                ok, image = self._cap.read()
            else:
                ok, image = self._cap.grab(), None
            camera_index += 1

            if not ok:
                logger.warning("Corte de señal en %s, reintentando en %.1fs", self.camera_id, self.reconnect_delay_seconds)
                self._cap.release()
                self._cap = None
                time.sleep(self.reconnect_delay_seconds)
                yield None
                continue

            if image is None:
                continue

            yield Frame(image=image, timestamp=datetime.utcnow(), frame_index=frame_index)
            frame_index += 1

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
