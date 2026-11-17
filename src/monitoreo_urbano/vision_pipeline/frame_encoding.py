from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

import cv2
import numpy as np

from monitoreo_urbano.common.logging import get_logger
from monitoreo_urbano.vision_pipeline.camera_source import Frame

logger = get_logger(__name__)

_SHARPNESS_WIDTH = 320


@dataclass
class EncodedFrame:

    jpeg: bytes
    timestamp: datetime
    frame_index: int


def compute_health_signals(image: np.ndarray) -> tuple[float, float]:
    """Desvío estándar del brillo y nitidez (varianza del Laplaciano) de un fotograma.

    Son las dos señales con que `CameraHealthMonitor` decide si la lente está tapada. Son funciones puras del
    fotograma, así que pueden calcularse en el hilo de captura y no en el del detector.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    std = float(gray.std())
    if gray.shape[1] > _SHARPNESS_WIDTH:
        height = round(gray.shape[0] * _SHARPNESS_WIDTH / gray.shape[1])
        gray = cv2.resize(gray, (_SHARPNESS_WIDTH, height), interpolation=cv2.INTER_AREA)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return std, sharpness


def encode_jpeg(image: np.ndarray, quality: int) -> bytes:
    ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("No se pudo comprimir el fotograma a JPEG")
    return buffer.tobytes()


def decode_image(frame: Frame | EncodedFrame) -> np.ndarray:
    if isinstance(frame, EncodedFrame):
        image = cv2.imdecode(np.frombuffer(frame.jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"No se pudo descomprimir el fotograma {frame.frame_index}")
        return image
    return frame.image


def stored_form(frame: Frame) -> Frame | EncodedFrame:
    if frame.jpeg is None:
        return frame
    return EncodedFrame(jpeg=frame.jpeg, timestamp=frame.timestamp, frame_index=frame.frame_index)


def make_capture_hook(jpeg_quality: int, compute_health: bool = True) -> Callable[[Frame], None]:
    """Trabajo por fotograma que se hace en el hilo de captura, antes de dejarlo en la cola.

    Comprime el fotograma (si `jpeg_quality` > 0) y calcula las señales de salud. Corre en paralelo en los
    hilos de cada cámara; el hilo del detector recibe el resultado ya hecho.
    """

    def hook(frame: Frame) -> None:
        if jpeg_quality > 0:
            frame.jpeg = encode_jpeg(frame.image, jpeg_quality)
        if compute_health:
            frame.health = compute_health_signals(frame.image)

    return hook
