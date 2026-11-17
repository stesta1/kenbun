from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from monitoreo_urbano.common.config import settings
from monitoreo_urbano.common.logging import get_logger
from monitoreo_urbano.common.runtime_config import RuntimeThresholds
from monitoreo_urbano.common.schemas import BoundingBox, Detection

logger = get_logger(__name__)


class Detector(ABC):
    @abstractmethod
    def detect(self, image: np.ndarray) -> list[Detection]:
        raise NotImplementedError

    def detect_batch(self, images: list[np.ndarray]) -> list[list[Detection]]:
        return [self.detect(image) for image in images]


class YOLODetector(Detector):
    def __init__(
        self,
        weights_path: str | None = None,
        device: str | None = None,
        confidence_threshold: float | None = None,
        classes: list[str] | None = None,
        thresholds: RuntimeThresholds | None = None,
    ) -> None:
        from ultralytics import YOLO

        self.weights_path = weights_path or str(settings.yolo_weights_path)
        self.device = device or settings.yolo_device
        self.thresholds = thresholds
        self._static_confidence_threshold = confidence_threshold or settings.yolo_confidence_threshold
        self.classes = classes or settings.detection_classes

        logger.info("Cargando pesos YOLO desde %s (device=%s)", self.weights_path, self.device)
        self._model = YOLO(self.weights_path)
        self._class_filter_ids = self._resolve_class_ids()

    @property
    def confidence_threshold(self) -> float:
        return self.thresholds.yolo_confidence_threshold if self.thresholds else self._static_confidence_threshold

    def _resolve_class_ids(self) -> list[int] | None:
        names: dict[int, str] = self._model.names  # type: ignore[attr-defined]
        wanted = set(self.classes)
        ids = [idx for idx, name in names.items() if name in wanted]
        return ids or None

    def _class_thresholds(self) -> dict[str, float]:
        general = self.confidence_threshold
        return {name: settings.yolo_confidence_by_class.get(name, general) for name in self._model.names.values()}

    def _predict(self, source):
        options: dict = {}
        if settings.yolo_half:
            options["half"] = True
        # Ultralytics filtra por un único umbral: se usa el más bajo y después se aplica el de cada clase.
        return self._model.predict(
            source=source,
            device=self.device,
            conf=min(self._class_thresholds().values()),
            classes=self._class_filter_ids,
            imgsz=settings.yolo_imgsz,
            verbose=False,
            **options,
        )

    def _to_detections(self, result) -> list[Detection]:
        names = result.names
        thresholds = self._class_thresholds()
        detections: list[Detection] = []
        for box in result.boxes:
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            if conf < thresholds.get(names[cls_id], 0.0):
                continue
            detections.append(
                Detection(
                    class_name=names[cls_id],
                    confidence=conf,
                    bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                )
            )
        return detections

    def detect(self, image: np.ndarray) -> list[Detection]:
        detections: list[Detection] = []
        for result in self._predict(image):
            detections.extend(self._to_detections(result))
        return detections

    def detect_batch(self, images: list[np.ndarray]) -> list[list[Detection]]:
        if not images:
            return []
        return [self._to_detections(result) for result in self._predict(list(images))]
