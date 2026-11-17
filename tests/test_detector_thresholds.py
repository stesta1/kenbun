from __future__ import annotations

from types import SimpleNamespace

import pytest

from monitoreo_urbano.common.config import settings
from monitoreo_urbano.vision_pipeline.detector import YOLODetector

NAMES = {0: "person", 1: "car", 2: "motorcycle", 3: "fire", 4: "smoke", 5: "accident"}


class _Tensor:
    def __init__(self, values):
        self._values = values

    def tolist(self):
        return list(self._values)

    def item(self):
        return self._values[0]

    def __getitem__(self, index):
        return _Tensor([self._values[index]]) if not isinstance(self._values[index], (list, tuple)) else _Tensor(self._values[index])


def _box(class_id: int, confidence: float):
    return SimpleNamespace(xyxy=[_Tensor([0, 0, 10, 10])], cls=[_Tensor([class_id])], conf=[_Tensor([confidence])])


def _detector(general: float = 0.35) -> YOLODetector:
    detector = YOLODetector.__new__(YOLODetector)
    detector._model = SimpleNamespace(names=NAMES)
    detector.thresholds = None
    detector._static_confidence_threshold = general
    return detector


@pytest.fixture(autouse=True)
def by_class(monkeypatch):
    monkeypatch.setattr(settings, "yolo_confidence_by_class", {"fire": 0.5, "accident": 0.6})


def test_classes_without_their_own_threshold_use_the_general_one():
    thresholds = _detector(0.35)._class_thresholds()

    assert thresholds["person"] == thresholds["car"] == thresholds["smoke"] == 0.35
    assert (thresholds["fire"], thresholds["accident"]) == (0.5, 0.6)


def test_the_prediction_uses_the_lowest_threshold_so_no_class_is_lost():
    detector = _detector(0.35)
    captured = {}
    detector._model.predict = lambda **kwargs: captured.update(kwargs) or []
    detector.device = "cpu"
    detector._class_filter_ids = None

    detector._predict("img")

    assert captured["conf"] == 0.35


def test_a_lower_general_threshold_lowers_the_prediction_floor():
    detector = _detector(0.2)
    captured = {}
    detector._model.predict = lambda **kwargs: captured.update(kwargs) or []
    detector.device = "cpu"
    detector._class_filter_ids = None

    detector._predict("img")

    assert captured["conf"] == 0.2


def test_detections_below_their_class_threshold_are_dropped():
    detector = _detector(0.35)
    result = SimpleNamespace(
        names=NAMES,
        boxes=[
            _box(5, 0.55),
            _box(5, 0.72),
            _box(3, 0.45),
            _box(0, 0.40),
            _box(0, 0.30),
        ],
    )

    kept = detector._to_detections(result)

    assert [(d.class_name, d.confidence) for d in kept] == [("accident", 0.72), ("person", 0.40)]
