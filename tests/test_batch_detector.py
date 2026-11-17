from __future__ import annotations

import queue
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from monitoreo_urbano.common.schemas import BoundingBox, Detection
from monitoreo_urbano.vision_pipeline.batch_detector import BatchDetector, DetectionResult
from monitoreo_urbano.vision_pipeline.camera_source import Frame
from monitoreo_urbano.vision_pipeline.camera_stream import StreamClosed
from monitoreo_urbano.vision_pipeline.detector import Detector


class FakeStream:
    def __init__(self) -> None:
        self._items: deque = deque()
        self._closed = False
        self._wake: threading.Event | None = None

    def push(self, item) -> None:
        self._items.append(item)
        if self._wake is not None:
            self._wake.set()

    def close(self) -> None:
        self._closed = True
        if self._wake is not None:
            self._wake.set()

    def get(self, timeout=None):
        if self._items:
            return self._items.popleft()
        if self._closed:
            raise StreamClosed
        raise queue.Empty

    def set_wake_event(self, event) -> None:
        self._wake = event


class FakeBackend(Detector):

    def __init__(self, fail_times: int = 0) -> None:
        self.calls: list[int] = []
        self.fail_times = fail_times

    def detect(self, image):
        return self.detect_batch([image])[0]

    def detect_batch(self, images):
        self.calls.append(len(images))
        if self.fail_times > 0:
            self.fail_times -= 1
            raise RuntimeError("falla simulada del detector")
        return [
            [Detection(class_name=f"m{int(image[0, 0, 0])}", confidence=0.9, bbox=BoundingBox(x1=0, y1=0, x2=1, y2=1))]
            for image in images
        ]


def _frame(marker: int, index: int = 0) -> Frame:
    return Frame(image=np.full((2, 2, 3), marker, dtype=np.uint8), timestamp=datetime.utcnow(), frame_index=index)


def _wait_for(condition, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.005)
    return condition()


@pytest.fixture
def engine_factory():
    created: list[BatchDetector] = []

    def make(backend=None, results=None, **kwargs):
        collected = results if results is not None else []
        kwargs.setdefault("fill_wait_seconds", 0.02)
        engine = BatchDetector(backend or FakeBackend(), collected.append, **kwargs)
        created.append(engine)
        return engine, collected

    yield make
    for engine in created:
        engine.stop(timeout=2)


def test_frames_from_several_cameras_go_through_the_detector_in_one_call(engine_factory):
    backend = FakeBackend()
    engine, results = engine_factory(backend)
    streams = {f"cam-{i}": FakeStream() for i in range(3)}
    for camera_id, stream in streams.items():
        engine.register(camera_id, stream)
    for index, stream in enumerate(streams.values()):
        stream.push(_frame(marker=10 + index))

    engine.start()

    assert _wait_for(lambda: len(results) == 3)
    assert backend.calls == [3]


def test_each_result_goes_back_to_its_own_camera_and_frame(engine_factory):
    engine, results = engine_factory()
    for index in range(3):
        stream = FakeStream()
        engine.register(f"cam-{index}", stream)
        stream.push(_frame(marker=20 + index, index=index))

    engine.start()

    assert _wait_for(lambda: len(results) == 3)
    by_camera: dict[str, DetectionResult] = {r.camera_id: r for r in results}
    for index in range(3):
        result = by_camera[f"cam-{index}"]
        assert result.detections[0].class_name == f"m{20 + index}"
        assert result.frame.frame_index == index


def test_batch_never_exceeds_the_maximum(engine_factory):
    backend = FakeBackend()
    engine, results = engine_factory(backend, max_batch=4)
    for index in range(10):
        stream = FakeStream()
        engine.register(f"cam-{index}", stream)
        stream.push(_frame(marker=index))

    engine.start()

    assert _wait_for(lambda: len(results) == 10)
    assert backend.calls == [4, 4, 2]


def test_a_camera_contributes_one_frame_per_batch(engine_factory):
    backend = FakeBackend()
    engine, results = engine_factory(backend)
    busy, quiet = FakeStream(), FakeStream()
    engine.register("busy", busy)
    engine.register("quiet", quiet)
    for marker in (1, 2, 3):
        busy.push(_frame(marker))
    quiet.push(_frame(9))

    engine.start()

    assert _wait_for(lambda: len(results) == 4)
    assert backend.calls[0] == 2
    assert [r.detections[0].class_name for r in results if r.camera_id == "busy"] == ["m1", "m2", "m3"]


def test_every_camera_is_served_when_the_batch_is_smaller_than_the_camera_count(engine_factory):
    engine, results = engine_factory(max_batch=3)
    streams = {f"cam-{i}": FakeStream() for i in range(6)}
    for camera_id, stream in streams.items():
        engine.register(camera_id, stream)
    for stream in streams.values():
        for marker in (1, 2, 3):
            stream.push(_frame(marker))

    engine.start()

    assert _wait_for(lambda: len(results) == 18)
    per_camera = {camera_id: [r for r in results if r.camera_id == camera_id] for camera_id in streams}
    assert all(len(items) == 3 for items in per_camera.values())
    first_nine = {r.camera_id for r in results[:9]}
    assert first_nine == set(streams)


def test_signal_loss_is_reported_and_never_sent_to_the_detector(engine_factory):
    backend = FakeBackend()
    lost: list[str] = []
    engine, results = engine_factory(backend, on_signal_loss=lost.append)
    stream = FakeStream()
    engine.register("cam-a", stream)
    stream.push(None)
    stream.push(_frame(5))

    engine.start()

    assert _wait_for(lambda: len(results) == 1 and lost == ["cam-a"])
    assert backend.calls == [1]


def test_a_closed_stream_is_reported_once_and_dropped(engine_factory):
    closed: list[str] = []
    engine, _ = engine_factory(on_stream_closed=closed.append)
    stream = FakeStream()
    engine.register("cam-a", stream)
    stream.close()

    engine.start()

    assert _wait_for(lambda: closed == ["cam-a"])
    assert _wait_for(lambda: engine.stats().cameras == 0)
    time.sleep(0.1)
    assert closed == ["cam-a"]


def test_a_detector_failure_is_survived_and_counted(engine_factory):
    backend = FakeBackend(fail_times=1)
    engine, results = engine_factory(backend)
    stream = FakeStream()
    engine.register("cam-a", stream)
    stream.push(_frame(1))

    engine.start()
    assert _wait_for(lambda: engine.stats().errors == 1)
    stream.push(_frame(2))

    assert _wait_for(lambda: len(results) == 1)
    assert engine.stats().running


def test_repeated_detector_failures_stop_the_engine_and_keep_the_error(engine_factory, monkeypatch):
    monkeypatch.setattr("monitoreo_urbano.vision_pipeline.batch_detector._ERROR_BACKOFF_SECONDS", 0.0)
    engine, _ = engine_factory(FakeBackend(fail_times=100), max_consecutive_errors=3)
    stream = FakeStream()
    engine.register("cam-a", stream)
    for marker in range(5):
        stream.push(_frame(marker))

    engine.start()

    assert _wait_for(lambda: not engine.is_alive)
    assert isinstance(engine.error, RuntimeError)
    assert engine.stats().errors == 3
    assert not engine.stats().running


def test_a_failing_result_callback_does_not_stop_the_engine():
    calls: list[str] = []

    def flaky(result: DetectionResult) -> None:
        calls.append(result.camera_id)
        if len(calls) == 1:
            raise ValueError("callback roto")

    engine = BatchDetector(FakeBackend(), flaky, fill_wait_seconds=0.01)
    try:
        stream = FakeStream()
        engine.register("cam-a", stream)
        stream.push(_frame(1))
        stream.push(_frame(2))
        engine.start()

        assert _wait_for(lambda: len(calls) == 2)
        assert engine.stats().callback_errors == 1
        assert engine.stats().running
    finally:
        engine.stop(timeout=2)


def test_stats_count_batches_images_and_latency(engine_factory):
    engine, results = engine_factory()
    stream = FakeStream()
    engine.register("cam-a", stream)
    for marker in range(4):
        stream.push(_frame(marker))

    engine.start()
    assert _wait_for(lambda: len(results) == 4)
    stats = engine.stats()

    assert stats.images == 4
    assert stats.batches >= 1
    assert stats.cameras == 1
    assert stats.mean_batch_size >= 1.0
    assert stats.inference_ms_p50 is not None
    assert stats.latency_ms_p50 is not None and stats.latency_ms_p50 >= 0


def test_a_camera_registered_twice_is_rejected(engine_factory):
    engine, _ = engine_factory()
    engine.register("cam-a", FakeStream())

    with pytest.raises(ValueError):
        engine.register("cam-a", FakeStream())


def test_unregistering_stops_polling_a_camera(engine_factory):
    engine, results = engine_factory()
    stream = FakeStream()
    engine.register("cam-a", stream)
    engine.unregister("cam-a")
    stream.push(_frame(1))

    engine.start()
    time.sleep(0.15)

    assert results == []


def test_stop_ends_the_thread(engine_factory):
    engine, _ = engine_factory()
    engine.start()

    engine.stop(timeout=2)

    assert not engine.is_alive


_WEIGHTS = Path(__file__).resolve().parents[1] / "data" / "models" / "yolo" / "baseline_v1.pt"


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(not _WEIGHTS.exists() or not _cuda_available(), reason="requiere los pesos de YOLO y una GPU")
def test_yolo_batch_detection_matches_single_image_detection():
    from monitoreo_urbano.vision_pipeline.detector import YOLODetector

    rng = np.random.default_rng(0)
    images = [rng.integers(0, 255, size=(360, 640, 3), dtype=np.uint8) for _ in range(3)]
    detector = YOLODetector(weights_path=str(_WEIGHTS), device="cuda:0", confidence_threshold=0.05)

    batch = detector.detect_batch(images)
    single = [detector.detect(image) for image in images]

    assert len(batch) == len(images)
    for from_batch, from_single in zip(batch, single):
        assert sorted(d.class_name for d in from_batch) == sorted(d.class_name for d in from_single)
