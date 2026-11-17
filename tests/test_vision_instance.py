from __future__ import annotations

import queue
import threading
import time
from collections import deque
from datetime import datetime

import numpy as np
import pytest

from monitoreo_urbano.common.schemas import HEALTH_OK, BoundingBox, CameraInfo, Detection, EventRecord
from monitoreo_urbano.vision_pipeline import camera_pipeline
from monitoreo_urbano.vision_pipeline.camera_source import Frame
from monitoreo_urbano.vision_pipeline.camera_stream import StreamClosed, StreamStats
from monitoreo_urbano.vision_pipeline.detector import Detector
from monitoreo_urbano.vision_pipeline.instance import VisionInstance


class FakeStore:
    def __init__(self) -> None:
        self.registered: list[str] = []
        self.health: list[tuple[str, str]] = []
        self.events: list[EventRecord] = []
        self.opened: list[EventRecord] = []
        self.refreshed = 0
        self.beats: list[tuple[list[str], str]] = []
        self.closed = False
        self.absorbed: list[tuple[str, str, float]] = []
        self._lock = threading.Lock()

    def register_camera(self, camera: CameraInfo) -> None:
        self.registered.append(camera.camera_id)

    def set_health(self, camera_id: str, status: str) -> None:
        with self._lock:
            self.health.append((camera_id, status))

    def insert_event(self, event: EventRecord) -> None:
        with self._lock:
            self.events.append(event)

    def open_event(self, event: EventRecord) -> None:
        with self._lock:
            self.opened.append(event)

    def absorb_review_events(self, camera_id: str, alert_event_id: str, within_seconds: float) -> int:
        with self._lock:
            self.absorbed.append((camera_id, alert_event_id, within_seconds))
        return 0

    def complete_event(self, event: EventRecord) -> bool:
        with self._lock:
            if not any(o.event_id == event.event_id for o in self.opened):
                return False
            self.events.append(event)
            return True

    def refresh_thresholds(self, thresholds) -> None:
        self.refreshed += 1

    def heartbeat(self, camera_ids, instance) -> None:
        with self._lock:
            self.beats.append((list(camera_ids), instance))

    def pending_description_count(self) -> int:
        return 0

    def close(self) -> None:
        self.closed = True


class FakeCameraStream:

    def __init__(self, camera_id: str, *, ready: bool = True, fps: float = 30.0) -> None:
        self.camera_id = camera_id
        self.reconnect_delay_seconds = 3.0
        self.error: BaseException | None = None
        self._fps = fps
        self._ready = threading.Event()
        if ready:
            self._ready.set()
        self._alive = False
        self._items: deque = deque()
        self._closed = False
        self._wake: threading.Event | None = None
        self.stopped = False

    def start(self) -> FakeCameraStream:
        self._alive = True
        return self

    def request_stop(self) -> None:
        self.stopped = True

    def stop(self, timeout: float = 5.0) -> None:
        self.stopped = True
        self._alive = False

    @property
    def is_alive(self) -> bool:
        return self._alive

    def wait_ready(self, timeout: float | None = None) -> bool:
        return self._ready.wait(timeout or 0)

    def make_ready(self) -> None:
        self._ready.set()

    @property
    def effective_fps(self) -> float | None:
        return self._fps if self._ready.is_set() else None

    def push(self, item) -> None:
        self._items.append(item)
        if self._wake is not None:
            self._wake.set()

    def close(self, error: BaseException | None = None) -> None:
        self.error = error
        self._closed = True
        self._alive = False
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

    def stats(self) -> StreamStats:
        return StreamStats(
            camera_id=self.camera_id, running=self._alive, native_fps=self._fps, effective_fps=self.effective_fps,
            stride=1, frames_delivered=0, frames_dropped=0, signal_losses=0, queue_depth=len(self._items),
            capture_fps=self._fps, last_frame_age_seconds=None,
        )


class FireEverywhere(Detector):
    def detect(self, image):
        return self.detect_batch([image])[0]

    def detect_batch(self, images):
        box = BoundingBox(x1=10, y1=10, x2=50, y2=50)
        return [[Detection(class_name="fire", confidence=0.9, bbox=box)] for _ in images]


def _camera(camera_id: str) -> CameraInfo:
    return CameraInfo(camera_id=camera_id, label=f"Esquina {camera_id}", source=f"rtsp://x/{camera_id}")


def _frame(index: int, seed: int | None = None) -> Frame:
    image = np.random.default_rng(index if seed is None else seed).integers(0, 255, size=(8, 8, 3), dtype=np.uint8)
    return Frame(image=image, timestamp=datetime.utcnow(), frame_index=index)


def _wait_for(condition, timeout: float = 6.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.01)
    return condition()


@pytest.fixture(autouse=True)
def short_clips(monkeypatch):
    monkeypatch.setattr(camera_pipeline.settings, "clip_post_event_seconds", 0.1)
    monkeypatch.setattr(camera_pipeline.settings, "frame_buffer_seconds", 1.0)
    monkeypatch.setattr("monitoreo_urbano.vision_pipeline.instance._CONTROL_TICK_SECONDS", 0.02)


@pytest.fixture
def make_instance():
    created: list[VisionInstance] = []

    def make(cameras, streams, store=None, clip_writer=None, **kwargs):
        store = store or FakeStore()
        writer = clip_writer or (lambda camera_id, event_id, frames, fps: f"{camera_id}/{event_id}.mp4")
        instance = VisionInstance(
            [_camera(c) for c in cameras],
            detector=kwargs.pop("detector", FireEverywhere()),
            store=store,
            clip_writer=writer,
            stream_factory=lambda camera: streams[camera.camera_id],
            **kwargs,
        )
        created.append(instance)
        return instance, store

    yield make
    for instance in created:
        instance.stop(clip_timeout=2)


def test_every_camera_is_registered_when_the_instance_starts(make_instance):
    streams = {"cam-a": FakeCameraStream("cam-a", ready=False), "cam-b": FakeCameraStream("cam-b", ready=False)}
    instance, store = make_instance(["cam-a", "cam-b"], streams)

    instance.start()

    assert store.registered == ["cam-a", "cam-b"]
    assert store.refreshed >= 1
    assert all(stream.is_alive for stream in streams.values())


def test_a_camera_is_marked_healthy_once_it_connects(make_instance):
    streams = {"cam-a": FakeCameraStream("cam-a", ready=False)}
    instance, store = make_instance(["cam-a"], streams)
    instance.start()
    time.sleep(0.15)
    assert ("cam-a", HEALTH_OK) not in store.health

    streams["cam-a"].make_ready()

    assert _wait_for(lambda: ("cam-a", HEALTH_OK) in store.health)
    assert instance.stats().cameras[0].connected


def test_a_camera_that_never_connects_does_not_hold_back_the_others(make_instance):
    streams = {"down": FakeCameraStream("down", ready=False), "up": FakeCameraStream("up", ready=True)}
    instance, store = make_instance(["down", "up"], streams)
    instance.start()

    for index in range(60):
        streams["up"].push(_frame(index))

    assert _wait_for(lambda: len(store.events) >= 1)
    assert store.events[0].camera_id == "up"
    assert ("down", HEALTH_OK) not in store.health


def test_an_incident_ends_up_as_an_event_with_its_clip(make_instance):
    written: list[dict] = []

    def clip_writer(camera_id, event_id, frames, fps):
        written.append({"camera_id": camera_id, "event_id": event_id, "frames": len(frames), "fps": fps})
        return f"{camera_id}/{event_id}.mp4"

    streams = {"cam-a": FakeCameraStream("cam-a", fps=30.0)}
    instance, store = make_instance(["cam-a"], streams, clip_writer=clip_writer)
    instance.start()

    for index in range(60):
        streams["cam-a"].push(_frame(index))

    assert _wait_for(lambda: len(store.events) == 1)
    event = store.events[0]
    assert event.camera_id == "cam-a"
    assert event.camera_label == "Esquina cam-a"
    assert event.trigger_reason == "fire"
    assert event.object_classes == ["fire"]
    assert event.clip_path == f"cam-a/{event.event_id}.mp4"
    assert event.end_ts >= event.start_ts
    assert written and written[0]["fps"] == 30.0 and written[0]["frames"] >= 8


def test_cameras_keep_independent_events(make_instance):
    streams = {name: FakeCameraStream(name) for name in ("cam-a", "cam-b", "cam-c")}
    instance, store = make_instance(list(streams), streams)
    instance.start()

    for index in range(60):
        streams["cam-a"].push(_frame(index))
        streams["cam-c"].push(_frame(index, seed=100 + index))

    assert _wait_for(lambda: {e.camera_id for e in store.events} == {"cam-a", "cam-c"})
    assert all(e.camera_id != "cam-b" for e in store.events)


def test_a_clip_that_fails_to_write_does_not_stop_the_instance(make_instance):
    calls = {"n": 0}

    def flaky_writer(camera_id, event_id, frames, fps):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("MinIO no responde")
        return f"{camera_id}/{event_id}.mp4"

    streams = {"cam-a": FakeCameraStream("cam-a"), "cam-b": FakeCameraStream("cam-b")}
    instance, store = make_instance(["cam-a", "cam-b"], streams, clip_writer=flaky_writer)
    instance.start()
    for index in range(60):
        streams["cam-a"].push(_frame(index))
        streams["cam-b"].push(_frame(index, seed=50 + index))

    assert _wait_for(lambda: instance.stats().clips.failed == 1)
    assert _wait_for(lambda: len(store.events) == 1)


def test_signal_loss_is_reported_as_a_camera_cut_after_the_sustained_attempts(make_instance):
    streams = {"cam-a": FakeCameraStream("cam-a")}
    instance, store = make_instance(["cam-a"], streams)
    instance.start()
    assert _wait_for(lambda: ("cam-a", HEALTH_OK) in store.health)

    for _ in range(3):
        streams["cam-a"].push(None)

    assert _wait_for(lambda: ("cam-a", "corte_de_señal") in store.health)


def test_a_stream_that_ends_marks_its_camera_as_cut_and_the_others_go_on(make_instance):
    streams = {"cam-a": FakeCameraStream("cam-a"), "cam-b": FakeCameraStream("cam-b")}
    instance, store = make_instance(["cam-a", "cam-b"], streams)
    instance.start()
    assert _wait_for(lambda: ("cam-b", HEALTH_OK) in store.health)

    streams["cam-a"].close(error=RuntimeError("decodificación"))

    assert _wait_for(lambda: ("cam-a", "corte_de_señal") in store.health)
    assert instance.wait(0.1) is False
    assert instance.stats().cameras[0].closed


def test_the_instance_stops_by_itself_when_every_camera_has_ended(make_instance):
    streams = {"cam-a": FakeCameraStream("cam-a"), "cam-b": FakeCameraStream("cam-b")}
    instance, _ = make_instance(["cam-a", "cam-b"], streams)
    instance.start()

    for stream in streams.values():
        stream.close()

    assert instance.wait(5) is True


def test_a_capture_that_dies_before_connecting_is_given_up_on(make_instance):
    streams = {"cam-a": FakeCameraStream("cam-a", ready=False)}
    instance, store = make_instance(["cam-a"], streams)
    instance.start()

    streams["cam-a"]._alive = False
    streams["cam-a"].error = OSError("URL inválida")

    assert _wait_for(lambda: ("cam-a", "corte_de_señal") in store.health)
    assert instance.wait(5) is True


def test_stop_stops_the_streams_waits_for_the_clips_and_closes_the_store(make_instance):
    streams = {"cam-a": FakeCameraStream("cam-a")}
    instance, store = make_instance(["cam-a"], streams)
    instance.start()
    for index in range(60):
        streams["cam-a"].push(_frame(index))
    assert _wait_for(lambda: len(store.events) == 1)

    instance.stop(clip_timeout=3)

    assert streams["cam-a"].stopped
    assert store.closed


def test_thresholds_are_refreshed_periodically(make_instance, monkeypatch):
    monkeypatch.setattr("monitoreo_urbano.vision_pipeline.instance._THRESHOLDS_REFRESH_SECONDS", 0.05)
    streams = {"cam-a": FakeCameraStream("cam-a")}
    instance, store = make_instance(["cam-a"], streams)
    instance.start()

    assert _wait_for(lambda: store.refreshed >= 3)


def test_an_instance_needs_cameras_and_rejects_duplicates():
    store = FakeStore()
    with pytest.raises(ValueError):
        VisionInstance([], detector=FireEverywhere(), store=store)
    with pytest.raises(ValueError):
        VisionInstance([_camera("a"), _camera("a")], detector=FireEverywhere(), store=store)


def test_stats_summarize_the_instance(make_instance):
    streams = {"cam-a": FakeCameraStream("cam-a"), "cam-b": FakeCameraStream("cam-b", ready=False)}
    instance, _ = make_instance(["cam-a", "cam-b"], streams)
    instance.start()
    assert _wait_for(lambda: instance.stats().cameras[0].connected)

    stats = instance.stats()

    assert [c.camera_id for c in stats.cameras] == ["cam-a", "cam-b"]
    assert [c.connected for c in stats.cameras] == [True, False]
    assert stats.clips.workers >= 1


def test_the_instance_beats_for_all_its_cameras_when_it_starts_and_then_periodically(make_instance, monkeypatch):
    monkeypatch.setattr(camera_pipeline.settings, "heartbeat_interval_seconds", 0.05)
    streams = {"cam-a": FakeCameraStream("cam-a", ready=False), "cam-b": FakeCameraStream("cam-b", ready=False)}
    instance, store = make_instance(["cam-a", "cam-b"], streams, name="vision-7")

    instance.start()

    assert store.beats[0] == (["cam-a", "cam-b"], "vision-7")
    assert _wait_for(lambda: len(store.beats) >= 4)


def test_a_failing_heartbeat_does_not_stop_the_instance(make_instance, monkeypatch):
    monkeypatch.setattr(camera_pipeline.settings, "heartbeat_interval_seconds", 0.02)
    streams = {"cam-a": FakeCameraStream("cam-a")}
    instance, store = make_instance(["cam-a"], streams)

    def broken(camera_ids, instance_name):
        raise OSError("base de datos caída")

    store.heartbeat = broken
    instance.start()
    time.sleep(0.2)
    for index in range(60):
        streams["cam-a"].push(_frame(index))

    assert _wait_for(lambda: len(store.events) == 1)


def test_the_event_is_registered_when_it_fires_and_completed_when_the_clip_closes(make_instance):
    streams = {"cam-a": FakeCameraStream("cam-a", fps=30.0)}
    instance, store = make_instance(["cam-a"], streams)
    instance.start()

    for index in range(60):
        streams["cam-a"].push(_frame(index))

    assert _wait_for(lambda: len(store.events) == 1)
    assert len(store.opened) == 1
    assert store.opened[0].event_id == store.events[0].event_id
    assert store.opened[0].clip_path == f"cam-a/{store.opened[0].event_id}.mp4"
