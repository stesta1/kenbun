from __future__ import annotations

import queue
import threading
import time
from datetime import datetime

import numpy as np
import pytest

from monitoreo_urbano.vision_pipeline import camera_source
from monitoreo_urbano.vision_pipeline.camera_source import CameraSource, CaptureStopped, Frame
from monitoreo_urbano.vision_pipeline.camera_stream import CameraStream, StreamClosed, _Mailbox


class CountingCapture:
    def __init__(self, opens: bool = True, frames: int = 0, fps: float = 30.0) -> None:
        self._opens = opens
        self._remaining = frames
        self._fps = fps
        self.reads = 0
        self.grabs = 0

    def isOpened(self) -> bool:
        return self._opens

    def _next(self) -> bool:
        if self._remaining > 0:
            self._remaining -= 1
            return True
        return False

    def read(self):
        self.reads += 1
        return (True, np.zeros((4, 4, 3), dtype=np.uint8)) if self._next() else (False, None)

    def grab(self) -> bool:
        self.grabs += 1
        return self._next()

    def release(self) -> None:
        pass

    def get(self, prop) -> float:
        return self._fps


@pytest.fixture
def captures(monkeypatch):
    pending: list[CountingCapture] = []
    monkeypatch.setattr(camera_source.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(camera_source.cv2, "VideoCapture", lambda *args, **kwargs: pending.pop(0))
    return pending


def _take(source: CameraSource, count: int, stride: int = 1) -> list[Frame | None]:
    generator = source.frames(stride)
    return [next(generator) for _ in range(count)]


def test_stride_delivers_one_of_every_n_camera_frames(captures):
    capture = CountingCapture(frames=10)
    captures.append(capture)

    delivered = _take(CameraSource("cam", "rtsp://x/cam"), 4, stride=3)

    assert all(isinstance(frame, Frame) for frame in delivered)
    assert capture.reads == 4
    assert capture.grabs == 6


def test_stride_one_never_uses_grab(captures):
    capture = CountingCapture(frames=5)
    captures.append(capture)

    _take(CameraSource("cam", "rtsp://x/cam"), 5, stride=1)

    assert (capture.reads, capture.grabs) == (5, 0)


def test_frame_index_counts_delivered_frames(captures):
    captures.append(CountingCapture(frames=20))

    delivered = _take(CameraSource("cam", "rtsp://x/cam"), 4, stride=5)

    assert [frame.frame_index for frame in delivered] == [0, 1, 2, 3]


def test_first_frame_after_a_reconnection_is_always_delivered(captures):
    first = CountingCapture(frames=2)
    second = CountingCapture(frames=4)
    captures.extend([first, second])

    kinds = ["frame" if isinstance(f, Frame) else "none" for f in _take(CameraSource("cam", "rtsp://x/cam"), 3, stride=2)]

    assert kinds == ["frame", "none", "frame"]
    assert second.reads == 1


def test_stop_request_ends_the_frame_loop(captures):
    captures.append(CountingCapture(frames=100))
    source = CameraSource("cam", "rtsp://x/cam")
    generator = source.frames()
    next(generator)

    source.request_stop()

    with pytest.raises(StopIteration):
        next(generator)


def test_stop_request_interrupts_the_wait_for_a_camera_that_never_opens(captures):
    captures.extend(CountingCapture(opens=False) for _ in range(2))
    source = CameraSource("cam", "rtsp://x/cam")
    source.request_stop()

    with pytest.raises(CaptureStopped):
        source.fps


def _recording_video_capture(monkeypatch):
    calls: list[tuple] = []

    def fake(*args, **kwargs):
        calls.append(args)
        return CountingCapture(frames=1)

    monkeypatch.setattr(camera_source.cv2, "VideoCapture", fake)
    monkeypatch.setattr(camera_source, "with_stream_credentials", lambda url: url)
    return calls


def test_opencv_is_opened_without_extra_parameters_when_nothing_is_configured(monkeypatch):
    calls = _recording_video_capture(monkeypatch)
    monkeypatch.setattr(camera_source.settings, "capture_decoder_threads", 0)

    CameraSource("cam", "rtsp://x/cam")._connect()

    assert calls == [("rtsp://x/cam",)]


def test_each_decoder_uses_one_thread_by_default(monkeypatch):
    calls = _recording_video_capture(monkeypatch)

    CameraSource("cam", "rtsp://x/cam")._connect()

    assert calls == [("rtsp://x/cam", camera_source.cv2.CAP_FFMPEG, [camera_source.cv2.CAP_PROP_N_THREADS, 1])]


def test_hardware_acceleration_is_passed_to_opencv_when_configured(monkeypatch):
    calls = _recording_video_capture(monkeypatch)
    monkeypatch.setattr(camera_source.settings, "capture_hw_acceleration", "any")
    monkeypatch.setattr(camera_source.settings, "capture_decoder_threads", 0)

    CameraSource("cam", "rtsp://x/cam")._connect()

    assert calls == [("rtsp://x/cam", camera_source.cv2.CAP_FFMPEG, [camera_source.cv2.CAP_PROP_HW_ACCELERATION, camera_source.cv2.VIDEO_ACCELERATION_ANY])]


def test_acceleration_and_thread_count_can_be_combined(monkeypatch):
    calls = _recording_video_capture(monkeypatch)
    monkeypatch.setattr(camera_source.settings, "capture_hw_acceleration", "any")
    monkeypatch.setattr(camera_source.settings, "capture_decoder_threads", 2)

    CameraSource("cam", "rtsp://x/cam")._connect()

    cv2 = camera_source.cv2
    assert calls[0][2] == [cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY, cv2.CAP_PROP_N_THREADS, 2]


def test_an_unknown_acceleration_name_is_rejected(monkeypatch):
    _recording_video_capture(monkeypatch)
    monkeypatch.setattr(camera_source.settings, "capture_hw_acceleration", "cuda-magico")

    with pytest.raises(ValueError, match="capture_hw_acceleration"):
        CameraSource("cam", "rtsp://x/cam")._connect()


def _frame(index: int) -> Frame:
    return Frame(image=np.zeros((2, 2, 3), dtype=np.uint8), timestamp=datetime.utcnow(), frame_index=index)


def test_mailbox_is_fifo():
    mailbox = _Mailbox(4)
    for index in range(3):
        mailbox.put(_frame(index))

    assert [mailbox.get(timeout=0).frame_index for _ in range(3)] == [0, 1, 2]


def test_mailbox_drops_the_oldest_frame_when_full():
    mailbox = _Mailbox(2)
    drops = [mailbox.put(_frame(index)) for index in range(4)]

    assert drops == [0, 0, 1, 1]
    assert [mailbox.get(timeout=0).frame_index for _ in range(2)] == [2, 3]


def test_mailbox_never_drops_a_signal_loss_marker():
    mailbox = _Mailbox(2)
    mailbox.put(None)
    mailbox.put(_frame(1))
    dropped = mailbox.put(_frame(2))

    assert dropped == 1
    assert mailbox.get(timeout=0) is None
    assert mailbox.get(timeout=0).frame_index == 2


def test_mailbox_get_times_out_when_empty():
    with pytest.raises(queue.Empty):
        _Mailbox(2).get(timeout=0)


def test_mailbox_raises_stream_closed_once_closed_and_drained():
    mailbox = _Mailbox(2)
    mailbox.put(_frame(0))
    mailbox.close()

    assert mailbox.get(timeout=0).frame_index == 0
    with pytest.raises(StreamClosed):
        mailbox.get(timeout=0)


class FakeSource:
    def __init__(self, items=(), native_fps: float = 30.0, fps_gate: threading.Event | None = None) -> None:
        self._items = list(items)
        self._native_fps = native_fps
        self._fps_gate = fps_gate
        self._stop = threading.Event()
        self.stride_seen: int | None = None
        self.closed = False

    @property
    def fps(self) -> float:
        if self._fps_gate is not None and not self._fps_gate.wait(5):
            raise TimeoutError
        if self._stop.is_set():
            raise CaptureStopped
        return self._native_fps

    def frames(self, stride: int = 1):
        self.stride_seen = stride
        yield from self._items
        while not self._stop.is_set():
            time.sleep(0.005)

    def request_stop(self) -> None:
        self._stop.set()

    def close(self) -> None:
        self.closed = True


def _drain(stream: CameraStream, count: int, timeout: float = 2.0):
    return [stream.get(timeout=timeout) for _ in range(count)]


def test_stream_delivers_frames_in_order_and_signal_losses_as_none():
    items = [_frame(0), None, _frame(1)]
    stream = CameraStream("cam", FakeSource(items), queue_size=8).start()
    try:
        received = _drain(stream, 3)
    finally:
        stream.stop()

    assert [None if item is None else item.frame_index for item in received] == [0, None, 1]


def test_stream_computes_the_stride_from_the_analysis_frequency():
    source = FakeSource([_frame(0)], native_fps=30.0)
    stream = CameraStream("cam", source, analysis_fps=6.0).start()
    try:
        assert stream.wait_ready(2)
    finally:
        stream.stop()

    assert source.stride_seen == 5
    assert stream.stride == 5
    assert stream.native_fps == 30.0
    assert stream.effective_fps == pytest.approx(6.0)


def test_stream_without_analysis_frequency_delivers_every_frame():
    source = FakeSource([_frame(0)], native_fps=25.0)
    stream = CameraStream("cam", source).start()
    try:
        stream.wait_ready(2)
    finally:
        stream.stop()

    assert (stream.stride, stream.effective_fps) == (1, 25.0)


def test_analysis_frequency_above_the_camera_frequency_never_gives_a_stride_below_one():
    stream = CameraStream("cam", FakeSource([_frame(0)], native_fps=15.0), analysis_fps=30.0).start()
    try:
        stream.wait_ready(2)
    finally:
        stream.stop()

    assert stream.stride == 1


def test_start_does_not_block_when_the_camera_is_down():
    gate = threading.Event()
    stream = CameraStream("cam", FakeSource([_frame(0)], fps_gate=gate))

    started = time.monotonic()
    stream.start()
    elapsed = time.monotonic() - started

    try:
        assert elapsed < 0.5
        assert stream.wait_ready(0.05) is False
        assert stream.effective_fps is None

        gate.set()
        assert stream.wait_ready(2) is True
        assert stream.get(timeout=2).frame_index == 0
    finally:
        gate.set()
        stream.stop()


def test_slow_consumer_makes_the_stream_drop_the_oldest_frames_and_count_them():
    items = [_frame(index) for index in range(10)]
    stream = CameraStream("cam", FakeSource(items), queue_size=2).start()
    try:
        deadline = time.monotonic() + 2
        while stream.stats().frames_delivered < 10 and time.monotonic() < deadline:
            time.sleep(0.01)
        stats = stream.stats()
        remaining = [stream.get(timeout=0).frame_index for _ in range(stats.queue_depth)]
    finally:
        stream.stop()

    assert stats.frames_delivered == 10
    assert stats.frames_dropped == 8
    assert remaining == [8, 9]


def test_stats_count_signal_losses_separately_from_frames():
    stream = CameraStream("cam", FakeSource([None, None, _frame(0)]), queue_size=8).start()
    try:
        _drain(stream, 3)
        stats = stream.stats()
    finally:
        stream.stop()

    assert (stats.frames_delivered, stats.signal_losses, stats.frames_dropped) == (1, 2, 0)


def test_stop_ends_the_thread_and_releases_the_source():
    source = FakeSource([_frame(0)])
    stream = CameraStream("cam", source).start()
    stream.wait_ready(2)

    stream.stop()

    assert not stream.is_alive
    assert source.closed
    assert stream.stats().running is False


def test_get_raises_stream_closed_after_stop_once_drained():
    stream = CameraStream("cam", FakeSource([])).start()
    stream.wait_ready(2)
    stream.stop()

    with pytest.raises(StreamClosed):
        stream.get(timeout=0)


def test_an_error_in_the_source_is_kept_and_closes_the_stream():
    class Broken(FakeSource):
        def frames(self, stride: int = 1):
            raise RuntimeError("falla de decodificación")
            yield  # pragma: no cover

    stream = CameraStream("cam", Broken()).start()
    deadline = time.monotonic() + 2
    while stream.is_alive and time.monotonic() < deadline:
        time.sleep(0.01)

    assert isinstance(stream.error, RuntimeError)
    with pytest.raises(StreamClosed):
        stream.get(timeout=0)
