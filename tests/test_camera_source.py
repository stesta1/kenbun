from __future__ import annotations

import numpy as np
import pytest

from monitoreo_urbano.vision_pipeline import camera_source
from monitoreo_urbano.vision_pipeline.camera_source import CameraSource, Frame


class FakeCapture:
    def __init__(self, opens: bool = True, reads: int = 0) -> None:
        self._opens = opens
        self._reads = reads

    def isOpened(self) -> bool:
        return self._opens

    def read(self):
        if self._reads > 0:
            self._reads -= 1
            return True, np.zeros((4, 4, 3), dtype=np.uint8)
        return False, None

    def release(self) -> None:
        pass

    def get(self, prop) -> float:
        return 25.0


@pytest.fixture
def captures(monkeypatch):
    queue: list[FakeCapture] = []
    monkeypatch.setattr(camera_source.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(camera_source.cv2, "VideoCapture", lambda *args, **kwargs: queue.pop(0))
    return queue


def _kinds(source: CameraSource, count: int) -> list[str]:
    frames = source.frames()
    return ["frame" if isinstance(next(frames), Frame) else "none" for _ in range(count)]


def test_reports_every_failed_reconnect_attempt(captures):
    captures.extend(FakeCapture(opens=False) for _ in range(3))
    assert _kinds(CameraSource("cam-01", "rtsp://x/cam-01"), 3) == ["none", "none", "none"]


def test_reports_the_cut_and_each_attempt_that_follows(captures):
    captures.extend([FakeCapture(reads=2), FakeCapture(opens=False), FakeCapture(opens=False)])
    assert _kinds(CameraSource("cam-01", "rtsp://x/cam-01"), 5) == ["frame", "frame", "none", "none", "none"]


def test_resumes_delivering_frames_once_the_camera_returns(captures):
    captures.extend([FakeCapture(reads=1), FakeCapture(opens=False), FakeCapture(reads=2)])
    assert _kinds(CameraSource("cam-01", "rtsp://x/cam-01"), 5) == ["frame", "none", "none", "frame", "frame"]


def test_initial_open_still_waits_for_the_camera(captures):
    captures.extend([FakeCapture(opens=False), FakeCapture(opens=False), FakeCapture(reads=1)])
    assert CameraSource("cam-01", "rtsp://x/cam-01").fps == 25.0
    assert captures == []
