from __future__ import annotations

from collections import deque

from monitoreo_urbano.vision_pipeline.camera_source import Frame


class FrameRingBuffer:
    def __init__(self, max_seconds: float, fps: float) -> None:
        self.max_seconds = max_seconds
        self.fps = fps
        capacity = max(1, int(max_seconds * fps))
        self._buffer: deque[Frame] = deque(maxlen=capacity)

    def append(self, frame: Frame) -> None:
        self._buffer.append(frame)

    def snapshot(self) -> list[Frame]:
        return list(self._buffer)
