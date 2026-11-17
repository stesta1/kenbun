from __future__ import annotations

import cv2

from monitoreo_urbano.common.runtime_config import RuntimeThresholds
from monitoreo_urbano.common.schemas import HEALTH_OK
from monitoreo_urbano.vision_pipeline.camera_source import Frame


_SHARPNESS_WIDTH = 320


class CameraHealthMonitor:

    def __init__(
        self,
        fps: float,
        reconnect_delay_seconds: float,
        blank_std_threshold: float = 6.0,
        covered_sharpness_threshold: float = 15.0,
        sustained_seconds: float = 5.0,
        signal_loss_sustained_seconds: float = 9.0,
        thresholds: RuntimeThresholds | None = None,
    ) -> None:
        self.fps = fps
        self.reconnect_delay_seconds = reconnect_delay_seconds
        self.thresholds = thresholds
        self._static_blank_std_threshold = blank_std_threshold
        self._static_covered_sharpness_threshold = covered_sharpness_threshold
        self._static_sustained_seconds = sustained_seconds
        self._static_signal_loss_sustained_seconds = signal_loss_sustained_seconds

        self._blank_streak = 0
        self._signal_loss_streak = 0
        self._active_reason: str | None = None

    @property
    def blank_std_threshold(self) -> float:
        if self.thresholds:
            return self.thresholds.camera_blank_std_threshold
        return self._static_blank_std_threshold

    @property
    def covered_sharpness_threshold(self) -> float:
        if self.thresholds:
            return self.thresholds.camera_covered_sharpness_threshold
        return self._static_covered_sharpness_threshold

    @property
    def sustained_frames(self) -> int:
        seconds = self.thresholds.camera_health_sustained_seconds if self.thresholds else self._static_sustained_seconds
        return max(1, round(seconds * self.fps))

    @property
    def sustained_reconnect_attempts(self) -> int:
        if self.thresholds:
            seconds = self.thresholds.camera_signal_loss_sustained_seconds
        else:
            seconds = self._static_signal_loss_sustained_seconds
        return max(1, round(seconds / self.reconnect_delay_seconds))

    @staticmethod
    def _sharpness(gray) -> float:
        if gray.shape[1] > _SHARPNESS_WIDTH:
            height = round(gray.shape[0] * _SHARPNESS_WIDTH / gray.shape[1])
            gray = cv2.resize(gray, (_SHARPNESS_WIDTH, height), interpolation=cv2.INTER_AREA)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def check(self, frame: Frame) -> str | None:
        if frame.health is not None:
            std, sharpness = frame.health
        else:
            gray = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)
            std, sharpness = float(gray.std()), self._sharpness(gray)

        is_blank = std < self.blank_std_threshold or sharpness < self.covered_sharpness_threshold
        self._blank_streak = self._blank_streak + 1 if is_blank else 0

        self._signal_loss_streak = 0

        if self._blank_streak >= self.sustained_frames:
            return self._maybe_notify("tapada")

        if not is_blank and self._active_reason is not None:
            self._active_reason = None
            return HEALTH_OK
        return None

    def on_signal_loss(self) -> str | None:
        self._blank_streak = 0
        self._signal_loss_streak += 1
        if self._signal_loss_streak >= self.sustained_reconnect_attempts:
            return self._maybe_notify("corte_de_señal")
        return None

    def _maybe_notify(self, reason: str) -> str | None:
        if self._active_reason == reason:
            return None
        self._active_reason = reason
        return reason
