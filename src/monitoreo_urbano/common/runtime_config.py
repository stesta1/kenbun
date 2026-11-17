from __future__ import annotations

import psycopg

from monitoreo_urbano.common.config import settings
from monitoreo_urbano.common.repository import get_runtime_overrides

ADJUSTABLE_THRESHOLD_KEYS = (
    "yolo_confidence_threshold",
    "event_dedup_radius_px",
    "event_dedup_window_seconds",
    "camera_blank_std_threshold",
    "camera_covered_sharpness_threshold",
    "camera_health_sustained_seconds",
    "camera_signal_loss_sustained_seconds",
)


class RuntimeThresholds:

    def __init__(self) -> None:
        self._overrides: dict[str, float] = {}

    def refresh(self, conn: psycopg.Connection) -> None:
        raw = get_runtime_overrides(conn)
        self._overrides = {key: float(raw[key]) for key in ADJUSTABLE_THRESHOLD_KEYS if key in raw}

    def _get(self, key: str) -> float:
        return self._overrides.get(key, getattr(settings, key))

    @property
    def yolo_confidence_threshold(self) -> float:
        return self._get("yolo_confidence_threshold")

    @property
    def event_dedup_radius_px(self) -> float:
        return self._get("event_dedup_radius_px")

    @property
    def event_dedup_window_seconds(self) -> float:
        return self._get("event_dedup_window_seconds")

    @property
    def camera_blank_std_threshold(self) -> float:
        return self._get("camera_blank_std_threshold")

    @property
    def camera_covered_sharpness_threshold(self) -> float:
        return self._get("camera_covered_sharpness_threshold")

    @property
    def camera_health_sustained_seconds(self) -> float:
        return self._get("camera_health_sustained_seconds")

    @property
    def camera_signal_loss_sustained_seconds(self) -> float:
        return self._get("camera_signal_loss_sustained_seconds")
