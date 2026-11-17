from __future__ import annotations

import threading
from typing import Protocol

from monitoreo_urbano.common.db import get_connection, init_db
from monitoreo_urbano.common.logging import get_logger
from monitoreo_urbano.common.repository import (
    absorb_review_events,
    beat_cameras,
    complete_recording_event,
    insert_pending_event,
    insert_recording_event,
    set_camera_health,
    upsert_camera,
)
from monitoreo_urbano.common.runtime_config import RuntimeThresholds
from monitoreo_urbano.common.schemas import HEALTH_OK, CameraInfo, EventRecord, EventStatus

logger = get_logger(__name__)


class InstanceStore(Protocol):

    def register_camera(self, camera: CameraInfo) -> None: ...

    def set_health(self, camera_id: str, status: str) -> None: ...

    def insert_event(self, event: EventRecord) -> None: ...

    def open_event(self, event: EventRecord) -> None: ...

    def absorb_review_events(self, camera_id: str, alert_event_id: str, within_seconds: float) -> int: ...

    def complete_event(self, event: EventRecord) -> bool: ...

    def heartbeat(self, camera_ids: list[str], instance: str) -> None: ...

    def refresh_thresholds(self, thresholds: RuntimeThresholds) -> None: ...

    def pending_description_count(self) -> int: ...

    def close(self) -> None: ...


class PostgresInstanceStore:
    """Persistencia de una instancia sobre Postgres.

    Una instancia escribe desde varios hilos (el del detector, los de clips y el de control). Se usa una sola
    conexión y un candado: son pocas escrituras por segundo y así no se entrelazan transacciones.
    """

    def __init__(self) -> None:
        self._conn = get_connection()
        init_db(self._conn)
        self._lock = threading.Lock()

    def register_camera(self, camera: CameraInfo) -> None:
        with self._lock:
            upsert_camera(self._conn, camera)
            set_camera_health(self._conn, camera.camera_id, "corte_de_señal")

    def set_health(self, camera_id: str, status: str) -> None:
        with self._lock:
            set_camera_health(self._conn, camera_id, status)
        if status == HEALTH_OK:
            logger.info("Cámara %s recuperada", camera_id)
        else:
            logger.warning("Problema de cámara detectado en %s: %s", camera_id, status)

    def insert_event(self, event: EventRecord) -> None:
        with self._lock:
            insert_pending_event(self._conn, event)

    def open_event(self, event: EventRecord) -> None:
        with self._lock:
            insert_recording_event(self._conn, event)

    def absorb_review_events(self, camera_id: str, alert_event_id: str, within_seconds: float) -> int:
        with self._lock:
            return absorb_review_events(self._conn, camera_id, alert_event_id, within_seconds)

    def complete_event(self, event: EventRecord) -> bool:
        with self._lock:
            return complete_recording_event(self._conn, event)

    def heartbeat(self, camera_ids: list[str], instance: str) -> None:
        with self._lock:
            beat_cameras(self._conn, camera_ids, instance)

    def refresh_thresholds(self, thresholds: RuntimeThresholds) -> None:
        with self._lock:
            thresholds.refresh(self._conn)

    def pending_description_count(self) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS count FROM events WHERE status = %s", (EventStatus.PENDING_DESCRIPTION.value,)
            ).fetchone()
        return int(row["count"])

    def close(self) -> None:
        with self._lock:
            self._conn.close()
