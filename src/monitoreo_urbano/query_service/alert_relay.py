from __future__ import annotations

import asyncio
from datetime import datetime

from fastapi import WebSocket

from monitoreo_urbano.common.config import settings
from monitoreo_urbano.common.db import get_connection, init_db
from monitoreo_urbano.common.logging import get_logger
from monitoreo_urbano.common.repository import get_events_updated_after
from monitoreo_urbano.common.schemas import ALERTABLE_REASONS, AlertMessage, CameraHealth, EventRecord, ReviewStatus

logger = get_logger(__name__)


class ConnectionManager:
    def __init__(self) -> None:
        # None = el cliente quiere las alertas de todas las cámaras; un conjunto (incluso vacío) = solo esas.
        self._connections: dict[WebSocket, frozenset[str] | None] = {}

    async def connect(self, websocket: WebSocket, camera_ids: frozenset[str] | None = None) -> None:
        await websocket.accept()
        self.register(websocket, camera_ids)

    def register(self, websocket: WebSocket, camera_ids: frozenset[str] | None = None) -> None:
        self._connections[websocket] = camera_ids

    def disconnect(self, websocket: WebSocket) -> None:
        self._connections.pop(websocket, None)

    async def broadcast(self, message: AlertMessage | CameraHealth) -> None:
        stale: list[WebSocket] = []
        for connection, camera_ids in list(self._connections.items()):
            if camera_ids is not None and message.camera_id not in camera_ids:
                continue
            try:
                await connection.send_json(message.model_dump(mode="json"))
            except Exception:
                stale.append(connection)
        for connection in stale:
            self.disconnect(connection)


def should_alert(event: EventRecord, already_alerted: set[str]) -> bool:
    return (
        event.trigger_reason in ALERTABLE_REASONS
        and event.review_status == ReviewStatus.PENDING
        and event.event_id not in already_alerted
    )


async def run_alert_poll_loop(manager: ConnectionManager) -> None:
    conn = get_connection()
    init_db(conn)
    last_checked = datetime.utcnow()
    already_alerted: set[str] = set()

    try:
        while True:
            await asyncio.sleep(settings.alert_poll_interval_seconds)
            events = get_events_updated_after(conn, last_checked)
            for event in events:
                last_checked = max(last_checked, event.updated_at)
                if not should_alert(event, already_alerted):
                    continue
                already_alerted.add(event.event_id)
                await manager.broadcast(
                    AlertMessage(
                        event_id=event.event_id,
                        camera_id=event.camera_id,
                        camera_label=event.camera_label,
                        timestamp=event.start_ts,
                        status=event.status,
                        trigger_reason=event.trigger_reason,
                        description_nl=event.description_nl,
                    )
                )
    finally:
        conn.close()
