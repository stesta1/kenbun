from __future__ import annotations

import asyncio
from datetime import datetime

from monitoreo_urbano.common.config import settings
from monitoreo_urbano.common.db import get_connection, init_db
from monitoreo_urbano.common.repository import get_camera_health, get_camera_health_updated_after
from monitoreo_urbano.common.schemas import HEALTH_NO_ANALYSIS
from monitoreo_urbano.query_service.alert_relay import ConnectionManager


async def run_health_poll_loop(manager: ConnectionManager) -> None:
    conn = get_connection()
    init_db(conn)
    last_checked = datetime.utcnow()
    without_analysis: set[str] = set()

    try:
        while True:
            await asyncio.sleep(settings.alert_poll_interval_seconds)
            sent: set[str] = set()
            for health in get_camera_health_updated_after(conn, last_checked):
                last_checked = max(last_checked, health.updated_at)
                sent.add(health.camera_id)
                await manager.broadcast(health)

            # 'sin_analisis' no se escribe: aparece cuando vence el latido y se va cuando vuelve. Ninguna de
            # las dos transiciones cambia `health_updated_at`, así que se detectan comparando con el ciclo anterior.
            snapshot = get_camera_health(conn)
            now_without = {h.camera_id for h in snapshot if h.status == HEALTH_NO_ANALYSIS}
            for health in snapshot:
                if health.camera_id in sent:
                    continue
                if health.camera_id in now_without ^ without_analysis:
                    await manager.broadcast(health)
            without_analysis = now_without
    finally:
        conn.close()
