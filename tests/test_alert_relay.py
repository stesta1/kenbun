from __future__ import annotations

import asyncio
from datetime import datetime

from monitoreo_urbano.common.schemas import AlertMessage, CameraHealth, EventStatus
from monitoreo_urbano.query_service.alert_relay import ConnectionManager


class FakeWebSocket:
    def __init__(self) -> None:
        self.received: list[dict] = []

    async def accept(self) -> None:
        pass

    async def send_json(self, payload: dict) -> None:
        self.received.append(payload)


def _alert(camera_id: str) -> AlertMessage:
    return AlertMessage(
        event_id=f"ev-{camera_id}",
        camera_id=camera_id,
        camera_label=camera_id,
        timestamp=datetime(2026, 10, 5, 12, 0, 0),
        status=EventStatus.PENDING_DESCRIPTION,
        trigger_reason="fire",
    )


def _run(manager: ConnectionManager, *cameras: str) -> None:
    async def scenario() -> None:
        for camera in cameras:
            await manager.broadcast(_alert(camera))

    asyncio.run(scenario())


def _connect(manager: ConnectionManager, camera_ids: frozenset[str] | None) -> FakeWebSocket:
    socket = FakeWebSocket()
    asyncio.run(manager.connect(socket, camera_ids))
    return socket


def test_sin_filtro_recibe_todas_las_camaras():
    manager = ConnectionManager()
    socket = _connect(manager, None)

    _run(manager, "cam-01", "cam-02")

    assert [m["camera_id"] for m in socket.received] == ["cam-01", "cam-02"]


def test_con_filtro_solo_recibe_las_camaras_elegidas():
    manager = ConnectionManager()
    socket = _connect(manager, frozenset({"cam-02"}))

    _run(manager, "cam-01", "cam-02", "cam-03")

    assert [m["camera_id"] for m in socket.received] == ["cam-02"]


def test_filtro_vacio_no_recibe_ninguna():
    manager = ConnectionManager()
    socket = _connect(manager, frozenset())

    _run(manager, "cam-01")

    assert socket.received == []


def test_cada_cliente_tiene_su_propio_filtro():
    manager = ConnectionManager()
    solo_1 = _connect(manager, frozenset({"cam-01"}))
    todas = _connect(manager, None)

    _run(manager, "cam-01", "cam-02")

    assert [m["camera_id"] for m in solo_1.received] == ["cam-01"]
    assert [m["camera_id"] for m in todas.received] == ["cam-01", "cam-02"]


def test_el_estado_de_camara_tambien_se_filtra_por_camara():
    manager = ConnectionManager()
    solo_2 = _connect(manager, frozenset({"cam-02"}))
    health = CameraHealth(
        camera_id="cam-01",
        camera_label="cam-01",
        status="corte_de_señal",
        since=datetime(2026, 10, 6, 12, 0, 0),
        updated_at=datetime(2026, 10, 6, 12, 0, 0),
    )

    asyncio.run(manager.broadcast(health))

    assert solo_2.received == []
