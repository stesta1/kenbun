from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from monitoreo_urbano.common.config import settings
from monitoreo_urbano.common.repository import (
    get_camera_health,
    get_camera_health_updated_after,
    set_camera_health,
    upsert_camera,
)
from monitoreo_urbano.common.schemas import CameraHealth, CameraInfo
from monitoreo_urbano.query_service import api
from monitoreo_urbano.query_service.alert_relay import ConnectionManager

OPERATORS_YAML = """
operators:
  - name: ana
    key: clave-ana
    cameras: [cam-01, cam-02]
  - name: beto
    key: clave-beto
"""

AT = datetime(2026, 10, 6, 12, 0, 0)


def _health(camera_id: str, status: str = "ok") -> CameraHealth:
    return CameraHealth(camera_id=camera_id, camera_label=camera_id, status=status, since=AT, updated_at=AT)


STATES = [_health("cam-01"), _health("cam-02", "corte_de_señal"), _health("cam-03", "tapada")]


@pytest.fixture(autouse=True)
def operadores(tmp_path, monkeypatch):
    path = tmp_path / "operators.yaml"
    path.write_text(OPERATORS_YAML, encoding="utf-8")
    monkeypatch.setattr(settings, "operators_config_path", path)
    monkeypatch.setattr(settings, "api_key", "clave-servicio")


@pytest.fixture
def client(monkeypatch):
    @contextmanager
    def connection():
        yield object()

    class FakePool:
        def connection(self):
            return connection()

    monkeypatch.setattr(api, "get_camera_health", lambda conn: STATES)
    api.app.state.pool = FakePool()
    api.app.state.health_manager = ConnectionManager()
    return TestClient(api.app, raise_server_exceptions=False)


def con(key: str) -> dict:
    return {"X-API-Key": key}


def _snapshot(client, key: str, path: str = "/camera-status", expected: int = 0) -> list[dict]:
    with client.websocket_connect(path) as socket:
        socket.send_json({"api_key": key})
        return [socket.receive_json() for _ in range(expected)]


def test_health_endpoint_requires_a_key(client):
    assert client.get("/cameras/health").status_code == 401


def test_health_endpoint_only_returns_the_operators_cameras(client):
    ana = [h["camera_id"] for h in client.get("/cameras/health", headers=con("clave-ana")).json()]
    beto = [h["camera_id"] for h in client.get("/cameras/health", headers=con("clave-beto")).json()]
    assert ana == ["cam-01", "cam-02"]
    assert beto == ["cam-01", "cam-02", "cam-03"]


def test_health_endpoint_reports_status_and_since(client):
    body = client.get("/cameras/health", headers=con("clave-ana")).json()
    faulty = next(h for h in body if h["camera_id"] == "cam-02")
    assert faulty["status"] == "corte_de_señal"
    assert faulty["since"].startswith("2026-10-06T12:00")


def test_status_channel_sends_the_current_state_on_connect(client):
    messages = _snapshot(client, "clave-ana", expected=2)
    assert [(m["camera_id"], m["status"]) for m in messages] == [("cam-01", "ok"), ("cam-02", "corte_de_señal")]


def test_status_channel_never_sends_cameras_outside_the_operators_scope(client):
    messages = _snapshot(client, "clave-ana", expected=2)
    assert "cam-03" not in {m["camera_id"] for m in messages}


def test_status_channel_respects_the_requested_cameras(client):
    messages = _snapshot(client, "clave-beto", path="/camera-status?cameras=cam-03", expected=1)
    assert [m["camera_id"] for m in messages] == ["cam-03"]


def test_status_channel_with_full_access_sends_every_camera(client):
    messages = _snapshot(client, "clave-beto", expected=3)
    assert [m["camera_id"] for m in messages] == ["cam-01", "cam-02", "cam-03"]


def test_status_channel_closes_with_4401_on_a_bad_key(client):
    with client.websocket_connect("/camera-status") as socket:
        socket.send_json({"api_key": "mala"})
        with pytest.raises(WebSocketDisconnect) as closed:
            socket.receive_text()
    assert closed.value.code == 4401


def _camera(camera_id: str = "cam-01") -> CameraInfo:
    return CameraInfo(camera_id=camera_id, label=f"Cámara {camera_id}", source=f"rtsp://x/{camera_id}")


def test_a_new_camera_starts_healthy(db_conn):
    upsert_camera(db_conn, _camera())
    [health] = get_camera_health(db_conn)
    assert health.status == "ok"
    assert health.camera_label == "Cámara cam-01"


def test_set_camera_health_changes_the_status(db_conn):
    upsert_camera(db_conn, _camera())
    set_camera_health(db_conn, "cam-01", "corte_de_señal")
    assert get_camera_health(db_conn)[0].status == "corte_de_señal"
    set_camera_health(db_conn, "cam-01", "ok")
    assert get_camera_health(db_conn)[0].status == "ok"


def test_since_moves_only_when_the_status_changes(db_conn):
    upsert_camera(db_conn, _camera())
    set_camera_health(db_conn, "cam-01", "tapada")
    first = get_camera_health(db_conn)[0].since
    set_camera_health(db_conn, "cam-01", "tapada")
    assert get_camera_health(db_conn)[0].since == first
    set_camera_health(db_conn, "cam-01", "ok")
    assert get_camera_health(db_conn)[0].since > first


def test_updated_after_only_returns_cameras_that_changed(db_conn):
    upsert_camera(db_conn, _camera("cam-01"))
    upsert_camera(db_conn, _camera("cam-02"))
    before = datetime.utcnow() - timedelta(seconds=1)
    set_camera_health(db_conn, "cam-02", "corte_de_señal")
    changed = get_camera_health_updated_after(db_conn, before)
    assert [h.camera_id for h in changed] == ["cam-02"]


def test_health_does_not_create_events(db_conn):
    upsert_camera(db_conn, _camera())
    set_camera_health(db_conn, "cam-01", "corte_de_señal")
    assert db_conn.execute("SELECT COUNT(*) AS n FROM events").fetchone()["n"] == 0
