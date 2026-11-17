from __future__ import annotations

import time
from contextlib import contextmanager
from datetime import datetime

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from monitoreo_urbano.common.config import settings
from monitoreo_urbano.common.schemas import CameraHealth
from monitoreo_urbano.query_service import api
from monitoreo_urbano.query_service.alert_relay import ConnectionManager

from .conftest import make_event

OPERATORS_YAML = """
operators:
  - name: ana
    key: clave-ana
    cameras: [cam-01]
  - name: beto
    key: clave-beto
"""


def con(key: str) -> dict:
    return {"X-API-Key": key}


@pytest.fixture(autouse=True)
def operadores(tmp_path, monkeypatch):
    path = tmp_path / "operators.yaml"
    path.write_text(OPERATORS_YAML, encoding="utf-8")
    monkeypatch.setattr(settings, "operators_config_path", path)
    monkeypatch.setattr(settings, "api_key", "clave-servicio")


@pytest.fixture
def estado(monkeypatch):
    events = {
        "ev-1": make_event(event_id="ev-1", camera_id="cam-01", trigger_reason="accident", description_nl="original"),
        "ev-2": make_event(event_id="ev-2", camera_id="cam-01", description_nl=None),
    }
    written: dict = {"overrides": {}}

    @contextmanager
    def connection():
        yield object()

    class FakePool:
        def connection(self):
            return connection()

    monkeypatch.setattr(api, "get_by_id", lambda conn, event_id: events.get(event_id))
    monkeypatch.setattr(api, "set_category", lambda conn, event_id, reason, by: written.update(category=(event_id, reason, by)))
    monkeypatch.setattr(api, "correct_description", lambda conn, event_id, text, by: written.update(correction=(event_id, text, by)))
    monkeypatch.setattr(api, "get_runtime_overrides", lambda conn: dict(written["overrides"]))
    monkeypatch.setattr(api, "set_runtime_override", lambda conn, key, value: written["overrides"].__setitem__(key, value))
    monkeypatch.setattr(api, "delete_runtime_override", lambda conn, key: written["overrides"].pop(key, None))
    api.app.state.pool = FakePool()
    api.app.state.connection_manager = ConnectionManager()
    api.app.state.health_manager = ConnectionManager()
    return written


@pytest.fixture
def client(estado):
    return TestClient(api.app, raise_server_exceptions=False)


def test_health_endpoint_answers_without_credentials(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_the_root_redirects_to_the_interface(client):
    response = client.get("/", follow_redirects=False)

    assert response.status_code in (302, 307)
    assert response.headers["location"] == "/ui/"


def test_clip_endpoints_hand_out_a_temporary_link(client, monkeypatch):
    monkeypatch.setattr(api, "_presigned_clip_url", lambda clip_path: f"https://minio.test/{clip_path}?firma=1")

    link = client.get("/clip-url/cam-01/x.mp4", headers=con("clave-ana"))
    redirect = client.get("/clips/cam-01/x.mp4", headers=con("clave-ana"), follow_redirects=False)

    assert link.json() == {"url": "https://minio.test/cam-01/x.mp4?firma=1"}
    assert redirect.status_code in (302, 307)
    assert redirect.headers["location"].startswith("https://minio.test/cam-01/x.mp4")


def test_the_presigned_link_uses_the_configured_expiry(monkeypatch):
    seen = {}

    class FakeClient:
        def presigned_get_object(self, bucket, key, expires):
            seen.update(bucket=bucket, key=key, expires=expires)
            return "https://minio.test/firmado"

    monkeypatch.setattr(api, "get_clips_client", lambda: FakeClient())

    assert api._presigned_clip_url("cam-01/x.mp4") == "https://minio.test/firmado"
    assert seen["bucket"] == settings.s3_bucket_clips
    assert seen["expires"].total_seconds() == settings.s3_presigned_url_expiry_minutes * 60


def test_changing_to_an_invalid_category_is_rejected(client, estado):
    response = client.post("/events/ev-1/category", json={"trigger_reason": "inventada"}, headers=con("clave-beto"))

    assert response.status_code == 400
    assert "category" not in estado


def test_changing_the_category_records_who_did_it(client, estado):
    response = client.post("/events/ev-1/category", json={"trigger_reason": "fire"}, headers=con("clave-ana"))

    assert response.status_code == 200
    assert estado["category"] == ("ev-1", "fire", "ana")


def test_choosing_the_current_category_changes_nothing(client, estado):
    client.post("/events/ev-1/category", json={"trigger_reason": "accident"}, headers=con("clave-ana"))

    assert "category" not in estado


def test_an_event_without_description_cannot_be_corrected(client, estado):
    response = client.post("/events/ev-2/description", json={"description_nl": "texto"}, headers=con("clave-ana"))

    assert response.status_code == 409
    assert "correction" not in estado


def test_runtime_config_lists_defaults_and_overrides(client, estado):
    estado["overrides"]["yolo_confidence_threshold"] = 0.4

    values = {v["key"]: v for v in client.get("/config/runtime", headers=con("clave-beto")).json()["values"]}

    assert values["yolo_confidence_threshold"]["is_override"] is True
    assert values["yolo_confidence_threshold"]["value"] == 0.4
    assert values["event_dedup_window_seconds"]["is_override"] is False
    assert values["event_dedup_window_seconds"]["value"] == values["event_dedup_window_seconds"]["default"]


def test_a_threshold_can_be_overridden_and_then_reset(client, estado):
    set_response = client.post(
        "/config/runtime", json={"key": "event_dedup_radius_px", "value": 55.0}, headers=con("clave-beto")
    )
    assert set_response.json()["is_override"] is True
    assert estado["overrides"] == {"event_dedup_radius_px": 55.0}

    reset_response = client.delete("/config/runtime/event_dedup_radius_px", headers=con("clave-beto"))

    assert reset_response.json()["is_override"] is False
    assert estado["overrides"] == {}


def test_unknown_thresholds_are_rejected(client, estado):
    update = client.post("/config/runtime", json={"key": "api_key", "value": 1.0}, headers=con("clave-beto"))
    reset = client.delete("/config/runtime/api_key", headers=con("clave-beto"))

    assert update.status_code == 400
    assert reset.status_code == 400
    assert estado["overrides"] == {}


def test_a_restricted_operator_cannot_touch_the_thresholds(client):
    assert client.post("/config/runtime", json={"key": "event_dedup_radius_px", "value": 1.0}, headers=con("clave-ana")).status_code == 403
    assert client.delete("/config/runtime/event_dedup_radius_px", headers=con("clave-ana")).status_code == 403


def test_a_websocket_can_authenticate_with_its_first_message(client, estado):
    with client.websocket_connect("/alerts") as websocket:
        websocket.send_text('{"api_key": "clave-beto"}')
        deadline = time.time() + 3
        while not api.app.state.connection_manager._connections and time.time() < deadline:
            time.sleep(0.01)
        assert len(api.app.state.connection_manager._connections) == 1


def test_a_websocket_with_a_wrong_key_is_closed(client):
    with client.websocket_connect("/alerts") as websocket:
        websocket.send_text('{"api_key": "clave-falsa"}')
        with pytest.raises(WebSocketDisconnect) as closed:
            websocket.receive_text()

    assert closed.value.code == 4401


def test_a_websocket_with_a_malformed_first_message_is_closed(client):
    with client.websocket_connect("/alerts") as websocket:
        websocket.send_text("esto no es json")
        with pytest.raises(WebSocketDisconnect) as closed:
            websocket.receive_text()

    assert closed.value.code == 4401


def test_camera_status_channel_starts_with_a_snapshot_limited_to_the_operator(client, monkeypatch):
    now = datetime.utcnow()
    snapshot = [
        CameraHealth(camera_id="cam-01", camera_label="Cámara 1", status="ok", since=now, updated_at=now),
        CameraHealth(camera_id="cam-02", camera_label="Cámara 2", status="tapada", since=now, updated_at=now),
    ]
    monkeypatch.setattr(api, "get_camera_health", lambda conn: snapshot)

    with client.websocket_connect("/camera-status", headers=con("clave-ana")) as websocket:
        first = websocket.receive_json()

    assert first["camera_id"] == "cam-01"
    assert first["status"] == "ok"


def test_the_service_builds_its_components_on_startup_and_releases_them_on_shutdown(monkeypatch):
    closed = []

    class FakeStore:
        def __init__(self, pool=None) -> None:
            self.pool = pool

        def close(self) -> None:
            closed.append("store")

    class FakeLlm:
        def close(self) -> None:
            closed.append("llm")

    async def idle_loop(manager) -> None:
        import asyncio

        await asyncio.sleep(3600)

    monkeypatch.setattr(api, "get_pool", lambda: "pool")
    monkeypatch.setattr(api, "EventStore", FakeStore)
    monkeypatch.setattr(api, "OpenAICompatibleLLMClient", FakeLlm)
    monkeypatch.setattr(api, "RAGPipeline", lambda store, llm: ("rag", store, llm))
    monkeypatch.setattr(api, "run_alert_poll_loop", idle_loop)
    monkeypatch.setattr(api, "run_health_poll_loop", idle_loop)

    with TestClient(api.app) as client:
        assert client.get("/health").status_code == 200
        assert api.app.state.pool == "pool"
        assert api.app.state.rag_pipeline[0] == "rag"

    assert sorted(closed) == ["llm", "store"]
