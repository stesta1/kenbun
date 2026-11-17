from __future__ import annotations

import time
from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from monitoreo_urbano.common import auth
from monitoreo_urbano.common.config import settings
from monitoreo_urbano.common.schemas import CameraInfo, ChatResponse
from monitoreo_urbano.query_service import api
from monitoreo_urbano.query_service.alert_relay import ConnectionManager

from .conftest import make_event

OPERATORS_YAML = """
operators:
  - name: ana
    key: clave-ana
    cameras: [cam-01, cam-02]
  - name: beto
    key: clave-beto
"""

CAMERAS = [CameraInfo(camera_id=c, label=c, source=f"rtsp://x/{c}") for c in ("cam-01", "cam-02", "cam-03")]


@pytest.fixture(autouse=True)
def operadores(tmp_path, monkeypatch):
    path = tmp_path / "operators.yaml"
    path.write_text(OPERATORS_YAML, encoding="utf-8")
    monkeypatch.setattr(settings, "operators_config_path", path)
    monkeypatch.setattr(settings, "api_key", "clave-servicio")


class FakeRag:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def answer(self, **kwargs):
        self.calls.append(kwargs)
        return ChatResponse(answer="ok", events=[])


@pytest.fixture
def estado(monkeypatch):
    events = {
        "ev-1": make_event(event_id="ev-1", camera_id="cam-01", description_nl="descripcion original"),
        "ev-3": make_event(event_id="ev-3", camera_id="cam-03"),
    }
    written: dict = {}

    @contextmanager
    def connection():
        yield object()

    class FakePool:
        def connection(self):
            return connection()

    monkeypatch.setattr(api, "get_all_cameras", lambda conn: CAMERAS)
    monkeypatch.setattr(api, "get_by_id", lambda conn, event_id: events.get(event_id))
    monkeypatch.setattr(api, "get_events_pending_review", lambda conn: list(events.values()))
    monkeypatch.setattr(api, "set_review", lambda conn, event_id, status, by: written.update(review=(event_id, by)))
    monkeypatch.setattr(
        api, "correct_description", lambda conn, event_id, text, by: written.update(correction=(event_id, by))
    )
    api.app.state.pool = FakePool()
    api.app.state.rag_pipeline = FakeRag()
    api.app.state.connection_manager = ConnectionManager()
    written["rag"] = api.app.state.rag_pipeline
    return written


@pytest.fixture
def client(estado):
    return TestClient(api.app, raise_server_exceptions=False)


def esperar_registro(manager: ConnectionManager, cantidad: int = 1) -> None:
    deadline = time.time() + 3
    while len(manager._connections) < cantidad and time.time() < deadline:
        time.sleep(0.01)


def con(key: str) -> dict:
    return {"X-API-Key": key}


def test_la_clave_identifica_al_operador():
    assert auth.identify("clave-ana") == auth.Operator("ana", frozenset({"cam-01", "cam-02"}))
    assert auth.identify("clave-beto") == auth.Operator("beto", None)
    assert auth.identify("clave-servicio") == auth.Operator("servicio", None)
    assert auth.identify("otra") is None
    assert auth.identify(None) is None


def test_auth_check_dice_quien_es_y_que_ve(client):
    assert client.get("/auth/check", headers=con("clave-ana")).json() == {
        "ok": True,
        "operator": "ana",
        "cameras": ["cam-01", "cam-02"],
    }
    assert client.get("/auth/check", headers=con("clave-beto")).json()["cameras"] is None


def test_nombres_repetidos_o_reservados_se_rechazan(tmp_path):
    path = tmp_path / "ops.yaml"
    path.write_text("operators:\n  - {name: a, key: k1}\n  - {name: a, key: k2}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        auth.load_operators(path)
    path.write_text("operators:\n  - {name: servicio, key: k1}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        auth.load_operators(path)


def test_cameras_solo_devuelve_las_del_operador(client):
    ana = [c["camera_id"] for c in client.get("/cameras", headers=con("clave-ana")).json()]
    beto = [c["camera_id"] for c in client.get("/cameras", headers=con("clave-beto")).json()]
    assert ana == ["cam-01", "cam-02"]
    assert beto == ["cam-01", "cam-02", "cam-03"]


def test_evento_de_otra_camara_responde_como_si_no_existiera(client):
    assert client.get("/events/ev-1", headers=con("clave-ana")).status_code == 200
    assert client.get("/events/ev-3", headers=con("clave-ana")).status_code == 404
    assert client.get("/events/ev-3", headers=con("clave-beto")).status_code == 200


def test_pendientes_filtra_por_camara(client):
    ana = [e["event_id"] for e in client.get("/events/pending-review", headers=con("clave-ana")).json()]
    assert ana == ["ev-1"]


def test_clips_de_otra_camara_se_rechazan(client):
    assert client.get("/clip-url/cam-03/x.mp4", headers=con("clave-ana")).status_code == 403
    assert client.get("/clips/cam-03/x.mp4", headers=con("clave-ana")).status_code == 403


def test_chat_acota_la_busqueda_a_las_camaras_de_la_clave(client, estado):
    client.post("/chat", json={"query": "choque"}, headers=con("clave-ana"))
    client.post("/chat", json={"query": "choque"}, headers=con("clave-beto"))
    ana, beto = estado["rag"].calls
    assert ana["camera_ids"] == frozenset({"cam-01", "cam-02"})
    assert beto["camera_ids"] is None


def test_config_runtime_solo_para_acceso_total(client):
    assert client.get("/config/runtime", headers=con("clave-ana")).status_code == 403


def test_la_revision_se_registra_a_nombre_del_operador_de_la_clave(client, estado):
    response = client.post(
        "/events/ev-1/review",
        json={"review_status": "dismissed", "reviewed_by": "otro-nombre"},
        headers=con("clave-ana"),
    )
    assert response.status_code == 200
    assert estado["review"] == ("ev-1", "ana")


def test_la_correccion_se_registra_a_nombre_del_operador_de_la_clave(client, estado):
    client.post("/events/ev-1/description", json={"description_nl": "texto nuevo"}, headers=con("clave-beto"))
    assert estado["correction"] == ("ev-1", "beto")


def test_no_se_puede_revisar_un_evento_de_otra_camara(client, estado):
    response = client.post("/events/ev-3/review", json={"review_status": "dismissed"}, headers=con("clave-ana"))
    assert response.status_code == 404
    assert "review" not in estado


def test_mediamtx_solo_deja_leer_las_camaras_del_operador(client):
    def pedir(path, key="clave-ana", action="read"):
        return client.post("/auth/mediamtx", json={"password": key, "path": path, "action": action}).status_code

    assert pedir("cam-01") == 200
    assert pedir("cam-03") == 403
    assert pedir("cam-01", key="mala") == 401
    assert pedir("cam-03", key="clave-beto") == 200


def test_mediamtx_solo_deja_publicar_a_claves_sin_restriccion(client):
    def publicar(key):
        body = {"password": key, "path": "cam-01", "action": "publish"}
        return client.post("/auth/mediamtx", json=body).status_code

    assert publicar("clave-ana") == 403
    assert publicar("clave-servicio") == 200


def test_alertas_nunca_llegan_de_camaras_fuera_del_alcance(client):
    manager = api.app.state.connection_manager
    with client.websocket_connect("/alerts?cameras=cam-02,cam-03") as socket:
        socket.send_json({"api_key": "clave-ana"})
        esperar_registro(manager)
        assert list(manager._connections.values()) == [frozenset({"cam-02"})]


def test_alertas_sin_filtro_usan_el_alcance_de_la_clave(client):
    manager = api.app.state.connection_manager
    with client.websocket_connect("/alerts") as socket:
        socket.send_json({"api_key": "clave-ana"})
        esperar_registro(manager)
        assert list(manager._connections.values()) == [frozenset({"cam-01", "cam-02"})]


def test_alertas_con_acceso_total_y_sin_filtro_reciben_todas(client):
    manager = api.app.state.connection_manager
    with client.websocket_connect("/alerts") as socket:
        socket.send_json({"api_key": "clave-beto"})
        esperar_registro(manager)
        assert list(manager._connections.values()) == [None]


def test_alertas_con_clave_mala_se_cierran(client):
    with client.websocket_connect("/alerts") as socket:
        socket.send_json({"api_key": "mala"})
        with pytest.raises(WebSocketDisconnect) as closed:
            socket.receive_text()
    assert closed.value.code == 4401
