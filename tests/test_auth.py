from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from monitoreo_urbano.common.auth import is_valid_key, with_stream_credentials
from monitoreo_urbano.common.config import settings
from monitoreo_urbano.query_service.alert_relay import ConnectionManager
from monitoreo_urbano.query_service.api import app

KEY = "clave-de-prueba"


@pytest.fixture
def client():
    app.state.connection_manager = ConnectionManager()
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture(autouse=True)
def sin_operadores(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "operators_config_path", tmp_path / "no-existe.yaml")


@pytest.fixture
def con_clave(monkeypatch):
    monkeypatch.setattr(settings, "api_key", KEY)


@pytest.fixture
def sin_clave(monkeypatch):
    monkeypatch.setattr(settings, "api_key", "")


def test_sin_clave_configurada_todo_pasa(sin_clave):
    assert is_valid_key(None)
    assert is_valid_key("cualquiera")


def test_con_clave_configurada_solo_pasa_la_correcta(con_clave):
    assert is_valid_key(KEY)
    assert not is_valid_key("otra")
    assert not is_valid_key("")
    assert not is_valid_key(None)


ENDPOINTS_PROTEGIDOS = [
    ("get", "/auth/check"),
    ("get", "/cameras"),
    ("get", "/events/pending-review"),
    ("get", "/events/abc"),
    ("get", "/clips/cam-01/x.mp4"),
    ("get", "/clip-url/cam-01/x.mp4"),
    ("get", "/config/runtime"),
    ("post", "/chat"),
    ("post", "/events/abc/review"),
    ("post", "/events/abc/description"),
]


@pytest.mark.parametrize(("method", "path"), ENDPOINTS_PROTEGIDOS)
def test_endpoints_protegidos_rechazan_sin_clave(client, con_clave, method, path):
    assert getattr(client, method)(path).status_code == 401


@pytest.mark.parametrize(("method", "path"), ENDPOINTS_PROTEGIDOS)
def test_endpoints_protegidos_rechazan_clave_incorrecta(client, con_clave, method, path):
    assert getattr(client, method)(path, headers={"X-API-Key": "mala"}).status_code == 401


def test_clave_correcta_pasa_la_autenticacion(client, con_clave):
    response = client.get("/auth/check", headers={"X-API-Key": KEY})
    assert response.status_code == 200
    assert response.json() == {"ok": True, "operator": "servicio", "cameras": None}


def test_sin_clave_configurada_no_se_exige(client, sin_clave):
    assert client.get("/auth/check").status_code == 200


def test_health_y_la_interfaz_quedan_abiertos(client, con_clave):
    assert client.get("/health").status_code == 200
    assert client.get("/ui/").status_code == 200
    assert client.get("/ui/js/common.js").status_code == 200


def test_mediamtx_acepta_la_clave_como_password(client, con_clave):
    body = {"user": "operador", "password": KEY, "action": "read", "path": "cam-01", "protocol": "rtsp"}
    assert client.post("/auth/mediamtx", json=body).status_code == 200


def test_mediamtx_acepta_la_clave_como_token(client, con_clave):
    assert client.post("/auth/mediamtx", json={"token": KEY, "action": "read"}).status_code == 200


def test_mediamtx_rechaza_clave_mala_o_ausente(client, con_clave):
    assert client.post("/auth/mediamtx", json={"user": "operador", "password": "mala"}).status_code == 401
    assert client.post("/auth/mediamtx", json={"user": "operador"}).status_code == 401


def test_mediamtx_ignora_campos_que_no_conoce(client, con_clave):
    body = {"password": KEY, "id": "abc", "query": "x=1", "campo_nuevo": 1}
    assert client.post("/auth/mediamtx", json=body).status_code == 200


def _esperar_conexion():
    deadline = time.time() + 3
    while not app.state.connection_manager._connections and time.time() < deadline:
        time.sleep(0.01)


def test_websocket_acepta_la_clave_como_primer_mensaje(client, con_clave):
    with client.websocket_connect("/alerts") as socket:
        socket.send_json({"api_key": KEY})
        _esperar_conexion()
        assert len(app.state.connection_manager._connections) == 1


def test_websocket_acepta_la_clave_en_el_encabezado(client, con_clave):
    with client.websocket_connect("/alerts", headers={"X-API-Key": KEY}):
        assert len(app.state.connection_manager._connections) == 1


def test_websocket_cierra_con_4401_si_la_clave_es_mala(client, con_clave):
    with client.websocket_connect("/alerts") as socket:
        socket.send_json({"api_key": "mala"})
        with pytest.raises(WebSocketDisconnect) as closed:
            socket.receive_text()
    assert closed.value.code == 4401
    assert len(app.state.connection_manager._connections) == 0


def test_websocket_sin_clave_configurada_no_pide_nada(client, sin_clave):
    with client.websocket_connect("/alerts"):
        assert len(app.state.connection_manager._connections) == 1


def test_with_stream_credentials_agrega_usuario_y_clave(con_clave):
    assert with_stream_credentials("rtsp://localhost:8554/cam-01") == f"rtsp://operador:{KEY}@localhost:8554/cam-01"


def test_with_stream_credentials_codifica_caracteres_especiales(monkeypatch):
    monkeypatch.setattr(settings, "api_key", "a/b@c:d")
    assert with_stream_credentials("rtsp://h:8554/c") == "rtsp://operador:a%2Fb%40c%3Ad@h:8554/c"


def test_with_stream_credentials_no_toca_lo_que_no_corresponde(con_clave):
    assert with_stream_credentials("rtsp://u:p@h:8554/c") == "rtsp://u:p@h:8554/c"
    assert with_stream_credentials("0") == "0"
    assert with_stream_credentials("C:/videos/a.mp4") == "C:/videos/a.mp4"


def test_with_stream_credentials_sin_clave_no_cambia_nada(sin_clave):
    assert with_stream_credentials("rtsp://localhost:8554/cam-01") == "rtsp://localhost:8554/cam-01"
