from __future__ import annotations

import json
from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient

from monitoreo_urbano.common.config import settings
from monitoreo_urbano.common.repository import (
    get_all_cameras,
    get_by_id,
    insert_described_event,
    set_event_details,
    upsert_camera,
)
from monitoreo_urbano.common.schemas import CameraInfo, EventStatus, ReviewStatus, Vehicle
from monitoreo_urbano.event_store.hybrid_search import _scope_filters
from monitoreo_urbano.query_service import api
from monitoreo_urbano.query_service.alert_relay import ConnectionManager
from monitoreo_urbano.query_service.rag_pipeline import _describe, _parse_plan

from .conftest import make_event


def con(key: str) -> dict:
    return {"X-API-Key": key}


def test_el_color_se_normaliza_y_el_tipo_es_cerrado():
    assert Vehicle(vehicle_type="camion", color="  Blanco   Perla ").color == "blanco perla"
    assert Vehicle(vehicle_type="camion", color="Blanco").label == "camión blanco"
    assert Vehicle(vehicle_type="moto").label == "moto"
    with pytest.raises(ValueError):
        Vehicle(vehicle_type="bicicleta", color="rojo")


@pytest.fixture(autouse=True)
def operadores(tmp_path, monkeypatch):
    path = tmp_path / "operators.yaml"
    path.write_text("operators:\n  - name: ana\n    key: clave-ana\n", encoding="utf-8")
    monkeypatch.setattr(settings, "operators_config_path", path)
    monkeypatch.setattr(settings, "api_key", "clave-servicio")


@pytest.fixture
def estado(monkeypatch):
    events = {
        "choque": make_event(event_id="choque", trigger_reason="accident", description_nl="Dos autos chocan."),
        "incendio": make_event(event_id="incendio", trigger_reason="fire", description_nl="Se quema un galpón."),
        "pendiente": make_event(event_id="pendiente", trigger_reason="accident", description_nl=None),
        "descartado": make_event(
            event_id="descartado", trigger_reason="accident", description_nl="x", review_status=ReviewStatus.DISMISSED
        ),
    }
    written: dict = {}

    @contextmanager
    def connection():
        yield object()

    class FakePool:
        def connection(self):
            return connection()

    monkeypatch.setattr(api, "get_by_id", lambda conn, event_id: events.get(event_id))
    monkeypatch.setattr(
        api,
        "set_event_details",
        lambda conn, event_id, text, vehicles, by: written.update(details=(event_id, text, [v.label for v in vehicles], by)),
    )
    monkeypatch.setattr(api, "set_category", lambda conn, event_id, reason, by: written.update(category=(event_id, reason)))
    monkeypatch.setattr(api, "set_review", lambda conn, event_id, status, by: written.update(review=(event_id, status)))
    api.app.state.pool = FakePool()
    api.app.state.connection_manager = ConnectionManager()
    return written


@pytest.fixture
def client(estado):
    return TestClient(api.app, raise_server_exceptions=False)


def detalles(vehicles=None, **extra):
    return {"description_nl": "Choque entre dos autos.", "vehicles": vehicles if vehicles is not None else [], **extra}


def test_un_choque_se_confirma_con_sus_vehiculos(client, estado):
    vehicles = [{"vehicle_type": "camion", "color": "Blanco"}, {"vehicle_type": "auto", "color": "rojo"}]
    response = client.post("/events/choque/details", json=detalles(vehicles), headers=con("clave-ana"))
    assert response.status_code == 200
    assert estado["details"] == ("choque", "Choque entre dos autos.", ["camión blanco", "auto rojo"], "ana")
    assert "category" not in estado


def test_un_choque_sin_vehiculos_no_se_confirma(client, estado):
    response = client.post("/events/choque/details", json=detalles([]), headers=con("clave-ana"))
    assert response.status_code == 422
    assert "details" not in estado


def test_un_posible_choque_tambien_exige_vehiculos(client, estado, monkeypatch):
    posible = make_event(event_id="posible", trigger_reason="posible_choque", description_nl="algo")
    monkeypatch.setattr(api, "get_by_id", lambda conn, event_id: posible)
    assert client.post("/events/posible/details", json=detalles([]), headers=con("clave-ana")).status_code == 422
    ok = client.post("/events/posible/details", json=detalles([{"vehicle_type": "moto"}]), headers=con("clave-ana"))
    assert ok.status_code == 200


def test_un_incendio_se_confirma_solo_con_descripcion_e_ignora_vehiculos(client, estado):
    response = client.post(
        "/events/incendio/details", json=detalles([{"vehicle_type": "auto", "color": "rojo"}]), headers=con("clave-ana")
    )
    assert response.status_code == 200
    assert estado["details"][2] == []


def test_cambiar_la_categoria_a_choque_exige_vehiculos(client, estado):
    sin = client.post("/events/incendio/details", json=detalles([], trigger_reason="accident"), headers=con("clave-ana"))
    assert sin.status_code == 422
    con_vehiculo = client.post(
        "/events/incendio/details",
        json=detalles([{"vehicle_type": "pickup"}], trigger_reason="accident"),
        headers=con("clave-ana"),
    )
    assert con_vehiculo.status_code == 200
    assert estado["category"] == ("incendio", "accident")


def test_categoria_invalida_y_tipo_invalido(client):
    assert client.post("/events/choque/details", json=detalles([], trigger_reason="humo"), headers=con("clave-ana")).status_code == 400
    assert (
        client.post("/events/choque/details", json=detalles([{"vehicle_type": "bicicleta"}]), headers=con("clave-ana")).status_code
        == 422
    )


def test_no_se_confirma_un_evento_sin_descripcion_ni_uno_descartado(client):
    vehicle = [{"vehicle_type": "auto"}]
    assert client.post("/events/pendiente/details", json=detalles(vehicle), headers=con("clave-ana")).status_code == 409
    assert client.post("/events/descartado/details", json=detalles(vehicle), headers=con("clave-ana")).status_code == 409


def test_confirmar_por_review_ya_no_se_permite_pero_descartar_si(client, estado):
    assert client.post("/events/choque/review", json={"review_status": "confirmed"}, headers=con("clave-ana")).status_code == 400
    assert "review" not in estado
    assert client.post("/events/choque/review", json={"review_status": "dismissed"}, headers=con("clave-ana")).status_code == 200
    assert estado["review"] == ("choque", ReviewStatus.DISMISSED)


def test_el_filtro_de_vehiculo_sale_del_texto_del_operador():
    raw = json.dumps(
        {"busquedas": [{"consulta": "choque", "vehiculo_tipo": "camion", "vehiculo_color": "Blanco", "camara": ["cam-01", "cam-02", "x"]}]}
    )
    search = _parse_plan(raw, "choque con un camión blanco en cam-01 y cam-02", {"cam-01", "cam-02"}, 0).searches[0]
    assert (search.vehicle_type, search.vehicle_color) == ("camion", "blanco")
    assert search.cameras == frozenset({"cam-01", "cam-02"})


def test_una_camara_que_el_operador_no_nombro_se_descarta():
    raw = json.dumps({"busquedas": [{"consulta": "choque", "camara": ["cam-01"]}]})
    places = {"cam-01": "Av. Independencia y Sarandí"}
    for dicho in ("qué pasó en esta cámara en ese período", "qué choques hubo hoy"):
        assert _parse_plan(raw, dicho, {"cam-01"}, 0, places=places).searches[0].cameras == frozenset()
    for dicho in ("qué pasó en cam-01", "qué pasó en la cámara 1", "choques en Independencia", "choques en Av. Independencia y Sarandí"):
        assert _parse_plan(raw, dicho, {"cam-01"}, 0, places=places).searches[0].cameras == frozenset({"cam-01"}), dicho
    assert _parse_plan(raw, "qué pasó en la cámara 10", {"cam-01"}, 0, places=places).searches[0].cameras == frozenset()


def test_un_color_que_el_operador_no_dijo_se_descarta():
    raw = json.dumps({"busquedas": [{"consulta": "choque", "vehiculo_tipo": "bicicleta", "vehiculo_color": "verde"}]})
    search = _parse_plan(raw, "choques de ayer", set(), 0).searches[0]
    assert (search.vehicle_type, search.vehicle_color) == (None, None)


def test_el_evento_muestra_los_vehiculos_que_cargo_el_operador():
    event = make_event(description_nl="Dos autos chocan.", vehicles=[Vehicle(vehicle_type="camion", color="blanco")])
    assert "vehículos: camión blanco" in _describe(event, 1)
    assert "vehículos" not in _describe(make_event(description_nl="Dos autos chocan."), 1)


def test_el_evento_muestra_la_ubicacion_de_su_camara():
    event = make_event(camera_id="cam-03", description_nl="Choque.")
    assert "— Av. Corrientes y Callao" in _describe(event, 1, places={"cam-03": "Av. Corrientes y Callao"})
    assert "—" not in _describe(event, 1, places={"cam-09": "otra"})


def test_el_planificador_ve_la_ubicacion_de_cada_camara():
    from monitoreo_urbano.query_service.rag_pipeline import RAGPipeline

    class Llm:
        prompts: list[str] = []

        def generate(self, system, user, json_mode=False):
            self.prompts.append(user)
            return json.dumps({"busquedas": [{"consulta": "choque", "camara": ["cam-03"]}]})

    llm = Llm()
    camera = CameraInfo(camera_id="cam-03", label="Cámara 3", source="rtsp://x", ubicacion="Av. Corrientes y Callao")
    plan = RAGPipeline(store=None, llm_client=llm)._plan("choque en Callao", [], [], [camera], None)
    assert "ubicación: Av. Corrientes y Callao" in llm.prompts[0]
    assert plan.searches[0].cameras == frozenset({"cam-03"})


def test_el_filtro_de_vehiculo_va_como_subconsulta_con_parametros():
    sql, params = _scope_filters(None, None, None, vehicle_type="camion", vehicle_color="Blanco")
    assert "event_vehicles" in sql and "v.vehicle_type = %s" in sql
    assert "camion" in params and "%blanco%" in params
    assert "event_vehicles" not in _scope_filters(None, None, None)[0]


def test_los_vehiculos_se_guardan_en_la_tabla_y_en_el_evento(db_conn):
    event = make_event(trigger_reason="accident")
    insert_described_event(db_conn, event.model_copy(update={"description_nl": "Dos autos chocan.", "status": EventStatus.INDEXED}))
    set_event_details(
        db_conn,
        event.event_id,
        "Choque entre un camión y un auto.",
        [Vehicle(vehicle_type="camion", color="blanco"), Vehicle(vehicle_type="auto", color="rojo")],
        "ana",
    )
    fetched = get_by_id(db_conn, event.event_id)
    assert [v.label for v in fetched.vehicles] == ["camión blanco", "auto rojo"]
    assert fetched.status == EventStatus.DESCRIBED
    assert fetched.review_status == ReviewStatus.CONFIRMED and fetched.reviewed_by == "ana"
    assert fetched.description_edited and fetched.description_nl_original == "Dos autos chocan."
    rows = db_conn.execute(
        "SELECT position, vehicle_type, color FROM event_vehicles WHERE event_id = %s ORDER BY position", (event.event_id,)
    ).fetchall()
    assert [(r["position"], r["vehicle_type"], r["color"]) for r in rows] == [(1, "camion", "blanco"), (2, "auto", "rojo")]


def test_guardar_de_nuevo_reemplaza_la_lista_y_sin_cambiar_el_texto_no_marca_correccion(db_conn):
    event = make_event(trigger_reason="accident")
    insert_described_event(db_conn, event.model_copy(update={"description_nl": "Dos autos chocan.", "status": EventStatus.INDEXED}))
    set_event_details(db_conn, event.event_id, "Dos autos chocan.", [Vehicle(vehicle_type="auto"), Vehicle(vehicle_type="moto")], "ana")
    set_event_details(db_conn, event.event_id, "Dos autos chocan.", [Vehicle(vehicle_type="pickup", color="gris")], "ana")
    fetched = get_by_id(db_conn, event.event_id)
    assert [v.label for v in fetched.vehicles] == ["pickup gris"]
    assert not fetched.description_edited and fetched.description_nl_original is None
    count = db_conn.execute("SELECT count(*) AS n FROM event_vehicles WHERE event_id = %s", (event.event_id,)).fetchone()["n"]
    assert count == 1


def test_los_vehiculos_entran_al_indice_de_texto(db_conn):
    event = make_event(trigger_reason="accident")
    insert_described_event(db_conn, event.model_copy(update={"description_nl": "Dos vehículos chocan.", "status": EventStatus.INDEXED}))
    set_event_details(db_conn, event.event_id, "Dos vehículos chocan.", [Vehicle(vehicle_type="camioneta", color="verde")], "ana")
    found = db_conn.execute(
        "SELECT event_id FROM events WHERE search_vector @@ to_tsquery('spanish', 'camioneta & verde')"
    ).fetchall()
    assert [r["event_id"] for r in found] == [event.event_id]


def test_la_ubicacion_de_la_camara_se_guarda_y_entra_al_indice(db_conn):
    upsert_camera(
        db_conn,
        CameraInfo(camera_id="cam-01", label="Cámara 1", source="rtsp://x", ubicacion="Av. Santa Fe y Callao"),
    )
    assert get_all_cameras(db_conn)[0].ubicacion == "Av. Santa Fe y Callao"
    insert_described_event(db_conn, make_event(description_nl="Choque."))
    found = db_conn.execute(
        "SELECT 1 FROM events WHERE search_vector @@ to_tsquery('spanish', 'callao')"
    ).fetchall()
    assert len(found) == 1


def test_el_filtro_exacto_por_tipo_y_color_encuentra_el_evento(db_conn):
    from monitoreo_urbano.event_store.hybrid_search import EventStore

    event = make_event(trigger_reason="accident")
    insert_described_event(db_conn, event.model_copy(update={"description_nl": "Dos autos chocan.", "status": EventStatus.INDEXED}))
    set_event_details(db_conn, event.event_id, "Dos autos chocan.", [Vehicle(vehicle_type="camion", color="Blanco")], "ana")
    db_conn.execute("UPDATE events SET status = 'indexed' WHERE event_id = %s", (event.event_id,))

    class Pool:
        @contextmanager
        def connection(self):
            yield db_conn

    store = EventStore.__new__(EventStore)
    store.pool = Pool()
    assert [e.event_id for e in store.recent_events(vehicle_type="camion", vehicle_color="blanco")] == [event.event_id]
    assert store.recent_events(vehicle_type="moto") == []
    assert store.recent_events(vehicle_type="camion", vehicle_color="rojo") == []
