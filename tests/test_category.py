from __future__ import annotations

from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient

from monitoreo_urbano.common.config import settings
from monitoreo_urbano.common.repository import (
    claim_next_indexable,
    get_by_id,
    get_events_pending_review,
    insert_described_event,
    set_category,
)
from monitoreo_urbano.common.schemas import EventStatus, ReviewStatus
from monitoreo_urbano.query_service import api
from monitoreo_urbano.query_service.alert_relay import ConnectionManager, should_alert

from .conftest import make_event

OPERATORS_YAML = """
operators:
  - name: ana
    key: clave-ana
    cameras: [cam-01]
  - name: beto
    key: clave-beto
"""


def _uncertain(**overrides):
    defaults = dict(
        trigger_reason="posible_altercado",
        status=EventStatus.DESCRIBED,
        description_nl="Dos personas forcejean.",
    )
    defaults.update(overrides)
    return make_event(**defaults)


def test_changing_the_category_updates_it_and_confirms_the_event(db_conn):
    event = _uncertain()
    insert_described_event(db_conn, event)

    set_category(db_conn, event.event_id, "fight", "ana")

    updated = get_by_id(db_conn, event.event_id)
    assert updated.trigger_reason == "fight"
    assert updated.review_status == ReviewStatus.CONFIRMED
    assert updated.reviewed_by == "ana"


def test_the_original_category_is_kept_only_the_first_time(db_conn):
    event = _uncertain()
    insert_described_event(db_conn, event)

    set_category(db_conn, event.event_id, "fight", "ana")
    set_category(db_conn, event.event_id, "accident", "beto")

    updated = get_by_id(db_conn, event.event_id)
    assert updated.trigger_reason == "accident"
    assert updated.trigger_reason_original == "posible_altercado"


def test_an_event_never_reclassified_has_no_original_category(db_conn):
    event = _uncertain()
    insert_described_event(db_conn, event)
    assert get_by_id(db_conn, event.event_id).trigger_reason_original is None


def test_a_reclassified_event_leaves_the_review_queue_and_becomes_indexable(db_conn):
    event = _uncertain()
    insert_described_event(db_conn, event)
    assert [e.event_id for e in get_events_pending_review(db_conn)] == [event.event_id]
    assert claim_next_indexable(db_conn) is None

    set_category(db_conn, event.event_id, "fight", "ana")

    assert get_events_pending_review(db_conn) == []
    claimed = claim_next_indexable(db_conn)
    assert claimed is not None and claimed.event_id == event.event_id


def test_only_a_pending_incident_that_was_not_alerted_yet_triggers_an_alert():
    pending = make_event(trigger_reason="fire", review_status=ReviewStatus.PENDING)
    assert should_alert(pending, set())
    assert not should_alert(pending, {pending.event_id})


def test_an_event_already_reviewed_does_not_alert_again_when_it_is_reclassified():
    reclassified = make_event(trigger_reason="fight", review_status=ReviewStatus.CONFIRMED)
    assert not should_alert(reclassified, set())


def test_a_category_that_does_not_alert_never_triggers_an_alert():
    assert not should_alert(make_event(trigger_reason="posible_altercado"), set())


@pytest.fixture(autouse=True)
def operadores(tmp_path, monkeypatch):
    path = tmp_path / "operators.yaml"
    path.write_text(OPERATORS_YAML, encoding="utf-8")
    monkeypatch.setattr(settings, "operators_config_path", path)
    monkeypatch.setattr(settings, "api_key", "clave-servicio")


@pytest.fixture
def state(monkeypatch):
    events = {
        "uncertain": _uncertain(event_id="uncertain", camera_id="cam-01"),
        "other-cam": _uncertain(event_id="other-cam", camera_id="cam-02"),
    }
    calls: list[tuple] = []

    @contextmanager
    def connection():
        yield object()

    class FakePool:
        def connection(self):
            return connection()

    def fake_set_category(conn, event_id, reason, by):
        calls.append((event_id, reason, by))
        events[event_id] = events[event_id].model_copy(update={"trigger_reason": reason})

    monkeypatch.setattr(api, "get_by_id", lambda conn, event_id: events.get(event_id))
    monkeypatch.setattr(api, "set_category", fake_set_category)
    api.app.state.pool = FakePool()
    api.app.state.connection_manager = ConnectionManager()
    return calls


@pytest.fixture
def client(state):
    return TestClient(api.app, raise_server_exceptions=False)


def post(client, event_id: str, reason: str, key: str = "clave-beto"):
    return client.post(f"/events/{event_id}/category", json={"trigger_reason": reason}, headers={"X-API-Key": key})


def test_the_endpoint_changes_the_category_in_the_name_of_the_operator(client, state):
    response = post(client, "uncertain", "fight")
    assert response.status_code == 200
    assert response.json()["trigger_reason"] == "fight"
    assert state == [("uncertain", "fight", "beto")]


def test_every_selectable_category_is_accepted(client):
    for reason in ("posible_altercado", "fight", "accident", "fire"):
        assert post(client, "uncertain", reason).status_code == 200


def test_an_unknown_category_is_rejected(client, state):
    response = post(client, "uncertain", "robo")
    assert response.status_code == 400
    assert state == []


def test_choosing_the_current_category_changes_nothing(client, state):
    assert post(client, "uncertain", "posible_altercado").status_code == 200
    assert state == []


def test_an_operator_cannot_reclassify_an_event_from_a_camera_it_cannot_see(client, state):
    assert post(client, "other-cam", "fight", key="clave-ana").status_code == 404
    assert state == []


def test_the_category_endpoint_requires_a_key(client):
    response = client.post("/events/uncertain/category", json={"trigger_reason": "fight"})
    assert response.status_code == 401
