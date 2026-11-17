from __future__ import annotations

from datetime import datetime, timedelta

from monitoreo_urbano.common.repository import (
    claim_next_by_status,
    absorb_review_events,
    claim_next_indexable,
    correct_description,
    get_by_id,
    get_events_pending_review,
    get_events_updated_after,
    insert_pending_event,
    set_review,
    update_description,
)
from monitoreo_urbano.common.schemas import EventAttributes, EventStatus, ReviewStatus
from tests.conftest import make_event


def test_insert_and_get_roundtrip(db_conn):
    event = make_event()
    insert_pending_event(db_conn, event)

    fetched = get_by_id(db_conn, event.event_id)
    assert fetched is not None
    assert fetched.event_id == event.event_id
    assert fetched.status == EventStatus.PENDING_DESCRIPTION


def test_claim_next_by_status_moves_to_processing(db_conn):
    event = make_event()
    insert_pending_event(db_conn, event)

    claimed = claim_next_by_status(db_conn, from_status=EventStatus.PENDING_DESCRIPTION)
    assert claimed is not None
    assert claimed.event_id == event.event_id
    assert claimed.status == EventStatus.PROCESSING

    assert claim_next_by_status(db_conn, from_status=EventStatus.PENDING_DESCRIPTION) is None


def test_update_description_moves_to_described_and_is_searchable_by_fts(db_conn):
    event = make_event()
    insert_pending_event(db_conn, event)
    claim_next_by_status(db_conn, from_status=EventStatus.PENDING_DESCRIPTION)

    update_description(
        db_conn,
        event_id=event.event_id,
        description_nl="Un auto azul colisiona con uno blanco en la esquina.",
        attributes=EventAttributes(vehicle_colors=["azul", "blanco"]),
        confidence=0.8,
    )

    fetched = get_by_id(db_conn, event.event_id)
    assert fetched.status == EventStatus.DESCRIBED
    assert "colisiona" in fetched.description_nl
    assert fetched.attributes.vehicle_colors == ["azul", "blanco"]

    cur = db_conn.execute("SELECT id FROM events WHERE search_vector @@ to_tsquery('spanish', 'colisiona')")
    assert cur.fetchone() is not None


def test_set_review_marks_status_without_touching_description(db_conn):
    event = make_event()
    insert_pending_event(db_conn, event)
    claim_next_by_status(db_conn, from_status=EventStatus.PENDING_DESCRIPTION)
    update_description(
        db_conn,
        event_id=event.event_id,
        description_nl="Un auto azul colisiona con uno blanco en la esquina.",
        attributes=EventAttributes(vehicle_colors=["azul", "blanco"]),
        confidence=0.8,
    )

    set_review(db_conn, event.event_id, ReviewStatus.DISMISSED, reviewed_by="operadora-1")

    fetched = get_by_id(db_conn, event.event_id)
    assert fetched.review_status == ReviewStatus.DISMISSED
    assert fetched.reviewed_by == "operadora-1"
    assert fetched.description_edited is False
    assert fetched.description_nl_original is None
    assert fetched.status == EventStatus.DESCRIBED


def test_correct_description_preserves_original_and_requeues_for_indexing(db_conn):
    event = make_event()
    insert_pending_event(db_conn, event)
    claim_next_by_status(db_conn, from_status=EventStatus.PENDING_DESCRIPTION)
    update_description(
        db_conn,
        event_id=event.event_id,
        description_nl="Un auto azul colisiona con uno blanco en la esquina.",
        attributes=EventAttributes(vehicle_colors=["azul", "blanco"]),
        confidence=0.8,
    )
    claim_next_by_status(db_conn, from_status=EventStatus.DESCRIBED)
    fetched = get_by_id(db_conn, event.event_id)
    from monitoreo_urbano.common.repository import set_embedding, set_indexed

    set_embedding(db_conn, event.event_id, [0.1] * 1024)
    set_indexed(db_conn, event.event_id)

    correct_description(
        db_conn, event.event_id, "Un auto rojo colisiona con una moto en la esquina.", reviewed_by="operadora-1"
    )

    fetched = get_by_id(db_conn, event.event_id)
    assert fetched.description_nl == "Un auto rojo colisiona con una moto en la esquina."
    assert fetched.description_nl_original == "Un auto azul colisiona con uno blanco en la esquina."
    assert fetched.description_edited is True
    assert fetched.review_status == ReviewStatus.CONFIRMED
    assert fetched.reviewed_by == "operadora-1"
    assert fetched.status == EventStatus.DESCRIBED

    cur = db_conn.execute("SELECT id FROM events WHERE search_vector @@ to_tsquery('spanish', 'moto')")
    assert cur.fetchone() is not None

    correct_description(db_conn, event.event_id, "Tercera versión de la descripción.", reviewed_by="operadora-2")
    fetched = get_by_id(db_conn, event.event_id)
    assert fetched.description_nl_original == "Un auto azul colisiona con uno blanco en la esquina."


def _describe(db_conn, event, description_nl: str = "Se observa un incidente en la esquina."):
    insert_pending_event(db_conn, event)
    claim_next_by_status(db_conn, from_status=EventStatus.PENDING_DESCRIPTION)
    update_description(
        db_conn,
        event_id=event.event_id,
        description_nl=description_nl,
        attributes=EventAttributes(),
        confidence=0.6,
    )


def test_claim_next_indexable_does_not_claim_an_unconfirmed_alert(db_conn):
    event = make_event(trigger_reason="fight")
    _describe(db_conn, event)

    assert claim_next_indexable(db_conn) is None
    assert get_by_id(db_conn, event.event_id).status == EventStatus.DESCRIBED


def test_claim_next_indexable_claims_a_confirmed_alert(db_conn):
    event = make_event(trigger_reason="fight")
    _describe(db_conn, event)
    set_review(db_conn, event.event_id, ReviewStatus.CONFIRMED, reviewed_by="operadora-1")

    claimed = claim_next_indexable(db_conn)
    assert claimed is not None
    assert claimed.event_id == event.event_id
    assert claimed.status == EventStatus.PROCESSING


def test_claim_next_indexable_skips_unconfirmed_non_alertable_event(db_conn):
    event = make_event(trigger_reason="posible_altercado")
    _describe(db_conn, event)

    assert claim_next_indexable(db_conn) is None
    fetched = get_by_id(db_conn, event.event_id)
    assert fetched.status == EventStatus.DESCRIBED


def test_claim_next_indexable_claims_confirmed_non_alertable_event(db_conn):
    event = make_event(trigger_reason="posible_altercado")
    _describe(db_conn, event)
    set_review(db_conn, event.event_id, ReviewStatus.CONFIRMED, reviewed_by="operadora-1")

    claimed = claim_next_indexable(db_conn)
    assert claimed is not None
    assert claimed.event_id == event.event_id
    assert claimed.status == EventStatus.PROCESSING


def test_claim_next_indexable_never_claims_dismissed_non_alertable_event(db_conn):
    event = make_event(trigger_reason="posible_altercado")
    _describe(db_conn, event)
    set_review(db_conn, event.event_id, ReviewStatus.DISMISSED, reviewed_by="operadora-1")

    assert claim_next_indexable(db_conn) is None
    fetched = get_by_id(db_conn, event.event_id)
    assert fetched.status == EventStatus.DESCRIBED


def test_get_events_pending_review_lists_the_unconfirmed_alerts_and_possible_events(db_conn):
    alertable = make_event(trigger_reason="fight")
    _describe(db_conn, alertable)

    pending = make_event(trigger_reason="posible_altercado")
    _describe(db_conn, pending)

    confirmed = make_event(trigger_reason="posible_altercado")
    _describe(db_conn, confirmed)
    set_review(db_conn, confirmed.event_id, ReviewStatus.CONFIRMED, reviewed_by="operadora-1")

    results = get_events_pending_review(db_conn)
    result_ids = {e.event_id for e in results}
    assert result_ids == {alertable.event_id, pending.event_id}


def test_get_events_updated_after_only_returns_recent_changes(db_conn):
    from datetime import datetime, timedelta

    event = make_event()
    insert_pending_event(db_conn, event)

    cutoff = datetime.utcnow() + timedelta(seconds=1)
    assert get_events_updated_after(db_conn, cutoff) == []


def test_claim_next_indexable_never_claims_a_dismissed_alertable_event(db_conn):
    event = make_event(trigger_reason="accident")
    _describe(db_conn, event)
    set_review(db_conn, event.event_id, ReviewStatus.DISMISSED, reviewed_by="operadora-1")

    assert claim_next_indexable(db_conn) is None


def test_an_alert_absorbs_the_recent_pending_reviews_of_the_same_camera_only(db_conn):
    now = datetime.utcnow()
    reciente = make_event(trigger_reason="posible_choque", camera_id="cam-01", created_at=now - timedelta(seconds=5))
    antiguo = make_event(trigger_reason="posible_choque", camera_id="cam-01", created_at=now - timedelta(seconds=300))
    otra_camara = make_event(trigger_reason="posible_choque", camera_id="cam-02", created_at=now - timedelta(seconds=5))
    confirmado = make_event(trigger_reason="posible_choque", camera_id="cam-01", created_at=now - timedelta(seconds=5))
    alerta = make_event(trigger_reason="accident", camera_id="cam-01")
    for event in (reciente, antiguo, otra_camara, confirmado, alerta):
        _describe(db_conn, event)
    set_review(db_conn, confirmado.event_id, ReviewStatus.CONFIRMED, reviewed_by="operadora-1")

    assert absorb_review_events(db_conn, "cam-01", alerta.event_id, within_seconds=30.0) == 1
    assert get_by_id(db_conn, reciente.event_id).review_status == ReviewStatus.DISMISSED
    assert get_by_id(db_conn, antiguo.event_id).review_status == ReviewStatus.PENDING
    assert get_by_id(db_conn, otra_camara.event_id).review_status == ReviewStatus.PENDING
    assert get_by_id(db_conn, confirmado.event_id).review_status == ReviewStatus.CONFIRMED
    assert get_by_id(db_conn, alerta.event_id).review_status == ReviewStatus.PENDING
