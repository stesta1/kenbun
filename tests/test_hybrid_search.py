from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta

import pytest

from monitoreo_urbano.common.config import settings
from monitoreo_urbano.common.repository import insert_described_event, set_embedding, set_indexed, set_review
from monitoreo_urbano.common.schemas import EventStatus, ReviewStatus
from monitoreo_urbano.event_store.hybrid_search import EventStore

from .conftest import make_event

DIMENSIONS = 1024
BASE = datetime(2026, 10, 6, 12, 0, 0)


def _unit(axis: int) -> list[float]:
    vector = [0.0] * DIMENSIONS
    vector[axis] = 1.0
    return vector


class FakeEmbedder:
    def embed(self, texts):
        return [_unit(0) for _ in texts]

    def embed_query(self, text):
        return _unit(0)


class FakePool:
    def __init__(self, conn) -> None:
        self._conn = conn

    @contextmanager
    def connection(self):
        yield self._conn


@pytest.fixture
def store(db_conn, monkeypatch):
    monkeypatch.setattr(settings, "hybrid_search_top_k_vector", 3)
    monkeypatch.setattr(settings, "hybrid_search_top_k_fts", 3)
    monkeypatch.setattr(settings, "hybrid_search_final_k", 20)
    return EventStore(embedder=FakeEmbedder(), pool=FakePool(db_conn))


def _indexed(db_conn, event_id: str, camera_id: str, minutes: int, description: str, axis: int) -> None:
    start = BASE + timedelta(minutes=minutes)
    event = make_event(
        event_id=event_id,
        camera_id=camera_id,
        start_ts=start,
        end_ts=start + timedelta(seconds=30),
        status=EventStatus.DESCRIBED,
        description_nl=description,
    )
    insert_described_event(db_conn, event)
    set_embedding(db_conn, event_id, _unit(axis))
    set_indexed(db_conn, event_id)
    set_review(db_conn, event_id, ReviewStatus.CONFIRMED, reviewed_by="operadora-1")


def _ids(results) -> list[str]:
    return sorted(item.event.event_id for item in results)


def _crowd(db_conn, count: int = 6) -> None:
    for i in range(count):
        _indexed(db_conn, f"later-{i}", "cam-01", 600 + i, "choque entre dos autos en la esquina", axis=0)


def test_time_range_is_applied_before_the_ranking(db_conn, store):
    _crowd(db_conn)
    _indexed(db_conn, "inside", "cam-01", 10, "persona caminando", axis=5)

    results = store.hybrid_search("choque", start_ts=BASE, end_ts=BASE + timedelta(minutes=60))

    assert _ids(results) == ["inside"]


def test_without_a_time_range_the_better_matches_win(db_conn, store):
    _crowd(db_conn)
    _indexed(db_conn, "inside", "cam-01", 10, "persona caminando", axis=5)

    results = store.hybrid_search("choque")

    assert "inside" not in _ids(results)
    assert len(results) == 3


def test_start_bound_excludes_events_that_began_earlier(db_conn, store):
    _indexed(db_conn, "early", "cam-01", 0, "choque", axis=0)
    _indexed(db_conn, "late", "cam-01", 30, "choque", axis=0)

    results = store.hybrid_search("choque", start_ts=BASE + timedelta(minutes=10))

    assert _ids(results) == ["late"]


def test_end_bound_excludes_events_that_ended_later(db_conn, store):
    _indexed(db_conn, "early", "cam-01", 0, "choque", axis=0)
    _indexed(db_conn, "late", "cam-01", 30, "choque", axis=0)

    results = store.hybrid_search("choque", end_ts=BASE + timedelta(minutes=10))

    assert _ids(results) == ["early"]


def test_a_window_with_no_events_returns_nothing(db_conn, store):
    _crowd(db_conn)

    results = store.hybrid_search("choque", start_ts=BASE, end_ts=BASE + timedelta(minutes=5))

    assert results == []


def test_camera_and_time_range_combine(db_conn, store):
    _crowd(db_conn)
    _indexed(db_conn, "cam-01-inside", "cam-01", 10, "persona caminando", axis=5)
    _indexed(db_conn, "cam-02-inside", "cam-02", 12, "persona caminando", axis=5)

    results = store.hybrid_search(
        "choque", camera_id="cam-02", start_ts=BASE, end_ts=BASE + timedelta(minutes=60)
    )

    assert _ids(results) == ["cam-02-inside"]


def test_operator_scope_and_time_range_combine(db_conn, store):
    _indexed(db_conn, "cam-01-inside", "cam-01", 10, "choque", axis=0)
    _indexed(db_conn, "cam-03-inside", "cam-03", 11, "choque", axis=0)

    results = store.hybrid_search(
        "choque",
        start_ts=BASE,
        end_ts=BASE + timedelta(minutes=60),
        camera_ids=frozenset({"cam-01"}),
    )

    assert _ids(results) == ["cam-01-inside"]


def test_excluded_events_never_appear_in_the_results(db_conn, store):
    _indexed(db_conn, "keep", "cam-01", 10, "choque", axis=0)
    _indexed(db_conn, "drop", "cam-01", 11, "choque", axis=0)

    results = store.hybrid_search("choque", exclude_event_ids=frozenset({"drop"}))

    assert _ids(results) == ["keep"]


def test_exclusion_is_applied_before_the_ranking_so_the_next_best_takes_its_place(db_conn, store):
    _crowd(db_conn, count=3)
    _indexed(db_conn, "other", "cam-01", 10, "choque", axis=0)

    without = {item.event.event_id for item in store.hybrid_search("choque")}
    results = store.hybrid_search("choque", exclude_event_ids=frozenset(without))

    assert not (set(_ids(results)) & without)


def test_get_events_respects_the_camera_scope(db_conn, store):
    _indexed(db_conn, "visible", "cam-01", 10, "choque", axis=0)
    _indexed(db_conn, "hidden", "cam-02", 11, "choque", axis=0)

    found = store.get_events(["visible", "hidden", "no-existe"], camera_ids=frozenset({"cam-01"}))

    assert [e.event_id for e in found] == ["visible"]
    assert [e.event_id for e in store.get_events(["visible", "hidden"])] == ["visible", "hidden"]


def test_the_category_filter_applies_to_the_search(db_conn, store):
    _indexed(db_conn, "choque", "cam-01", 10, "choque", axis=0)
    _indexed(db_conn, "fuego", "cam-01", 11, "choque", axis=0)
    db_conn.execute("UPDATE events SET trigger_reason = 'fire' WHERE event_id = 'fuego'")

    results = store.hybrid_search("choque", categories=frozenset({"fire"}))

    assert _ids(results) == ["fuego"]


def test_recent_events_are_listed_newest_first_within_the_range_and_scope(db_conn, store):
    _indexed(db_conn, "viejo", "cam-01", 0, "uno", axis=1)
    _indexed(db_conn, "medio", "cam-01", 30, "dos", axis=2)
    _indexed(db_conn, "nuevo", "cam-01", 60, "tres", axis=3)
    _indexed(db_conn, "otra-camara", "cam-02", 45, "cuatro", axis=4)

    events = store.recent_events(
        start_ts=BASE + timedelta(minutes=10),
        camera_ids=frozenset({"cam-01"}),
        exclude_event_ids=frozenset({"nuevo"}),
    )

    assert [e.event_id for e in events] == ["medio"]
    assert [e.event_id for e in store.recent_events(limit=2)] == ["nuevo", "otra-camara"]


def test_a_dismissed_event_never_appears_in_searches_even_if_it_was_already_indexed(db_conn, store):
    _indexed(db_conn, "ok", "cam-01", 0, "choque entre un auto rojo y una camioneta", 0)
    _indexed(db_conn, "descartado", "cam-01", 1, "choque entre un auto azul y un camion", 0)
    set_review(db_conn, "descartado", ReviewStatus.DISMISSED, reviewed_by="operadora-1")

    assert "descartado" not in _ids(store.hybrid_search("choque"))
    assert "ok" in _ids(store.hybrid_search("choque"))
    assert "descartado" not in [e.event_id for e in store.recent_events()]


def test_an_indexed_event_nobody_confirmed_never_appears_in_searches(store, db_conn):
    _indexed(db_conn, "confirmado", "cam-01", 0, "choque de dos autos en la esquina", 0)
    _indexed(db_conn, "sin-confirmar", "cam-01", 1, "choque de dos autos en la esquina", 0)
    db_conn.execute("UPDATE events SET review_status = %s WHERE event_id = %s", (ReviewStatus.PENDING.value, "sin-confirmar"))

    assert _ids(store.hybrid_search("choque de dos autos")) == ["confirmado"]
    assert [e.event_id for e in store.recent_events()] == ["confirmado"]
