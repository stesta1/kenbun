from __future__ import annotations

from contextlib import contextmanager

import pytest

from monitoreo_urbano.common.repository import get_by_id, insert_described_event, upsert_camera
from monitoreo_urbano.common.schemas import CameraInfo, EventStatus
from monitoreo_urbano.event_store import consumer, hybrid_search
from monitoreo_urbano.event_store.hybrid_search import EventStore, _build_tsquery, _fts_search

from .conftest import make_event
from .test_hybrid_search import FakeEmbedder, FakePool, store  # noqa: F401  (se reutiliza el almacén de prueba)


class StopLoop(Exception):
    pass


def test_the_text_query_is_split_into_alternative_terms():
    assert _build_tsquery("choque en la avenida") == "choque | en | la | avenida"


def test_a_query_without_words_builds_no_text_search():
    assert _build_tsquery("¿¿ ?? !!") == ""


def test_a_query_without_words_returns_no_text_matches(db_conn):
    assert _fts_search(db_conn, "¿¿ ??", top_k=5, camera_ids=None) == []


def test_completing_an_event_stores_its_embedding_and_marks_it_indexed(db_conn, store):
    event = make_event(status=EventStatus.DESCRIBED, description_nl="Dos autos chocan en la esquina.")
    insert_described_event(db_conn, event)

    store.complete_event_description(event)

    stored = get_by_id(db_conn, event.event_id)
    assert stored.status == EventStatus.INDEXED


def test_an_event_without_description_cannot_be_indexed(store):
    event = make_event(description_nl=None)

    with pytest.raises(AssertionError, match="descripción"):
        store.complete_event_description(event)


def test_an_operator_without_cameras_gets_no_results(db_conn, store):
    event = make_event(status=EventStatus.DESCRIBED, description_nl="choque en la esquina")
    insert_described_event(db_conn, event)
    store.complete_event_description(event)

    assert store.hybrid_search("choque", camera_ids=frozenset()) == []
    assert store.recent_events(camera_ids=frozenset()) == []


def test_asking_for_a_camera_outside_the_scope_returns_nothing(db_conn, store):
    event = make_event(camera_id="cam-01", status=EventStatus.DESCRIBED, description_nl="choque en la esquina")
    insert_described_event(db_conn, event)
    store.complete_event_description(event)

    assert store.hybrid_search("choque", camera_id="cam-01", camera_ids=frozenset({"cam-02"})) == []


def test_list_cameras_returns_the_catalog(db_conn, store):
    upsert_camera(db_conn, CameraInfo(camera_id="cam-z", label="Cámara Z", source="rtsp://x/z", comuna="3"))

    assert [camera.camera_id for camera in store.list_cameras()] == ["cam-z"]


def test_the_store_closes_its_connection_pool(monkeypatch):
    closed = []

    class Pool:
        @contextmanager
        def connection(self):
            yield object()

        def close(self) -> None:
            closed.append(True)

    monkeypatch.setattr(hybrid_search, "init_db", lambda conn: None)

    EventStore(embedder=FakeEmbedder(), pool=Pool()).close()

    assert closed == [True]


class IndexerStore:
    def __init__(self, events) -> None:
        self.pending = list(events)
        self.indexed: list[str] = []
        self.failing: set[str] = set()
        self.pool = self

    @contextmanager
    def connection(self):
        yield object()

    def complete_event_description(self, event) -> None:
        if event.event_id in self.failing:
            raise RuntimeError("no se pudo calcular el embedding")
        self.indexed.append(event.event_id)


@pytest.fixture
def indexer(monkeypatch):
    marked = []
    state = {"claims": 0}

    def sleep(_seconds):
        raise StopLoop

    def claim(conn):
        state["claims"] += 1
        return state["store"].pending.pop(0) if state["store"].pending else None

    monkeypatch.setattr(consumer.time, "sleep", sleep)
    monkeypatch.setattr(consumer, "claim_next_indexable", claim)
    monkeypatch.setattr(consumer, "mark_failed", lambda conn, event_id: marked.append(event_id))
    monkeypatch.setattr(consumer, "reset_stale_processing", lambda conn, **kwargs: 2)
    monkeypatch.setattr(consumer.settings, "stale_check_interval_seconds", 3600.0)
    state["marked"] = marked
    return state


def test_the_indexer_indexes_every_claimed_event_until_the_queue_is_empty(indexer):
    events = [make_event(event_id="ev-1"), make_event(event_id="ev-2")]
    indexer["store"] = store_ = IndexerStore(events)

    with pytest.raises(StopLoop):
        consumer.run_indexer_loop(store_, poll_interval_seconds=0.01)

    assert store_.indexed == ["ev-1", "ev-2"]
    assert indexer["marked"] == []


def test_the_indexer_marks_a_failed_event_and_keeps_going(indexer):
    events = [make_event(event_id="ev-1"), make_event(event_id="ev-2")]
    indexer["store"] = store_ = IndexerStore(events)
    store_.failing = {"ev-1"}

    with pytest.raises(StopLoop):
        consumer.run_indexer_loop(store_, poll_interval_seconds=0.01)

    assert indexer["marked"] == ["ev-1"]
    assert store_.indexed == ["ev-2"]


def test_the_indexer_recovers_stuck_events_when_it_starts(indexer):
    indexer["store"] = store_ = IndexerStore([])

    assert consumer.recover_stuck_events(store_) == 2
