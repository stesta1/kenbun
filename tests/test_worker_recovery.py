from __future__ import annotations

from contextlib import contextmanager

import pytest

from monitoreo_urbano.common.config import settings
from monitoreo_urbano.common.schemas import EventStatus
from monitoreo_urbano.event_store import consumer
from monitoreo_urbano.vlm_describer import worker


class StopLoop(Exception):
    pass


@pytest.fixture(autouse=True)
def fast_loops(monkeypatch):
    monkeypatch.setattr(settings, "stale_check_interval_seconds", 0.0)
    monkeypatch.setattr(settings, "vlm_worker_poll_interval_seconds", 0.0)


def test_the_vlm_worker_only_recovers_events_that_were_not_described_yet(monkeypatch):
    calls: list[dict] = []
    monkeypatch.setattr(worker, "reset_stale_processing", lambda conn, **kwargs: calls.append(kwargs) or 0)

    worker.recover_stuck_events(object())

    assert calls == [
        {
            "back_to_status": EventStatus.PENDING_DESCRIPTION,
            "older_than_minutes": settings.vlm_worker_stale_processing_minutes,
            "already_described": False,
        }
    ]


def test_the_vlm_worker_recovers_stuck_events_again_while_it_runs(monkeypatch):
    resets: list[int] = []
    claims = {"n": 0}
    monkeypatch.setattr(worker, "reset_stale_processing", lambda conn, **kwargs: resets.append(1) or 0)

    def claim(conn, from_status):
        claims["n"] += 1
        if claims["n"] >= 4:
            raise StopLoop
        return None

    monkeypatch.setattr(worker, "claim_next_by_status", claim)

    with pytest.raises(StopLoop):
        worker.run_worker_loop(object(), describer=None)

    assert len(resets) >= 4


def test_a_failed_recovery_does_not_stop_the_vlm_worker(monkeypatch):
    calls = {"n": 0}

    def reset(conn, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("conexión perdida")
        return 0

    monkeypatch.setattr(worker, "reset_stale_processing", reset)
    claims = {"n": 0}

    def claim(conn, from_status):
        claims["n"] += 1
        if claims["n"] >= 3:
            raise StopLoop
        return None

    monkeypatch.setattr(worker, "claim_next_by_status", claim)

    with pytest.raises(StopLoop):
        worker.run_worker_loop(object(), describer=None)

    assert calls["n"] >= 3


class FakeStore:
    class pool:  # noqa: N801
        @staticmethod
        @contextmanager
        def connection():
            yield object()


def test_the_indexer_only_recovers_events_that_were_already_described(monkeypatch):
    calls: list[dict] = []
    monkeypatch.setattr(consumer, "reset_stale_processing", lambda conn, **kwargs: calls.append(kwargs) or 0)

    consumer.recover_stuck_events(FakeStore())

    assert calls[0]["back_to_status"] == EventStatus.DESCRIBED
    assert calls[0]["already_described"] is True


def test_the_indexer_recovers_stuck_events_again_while_it_runs(monkeypatch):
    resets: list[int] = []
    claims = {"n": 0}
    monkeypatch.setattr(consumer, "reset_stale_processing", lambda conn, **kwargs: resets.append(1) or 0)

    def claim(conn):
        claims["n"] += 1
        if claims["n"] >= 4:
            raise StopLoop
        return None

    monkeypatch.setattr(consumer, "claim_next_indexable", claim)

    with pytest.raises(StopLoop):
        consumer.run_indexer_loop(FakeStore(), poll_interval_seconds=0.001)

    assert len(resets) >= 4
