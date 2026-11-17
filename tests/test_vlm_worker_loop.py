from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import make_event
from monitoreo_urbano.common.config import settings
from monitoreo_urbano.common.repository import get_by_id, insert_pending_event
from monitoreo_urbano.common.schemas import EventAttributes, EventStatus
from monitoreo_urbano.vlm_describer import worker
from monitoreo_urbano.vlm_describer.describer import DescriptionResult


class StopLoop(Exception):
    pass


@pytest.fixture(autouse=True)
def fast_loops(monkeypatch):
    monkeypatch.setattr(settings, "stale_check_interval_seconds", 3600.0)
    monkeypatch.setattr(settings, "vlm_worker_poll_interval_seconds", 0.0)

    def stop(_seconds):
        raise StopLoop

    monkeypatch.setattr(worker.time, "sleep", stop)


def test_the_worker_describes_a_pending_event_and_stores_the_description(monkeypatch, db_conn):
    event = make_event(trigger_reason="accident")
    insert_pending_event(db_conn, event)
    frames = [np.zeros((8, 8, 3), dtype=np.uint8)]
    seen = {}

    def fake_describe(describer, clip_frames, **kwargs):
        seen["frames"] = len(clip_frames)
        seen.update(kwargs)
        return DescriptionResult("Dos autos chocan en la esquina.", EventAttributes(vehicle_colors=["rojo"]), 0.7)

    monkeypatch.setattr(worker, "_read_clip_frames", lambda key: frames)
    monkeypatch.setattr(worker, "describe_without_physical_traits", fake_describe)

    with pytest.raises(StopLoop):
        worker.run_worker_loop(db_conn, describer=object())

    stored = get_by_id(db_conn, event.event_id)
    assert stored.status == EventStatus.DESCRIBED
    assert stored.description_nl == "Dos autos chocan en la esquina."
    assert stored.attributes.vehicle_colors == ["rojo"]
    assert seen["frames"] == 1
    assert seen["trigger_reason"] == "accident"


def test_the_worker_marks_the_event_as_failed_when_the_clip_has_no_frames(monkeypatch, db_conn):
    event = make_event()
    insert_pending_event(db_conn, event)
    monkeypatch.setattr(worker, "_read_clip_frames", lambda key: [])

    with pytest.raises(StopLoop):
        worker.run_worker_loop(db_conn, describer=object())

    assert get_by_id(db_conn, event.event_id).status == EventStatus.FAILED


def test_the_worker_marks_the_event_as_failed_when_the_describer_raises(monkeypatch, db_conn):
    event = make_event()
    insert_pending_event(db_conn, event)
    monkeypatch.setattr(worker, "_read_clip_frames", lambda key: [np.zeros((8, 8, 3), dtype=np.uint8)])

    def broken(*args, **kwargs):
        raise RuntimeError("modelo caído")

    monkeypatch.setattr(worker, "describe_without_physical_traits", broken)

    with pytest.raises(StopLoop):
        worker.run_worker_loop(db_conn, describer=object())

    assert get_by_id(db_conn, event.event_id).status == EventStatus.FAILED


def test_the_worker_keeps_running_if_the_periodic_recovery_fails(monkeypatch, db_conn):
    monkeypatch.setattr(settings, "stale_check_interval_seconds", 0.0)
    calls = {"n": 0}

    def recover(conn):
        calls["n"] += 1
        if calls["n"] > 1:
            raise RuntimeError("base no disponible")
        return 0

    monkeypatch.setattr(worker, "recover_stuck_events", recover)

    with pytest.raises(StopLoop):
        worker.run_worker_loop(db_conn, describer=object())

    assert calls["n"] == 2
