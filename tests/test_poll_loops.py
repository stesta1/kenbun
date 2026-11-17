from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta

import pytest

from tests.conftest import make_event
from monitoreo_urbano.common.repository import beat_cameras, insert_pending_event, set_camera_health, upsert_camera
from monitoreo_urbano.common.schemas import CameraInfo, ReviewStatus
from monitoreo_urbano.query_service import alert_relay, health_relay


class FakeConnection:

    def __init__(self, conn) -> None:
        self._conn = conn

    def __getattr__(self, name):
        return getattr(self._conn, name)

    def close(self) -> None:
        pass


class RecordingManager:
    def __init__(self) -> None:
        self.messages: list = []

    async def broadcast(self, message) -> None:
        self.messages.append(message)


def _run_loop(monkeypatch, module, loop, db_conn, steps) -> RecordingManager:
    manager = RecordingManager()
    monkeypatch.setattr(module, "get_connection", lambda: FakeConnection(db_conn))
    monkeypatch.setattr(module, "init_db", lambda conn: None)
    calls = {"n": 0}

    async def fake_sleep(_seconds):
        index = calls["n"]
        calls["n"] += 1
        if index >= len(steps):
            raise asyncio.CancelledError
        steps[index]()

    monkeypatch.setattr(module.asyncio, "sleep", fake_sleep)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(loop(manager))
    return manager


def test_alert_loop_broadcasts_a_new_pending_incident_once(monkeypatch, db_conn):
    event = make_event(trigger_reason="accident", created_at=datetime.utcnow(), confidence=0.8)
    inserted = []

    def insert():
        event.created_at = datetime.utcnow() + timedelta(seconds=1)
        insert_pending_event(db_conn, event)
        inserted.append(event.event_id)

    manager = _run_loop(monkeypatch, alert_relay, alert_relay.run_alert_poll_loop, db_conn, [insert, lambda: None])

    assert [m.event_id for m in manager.messages] == inserted
    assert manager.messages[0].trigger_reason == "accident"
    assert manager.messages[0].camera_id == "cam-01"


def test_alert_loop_skips_events_that_are_not_alertable(monkeypatch, db_conn):
    discarded = make_event(trigger_reason="accident", review_status=ReviewStatus.DISMISSED)
    weak = make_event(trigger_reason="posible_choque")

    def insert():
        for event in (discarded, weak):
            event.created_at = datetime.utcnow() + timedelta(seconds=1)
            insert_pending_event(db_conn, event)

    manager = _run_loop(monkeypatch, alert_relay, alert_relay.run_alert_poll_loop, db_conn, [insert])

    assert manager.messages == []


def test_health_loop_broadcasts_a_camera_that_changes_status(monkeypatch, db_conn):
    camera = CameraInfo(camera_id="cam-h", label="Cámara H", source="rtsp://localhost:8554/cam-h", comuna="1")
    upsert_camera(db_conn, camera)

    manager = _run_loop(
        monkeypatch,
        health_relay,
        health_relay.run_health_poll_loop,
        db_conn,
        [lambda: (time.sleep(0.05), set_camera_health(db_conn, "cam-h", "tapada"))],
    )

    assert any(m.camera_id == "cam-h" and m.status == "tapada" for m in manager.messages)


def test_health_loop_announces_when_a_camera_stops_and_resumes_being_analysed(monkeypatch, db_conn):
    camera = CameraInfo(camera_id="cam-s", label="Cámara S", source="rtsp://localhost:8554/cam-s", comuna="1")
    upsert_camera(db_conn, camera)

    def heartbeat_expires():
        db_conn.execute("UPDATE cameras SET analysis_heartbeat_at = %s WHERE camera_id = 'cam-s'", (datetime.utcnow() - timedelta(hours=1),))
        db_conn.commit()

    def heartbeat_returns():
        beat_cameras(db_conn, ["cam-s"], "vision-1")

    manager = _run_loop(monkeypatch, health_relay, health_relay.run_health_poll_loop, db_conn, [heartbeat_expires, heartbeat_returns])

    assert [m.status for m in manager.messages if m.camera_id == "cam-s"] == ["sin_analisis", "ok"]
