from __future__ import annotations

from datetime import datetime, timedelta

from tests.conftest import make_event
from monitoreo_urbano.common.runtime_config import RuntimeThresholds
from monitoreo_urbano.common.schemas import CameraInfo, ReviewStatus
from monitoreo_urbano.vision_pipeline import instance_store
from monitoreo_urbano.vision_pipeline.instance_store import PostgresInstanceStore


class FakeConnection:

    def __init__(self, conn) -> None:
        self._conn = conn
        self.closed = False

    def __getattr__(self, name):
        return getattr(self._conn, name)

    def close(self) -> None:
        self.closed = True


def _store(monkeypatch, db_conn) -> tuple[PostgresInstanceStore, FakeConnection]:
    fake = FakeConnection(db_conn)
    monkeypatch.setattr(instance_store, "get_connection", lambda: fake)
    monkeypatch.setattr(instance_store, "init_db", lambda conn: None)
    return PostgresInstanceStore(), fake


def _health(db_conn, camera_id: str) -> str:
    return db_conn.execute("SELECT health_status FROM cameras WHERE camera_id = %s", (camera_id,)).fetchone()["health_status"]


def test_register_camera_stores_it_with_signal_lost_until_frames_arrive(monkeypatch, db_conn):
    store, _ = _store(monkeypatch, db_conn)

    store.register_camera(CameraInfo(camera_id="cam-i", label="Cámara I", source="rtsp://localhost:8554/cam-i", comuna="1"))

    assert _health(db_conn, "cam-i") == "corte_de_señal"


def test_set_health_updates_the_camera_for_a_recovery_and_for_a_problem(monkeypatch, db_conn):
    store, _ = _store(monkeypatch, db_conn)
    store.register_camera(CameraInfo(camera_id="cam-i", label="Cámara I", source="rtsp://localhost:8554/cam-i", comuna="1"))

    store.set_health("cam-i", "ok")
    assert _health(db_conn, "cam-i") == "ok"

    store.set_health("cam-i", "tapada")
    assert _health(db_conn, "cam-i") == "tapada"


def test_event_lifecycle_from_recording_to_pending_description(monkeypatch, db_conn):
    store, _ = _store(monkeypatch, db_conn)
    event = make_event(trigger_reason="accident")

    store.open_event(event)
    assert store.pending_description_count() == 0

    event.clip_path = "cam-01/final.mp4"
    assert store.complete_event(event) is True
    assert store.pending_description_count() == 1
    assert store.complete_event(event) is False


def test_insert_event_goes_straight_to_the_description_queue(monkeypatch, db_conn):
    store, _ = _store(monkeypatch, db_conn)

    store.insert_event(make_event())

    assert store.pending_description_count() == 1


def test_absorb_review_events_dismisses_weak_signals_next_to_an_alert(monkeypatch, db_conn):
    store, _ = _store(monkeypatch, db_conn)
    now = datetime.utcnow()
    weak = make_event(trigger_reason="posible_choque", start_ts=now - timedelta(seconds=5), created_at=now - timedelta(seconds=5))
    alert = make_event(trigger_reason="accident", start_ts=now, created_at=now)
    store.insert_event(weak)
    store.insert_event(alert)

    absorbed = store.absorb_review_events("cam-01", alert.event_id, within_seconds=30)

    assert absorbed == 1
    row = db_conn.execute("SELECT review_status FROM events WHERE event_id = %s", (weak.event_id,)).fetchone()
    assert row["review_status"] == ReviewStatus.DISMISSED.value


def test_heartbeat_marks_the_cameras_as_analysed_by_the_instance(monkeypatch, db_conn):
    store, _ = _store(monkeypatch, db_conn)
    store.register_camera(CameraInfo(camera_id="cam-i", label="Cámara I", source="rtsp://localhost:8554/cam-i", comuna="1"))

    store.heartbeat(["cam-i"], "vision-1")

    row = db_conn.execute("SELECT analysis_instance FROM cameras WHERE camera_id = 'cam-i'").fetchone()
    assert row["analysis_instance"] == "vision-1"


def test_refresh_thresholds_reads_the_overrides_saved_in_the_database(monkeypatch, db_conn):
    store, _ = _store(monkeypatch, db_conn)
    refreshed = []
    thresholds = RuntimeThresholds()
    monkeypatch.setattr(thresholds, "refresh", lambda conn: refreshed.append(conn))

    store.refresh_thresholds(thresholds)

    assert len(refreshed) == 1


def test_close_releases_the_connection(monkeypatch, db_conn):
    store, fake = _store(monkeypatch, db_conn)

    store.close()

    assert fake.closed is True
