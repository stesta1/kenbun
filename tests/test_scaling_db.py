
from __future__ import annotations

import threading
import uuid
from datetime import datetime, timedelta

from monitoreo_urbano.common.config import settings
from monitoreo_urbano.common.db import get_connection
from monitoreo_urbano.common.repository import (
    beat_cameras,
    claim_next_by_status,
    claim_next_indexable,
    get_by_id,
    get_camera_health,
    get_camera_health_updated_after,
    insert_pending_event,
    reset_stale_processing,
    set_camera_health,
    set_review,
    update_description,
    upsert_camera,
)
from monitoreo_urbano.common.schemas import CameraInfo, EventAttributes, EventStatus, ReviewStatus
from tests.conftest import make_event


def _age(conn, event_id: str, minutes: float) -> None:
    conn.execute(
        "UPDATE events SET updated_at = %s WHERE event_id = %s",
        (datetime.utcnow() - timedelta(minutes=minutes), event_id),
    )


def test_two_workers_never_claim_the_same_event(db_conn):
    ids = []
    for _ in range(12):
        event = make_event()
        insert_pending_event(db_conn, event)
        ids.append(event.event_id)

    search_path = db_conn.execute("SHOW search_path").fetchone()["search_path"]
    claimed: list[str] = []
    lock = threading.Lock()

    def worker() -> None:
        conn = get_connection()
        conn.execute(f"SET search_path TO {search_path}")
        try:
            while (event := claim_next_by_status(conn, from_status=EventStatus.PENDING_DESCRIPTION)) is not None:
                with lock:
                    claimed.append(event.event_id)
        finally:
            conn.close()

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sorted(claimed) == sorted(ids)


def test_claiming_takes_the_oldest_event_first(db_conn):
    first, second = make_event(), make_event()
    insert_pending_event(db_conn, first)
    insert_pending_event(db_conn, second)

    assert claim_next_by_status(db_conn, from_status=EventStatus.PENDING_DESCRIPTION).event_id == first.event_id


def test_recovery_by_stage_does_not_step_on_the_other_worker(db_conn):
    describing = make_event()
    indexing = make_event(trigger_reason="fire")
    insert_pending_event(db_conn, describing)
    insert_pending_event(db_conn, indexing)

    claim_next_by_status(db_conn, from_status=EventStatus.PENDING_DESCRIPTION)
    claim_next_by_status(db_conn, from_status=EventStatus.PENDING_DESCRIPTION)
    update_description(db_conn, indexing.event_id, "Un incendio", EventAttributes(), 0.9)
    set_review(db_conn, indexing.event_id, ReviewStatus.CONFIRMED, reviewed_by="operadora-1")
    claim_next_indexable(db_conn)
    _age(db_conn, describing.event_id, 30)
    _age(db_conn, indexing.event_id, 30)

    reset = reset_stale_processing(
        db_conn, EventStatus.PENDING_DESCRIPTION, older_than_minutes=10, already_described=False
    )

    assert reset == 1
    assert get_by_id(db_conn, describing.event_id).status == EventStatus.PENDING_DESCRIPTION
    assert get_by_id(db_conn, indexing.event_id).status == EventStatus.PROCESSING

    reset = reset_stale_processing(db_conn, EventStatus.DESCRIBED, older_than_minutes=10, already_described=True)

    assert reset == 1
    assert get_by_id(db_conn, indexing.event_id).status == EventStatus.DESCRIBED


def test_recent_processing_is_left_alone(db_conn):
    event = make_event()
    insert_pending_event(db_conn, event)
    claim_next_by_status(db_conn, from_status=EventStatus.PENDING_DESCRIPTION)

    assert reset_stale_processing(db_conn, EventStatus.PENDING_DESCRIPTION, 10, already_described=False) == 0


def _camera(camera_id: str = "cam-01") -> CameraInfo:
    return CameraInfo(camera_id=camera_id, label=f"Cámara {camera_id}", source="rtsp://x")


def test_a_camera_that_never_beat_keeps_its_health_status(db_conn):
    upsert_camera(db_conn, _camera())
    set_camera_health(db_conn, "cam-01", "tapada")

    assert [h.status for h in get_camera_health(db_conn)] == ["tapada"]


def test_a_recent_heartbeat_keeps_the_health_status(db_conn):
    upsert_camera(db_conn, _camera())
    set_camera_health(db_conn, "cam-01", "ok")
    beat_cameras(db_conn, ["cam-01"], "vision-1")

    assert [h.status for h in get_camera_health(db_conn)] == ["ok"]


def test_an_expired_heartbeat_shows_the_camera_as_without_analysis(db_conn):
    upsert_camera(db_conn, _camera())
    set_camera_health(db_conn, "cam-01", "ok")
    beat_cameras(db_conn, ["cam-01"], "vision-1")
    old = datetime.utcnow() - timedelta(seconds=settings.heartbeat_stale_seconds + 10)
    db_conn.execute("UPDATE cameras SET analysis_heartbeat_at = %s", (old,))

    (health,) = get_camera_health(db_conn)

    assert health.status == "sin_analisis"
    assert health.since == old


def test_beating_again_clears_the_without_analysis_state(db_conn):
    upsert_camera(db_conn, _camera())
    set_camera_health(db_conn, "cam-01", "ok")
    db_conn.execute("UPDATE cameras SET analysis_heartbeat_at = %s", (datetime.utcnow() - timedelta(hours=1),))
    assert get_camera_health(db_conn)[0].status == "sin_analisis"

    beat_cameras(db_conn, ["cam-01"], "vision-1")

    assert get_camera_health(db_conn)[0].status == "ok"


def test_only_the_beating_cameras_are_updated(db_conn):
    upsert_camera(db_conn, _camera("cam-01"))
    upsert_camera(db_conn, _camera("cam-02"))
    beat_cameras(db_conn, ["cam-01"], f"vision-{uuid.uuid4().hex[:4]}")

    rows = db_conn.execute("SELECT camera_id, analysis_heartbeat_at FROM cameras ORDER BY camera_id").fetchall()

    assert rows[0]["analysis_heartbeat_at"] is not None
    assert rows[1]["analysis_heartbeat_at"] is None


def test_the_health_stream_uses_the_derived_status_too(db_conn):
    upsert_camera(db_conn, _camera())
    before = datetime.utcnow() - timedelta(minutes=1)
    set_camera_health(db_conn, "cam-01", "ok")
    db_conn.execute("UPDATE cameras SET analysis_heartbeat_at = %s", (datetime.utcnow() - timedelta(hours=1),))

    assert [h.status for h in get_camera_health_updated_after(db_conn, before)] == ["sin_analisis"]
