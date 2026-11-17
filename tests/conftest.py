from __future__ import annotations

import uuid
from datetime import datetime, timedelta

import psycopg
import pytest

from monitoreo_urbano.common.db import get_connection, init_db
from monitoreo_urbano.common.schemas import EventRecord


@pytest.fixture
def db_conn() -> psycopg.Connection:
    schema = f"test_{uuid.uuid4().hex}"
    conn = get_connection()
    conn.execute(f"CREATE SCHEMA {schema}")
    conn.execute(f"SET search_path TO {schema}, public")
    init_db(conn)
    yield conn
    conn.execute(f"DROP SCHEMA {schema} CASCADE")
    conn.close()


def make_event(**overrides) -> EventRecord:
    now = datetime.utcnow()
    defaults = dict(
        event_id=str(uuid.uuid4()),
        camera_id="cam-01",
        camera_label="Av. Independencia y Sarandí",
        track_id=1,
        object_classes=["car"],
        start_ts=now - timedelta(seconds=10),
        end_ts=now,
        clip_path="cam-01/test.mp4",
    )
    defaults.update(overrides)
    return EventRecord(**defaults)
