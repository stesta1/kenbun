from __future__ import annotations

import json
from datetime import datetime, timedelta

import psycopg

from monitoreo_urbano.common.config import settings
from monitoreo_urbano.common.schemas import (
    HEALTH_NO_ANALYSIS,
    CameraHealth,
    CameraInfo,
    EventAttributes,
    EventRecord,
    EventStatus,
    ReviewStatus,
    Vehicle,
)


def _row_to_event(row: dict) -> EventRecord:
    attrs_raw = json.loads(row["attributes"]) if row["attributes"] else {}
    return EventRecord(
        event_id=row["event_id"],
        camera_id=row["camera_id"],
        camera_label=row["camera_label"],
        track_id=row["track_id"],
        object_classes=json.loads(row["object_classes"]),
        trigger_reason=row["trigger_reason"],
        start_ts=row["start_ts"],
        end_ts=row["end_ts"],
        clip_path=row["clip_path"],
        status=EventStatus(row["status"]),
        description_nl=row["description_nl"],
        attributes=EventAttributes(**attrs_raw),
        confidence=row["confidence"],
        review_status=ReviewStatus(row["review_status"]),
        reviewed_by=row["reviewed_by"],
        description_nl_original=row["description_nl_original"],
        description_edited=row["description_edited"],
        trigger_reason_original=row["trigger_reason_original"],
        vehicles=[Vehicle(**vehicle) for vehicle in json.loads(row["vehicles"])] if row.get("vehicles") else [],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        described_at=row["described_at"],
    )


def insert_pending_event(conn: psycopg.Connection, event: EventRecord) -> None:
    conn.execute(
        """
        INSERT INTO events (
            event_id, camera_id, camera_label, track_id, object_classes, trigger_reason,
            start_ts, end_ts, clip_path, status, confidence,
            review_status, created_at, updated_at
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            event.event_id,
            event.camera_id,
            event.camera_label,
            event.track_id,
            json.dumps(event.object_classes),
            event.trigger_reason,
            event.start_ts,
            event.end_ts,
            event.clip_path,
            EventStatus.PENDING_DESCRIPTION.value,
            event.confidence,
            event.review_status.value,
            event.created_at,
            event.created_at,
        ),
    )
    conn.commit()


def insert_recording_event(conn: psycopg.Connection, event: EventRecord) -> None:
    """Registra el evento en el instante del disparo, antes de que cierre el clip: es lo que dispara el aviso."""
    conn.execute(
        """
        INSERT INTO events (
            event_id, camera_id, camera_label, track_id, object_classes, trigger_reason,
            start_ts, end_ts, clip_path, status, confidence,
            review_status, created_at, updated_at
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            event.event_id,
            event.camera_id,
            event.camera_label,
            event.track_id,
            json.dumps(event.object_classes),
            event.trigger_reason,
            event.start_ts,
            event.end_ts,
            event.clip_path,
            EventStatus.RECORDING.value,
            event.confidence,
            event.review_status.value,
            event.created_at,
            event.created_at,
        ),
    )
    conn.commit()


def complete_recording_event(conn: psycopg.Connection, event: EventRecord) -> bool:
    """Cierra un evento en grabación: fija el clip definitivo y lo deja en la cola de descripción.

    Devuelve False si el evento no estaba en grabación (por ejemplo, si no se pudo registrar al dispararse).
    """
    cur = conn.execute(
        """
        UPDATE events
        SET end_ts = %s, clip_path = %s, object_classes = %s, confidence = %s, status = %s, updated_at = %s
        WHERE event_id = %s AND status = %s
        """,
        (
            event.end_ts,
            event.clip_path,
            json.dumps(event.object_classes),
            event.confidence,
            EventStatus.PENDING_DESCRIPTION.value,
            datetime.utcnow(),
            event.event_id,
            EventStatus.RECORDING.value,
        ),
    )
    conn.commit()
    return cur.rowcount > 0


def insert_described_event(conn: psycopg.Connection, event: EventRecord) -> None:
    now = datetime.utcnow()
    conn.execute(
        """
        INSERT INTO events (
            event_id, camera_id, camera_label, track_id, object_classes, trigger_reason,
            start_ts, end_ts, clip_path, status, description_nl, attributes, confidence,
            review_status, created_at, updated_at, described_at
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            event.event_id,
            event.camera_id,
            event.camera_label,
            event.track_id,
            json.dumps(event.object_classes),
            event.trigger_reason,
            event.start_ts,
            event.end_ts,
            event.clip_path,
            EventStatus.DESCRIBED.value,
            event.description_nl,
            json.dumps(event.attributes.model_dump()),
            event.confidence,
            event.review_status.value,
            event.created_at,
            now,
            now,
        ),
    )
    conn.commit()


def get_runtime_overrides(conn: psycopg.Connection) -> dict[str, str]:
    cur = conn.execute("SELECT key, value FROM runtime_config")
    return {row["key"]: row["value"] for row in cur.fetchall()}


def set_runtime_override(conn: psycopg.Connection, key: str, value: float) -> None:
    conn.execute(
        """
        INSERT INTO runtime_config (key, value, updated_at) VALUES (%s, %s, %s)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
        """,
        (key, str(value), datetime.utcnow()),
    )
    conn.commit()


def delete_runtime_override(conn: psycopg.Connection, key: str) -> None:
    conn.execute("DELETE FROM runtime_config WHERE key = %s", (key,))
    conn.commit()


def claim_next_by_status(
    conn: psycopg.Connection, from_status: EventStatus, claim_status: EventStatus = EventStatus.PROCESSING
) -> EventRecord | None:
    # Un solo UPDATE con SKIP LOCKED: varios workers reclaman a la vez sin pisarse ni quedarse sin evento
    # porque otro les ganó entre el SELECT y el UPDATE.
    cur = conn.execute(
        """
        UPDATE events SET status = %s, updated_at = %s
        WHERE event_id = (
            SELECT event_id FROM events WHERE status = %s ORDER BY created_at ASC LIMIT 1 FOR UPDATE SKIP LOCKED
        )
        RETURNING event_id
        """,
        (claim_status.value, datetime.utcnow(), from_status.value),
    )
    row = cur.fetchone()
    conn.commit()
    if row is None:
        return None

    return get_by_id(conn, row["event_id"])


def claim_next_indexable(conn: psycopg.Connection) -> EventRecord | None:
    # Un evento described se indexa solo cuando un operador lo confirmó: la descripción la escribe un modelo
    # que no es del todo fiable, así que ningún evento, ni siquiera una alerta, queda buscable sin esa revisión.
    # Mientras no se confirme se queda en described: no es que la cola se trabe, es que todavía no es indexable.
    cur = conn.execute(
        """
        UPDATE events SET status = %s, updated_at = %s
        WHERE event_id = (
            SELECT event_id FROM events
            WHERE status = %s AND review_status = %s
            ORDER BY created_at ASC LIMIT 1 FOR UPDATE SKIP LOCKED
        )
        RETURNING event_id
        """,
        (
            EventStatus.PROCESSING.value,
            datetime.utcnow(),
            EventStatus.DESCRIBED.value,
            ReviewStatus.CONFIRMED.value,
        ),
    )
    row = cur.fetchone()
    conn.commit()
    if row is None:
        return None

    return get_by_id(conn, row["event_id"])


def get_events_pending_review(conn: psycopg.Connection) -> list[EventRecord]:
    cur = conn.execute(
        """
        SELECT * FROM events
        WHERE status = %s AND review_status = %s
        ORDER BY created_at DESC
        """,
        (EventStatus.DESCRIBED.value, ReviewStatus.PENDING.value),
    )
    return [_row_to_event(row) for row in cur.fetchall()]


def reset_stale_processing(
    conn: psycopg.Connection,
    back_to_status: EventStatus,
    older_than_minutes: float,
    *,
    already_described: bool | None = None,
) -> int:
    """Devuelve a su cola los eventos que un worker reclamó y dejó trabados en PROCESSING.

    Descripción e indexado comparten PROCESSING; los distingue `described_at`, que está vacío mientras el
    VLM no describió el evento. Cada worker pasa `already_described` para recuperar solo lo suyo y no pisar
    el trabajo en curso del otro. None recupera todo.
    """
    stage = ""
    if already_described is True:
        stage = " AND described_at IS NOT NULL"
    elif already_described is False:
        stage = " AND described_at IS NULL"
    cur = conn.execute(
        f"""
        UPDATE events SET status = %s, updated_at = %s
        WHERE status = %s AND updated_at <= (now() - (%s * interval '1 minute')){stage}
        """,
        (
            back_to_status.value,
            datetime.utcnow(),
            EventStatus.PROCESSING.value,
            older_than_minutes,
        ),
    )
    conn.commit()
    return cur.rowcount


def update_description(
    conn: psycopg.Connection,
    event_id: str,
    description_nl: str,
    attributes: EventAttributes,
    confidence: float | None,
) -> None:
    now = datetime.utcnow()
    conn.execute(
        """
        UPDATE events
        SET description_nl = %s, attributes = %s, confidence = COALESCE(%s, confidence), status = %s, described_at = %s, updated_at = %s
        WHERE event_id = %s
        """,
        (
            description_nl,
            json.dumps(attributes.model_dump()),
            confidence,
            EventStatus.DESCRIBED.value,
            now,
            now,
            event_id,
        ),
    )
    conn.commit()


def mark_failed(conn: psycopg.Connection, event_id: str) -> None:
    conn.execute(
        "UPDATE events SET status = %s, updated_at = %s WHERE event_id = %s",
        (EventStatus.FAILED.value, datetime.utcnow(), event_id),
    )
    conn.commit()


def set_indexed(conn: psycopg.Connection, event_id: str) -> None:
    conn.execute(
        "UPDATE events SET status = %s, updated_at = %s WHERE event_id = %s",
        (EventStatus.INDEXED.value, datetime.utcnow(), event_id),
    )
    conn.commit()


def set_embedding(conn: psycopg.Connection, event_id: str, embedding) -> None:
    vector_literal = "[" + ",".join(str(float(v)) for v in embedding) + "]"
    conn.execute("UPDATE events SET embedding = %s::vector WHERE event_id = %s", (vector_literal, event_id))
    conn.commit()


def get_by_id(conn: psycopg.Connection, event_id: str) -> EventRecord | None:
    cur = conn.execute("SELECT * FROM events WHERE event_id = %s", (event_id,))
    row = cur.fetchone()
    return _row_to_event(row) if row else None


def get_events_updated_after(conn: psycopg.Connection, after: datetime) -> list[EventRecord]:
    cur = conn.execute(
        "SELECT * FROM events WHERE updated_at > %s ORDER BY updated_at ASC",
        (after,),
    )
    return [_row_to_event(row) for row in cur.fetchall()]


def absorb_review_events(conn: psycopg.Connection, camera_id: str, alert_event_id: str, within_seconds: float) -> int:
    """Descarta los pendientes de revisión de un choque que la alerta ya cubre: el clip de la alerta incluye los segundos previos.

    Alcanza a los `posible_choque` todavía sin revisar de la misma cámara creados en los últimos `within_seconds`. Los deja descartados
    (no figuran en la cola ni en las búsquedas), con la nota de qué alerta los absorbió. Devuelve cuántos fueron.
    """
    now = datetime.utcnow()
    cur = conn.execute(
        """
        UPDATE events SET review_status = %s, reviewed_by = %s, updated_at = %s
        WHERE camera_id = %s AND trigger_reason = %s AND review_status = %s AND created_at >= %s AND event_id <> %s
        """,
        (ReviewStatus.DISMISSED.value, f"sistema: absorbido por la alerta {alert_event_id[:8]}", now, camera_id, "posible_choque",
         ReviewStatus.PENDING.value, now - timedelta(seconds=within_seconds), alert_event_id),
    )
    conn.commit()
    return cur.rowcount


def set_review(conn: psycopg.Connection, event_id: str, review_status: ReviewStatus, reviewed_by: str) -> None:
    conn.execute(
        "UPDATE events SET review_status = %s, reviewed_by = %s, updated_at = %s WHERE event_id = %s",
        (review_status.value, reviewed_by, datetime.utcnow(), event_id),
    )
    conn.commit()


def set_category(conn: psycopg.Connection, event_id: str, trigger_reason: str, reviewed_by: str) -> None:
    conn.execute(
        """
        UPDATE events
        SET trigger_reason_original = COALESCE(trigger_reason_original, trigger_reason),
            trigger_reason = %s,
            review_status = %s,
            reviewed_by = %s,
            updated_at = %s
        WHERE event_id = %s
        """,
        (trigger_reason, ReviewStatus.CONFIRMED.value, reviewed_by, datetime.utcnow(), event_id),
    )
    conn.commit()


def correct_description(conn: psycopg.Connection, event_id: str, description_nl: str, reviewed_by: str) -> None:
    # description_nl_original solo se fija la primera vez que se corrige (COALESCE contra el valor
    # previo a este UPDATE); correcciones posteriores no lo vuelven a pisar. El status vuelve a
    # DESCRIBED para que event_store lo reclame y recalcule embedding + search_vector con el texto
    # nuevo, reusando el mismo camino de indexado que corre para cualquier evento.
    now = datetime.utcnow()
    conn.execute(
        """
        UPDATE events
        SET description_nl_original = COALESCE(description_nl_original, description_nl),
            description_nl = %s,
            description_edited = TRUE,
            review_status = %s,
            reviewed_by = %s,
            status = %s,
            updated_at = %s
        WHERE event_id = %s
        """,
        (
            description_nl,
            ReviewStatus.CONFIRMED.value,
            reviewed_by,
            EventStatus.DESCRIBED.value,
            now,
            event_id,
        ),
    )
    conn.commit()


def set_event_details(
    conn: psycopg.Connection,
    event_id: str,
    description_nl: str,
    vehicles: list[Vehicle],
    reviewed_by: str,
) -> None:
    """Guarda la revisión completa de un evento: descripción final, vehículos y confirmación.

    Reemplaza la lista de vehículos entera (tabla y copias en el evento) en una sola transacción. El evento vuelve a
    DESCRIBED para que event_store recalcule embedding y search_vector con el texto y los vehículos nuevos.
    description_edited solo se marca si el texto cambió respecto del que había (el que escribió el VLM o una
    corrección previa); cargar solo los vehículos no cuenta como corregir la descripción.
    """
    now = datetime.utcnow()
    vehicles_json = json.dumps([vehicle.model_dump() for vehicle in vehicles], ensure_ascii=False) if vehicles else None
    vehicles_text = "; ".join(vehicle.label for vehicle in vehicles) if vehicles else None
    with conn.transaction():
        conn.execute(
            """
            UPDATE events
            SET description_nl_original = CASE WHEN description_nl IS DISTINCT FROM %s
                                               THEN COALESCE(description_nl_original, description_nl)
                                               ELSE description_nl_original END,
                description_edited = description_edited OR description_nl IS DISTINCT FROM %s,
                description_nl = %s,
                vehicles = %s,
                vehicles_text = %s,
                review_status = %s,
                reviewed_by = %s,
                status = %s,
                updated_at = %s
            WHERE event_id = %s
            """,
            (
                description_nl,
                description_nl,
                description_nl,
                vehicles_json,
                vehicles_text,
                ReviewStatus.CONFIRMED.value,
                reviewed_by,
                EventStatus.DESCRIBED.value,
                now,
                event_id,
            ),
        )
        conn.execute("DELETE FROM event_vehicles WHERE event_id = %s", (event_id,))
        for position, vehicle in enumerate(vehicles, start=1):
            conn.execute(
                "INSERT INTO event_vehicles (event_id, position, vehicle_type, color) VALUES (%s, %s, %s, %s)",
                (event_id, position, vehicle.vehicle_type, vehicle.color),
            )


def upsert_camera(conn: psycopg.Connection, camera: CameraInfo) -> None:
    conn.execute(
        """
        INSERT INTO cameras (camera_id, label, source, comuna, lat, lon, ubicacion, updated_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (camera_id) DO UPDATE SET
            label = excluded.label, source = excluded.source, comuna = excluded.comuna,
            lat = excluded.lat, lon = excluded.lon, ubicacion = excluded.ubicacion, updated_at = excluded.updated_at
        """,
        (camera.camera_id, camera.label, camera.source, camera.comuna, camera.lat, camera.lon, camera.ubicacion, datetime.utcnow()),
    )
    conn.commit()


def set_camera_health(conn: psycopg.Connection, camera_id: str, status: str) -> None:
    now = datetime.utcnow()
    conn.execute(
        """
        UPDATE cameras
        SET health_since = CASE WHEN health_status = %s THEN COALESCE(health_since, %s) ELSE %s END,
            health_status = %s,
            health_updated_at = %s
        WHERE camera_id = %s
        """,
        (status, now, now, status, now, camera_id),
    )
    conn.commit()


def beat_cameras(conn: psycopg.Connection, camera_ids: list[str], instance: str) -> None:
    conn.execute(
        "UPDATE cameras SET analysis_heartbeat_at = %s, analysis_instance = %s WHERE camera_id = ANY(%s)",
        (datetime.utcnow(), instance, camera_ids),
    )
    conn.commit()


# 'sin_analisis' se deriva al leer: si la instancia cae no hay quien escriba el estado. Una cámara que
# nunca latió (base anterior a los latidos) conserva su estado de salud.
_HEALTH_COLUMNS = """
    camera_id, label AS camera_label,
    CASE WHEN analysis_heartbeat_at < %(stale_before)s THEN %(no_analysis)s ELSE health_status END AS status,
    CASE WHEN analysis_heartbeat_at < %(stale_before)s THEN analysis_heartbeat_at
         ELSE COALESCE(health_since, updated_at) END AS since,
    CASE WHEN analysis_heartbeat_at < %(stale_before)s THEN analysis_heartbeat_at
         ELSE COALESCE(health_updated_at, updated_at) END AS updated_at
"""


def _health_params(extra: dict | None = None) -> dict:
    params = {
        "stale_before": datetime.utcnow() - timedelta(seconds=settings.heartbeat_stale_seconds),
        "no_analysis": HEALTH_NO_ANALYSIS,
    }
    return {**params, **(extra or {})}


def get_camera_health(conn: psycopg.Connection) -> list[CameraHealth]:
    cur = conn.execute(f"SELECT {_HEALTH_COLUMNS} FROM cameras ORDER BY camera_id", _health_params())
    return [CameraHealth(**row) for row in cur.fetchall()]


def get_camera_health_updated_after(conn: psycopg.Connection, after: datetime) -> list[CameraHealth]:
    cur = conn.execute(
        f"SELECT {_HEALTH_COLUMNS} FROM cameras WHERE health_updated_at > %(after)s ORDER BY health_updated_at ASC",
        _health_params({"after": after}),
    )
    return [CameraHealth(**row) for row in cur.fetchall()]


def get_all_cameras(conn: psycopg.Connection) -> list[CameraInfo]:
    cur = conn.execute("SELECT camera_id, label, source, comuna, lat, lon, ubicacion FROM cameras ORDER BY camera_id")
    return [CameraInfo(**row) for row in cur.fetchall()]
