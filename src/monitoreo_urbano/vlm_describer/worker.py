from __future__ import annotations

import tempfile
import time
from pathlib import Path

import cv2
import psycopg

from monitoreo_urbano.common.config import settings
from monitoreo_urbano.common.logging import get_logger
from monitoreo_urbano.common.object_storage import get_clips_client
from monitoreo_urbano.common.repository import (
    claim_next_by_status,
    mark_failed,
    reset_stale_processing,
    update_description,
)
from monitoreo_urbano.common.schemas import EventStatus
from monitoreo_urbano.vlm_describer.describer import (
    VLMDescriber,
    describe_without_physical_traits,
    select_event_frames,
)

logger = get_logger(__name__)


def trigger_timing(event) -> tuple[float | None, float | None]:
    try:
        start = event.start_ts.replace(tzinfo=None)
        end = event.end_ts.replace(tzinfo=None)
        created = event.created_at.replace(tzinfo=None)
        return (created - start).total_seconds(), (end - start).total_seconds()
    except Exception:  # noqa: BLE001
        return None, None


def _read_clip_frames(clip_object_key: str):
    client = get_clips_client()
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / "clip.mp4"
        client.fget_object(settings.s3_bucket_clips, clip_object_key, str(tmp_path))

        cap = cv2.VideoCapture(str(tmp_path))
        frames = []
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(frame)
        finally:
            cap.release()
    return frames


def recover_stuck_events(conn: psycopg.Connection) -> int:
    """Devuelve a la cola los eventos que un worker reclamó para describir y nunca terminó.

    Corre al arrancar y cada `stale_check_interval_seconds`, porque con varios workers uno puede caer mientras
    los demás siguen. Solo toca los que todavía no tienen descripción: los que están siendo indexados son de
    `event_store`.
    """
    reset_count = reset_stale_processing(
        conn,
        back_to_status=EventStatus.PENDING_DESCRIPTION,
        older_than_minutes=settings.vlm_worker_stale_processing_minutes,
        already_described=False,
    )
    if reset_count:
        logger.info("Se recuperaron %d eventos que habían quedado trabados en PROCESSING", reset_count)
    return reset_count


def run_worker_loop(conn: psycopg.Connection, describer: VLMDescriber) -> None:
    recover_stuck_events(conn)
    last_recovery = time.monotonic()

    logger.info("vlm_describer listo, esperando eventos en %s", EventStatus.PENDING_DESCRIPTION.value)
    while True:
        if time.monotonic() - last_recovery >= settings.stale_check_interval_seconds:
            last_recovery = time.monotonic()
            try:
                recover_stuck_events(conn)
            except Exception:  # noqa: BLE001
                logger.exception("No se pudieron recuperar los eventos trabados")

        event = claim_next_by_status(conn, from_status=EventStatus.PENDING_DESCRIPTION)
        if event is None:
            time.sleep(settings.vlm_worker_poll_interval_seconds)
            continue

        logger.info("Describiendo evento %s (%s, track %s)", event.event_id, event.camera_label, event.track_id)
        try:
            frames = _read_clip_frames(event.clip_path)
            if not frames:
                raise RuntimeError(f"El clip {event.clip_path} no tiene frames legibles")

            trigger_seconds, duration_seconds = trigger_timing(event)
            result = describe_without_physical_traits(
                describer,
                select_event_frames(frames, trigger_seconds, duration_seconds),
                camera_label=event.camera_label,
                object_classes=event.object_classes,
                trigger_reason=event.trigger_reason,
            )
            update_description(
                conn,
                event_id=event.event_id,
                description_nl=result.description_nl,
                attributes=result.attributes,
                confidence=result.confidence,
            )
            logger.info("Evento %s descrito: [OMITIDO]", event.event_id)
        except Exception:
            logger.exception("Falló la descripción del evento %s, se marca FAILED", event.event_id)
            mark_failed(conn, event.event_id)
