from __future__ import annotations

import time

from monitoreo_urbano.common.config import settings
from monitoreo_urbano.common.logging import get_logger
from monitoreo_urbano.common.repository import claim_next_indexable, mark_failed, reset_stale_processing
from monitoreo_urbano.common.schemas import EventStatus
from monitoreo_urbano.event_store.hybrid_search import EventStore

logger = get_logger(__name__)


def recover_stuck_events(store: EventStore) -> int:
    """Devuelve a DESCRIBED los eventos que quedaron trabados al indexar.

    Solo toca los que ya tienen descripción; los que siguen en descripción son del `vlm_describer`.
    """
    with store.pool.connection() as conn:
        reset_count = reset_stale_processing(
            conn,
            back_to_status=EventStatus.DESCRIBED,
            older_than_minutes=settings.vlm_worker_stale_processing_minutes,
            already_described=True,
        )
    if reset_count:
        logger.info("Se recuperaron %d eventos trabados antes de indexar", reset_count)
    return reset_count


def run_indexer_loop(store: EventStore, poll_interval_seconds: float | None = None) -> None:
    poll_interval = poll_interval_seconds or settings.vlm_worker_poll_interval_seconds

    recover_stuck_events(store)
    last_recovery = time.monotonic()

    logger.info("event_store listo, esperando eventos en %s", EventStatus.DESCRIBED.value)
    while True:
        if time.monotonic() - last_recovery >= settings.stale_check_interval_seconds:
            last_recovery = time.monotonic()
            try:
                recover_stuck_events(store)
            except Exception:  # noqa: BLE001
                logger.exception("No se pudieron recuperar los eventos trabados")

        with store.pool.connection() as conn:
            event = claim_next_indexable(conn)
        if event is None:
            time.sleep(poll_interval)
            continue

        try:
            store.complete_event_description(event)
            logger.info("Evento %s indexado (buscable desde ahora)", event.event_id)
        except Exception:
            logger.exception("Falló la indexación del evento %s, se marca FAILED", event.event_id)
            with store.pool.connection() as conn:
                mark_failed(conn, event.event_id)
