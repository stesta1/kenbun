from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2  # noqa: E402

from monitoreo_urbano.common.db import get_connection, init_db  # noqa: E402
from monitoreo_urbano.common.repository import get_events_updated_after  # noqa: E402
from monitoreo_urbano.common.schemas import EventStatus  # noqa: E402
from monitoreo_urbano.vision_pipeline.visualizer import render_text_panel  # noqa: E402

_REFRESH_SECONDS = 1.0
_LOG_WINDOW = "Log de eventos"
_DESC_WINDOW = "Descripciones"

_STATUS_LABEL = {
    EventStatus.PENDING_DESCRIPTION: "clip nuevo, esperando al VLM",
    EventStatus.PROCESSING: "en proceso",
    EventStatus.DESCRIBED: "descripcion lista",
    EventStatus.INDEXED: "indexado, ya se puede consultar",
    EventStatus.FAILED: "fallo",
}


def main() -> None:
    conn = get_connection()
    init_db(conn)

    log_entries: list[str] = []
    description_entries: list[str] = []
    last_checked = datetime.utcnow()

    print("Mirando la base de eventos... 'q' en cualquiera de las dos ventanas para salir.")
    try:
        while True:
            events = get_events_updated_after(conn, last_checked)
            for event in events:
                last_checked = max(last_checked, event.updated_at)
                label = _STATUS_LABEL.get(event.status, event.status.value)
                log_entries.append(
                    f"[{event.updated_at:%H:%M:%S}] {event.camera_label} | {event.event_id[:8]} | {label}"
                )

                if event.status == EventStatus.DESCRIBED and event.description_nl:
                    description_entries.append(
                        f"[{event.updated_at:%H:%M:%S}] {event.camera_label} ({event.start_ts:%H:%M:%S})\n"
                        f"{event.description_nl}"
                    )

            cv2.imshow(_LOG_WINDOW, render_text_panel(log_entries, title=_LOG_WINDOW))
            cv2.imshow(_DESC_WINDOW, render_text_panel(description_entries, title=_DESC_WINDOW, font_scale=0.55))

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            time.sleep(_REFRESH_SECONDS)
    finally:
        cv2.destroyAllWindows()
        conn.close()


if __name__ == "__main__":
    main()
