from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from monitoreo_urbano.common.db import get_connection, init_db  # noqa: E402
from monitoreo_urbano.common.repository import get_events_updated_after  # noqa: E402
from monitoreo_urbano.common.schemas import EventStatus  # noqa: E402

_STATUS_LABEL = {
    EventStatus.PENDING_DESCRIPTION: "clip nuevo, esperando al VLM",
    EventStatus.PROCESSING: "en proceso",
    EventStatus.DESCRIBED: "descripción lista",
    EventStatus.INDEXED: "indexado, ya se puede consultar",
    EventStatus.FAILED: "falló",
}


def main() -> None:
    conn = get_connection()
    init_db(conn)
    last_checked = datetime.utcnow()

    print("Mirando la tabla de eventos... (Ctrl+C para salir)\n")
    try:
        while True:
            events = get_events_updated_after(conn, last_checked)
            for event in events:
                last_checked = max(last_checked, event.updated_at)
                label = _STATUS_LABEL.get(event.status, event.status.value)
                print(f"[{event.updated_at:%H:%M:%S}] {event.camera_label} · {event.event_id[:8]} · {label}")
                if event.status == EventStatus.DESCRIBED:
                    print(f"    → {event.description_nl}\n")
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        conn.close()


if __name__ == "__main__":
    main()
