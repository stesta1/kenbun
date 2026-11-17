"""Carga el conjunto sintético de la prueba de consulta en una base SEPARADA de la operativa.

Uso:
    python scripts/load_synthetic_events.py --reset
    python scripts/load_synthetic_events.py --db monitoreo_sintetico --dir ../reentreno/medicion/consulta_sintetica
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import psycopg

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
DIR_POR_DEFECTO = ROOT.parent / "reentreno" / "medicion" / "consulta_sintetica"
CLASES = {"fire": ["fire"], "smoke": ["smoke"], "fight": ["person", "person"], "posible_altercado": ["person", "person"]}


def url_de(base_url: str, nombre_base: str) -> str:
    partes = urlsplit(base_url)
    return urlunsplit(partes._replace(path="/" + nombre_base))


def preparar_base(base_url: str, nombre: str, reset: bool) -> str:
    destino = url_de(base_url, nombre)
    with psycopg.connect(url_de(base_url, "postgres"), autocommit=True) as conn:
        existe = conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", (nombre,)).fetchone() is not None
        if existe and reset:
            conn.execute(f'DROP DATABASE "{nombre}" WITH (FORCE)')
            existe = False
        if not existe:
            conn.execute(f'CREATE DATABASE "{nombre}"')
    return destino


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="monitoreo_sintetico")
    parser.add_argument("--dir", type=Path, default=DIR_POR_DEFECTO)
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    from monitoreo_urbano.common.config import settings

    destino = preparar_base(settings.database_url, args.db, args.reset)
    os.environ["MU_DATABASE_URL"] = destino
    settings.database_url = destino

    from monitoreo_urbano.common.db import get_connection
    from monitoreo_urbano.common.repository import get_by_id, insert_described_event, set_event_details, upsert_camera
    from monitoreo_urbano.common.schemas import CameraInfo, EventAttributes, EventRecord, Vehicle
    from monitoreo_urbano.event_store.hybrid_search import EventStore

    store = EventStore()
    conn = get_connection()
    with (args.dir / "camaras.csv").open(encoding="utf-8", newline="") as f:
        for fila in csv.DictReader(f):
            upsert_camera(conn, CameraInfo(
                camera_id=fila["camera_id"], label=fila["label"], source=f"rtsp://localhost:8554/{fila['camera_id']}",
                comuna=fila["comuna"], ubicacion=fila["ubicacion"],
            ))

    with (args.dir / "eventos.csv").open(encoding="utf-8", newline="") as f:
        filas = list(csv.DictReader(f))
    for numero, fila in enumerate(filas, start=1):
        if get_by_id(conn, fila["event_id"]) is not None:
            continue
        inicio, fin = datetime.fromisoformat(fila["start_ts"]), datetime.fromisoformat(fila["end_ts"])
        evento = EventRecord(
            event_id=fila["event_id"], camera_id=fila["camera_id"], camera_label=fila["camera_label"], track_id=numero,
            object_classes=CLASES.get(fila["trigger_reason"], ["car", "car"]), trigger_reason=fila["trigger_reason"],
            start_ts=inicio, end_ts=fin, clip_path=fila["clip_path"], description_nl=fila["description_nl"],
            attributes=EventAttributes(), confidence=1.0,
        )
        insert_described_event(conn, evento)
        vehiculos = [Vehicle(vehicle_type=par.split(":")[0], color=par.split(":")[1]) for par in fila["vehiculos"].split(";") if par]
        set_event_details(conn, evento.event_id, fila["description_nl"], vehiculos, fila["reviewed_by"])
        store.complete_event_description(get_by_id(conn, evento.event_id))
        print(f"{numero:02d}/{len(filas)} {evento.event_id[:8]} {fila['trigger_reason']:<14} {fila['camera_id']}", flush=True)

    total = conn.execute("SELECT count(*) AS n FROM events WHERE status = 'indexed' AND review_status = 'confirmed'").fetchone()["n"]
    print(f"Base «{args.db}»: {total} eventos confirmados e indexados.")
    print(f"Para consultar: MU_DATABASE_URL={destino} python -m monitoreo_urbano.query_service.main")
    store.close()


if __name__ == "__main__":
    main()
