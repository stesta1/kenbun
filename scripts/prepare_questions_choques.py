"""Fija las 30 preguntas de la prueba de consulta sobre el corpus de choques reales, antes de ejecutar ninguna consulta.

Uso:
    python scripts/prepare_questions_choques.py --eventos ../reentreno/medicion/eventos_choques.json         --corrida ../reentreno/medicion/corrida_choques.json ../reentreno/medicion/corrida_choques_2.json         --out ../reentreno/medicion/preguntas_choques.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from monitoreo_urbano.common.db import get_connection  # noqa: E402

PREVIO_S = 70

CONTENIDO = [
    ("¿Hubo un incendio con humo?", lambda e: e["motivo"] == "fire"),
    ("¿Hubo un choque de noche?", lambda e: e["motivo"] == "accident" and e["hora"] == "night"),
    ("¿Se registró un choque durante una lluvia?", lambda e: e["motivo"] == "accident" and e["clima"] == "rain"),
    ("¿Hubo un choque en una autopista?", lambda e: e["motivo"] == "accident" and e["escena"] == "highway"),
    ("¿Hubo un choque en una intersección con semáforo?", lambda e: e["motivo"] == "accident" and e["escena"] == "signalized_intersection"),
    ("¿Hubo un choque en una autopista con lluvia?", lambda e: e["motivo"] == "accident" and e["escena"] == "highway" and e["clima"] == "rain"),
    ("¿Hubo un choque de noche en una intersección con semáforo?", lambda e: e["motivo"] == "accident" and e["hora"] == "night" and e["escena"] == "signalized_intersection"),
    ("¿Hubo un choque en un estacionamiento?", lambda e: e["motivo"] == "accident" and e["escena"] == "parking_lot"),
    ("¿Se registró un choque en una calle de la ciudad?", lambda e: e["motivo"] == "accident" and e["escena"] == "city_street"),
    ("¿Hubo un choque en una intersección sin semáforo?", lambda e: e["motivo"] == "accident" and e["escena"] == "simple_intersection"),
]

RESTRINGIDAS = [
    ("¿Qué incidentes hubo en esta cámara en ese período?", "lat-cam-1", (0, 2), None),
    ("¿Qué incidentes hubo en esta cámara en ese período?", "lat-cam-1", (3, 5), None),
    ("¿Qué incidentes hubo en esta cámara en ese período?", "lat-cam-1", (6, 9), None),
    ("¿Qué choques hubo en esta cámara en ese período?", "lat-cam-2", (0, 3), "accident"),
    ("¿Qué choques hubo en esta cámara en ese período?", "lat-cam-2", (4, 8), "accident"),
    ("¿Hubo un incendio en esta cámara en ese período?", "lat-cam-1", (15, 15), "fire"),
    ("¿Qué incidentes hubo en esta cámara en ese período?", "lat-cam-2", (9, 12), None),
    ("¿Qué choques hubo en esta cámara en ese período?", "lat-cam-1", (10, 14), "accident"),
    ("¿Qué incidentes hubo en esta cámara en ese período?", "lat-cam-2", (13, 17), None),
    ("¿Qué choques hubo en esta cámara en ese período?", "lat-cam-1", (16, 18), "accident"),
]

SIN_EVIDENCIA = [
    ("¿Hubo una pelea entre personas?", None, None, None),
    ("¿Se registró un camión volcado?", None, None, None),
    ("¿Una motocicleta chocó contra un auto?", None, None, None),
    ("¿Hubo una explosión?", None, None, None),
    ("¿Se detectó un incendio en esta cámara en ese período?", "lat-cam-2", (0, 3), "fire"),
    ("¿Hubo un choque en esta cámara en ese período?", "lat-cam-1", "fuera", None),
    ("¿Se vio a alguien robando algo?", None, None, None),
    ("¿Hubo una inundación en la avenida?", None, None, None),
    ("¿Se registró un incendio en esta cámara en ese período?", "lat-cam-1", (6, 9), "fire"),
    ("¿Hubo un choque en esta cámara en ese período?", "lat-cam-9", "todo", None),
]


def ventana(runs: list[dict], camera: str, span) -> tuple[str, str]:
    mine = sorted((r for r in runs if r["camara"] == camera), key=lambda r: r["t_launch_utc"])
    if span == "fuera":
        return "2026-01-01T00:00:00", "2026-01-02T00:00:00"
    if span == "todo":
        every = sorted(r["t_launch_utc"][:19] for r in runs)
        return every[0], (datetime.fromisoformat(max(r["t_fin_utc"][:19] for r in runs)) + timedelta(minutes=5)).isoformat()
    i, j = span
    j = min(j, len(mine) - 1)
    i = min(i, j)
    desde = datetime.fromisoformat(mine[i]["t_launch_utc"][:19]) - timedelta(seconds=PREVIO_S)
    return desde.isoformat(), mine[j]["t_fin_utc"][:19]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eventos", type=Path, required=True)
    parser.add_argument("--corrida", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    events = json.loads(args.eventos.read_text(encoding="utf-8"))["eventos"]
    runs = [r for path in args.corrida for r in json.loads(path.read_text(encoding="utf-8"))["corridas"]]
    conn = get_connection()

    def in_window(camera: str, desde: str, hasta: str, motivo: str | None) -> list[str]:
        sql = "SELECT event_id FROM events WHERE status = 'indexed' AND camera_id = %s AND start_ts >= %s AND end_ts <= %s"
        params: list = [camera, desde, hasta]
        if motivo:
            sql += " AND trigger_reason = %s"
            params.append(motivo)
        return sorted(r["event_id"] for r in conn.execute(sql, params).fetchall())

    preguntas: list[dict] = []
    movidas: list[str] = []
    sin = list(SIN_EVIDENCIA)
    n = 0
    for texto, pred in CONTENIDO:
        rel = sorted(e for e, info in events.items() if info["estado"] == "indexed" and pred(info))
        if not rel:
            movidas.append(texto)
            sin.append((texto, None, None, None))
            continue
        n += 1
        preguntas.append({"id": f"C{n:02d}", "tipo": "contenido", "pregunta": texto, "relevantes": rel})
    for k, (texto, cam, span, motivo) in enumerate(RESTRINGIDAS, 1):
        desde, hasta = ventana(runs, cam, span)
        preguntas.append({"id": f"R{k:02d}", "tipo": "restringida", "pregunta": texto, "camera_id": cam, "desde": desde, "hasta": hasta,
                          "motivo": motivo, "relevantes": in_window(cam, desde, hasta, motivo)})
    for k, (texto, cam, span, motivo) in enumerate(sin[:10 + len(movidas)], 1):
        rel: list[str] = []
        item = {"id": f"S{k:02d}", "tipo": "sin_evidencia", "pregunta": texto, "camera_id": cam, "desde": None, "hasta": None, "relevantes": rel}
        if cam and span:
            desde, hasta = ventana(runs, cam, span)
            item["desde"], item["hasta"] = desde, hasta
            rel.extend(in_window(cam, desde, hasta, motivo))
        preguntas.append(item)
    while len(preguntas) > 30:
        preguntas.pop()
    for p in preguntas:
        print(p["id"], p["tipo"], len(p["relevantes"]), p["pregunta"][:60])
    print("movidas a ausencia:", movidas)
    args.out.write_text(json.dumps({"preguntas": preguntas, "contenido_movido_a_ausencia": movidas}, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
