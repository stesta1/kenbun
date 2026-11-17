"""Fija las 30 preguntas de la prueba de consulta y sus eventos relevantes, antes de ejecutar ninguna consulta.

Uso:
    python scripts/prepare_questions.py --out ../reentreno/medicion/preguntas.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from monitoreo_urbano.common.db import get_connection  # noqa: E402

MEDICION = ROOT.parent / "reentreno" / "medicion"
GAP_S = 40

ESCENAS = {
    ("V3", 0): "curva_de_dia",
    ("V4", 0): "ruta_de_noche_autos_blancos",
    ("V4", 1): "ruta_de_noche_camioneta_por_detras",
    ("V5", 0): "interseccion_camion_blanco",
    ("V8", 0): "auto_naranja_camioneta_gris",
    ("V6", 0): "incendio",
    ("V1", 0): "auto_naranja_camioneta_gris",
    ("V1", 1): "camion_con_carga_y_polvo",
    ("V2", 0): "ruta_de_noche_autos_blancos",
    ("V2", 1): "interseccion_camion_blanco",
}

CONTENIDO = [
    ("¿Hubo un incendio con humo negro?", {"incendio"}),
    ("¿Se detectó fuego en un edificio?", {"incendio"}),
    ("¿Hubo un choque entre un auto naranja y una camioneta gris?", {"auto_naranja_camioneta_gris"}),
    ("¿Ocurrió un choque de noche en una ruta con varios carriles?", {"ruta_de_noche_autos_blancos", "ruta_de_noche_camioneta_por_detras"}),
    ("¿Un camión con carga chocó y levantó polvo?", {"camion_con_carga_y_polvo"}),
    ("¿Hubo un choque en una intersección con un camión blanco al costado?", {"interseccion_camion_blanco"}),
    ("¿Una camioneta chocó por detrás a un auto de noche?", {"ruta_de_noche_camioneta_por_detras"}),
    ("¿Hubo un choque en una curva de ruta durante el día?", {"curva_de_dia"}),
    ("Mostrame los eventos con llamas y una columna de humo.", {"incendio"}),
    ("¿Se registró algún choque entre dos autos blancos?", {"ruta_de_noche_autos_blancos", "curva_de_dia"}),
]

RESTRINGIDAS = [
    ("¿Qué incidentes hubo en esta cámara en ese período?", "lat-cam-1", "2026-10-15T23:28:00", "2026-10-15T23:31:30", None),
    ("¿Qué incidentes hubo en esta cámara en ese período?", "lat-cam-1", "2026-10-15T23:31:30", "2026-10-15T23:36:00", None),
    ("¿Qué incidentes hubo en esta cámara en ese período?", "lat-cam-1", "2026-10-15T23:42:00", "2026-10-15T23:46:30", None),
    ("¿Qué choques hubo en esta cámara en ese período?", "lat-cam-2", "2026-10-15T23:47:00", "2026-10-15T23:51:30", "accident"),
    ("¿Qué choques hubo en esta cámara en ese período?", "lat-cam-2", "2026-10-15T23:52:00", "2026-10-15T23:55:30", "accident"),
    ("¿Hubo un incendio en esta cámara en ese período?", "lat-cam-2", "2026-10-15T23:51:30", "2026-10-15T23:53:30", "fire"),
    ("¿Hubo un incendio en esta cámara en ese período?", "lat-cam-1", "2026-10-15T23:32:00", "2026-10-15T23:35:00", "fire"),
    ("¿Qué incidentes hubo en esta cámara en ese período?", "lat-cam-1", "2026-10-16T00:06:00", "2026-10-16T00:10:00", None),
    ("¿Qué incidentes hubo en esta cámara en ese período?", "lat-cam-2", "2026-10-16T00:08:00", "2026-10-16T00:11:00", None),
    ("¿Qué choques hubo en esta cámara en ese período?", "lat-cam-2", "2026-10-16T00:20:00", "2026-10-16T00:23:00", "accident"),
]

SIN_EVIDENCIA = [
    ("¿Hubo una pelea entre personas?", None, None, None),
    ("¿Se registró un camión volcado?", None, None, None),
    ("¿Una motocicleta chocó contra un auto?", None, None, None),
    ("¿Hubo una explosión?", None, None, None),
    ("¿Se detectó un incendio en esta cámara en ese período?", "lat-cam-2", "2026-10-15T23:47:00", "2026-10-15T23:51:30"),
    ("¿Hubo un choque en esta cámara en ese período?", "lat-cam-1", "2026-01-01T00:00:00", "2026-01-02T00:00:00"),
    ("¿Se vio a alguien robando algo?", None, None, None),
    ("¿Hubo una inundación en la avenida?", None, None, None),
    ("¿Se registró un incendio en esta cámara en ese período?", "lat-cam-1", "2026-10-16T00:06:00", "2026-10-16T00:10:00"),
    ("¿Hubo un choque en esta cámara en ese período?", "lat-cam-9", "2026-10-15T23:00:00", "2026-10-16T01:00:00"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    videos = json.loads((MEDICION / "eventos_videos.json").read_text(encoding="utf-8"))
    conn = get_connection()
    escena: dict[str, str] = {}
    for camera in ("lat-cam-1", "lat-cam-2"):
        rows = conn.execute("SELECT event_id, created_at FROM events WHERE camera_id = %s ORDER BY created_at", (camera,)).fetchall()
        group: list[dict] = []

        def flush() -> None:
            for index, row in enumerate(group):
                escena[row["event_id"]] = ESCENAS[(videos[row["event_id"]]["video"], index)]

        for row in rows:
            if group and (row["created_at"] - group[-1]["created_at"]).total_seconds() > GAP_S:
                flush()
                group = []
            group.append(row)
        flush()

    preguntas: list[dict] = []
    for n, (texto, escenas) in enumerate(CONTENIDO, 1):
        rel = sorted(e for e, s in escena.items() if s in escenas)
        preguntas.append({"id": f"C{n:02d}", "tipo": "contenido", "pregunta": texto, "escenas_relevantes": sorted(escenas), "relevantes": rel})
    for n, (texto, cam, desde, hasta, motivo) in enumerate(RESTRINGIDAS, 1):
        sql = "SELECT event_id FROM events WHERE camera_id = %s AND start_ts >= %s AND end_ts <= %s"
        params: list = [cam, desde, hasta]
        if motivo:
            sql += " AND trigger_reason = %s"
            params.append(motivo)
        rel = sorted(r["event_id"] for r in conn.execute(sql, params).fetchall())
        preguntas.append({"id": f"R{n:02d}", "tipo": "restringida", "pregunta": texto, "camera_id": cam, "desde": desde, "hasta": hasta, "relevantes": rel})
    for n, (texto, cam, desde, hasta) in enumerate(SIN_EVIDENCIA, 1):
        rel: list[str] = []
        if cam and desde:
            sql = "SELECT event_id FROM events WHERE camera_id = %s AND start_ts >= %s AND end_ts <= %s"
            rows = conn.execute(sql + (" AND trigger_reason = 'fire'" if "incendio" in texto else ""), [cam, desde, hasta]).fetchall()
            rel = [r["event_id"] for r in rows]
        preguntas.append({"id": f"S{n:02d}", "tipo": "sin_evidencia", "pregunta": texto, "camera_id": cam, "desde": desde, "hasta": hasta, "relevantes": rel})
    for p in preguntas:
        print(p["id"], len(p["relevantes"]))
    args.out.write_text(json.dumps({"preguntas": preguntas}, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
