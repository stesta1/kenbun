"""Compara las alertas de choque del sistema en vivo (corrida por RTSP) con las que predice la simulación offline.

Uso:
    python scripts/analyze_alert_run.py --corrida ../reentreno/medicion/corrida_alertas_dominio.json --carpeta ../test_ui/videos/choques_dominio
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

LEAD_IN = 8.0
TOL = 3.0


def main() -> None:
    import evaluate_alert_configs as e
    from monitoreo_urbano.common.config import settings
    from monitoreo_urbano.common.db import get_connection

    parser = argparse.ArgumentParser()
    parser.add_argument("--corrida", type=Path, required=True)
    parser.add_argument("--carpeta", type=Path, required=True)
    parser.add_argument("--retraso-s", type=float, default=2.0, help="Retraso estimado entre el lanzamiento y la primera imagen del video")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    runs = json.loads(args.corrida.read_text(encoding="utf-8"))["corridas"]
    mapa = {m["archivo"][:-4]: m for m in json.loads((args.carpeta / "mapa_videos.json").read_text(encoding="utf-8"))}
    conn = get_connection()
    events = conn.execute("SELECT event_id, camera_id, trigger_reason, status, created_at FROM events ORDER BY created_at").fetchall()

    cfg = {"nombre": "defaults", "fps": 6.0, "directo": settings.accident_direct_trigger,
           "conf": settings.yolo_confidence_by_class["accident"], "persist_s": settings.incident_min_seconds["accident"],
           "ventana_s": settings.incident_window_seconds, "dedup_s": settings.event_dedup_window_seconds,
           "dedup_modo": settings.incident_dedup_mode, "min_muestras": settings.trigger_min_samples}
    stems = [mapa[r["video"]]["origen"] for r in runs]
    offline = e.run_config((cfg, str(e.ROOT / "data" / "models" / "yolo" / "baseline_v6.pt"), stems))["videos"]

    rows = []
    for r in runs:
        m = mapa[r["video"]]
        t0, dur = m["accident_time"], m["duracion_s"]
        a = datetime.fromisoformat(r["t_launch_utc"]).replace(tzinfo=None)
        b = datetime.fromisoformat(r["t_fin_utc"]).replace(tzinfo=None)
        mine = []
        for ev in events:
            created = ev["created_at"].replace(tzinfo=None)
            if ev["camera_id"] == r["camara"] and a <= created <= b and ev["trigger_reason"] == "accident":
                mine.append(round((created - a).total_seconds() - LEAD_IN - args.retraso_s, 1))
        off = [round(x, 1) for x in offline[m["origen"]]["eventos"]]
        dentro = [x for x in mine if t0 - TOL - 1 <= x <= dur]
        fuera_negro = [x for x in mine if x < t0 - TOL - 1 and x < 0]
        fuera_antes = [x for x in mine if 0 <= x < t0 - TOL - 1]
        fuera_despues = [x for x in mine if x > dur]
        rows.append({"video": r["video"], "origen": m["origen"], "camara": r["camara"], "t0": round(t0, 1), "duracion": round(dur, 1),
                     "vivo": mine, "offline": off, "detectado_vivo": bool(dentro), "detectado_offline": any(x >= t0 - TOL for x in off),
                     "antes_del_choque": fuera_antes, "en_negro": fuera_negro, "despues_del_video": fuera_despues})
    n = len(rows)
    print(f"{n} videos publicados por RTSP en cámaras reales del sistema, valores por defecto\n")
    print("video  t0   dur  | vivo (s en el video)        | offline (s)")
    for x in rows:
        print(f"{x['video']}  {x['t0']:>4} {x['duracion']:>4} | {str(x['vivo']):<27} | {x['offline']}")
    vivo = sum(x["detectado_vivo"] for x in rows)
    off = sum(x["detectado_offline"] for x in rows)
    dup = sum(len(x["vivo"]) > 1 for x in rows)
    print(f"\nDetectados en vivo: {vivo}/{n}   |   predichos offline: {off}/{n}")
    print(f"Videos con más de un evento en vivo: {dup}")
    print(f"Eventos antes del choque (falsos previos): {sum(len(x['antes_del_choque']) for x in rows)} | en el negro inicial: {sum(len(x['en_negro']) for x in rows)} | en el tránsito posterior al video: {sum(len(x['despues_del_video']) for x in rows)}")
    if args.out:
        args.out.write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
