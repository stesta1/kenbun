"""Atribuye cada evento de la base de medición al video que lo originó, para el corpus de `generate_corpus_choques.py`.

Uso:
    python scripts/map_events_to_videos_choques.py --corrida ../reentreno/medicion/corrida_choques.json --out ../reentreno/medicion/eventos_choques.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from monitoreo_urbano.common.db import get_connection  # noqa: E402

MAPA = ROOT.parent / "test_ui" / "videos" / "choques" / "mapa_videos.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corrida", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    runs = [r for path in args.corrida for r in json.loads(path.read_text(encoding="utf-8"))["corridas"]]
    meta = {m["archivo"][:-4]: m for m in json.loads(MAPA.read_text(encoding="utf-8"))}
    conn = get_connection()
    events = conn.execute("SELECT event_id, camera_id, trigger_reason, status, created_at FROM events ORDER BY created_at").fetchall()
    mapping: dict[str, dict] = {}
    for run in runs:
        t0, t1 = datetime.fromisoformat(run["t_launch_utc"]).replace(tzinfo=None), datetime.fromisoformat(run["t_fin_utc"]).replace(tzinfo=None)
        for e in events:
            created = e["created_at"].replace(tzinfo=None)
            if e["camera_id"] == run["camara"] and t0 <= created <= t1:
                info = meta.get(run["video"], {})
                mapping[e["event_id"]] = {
                    "video": run["video"], "camera_id": e["camera_id"], "motivo": e["trigger_reason"], "estado": e["status"],
                    "escena": info.get("escena", "incendio" if run["video"] == "V6" else None), "hora": info.get("hora"), "clima": info.get("clima"),
                    "creado": e["created_at"].isoformat(),
                }
    por_video: dict[str, list[str]] = {}
    for eid, info in mapping.items():
        por_video.setdefault(info["video"], []).append(eid)
    sin = sorted(m for m in meta if m not in por_video)
    multiples = {v: len(e) for v, e in por_video.items() if len(e) > 1}
    sueltos = [e["event_id"] for e in events if e["event_id"] not in mapping]
    print(f"{len(mapping)} eventos atribuidos de {len(events)}; videos con evento: {len(por_video)}; sin evento: {len(sin)} {sin}; con más de uno: {multiples}; sin atribuir: {sueltos}")
    args.out.write_text(json.dumps({"eventos": mapping, "videos_sin_evento": sin, "videos_con_mas_de_un_evento": multiples}, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
