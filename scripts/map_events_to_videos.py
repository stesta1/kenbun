"""Asigna a cada evento de la base de medición el video que lo originó (verdad de referencia de las pruebas).

Uso:
    python scripts/map_events_to_videos.py --out ../reentreno/medicion/eventos_videos.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from monitoreo_urbano.common.db import get_connection  # noqa: E402

REENTRENO = ROOT.parent / "reentreno"
GAP_S = 40


def runs_for(camera: str) -> list[str]:
    seq: list[str] = []
    one = json.loads((REENTRENO / "latencia_alerta_1.json").read_text(encoding="utf-8"))
    two = json.loads((REENTRENO / "latencia_alerta_2.json").read_text(encoding="utf-8"))
    extra_path = REENTRENO / "medicion" / "corpus_extra.json"
    extra = json.loads(extra_path.read_text(encoding="utf-8")) if extra_path.exists() else {"filas": []}
    for data in (one, two, extra):
        seq += [r["video"] for r in data["filas"] if r["camara"] == camera]
    return seq


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    conn = get_connection()
    mapping: dict[str, dict] = {}
    for camera in ("lat-cam-1", "lat-cam-2"):
        rows = conn.execute(
            "SELECT event_id, created_at FROM events WHERE camera_id = %s ORDER BY created_at", (camera,)
        ).fetchall()
        groups: list[list[dict]] = []
        for row in rows:
            if groups and (row["created_at"] - groups[-1][-1]["created_at"]).total_seconds() <= GAP_S:
                groups[-1].append(row)
            else:
                groups.append([row])
        videos = runs_for(camera)
        print(f"{camera}: {len(rows)} eventos, {len(groups)} grupos, {len(videos)} corridas")
        if len(groups) != len(videos):
            sys.exit("La cantidad de grupos no coincide con las corridas; revisar a mano.")
        for group, video in zip(groups, videos):
            for row in group:
                mapping[row["event_id"]] = {"video": video, "camera_id": camera, "created_at": row["created_at"].isoformat()}
    args.out.write_text(json.dumps(mapping, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{len(mapping)} eventos asignados")


if __name__ == "__main__":
    main()
