"""Verifica la integridad de 100 eventos persistidos (apartado 5.7 de la tesis, RF6).

Uso:
    python scripts/verify_integrity.py --out ../reentreno/integridad_eventos.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
from collections import Counter
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from monitoreo_urbano.common.config import settings  # noqa: E402
from monitoreo_urbano.common.db import get_connection  # noqa: E402
from monitoreo_urbano.common.object_storage import get_clips_client  # noqa: E402
from monitoreo_urbano.common.schemas import ALERTABLE_REASONS  # noqa: E402

REQUIRED = ["event_id", "camera_id", "camera_label", "track_id", "object_classes", "trigger_reason", "start_ts", "end_ts", "clip_path", "status", "confidence", "created_at"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    conn = get_connection()
    rows = conn.execute(
        "SELECT *, (embedding IS NOT NULL) AS has_embedding, (search_vector IS NOT NULL) AS has_fts FROM events ORDER BY event_id"
    ).fetchall()
    sample = random.Random(args.seed).sample(rows, min(args.n, len(rows)))
    client = get_clips_client()
    report, fails = [], []
    for row in sample:
        problems: list[str] = []
        missing = [f for f in REQUIRED if row.get(f) in (None, "")]
        if missing:
            problems.append(f"campos vacíos: {missing}")
        if row["end_ts"] and row["start_ts"] and row["end_ts"] < row["start_ts"]:
            problems.append("end_ts anterior a start_ts")
        try:
            with tempfile.TemporaryDirectory() as tmp:
                target = Path(tmp) / "clip.mp4"
                client.fget_object(settings.s3_bucket_clips, row["clip_path"], str(target))
                cap = cv2.VideoCapture(str(target))
                frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                ok, _ = cap.read()
                cap.release()
                if frames <= 0 or not ok:
                    problems.append("clip sin fotogramas legibles")
        except Exception as exc:  # noqa: BLE001
            problems.append(f"clip ausente o ilegible: {exc.__class__.__name__}")
        if not (row["description_nl"] or "").strip():
            problems.append("sin descripción")
        should_index = row["trigger_reason"] in ALERTABLE_REASONS or row["review_status"] == "confirmed"
        if should_index and not (row["status"] == "indexed" and row["has_embedding"] and row["has_fts"]):
            problems.append(f"debía estar indexado y está en {row['status']} (vector {row['has_embedding']}, texto {row['has_fts']})")
        if not should_index and row["status"] == "indexed":
            problems.append("indexado sin ser alerta confiable ni estar confirmado")
        kind = "señal débil sin confirmar" if row["trigger_reason"] not in ALERTABLE_REASONS and row["review_status"] != "confirmed" else (
            "incidente descartado" if row["review_status"] == "dismissed" else "incidente con alerta")
        entry = {"event_id": row["event_id"], "tipo": kind, "review_status": row["review_status"], "status": row["status"], "problemas": problems}
        report.append(entry)
        if problems:
            fails.append(entry)
    summary = {"n": len(report), "integros": len(report) - len(fails), "con_fallas": len(fails), "por_tipo": dict(Counter(r["tipo"] for r in report)), "por_estado": dict(Counter(r["status"] for r in report))}
    print(json.dumps(summary, ensure_ascii=False))
    for entry in fails:
        print("FALLA", entry["event_id"], entry["problemas"])
    args.out.write_text(json.dumps({"resumen": summary, "eventos": report}, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
