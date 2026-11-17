"""Verificación de integridad de 100 eventos (OE2) con el flujo vigente: ningún evento se indexa sin la confirmación de un operador.

Uso (desde la raíz del repositorio, con las variables MU_* de la prueba):
    .venv/Scripts/python.exe scripts/reentreno_v10/integridad_flujo_vigente.py --out-dir reentreno/corrida_noche/<fecha>/integridad
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import cv2
import httpx

APP = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(APP / "src"))
from monitoreo_urbano.common.config import settings  # noqa: E402
from monitoreo_urbano.common.db import get_connection  # noqa: E402
from monitoreo_urbano.common.object_storage import get_clips_client  # noqa: E402
from monitoreo_urbano.common.schemas import ALERTABLE_REASONS  # noqa: E402

REQUIRED = ["event_id", "camera_id", "camera_label", "track_id", "object_classes", "trigger_reason", "start_ts", "end_ts",
            "clip_path", "status", "confidence", "created_at"]


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def wait_until(conn, sql: str, label: str, timeout: float, quiet: float = 60) -> None:
    start, zero_since = time.time(), None
    while time.time() - start < timeout:
        row = conn.execute(sql).fetchone()
        n = list(row.values())[0]
        if n == 0:
            zero_since = zero_since or time.time()
            if time.time() - zero_since >= quiet:
                log(f"{label}: listo")
                return
        else:
            zero_since = None
        time.sleep(10)
    log(f"{label}: se agotó la espera de {timeout:.0f} s; se sigue igual")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--espera-cola", type=float, default=1800)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    conn = get_connection()
    log(f"base: {settings.database_url.rsplit('/', 1)[-1]} · bucket: {settings.s3_bucket_clips}")

    busy = "SELECT count(*) FROM events WHERE status IN ('recording','pending_description','processing')"
    wait_until(conn, busy, "cola de descripción", args.espera_cola)

    rows = conn.execute("SELECT event_id, trigger_reason, description_nl FROM events WHERE status = 'described' AND review_status = 'pending' ORDER BY event_id").fetchall()
    weak = [r for r in rows if r["trigger_reason"] not in ALERTABLE_REASONS]
    alerts = [r for r in rows if r["trigger_reason"] in ALERTABLE_REASONS]
    rng = random.Random(args.seed)
    rng.shuffle(weak)
    rng.shuffle(alerts)
    plan = {"confirmar_debiles": weak[:20], "descartar_debiles": weak[20:40], "confirmar_alertas": alerts[:10], "descartar_alertas": alerts[10:20]}
    api = httpx.Client(base_url=f"http://127.0.0.1:{settings.api_port}", headers={"x-api-key": settings.api_key}, timeout=120)
    acciones = []
    for accion, eventos in plan.items():
        for e in eventos:
            if accion.startswith("confirmar"):
                body = {"description_nl": (e["description_nl"] or "Evento revisado por el operador.").strip(),
                        "vehicles": [{"vehicle_type": "auto", "color": "blanco"}]}
                r = api.post(f"/events/{e['event_id']}/details", json=body)
            else:
                r = api.post(f"/events/{e['event_id']}/review", json={"review_status": "dismissed"})
            acciones.append({"accion": accion, "event_id": e["event_id"], "motivo": e["trigger_reason"], "http": r.status_code})
    log("acciones: " + json.dumps(Counter(f"{a['accion']}:{a['http']}" for a in acciones), ensure_ascii=False))

    pending_idx = "SELECT count(*) FROM events WHERE review_status = 'confirmed' AND status <> 'indexed'"
    wait_until(conn, pending_idx, "indexación de confirmados", 900, quiet=30)

    historial = conn.execute("SELECT trigger_reason, review_status, status, count(*) AS n FROM events GROUP BY 1, 2, 3 ORDER BY 1, 2, 3").fetchall()
    candidates = conn.execute(
        "SELECT *, (embedding IS NOT NULL) AS has_embedding, (search_vector IS NOT NULL) AS has_fts FROM events "
        "WHERE status IN ('described','indexed') ORDER BY event_id").fetchall()
    sample = random.Random(args.seed).sample(candidates, min(args.n, len(candidates)))
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
        confirmed = row["review_status"] == "confirmed"
        indexed = row["status"] == "indexed" and row["has_embedding"] and row["has_fts"]
        if confirmed and not indexed:
            problems.append(f"confirmado y no indexado: {row['status']} (vector {row['has_embedding']}, texto {row['has_fts']})")
        if not confirmed and row["status"] == "indexed":
            problems.append(f"indexado sin confirmación ({row['review_status']})")
        tipo = ("alerta" if row["trigger_reason"] in ALERTABLE_REASONS else "señal débil") + " " + row["review_status"]
        entry = {"event_id": row["event_id"], "motivo": row["trigger_reason"], "tipo": tipo, "review_status": row["review_status"],
                 "status": row["status"], "problemas": problems}
        report.append(entry)
        if problems:
            fails.append(entry)
    summary = {"n": len(report), "integros": len(report) - len(fails), "con_fallas": len(fails), "candidatos": len(candidates),
               "por_tipo": dict(Counter(r["tipo"] for r in report)), "por_estado": dict(Counter(r["status"] for r in report)),
               "regla": "indexado si y solo si confirmado por un operador", "semilla": args.seed}
    log(json.dumps(summary, ensure_ascii=False))
    for entry in fails:
        log(f"FALLA {entry['event_id']} {entry['problemas']}")
    (args.out_dir / "integridad.json").write_text(json.dumps(
        {"resumen": summary, "historial": [dict(h) for h in historial], "acciones": acciones, "eventos": report},
        indent=2, ensure_ascii=False, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
