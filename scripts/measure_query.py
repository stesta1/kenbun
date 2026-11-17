"""Mide la latencia y la calidad de la consulta en lenguaje natural (apartados 5.4 y 5.6 de la tesis).

Uso:
    python scripts/measure_query.py --sources 1 --questions ../reentreno/medicion/preguntas.json --out ../reentreno/consulta_1.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from monitoreo_urbano.common.auth import with_stream_credentials  # noqa: E402
from monitoreo_urbano.common.config import settings  # noqa: E402

VIDEOS = ROOT.parent / "test_ui" / "videos"
CAMERAS = ["lat-cam-1", "lat-cam-2"]
WARMUP = [
    "¿Hubo algún choque?", "¿Qué pasó en las cámaras?", "¿Hubo fuego?", "Contame los incidentes.", "¿Algún accidente de tránsito?",
    "¿Qué se detectó hoy?", "¿Hubo humo?", "¿Qué pasó en la cámara 1?", "¿Hubo vehículos chocados?", "Resumí lo último.",
]


def stream_cmd(camera: str) -> list[str]:
    return [
        "ffmpeg", "-loglevel", "error", "-re", "-stream_loop", "-1", "-i", str(VIDEOS / "V7.mp4"),
        "-vf", "scale=960:720,fps=15", "-c:v", "libx264", "-preset", "veryfast", "-tune", "zerolatency", "-pix_fmt", "yuv420p",
        "-f", "rtsp", "-rtsp_transport", "tcp", with_stream_credentials(f"rtsp://localhost:8554/{camera}"),
    ]


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(p * len(ordered)) - 1)]


def ask(client: httpx.Client, question: dict) -> tuple[float, dict | None, str | None]:
    body = {"query": question["pregunta"]}
    for key_out, key_in in (("camera_id", "camera_id"), ("start_ts", "desde"), ("end_ts", "hasta")):
        if question.get(key_in):
            body[key_out] = question[key_in]
    t0 = time.perf_counter()
    try:
        response = client.post("/chat", json=body)
        elapsed = time.perf_counter() - t0
        response.raise_for_status()
        return elapsed, response.json(), None
    except Exception as exc:  # noqa: BLE001
        return time.perf_counter() - t0, None, repr(exc)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=int, choices=(1, 2), required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    questions = json.loads(args.questions.read_text(encoding="utf-8"))["preguntas"]
    procs = [subprocess.Popen(stream_cmd(c)) for c in CAMERAS[: args.sources]]
    rows: list[dict] = []
    try:
        time.sleep(75)
        with httpx.Client(base_url=f"http://127.0.0.1:{settings.api_port}", headers={"x-api-key": settings.api_key}, timeout=180) as client:
            for text in WARMUP:
                ask(client, {"pregunta": text})
            for rep in range(args.reps):
                for question in questions:
                    elapsed, data, error = ask(client, question)
                    rows.append({
                        "id": question["id"], "tipo": question["tipo"], "rep": rep + 1, "latencia_s": round(elapsed, 2),
                        "error": error,
                        "respuesta": data["answer"] if data else None,
                        "eventos": [e["event_id"] for e in data["events"]] if data else [],
                    })
                    print(question["id"], rep + 1, round(elapsed, 2), error or "", flush=True)
    finally:
        for proc in procs:
            proc.terminate()
    values = [r["latencia_s"] for r in rows if r["error"] is None]
    summary = {
        "n": len(rows), "errores": sum(1 for r in rows if r["error"]),
        "mediana_s": round(statistics.median(values), 2), "p95_s": round(percentile(values, 0.95), 2), "maximo_s": max(values),
    }
    print(summary)
    args.out.write_text(json.dumps({
        "fecha": datetime.now().isoformat(timespec="seconds"), "fuentes": args.sources, "resumen": summary, "filas": rows,
    }, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
