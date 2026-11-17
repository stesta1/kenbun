"""Mide el uso de memoria de video y de GPU con una y dos cámaras y los tres modelos compartiendo la GPU (apartado 6.4.3).

Uso:
    python scripts/measure_gpu.py --sources 1 --seconds 120 --out ../reentreno/uso_gpu_1.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from monitoreo_urbano.common.auth import with_stream_credentials  # noqa: E402
from monitoreo_urbano.common.config import settings  # noqa: E402

VIDEOS = ROOT.parent / "test_ui" / "videos"
CAMERAS = ["lat-cam-1", "lat-cam-2"]
QUERIES = ["¿Hubo algún choque?", "¿Qué pasó en las cámaras?", "¿Hubo fuego con humo?", "¿Algún accidente de noche?"]


def sample() -> tuple[float, float]:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True
    ).stdout.strip().split(",")
    return float(out[0]), float(out[1])


def stream(camera: str, video: str, seconds: float) -> subprocess.Popen:
    cmd = [
        "ffmpeg", "-loglevel", "error", "-re", "-stream_loop", "-1", "-t", str(seconds), "-i", str(VIDEOS / f"{video}.mp4"),
        "-vf", "scale=960:720,fps=15", "-c:v", "libx264", "-preset", "veryfast", "-tune", "zerolatency", "-pix_fmt", "yuv420p",
        "-f", "rtsp", "-rtsp_transport", "tcp", with_stream_credentials(f"rtsp://localhost:8554/{camera}"),
    ]
    return subprocess.Popen(cmd)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=int, choices=(1, 2), required=True)
    parser.add_argument("--seconds", type=float, default=120)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    idle_mem, idle_util = sample()
    procs = [stream(c, v, args.seconds + 20) for c, v in zip(CAMERAS[: args.sources], ("V1", "V2"))]
    stop = threading.Event()
    samples: list[tuple[float, float]] = []
    queries = {"n": 0}

    def sampler() -> None:
        while not stop.is_set():
            samples.append(sample())
            time.sleep(1)

    def asker() -> None:
        with httpx.Client(base_url=f"http://127.0.0.1:{settings.api_port}", headers={"x-api-key": settings.api_key}, timeout=180) as client:
            while not stop.is_set():
                client.post("/chat", json={"query": QUERIES[queries["n"] % len(QUERIES)]})
                queries["n"] += 1
                time.sleep(2)

    threads = [threading.Thread(target=sampler, daemon=True), threading.Thread(target=asker, daemon=True)]
    time.sleep(15)
    for t in threads:
        t.start()
    time.sleep(args.seconds)
    stop.set()
    for t in threads:
        t.join(5)
    for p in procs:
        p.terminate()
    mem = [m for m, _ in samples]
    util = [u for _, u in samples]
    report = {
        "fuentes": args.sources, "segundos": args.seconds, "muestras": len(samples), "consultas_lanzadas": queries["n"],
        "memoria_antes_mib": idle_mem, "memoria_pico_mib": max(mem), "memoria_media_mib": round(statistics.mean(mem)),
        "gpu_uso_medio_pct": round(statistics.mean(util), 1), "gpu_uso_maximo_pct": max(util), "memoria_total_mib": 16303,
    }
    print(json.dumps(report, ensure_ascii=False))
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
