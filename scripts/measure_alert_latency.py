"""Mide la latencia de alerta de punta a punta (apartado 5.3 de la tesis, RNF1).

Uso:
    python scripts/measure_alert_latency.py --sources 1 --out ../reentreno/latencia_alerta_1.json
    python scripts/measure_alert_latency.py --sources 2 --out ../reentreno/latencia_alerta_2.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import websockets

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from monitoreo_urbano.common.auth import with_stream_credentials  # noqa: E402
from monitoreo_urbano.common.config import settings  # noqa: E402

SOURCE = ROOT / "data" / "datasets" / "_sources" / "accident"
VIDEOS = SOURCE / "real_videos"
PYTHON = sys.executable
LEAD_IN = 8.0
W, H, FPS = 960, 720, 15
SELECCION = ROOT.parent / "reentreno" / "latencia_videos_seleccion.json"


def load_selection() -> tuple[dict[str, float], dict[str, float]]:
    rows = json.loads(SELECCION.read_text(encoding="utf-8"))
    return {r["video"]: float(r["impacto_s"]) for r in rows}, {r["video"]: float(r["alerta_offline_s"]) for r in rows}


REFERENCIA, DISPARO_OFFLINE = load_selection()
_NOMBRES = list(REFERENCIA)
PARES = [(_NOMBRES[i], _NOMBRES[i + 1]) for i in range(0, len(_NOMBRES) - 1, 2)]
CAMERAS = ["lat-cam-1", "lat-cam-2"]


def ffmpeg_cmd(camera: str, video: str | None, hold: float, lead_in: float, white: bool = False) -> list[str]:
    url = with_stream_credentials(f"rtsp://localhost:8554/{camera}")
    black = ["-re", "-f", "lavfi", "-t", str(lead_in), "-i", f"color=c=black:s={W}x{H}:r={FPS}"]
    if white:
        tail = ["-re", "-f", "lavfi", "-t", "3", "-i", f"color=c=white:s={W}x{H}:r={FPS}"]
        fc = f"[0:v]setsar=1,setpts=PTS-STARTPTS[a];[1:v]setsar=1,setpts=PTS-STARTPTS[b];[a][b]concat=n=2:v=1:a=0[o]"
    else:
        tail = ["-re", "-i", str(VIDEOS / f"{video}.mp4")]
        fit = f"scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={FPS},setpts=PTS-STARTPTS"
        fc = f"[0:v]setsar=1,setpts=PTS-STARTPTS[a];[1:v]{fit}[b];[a][b]concat=n=2:v=1:a=0[o]"
    return [
        "ffmpeg", "-loglevel", "error", *black, *tail, "-filter_complex", fc, "-map", "[o]",
        "-c:v", "libx264", "-preset", "veryfast", "-tune", "zerolatency", "-pix_fmt", "yuv420p",
        "-f", "rtsp", "-rtsp_transport", "tcp", url,
    ]


def calibrate(camera: str) -> float:
    t_launch = time.time()
    proc = subprocess.Popen(ffmpeg_cmd(camera, None, 0, LEAD_IN, white=True))
    time.sleep(1.0)
    cap = cv2.VideoCapture(with_stream_credentials(f"rtsp://localhost:8554/{camera}"))
    seen = None
    deadline = time.time() + 20
    while time.time() < deadline:
        ok, frame = cap.read()
        if ok and frame.mean() > 200:
            seen = time.time()
            break
    cap.release()
    proc.terminate()
    if seen is None:
        raise RuntimeError("No se vio el cuadro blanco")
    return seen - t_launch - LEAD_IN


async def listen(stop: asyncio.Event, sink: list[dict]) -> None:
    uri = f"ws://127.0.0.1:{settings.api_port}/alerts"
    async with websockets.connect(uri, additional_headers={"x-api-key": settings.api_key}) as ws:
        sink.append({"conectado": True})
        while not stop.is_set():
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            message = json.loads(raw)
            message["t_recv"] = time.time()
            sink.append(message)


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    import math

    return ordered[max(0, math.ceil(p * len(ordered)) - 1)]


async def run(args: argparse.Namespace) -> None:
    delta = statistics.median(calibrate(CAMERAS[0]) for _ in range(5))
    print(f"Retraso de arranque medido (RTSP + lectura): {delta:.2f} s", flush=True)
    sink: list[dict] = []
    stop = asyncio.Event()
    listener = asyncio.create_task(listen(stop, sink))
    await asyncio.sleep(1.5)
    hold = args.clip_post + 15
    cams = CAMERAS[: args.sources]
    plan = [(v,) for v in args.videos.split(",")] if args.sources == 1 else PARES
    if args.pairs:
        plan = [tuple(p.split(":")) for p in args.pairs.split(",")]
        cams = CAMERAS[: len(plan[0])]
    rows: list[dict] = []
    for rep in range(args.reps):
        for combo in plan:
            sink.clear()
            sink.append({"conectado": True})
            t_launch = time.time()
            procs = [subprocess.Popen(ffmpeg_cmd(cam, video, hold, LEAD_IN)) for cam, video in zip(cams, combo)]
            expected = {cam: t_launch + LEAD_IN + delta + REFERENCIA.get(v, 0.0) for cam, v in zip(cams, combo)}
            durations = []
            for v in combo:
                cap = cv2.VideoCapture(str(VIDEOS / f"{v}.mp4"))
                durations.append(cap.get(cv2.CAP_PROP_FRAME_COUNT) / (cap.get(cv2.CAP_PROP_FPS) or 30.0))
                cap.release()
            deadline = t_launch + LEAD_IN + max(durations) + 14
            await asyncio.sleep(max(0, deadline - time.time()))
            for proc in procs:
                proc.terminate()
            alerts = [m for m in sink if "t_recv" in m]
            for cam, video in zip(cams, combo):
                mine = [m for m in alerts if m.get("camera_id") == cam]
                first = min(mine, key=lambda m: m["t_recv"]) if mine else None
                rows.append({
                    "rep": rep + 1, "fuentes": args.sources, "t_launch_utc": datetime.utcfromtimestamp(t_launch).isoformat(timespec="seconds"), "camara": cam, "video": video,
                    "latencia_s": round(first["t_recv"] - expected[cam], 2) if first else None,
                    "latencia_regla_s": round(first["t_recv"] - expected[cam] - DISPARO_OFFLINE.get(video, 0.0), 2) if first else None,
                    "motivo": first.get("trigger_reason") if first else None,
                    "t_launch_epoch": round(t_launch, 3), "t_recv_epoch": round(first["t_recv"], 3) if first else None,
                    "alertas_recibidas": len(mine),
                })
                print(rows[-1], flush=True)
            await asyncio.sleep(args.gap)
    stop.set()
    await listener
    def summarize(key: str) -> dict:
        values = [r[key] for r in rows if r[key] is not None]
        return {
            "n": len(rows), "con_alerta": len(values), "omisiones": len(rows) - len(values),
            "mediana_s": round(statistics.median(values), 2) if values else None,
            "p95_s": round(percentile(values, 0.95), 2) if values else None,
            "maximo_s": max(values) if values else None, "minimo_s": min(values) if values else None,
        }

    summary = {"desde_impacto": summarize("latencia_s"), "desde_disparo_regla": summarize("latencia_regla_s")}
    print(summary, flush=True)
    Path(args.out).write_text(json.dumps({
        "fecha": datetime.now().isoformat(timespec="seconds"), "fuentes": args.sources, "clip_post_s": args.clip_post,
        "retraso_arranque_s": round(delta, 2), "referencia_s": REFERENCIA, "disparo_offline_s": DISPARO_OFFLINE, "resumen": summary, "filas": rows,
    }, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=int, choices=(1, 2), required=True)
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--clip-post", type=float, default=30.0, help="Debe coincidir con MU_CLIP_POST_EVENT_SECONDS del módulo 1")
    parser.add_argument("--gap", type=float, default=8.0, help="Pausa entre corridas (debe superar la ventana de deduplicación)")
    parser.add_argument("--videos", default=",".join(REFERENCIA), help="Solo con una fuente")
    parser.add_argument("--pairs", default=None, help="Pares cámara1:cámara2 de videos, separados por coma (genera eventos; la latencia no se interpreta si no hay referencia)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
