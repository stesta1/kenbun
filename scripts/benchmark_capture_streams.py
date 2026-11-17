"""Fase 3 del plan de detector compartido: ¿cuántas cámaras puede capturar una instancia a la vez?

Uso:
    python scripts/benchmark_capture_streams.py --out ../reentreno/fase3_captura.json
"""

from __future__ import annotations

import argparse
import json
import queue
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from monitoreo_urbano.vision_pipeline.camera_source import CameraSource  # noqa: E402
from monitoreo_urbano.vision_pipeline.camera_stream import CameraStream, StreamClosed  # noqa: E402

NATIVE_FPS = 30


class PacedCapture:

    def __init__(self, cap: cv2.VideoCapture, fps: float) -> None:
        self._cap = cap
        self._period = 1.0 / fps
        self._next = time.perf_counter()

    def _pace(self) -> None:
        now = time.perf_counter()
        if self._next > now:
            time.sleep(self._next - now)
        elif now - self._next > 0.5:
            self._next = now
        self._next += self._period

    def read(self):
        self._pace()
        return self._cap.read()

    def grab(self):
        self._pace()
        return self._cap.grab()

    def isOpened(self) -> bool:
        return self._cap.isOpened()

    def release(self) -> None:
        self._cap.release()

    def get(self, prop) -> float:
        return NATIVE_FPS if prop == cv2.CAP_PROP_FPS else self._cap.get(prop)


class PacedCameraSource(CameraSource):
    def _connect(self):
        cap = self._video_capture()
        return PacedCapture(cap, NATIVE_FPS) if cap.isOpened() else None


def gpu_used_mb() -> int | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True
        )
        return int(out.stdout.strip().split()[0])
    except Exception:  # noqa: BLE001
        return None


def make_test_video(source: Path, out_path: Path, seconds: int) -> None:
    command = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-stream_loop", "-1", "-i", str(source), "-t", str(seconds),
        "-vf", "crop='min(iw,ih*16/9)':'min(ih,iw*9/16)',scale=1280:720,fps=30,noise=alls=10:allf=t",
        "-c:v", "libx264", "-preset", "veryfast", "-g", "30",
        "-b:v", "2500k", "-maxrate", "2500k", "-bufsize", "5000k", "-pix_fmt", "yuv420p", "-an", str(out_path),
    ]
    subprocess.run(command, check=True)


def run_case(video: Path, cameras: int, analysis_fps: float, queue_size: int, seconds: float, warmup: float) -> dict:
    gpu_before = gpu_used_mb()
    streams = [
        CameraStream(f"cam-{i:03d}", PacedCameraSource(f"cam-{i:03d}", str(video)), analysis_fps=analysis_fps, queue_size=queue_size)
        for i in range(cameras)
    ]
    for stream in streams:
        stream.start()
    not_ready = [s.camera_id for s in streams if not s.wait_ready(30)]
    if not_ready:
        raise SystemExit(f"{len(not_ready)} cámaras no se conectaron: {not_ready[:3]}")

    latencies_ms: list[float] = []
    consumed = {s.camera_id: 0 for s in streams}
    stop = threading.Event()
    measuring = threading.Event()

    def consume() -> None:
        while not stop.is_set():
            idle = True
            for stream in streams:
                try:
                    frame = stream.get(timeout=0)
                except (queue.Empty, StreamClosed):
                    continue
                idle = False
                if frame is not None and measuring.is_set():
                    consumed[stream.camera_id] += 1
                    latencies_ms.append((datetime.utcnow() - frame.timestamp).total_seconds() * 1000)
            if idle:
                time.sleep(0.001)

    consumer = threading.Thread(target=consume, daemon=True)
    consumer.start()

    time.sleep(warmup)
    before = {s.camera_id: s.stats() for s in streams}
    measuring.set()
    cpu_start = time.process_time()
    wall_start = time.perf_counter()
    time.sleep(seconds)
    wall = time.perf_counter() - wall_start
    cpu = time.process_time() - cpu_start
    measuring.clear()
    after = {s.camera_id: s.stats() for s in streams}

    memory_mb = None
    try:
        import psutil

        memory_mb = round(psutil.Process().memory_info().rss / 2**20)
    except Exception:  # noqa: BLE001
        pass

    gpu_after = gpu_used_mb()
    stop.set()
    consumer.join(2)
    for stream in streams:
        stream.stop(timeout=3)

    per_camera_fps = [consumed[c] / wall for c in consumed]
    dropped = sum(after[c].frames_dropped - before[c].frames_dropped for c in after)
    delivered = sum(after[c].frames_delivered - before[c].frames_delivered for c in after)
    losses = sum(after[c].signal_losses - before[c].signal_losses for c in after)
    expected = streams[0].effective_fps or analysis_fps
    return {
        "camaras": cameras,
        "fps_de_analisis_pedido": analysis_fps,
        "fps_de_analisis_efectivo": round(expected, 2),
        "cola_por_camara": queue_size,
        "fps_recibidos_por_camara_media": round(statistics.mean(per_camera_fps), 2),
        "fps_recibidos_por_camara_minimo": round(min(per_camera_fps), 2),
        "cumplimiento_medio": round(statistics.mean(per_camera_fps) / expected, 3),
        "fotogramas_capturados": delivered,
        "fotogramas_descartados": dropped,
        "porcentaje_descartado": round(100 * dropped / delivered, 2) if delivered else None,
        "cortes_de_senal": losses,
        "demora_captura_a_consumo_ms_mediana": round(statistics.median(latencies_ms), 1) if latencies_ms else None,
        "demora_captura_a_consumo_ms_p95": round(float(np.percentile(latencies_ms, 95)), 1) if latencies_ms else None,
        "demora_captura_a_consumo_ms_max": round(max(latencies_ms), 1) if latencies_ms else None,
        "nucleos_cpu_usados": round(cpu / wall, 2),
        "memoria_proceso_mb": memory_mb,
        "vram_usada_por_las_capturas_mb": (gpu_after - gpu_before) if gpu_before is not None and gpu_after is not None else None,
        "hilos_de_captura": cameras,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", type=Path, default=ROOT / "data" / "raw_videos" / "sample_bus.mp4")
    parser.add_argument("--cameras", default="10,25,50,100")
    parser.add_argument("--analysis-fps", type=float, default=6.0)
    parser.add_argument("--queue-size", type=int, default=2)
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--warmup", type=float, default=4.0)
    parser.add_argument("--hw", default="none", help="Aceleración de decodificación: none, any, d3d11...")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    from monitoreo_urbano.common.config import settings

    settings.capture_hw_acceleration = args.hw

    report = {"parametros": vars(args) | {"video": str(args.video), "out": str(args.out)}, "resultados": []}
    with tempfile.TemporaryDirectory() as tmp:
        video = Path(tmp) / "test_720p30.mp4"
        total = int(args.seconds + args.warmup + 15)
        make_test_video(args.video, video, seconds=total)
        for cameras in [int(x) for x in args.cameras.split(",")]:
            print(f"\n== {cameras} cámaras, análisis a {args.analysis_fps:g} fps, cola de {args.queue_size}", flush=True)
            result = run_case(video, cameras, args.analysis_fps, args.queue_size, args.seconds, args.warmup)
            report["resultados"].append(result)
            print(
                f"   fps por cámara: media {result['fps_recibidos_por_camara_media']} (mín {result['fps_recibidos_por_camara_minimo']}) "
                f"de {result['fps_de_analisis_efectivo']} esperados = {result['cumplimiento_medio']:.0%}\n"
                f"   descartados: {result['fotogramas_descartados']} de {result['fotogramas_capturados']} ({result['porcentaje_descartado']}%) | cortes: {result['cortes_de_senal']}\n"
                f"   demora captura→consumo: mediana {result['demora_captura_a_consumo_ms_mediana']} ms, p95 {result['demora_captura_a_consumo_ms_p95']} ms, máx {result['demora_captura_a_consumo_ms_max']} ms\n"
                f"   CPU: {result['nucleos_cpu_usados']} núcleos | RAM: {result['memoria_proceso_mb']} MB | VRAM: {result['vram_usada_por_las_capturas_mb']} MB",
                flush=True,
            )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nResultados en {args.out}")


if __name__ == "__main__":
    main()
