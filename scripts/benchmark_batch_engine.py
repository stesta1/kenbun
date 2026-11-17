"""Fase 4 del plan de detector compartido: ¿cuántas cámaras sostiene UNA instancia de punta a punta?

Uso:
    python scripts/benchmark_batch_engine.py --cameras 25,50,75,100 --out ../reentreno/fase4_motor.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_capture_streams import PacedCameraSource, gpu_used_mb, make_test_video  # noqa: E402

from monitoreo_urbano.common.config import settings  # noqa: E402
from monitoreo_urbano.vision_pipeline.batch_detector import BatchDetector  # noqa: E402
from monitoreo_urbano.vision_pipeline.camera_stream import CameraStream  # noqa: E402
from monitoreo_urbano.vision_pipeline.detector import YOLODetector  # noqa: E402


class ProfilingDetector(YOLODetector):

    stages: list[tuple[float, float, float]] = []

    def detect_batch(self, images):
        started = time.perf_counter()
        results = self._predict(list(images))
        wall_ms = (time.perf_counter() - started) * 1000
        speed = results[0].speed
        self.stages.append((speed["preprocess"], speed["inference"], speed["postprocess"], wall_ms / len(images)))
        return [self._to_detections(r) for r in results]


def run_case(
    detector, video: Path, cameras: int, analysis_fps: float, queue_size: int, seconds: float, warmup: float,
    start_at: float | None = None,
) -> dict:
    gpu_before = gpu_used_mb()
    streams = [
        CameraStream(f"cam-{i:03d}", PacedCameraSource(f"cam-{i:03d}", str(video)), analysis_fps=analysis_fps, queue_size=queue_size)
        for i in range(cameras)
    ]
    per_camera: dict[str, int] = {s.camera_id: 0 for s in streams}
    counting = {"on": False}

    def on_result(result) -> None:
        if counting["on"]:
            per_camera[result.camera_id] += 1

    engine = BatchDetector(detector, on_result)
    for stream in streams:
        engine.register(stream.camera_id, stream)
    for stream in streams:
        stream.start()
    not_ready = [s.camera_id for s in streams if not s.wait_ready(30)]
    if not_ready:
        raise SystemExit(f"{len(not_ready)} cámaras no se conectaron")
    engine.start()

    time.sleep(warmup)
    if start_at is not None:
        time.sleep(max(0.0, start_at - time.time()))
    stats_before = engine.stats()
    stream_before = {s.camera_id: s.stats() for s in streams}
    counting["on"] = True
    cpu_start, wall_start = time.process_time(), time.perf_counter()
    time.sleep(seconds)
    wall = time.perf_counter() - wall_start
    cpu = time.process_time() - cpu_start
    counting["on"] = False
    stats = engine.stats()
    stream_after = {s.camera_id: s.stats() for s in streams}
    gpu_after = gpu_used_mb()

    memory_mb = None
    try:
        import psutil

        memory_mb = round(psutil.Process().memory_info().rss / 2**20)
    except Exception:  # noqa: BLE001
        pass

    engine.stop(timeout=5)
    for stream in streams:
        stream.stop(timeout=3)

    expected = streams[0].effective_fps or analysis_fps
    fps_per_camera = [count / wall for count in per_camera.values()]
    captured = sum(stream_after[c].frames_delivered - stream_before[c].frames_delivered for c in stream_after)
    dropped = sum(stream_after[c].frames_dropped - stream_before[c].frames_dropped for c in stream_after)
    batches = stats.batches - stats_before.batches
    images = stats.images - stats_before.images
    return {
        "camaras": cameras,
        "fps_de_analisis_efectivo": round(expected, 2),
        "imagenes_por_segundo_pedidas": round(cameras * expected, 1),
        "imagenes_por_segundo_procesadas": round(images / wall, 1),
        "fps_por_camara_media": round(statistics.mean(fps_per_camera), 2),
        "fps_por_camara_minimo": round(min(fps_per_camera), 2),
        "cumplimiento_medio": round(statistics.mean(fps_per_camera) / expected, 3),
        "fotogramas_capturados": captured,
        "fotogramas_descartados": dropped,
        "porcentaje_descartado": round(100 * dropped / captured, 2) if captured else None,
        "lotes": batches,
        "tamano_medio_de_lote": round(images / batches, 1) if batches else None,
        "tamano_maximo_de_lote": stats.max_batch_size,
        "inferencia_por_lote_ms_mediana": stats.inference_ms_p50,
        "inferencia_por_lote_ms_p95": stats.inference_ms_p95,
        "demora_captura_a_resultado_ms_mediana": stats.latency_ms_p50,
        "demora_captura_a_resultado_ms_p95": stats.latency_ms_p95,
        "errores_del_detector": stats.errors,
        "nucleos_cpu_usados": round(cpu / wall, 2),
        "ram_mb": memory_mb,
        "vram_usada_mb": (gpu_after - gpu_before) if gpu_before is not None and gpu_after is not None else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", type=Path, default=ROOT / "data" / "raw_videos" / "sample_bus.mp4")
    parser.add_argument("--weights", type=Path, default=ROOT / "data" / "models" / "yolo" / "baseline_v1.pt")
    parser.add_argument("--cameras", default="25,50,75,100")
    parser.add_argument("--analysis-fps", type=float, default=6.0)
    parser.add_argument("--queue-size", type=int, default=2)
    parser.add_argument("--max-batch", type=int, default=None)
    parser.add_argument("--fill-wait", type=float, default=None)
    parser.add_argument("--hw", default="none", help="Decodificación: none, any, d3d11...")
    parser.add_argument("--half", action="store_true", help="Inferencia en FP16")
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--warmup", type=float, default=5.0)
    parser.add_argument("--switch-interval", type=float, default=None, help="sys.setswitchinterval, en segundos")
    parser.add_argument("--profile", action="store_true", help="Desglosa el tiempo por etapa de Ultralytics")
    parser.add_argument("--torch-threads", type=int, default=None, help="torch.set_num_threads")
    parser.add_argument("--start-at", type=float, default=None, help="Instante (epoch) en que empieza la medición")
    parser.add_argument("--tag", default="", help="Etiqueta que se antepone a cada línea de salida")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if args.switch_interval:
        sys.setswitchinterval(args.switch_interval)
    if args.torch_threads:
        import torch

        torch.set_num_threads(args.torch_threads)
    settings.capture_hw_acceleration = args.hw
    settings.yolo_half = args.half
    if args.max_batch:
        settings.detector_max_batch = args.max_batch
    if args.fill_wait is not None:
        settings.detector_batch_fill_wait_seconds = args.fill_wait

    detector = (ProfilingDetector if args.profile else YOLODetector)(weights_path=str(args.weights), device="cuda:0")
    import numpy as np

    detector.detect_batch([np.zeros((720, 1280, 3), dtype=np.uint8)] * 4)

    report = {
        "parametros": {
            "fps_de_analisis": args.analysis_fps, "cola": args.queue_size, "decodificacion": args.hw, "fp16": args.half,
            "intervalo_de_cambio_de_hilo_s": args.switch_interval, "hilos_de_torch": args.torch_threads, "lote_maximo": settings.detector_max_batch, "espera_para_llenar_s": settings.detector_batch_fill_wait_seconds,
        },
        "resultados": [],
    }
    with tempfile.TemporaryDirectory() as tmp:
        video = Path(tmp) / "test_720p30.mp4"
        make_test_video(args.video, video, seconds=int(args.seconds + args.warmup + 15))
        for cameras in [int(x) for x in args.cameras.split(",")]:
            print(f"\n== {cameras} cámaras a {args.analysis_fps:g} fps | decodificación: {args.hw} | FP16: {args.half} | lote máx. {settings.detector_max_batch}", flush=True)
            result = run_case(
                detector, video, cameras, args.analysis_fps, args.queue_size, args.seconds, args.warmup, args.start_at
            )
            if args.profile and ProfilingDetector.stages:
                stages = ProfilingDetector.stages
                result["etapas_ms_por_imagen"] = {
                    "preprocesado": round(statistics.mean(x[0] for x in stages), 2),
                    "inferencia": round(statistics.mean(x[1] for x in stages), 2),
                    "posprocesado": round(statistics.mean(x[2] for x in stages), 2),
                    "total_medido_por_fuera": round(statistics.mean(x[3] for x in stages), 2),
                }
                print(f"   etapas (ms por imagen): {result['etapas_ms_por_imagen']}", flush=True)
                ProfilingDetector.stages.clear()
            report["resultados"].append(result)
            print(
                f"   imágenes/s: {result['imagenes_por_segundo_procesadas']} de {result['imagenes_por_segundo_pedidas']} pedidas"
                f" | por cámara: media {result['fps_por_camara_media']} (mín {result['fps_por_camara_minimo']}) = {result['cumplimiento_medio']:.0%}\n"
                f"   lotes: tamaño medio {result['tamano_medio_de_lote']} (máx {result['tamano_maximo_de_lote']}), inferencia {result['inferencia_por_lote_ms_mediana']} ms mediana / {result['inferencia_por_lote_ms_p95']} p95\n"
                f"   demora captura→resultado: {result['demora_captura_a_resultado_ms_mediana']} ms mediana, {result['demora_captura_a_resultado_ms_p95']} ms p95"
                f" | descartados: {result['fotogramas_descartados']} de {result['fotogramas_capturados']} ({result['porcentaje_descartado']}%)\n"
                f"   CPU {result['nucleos_cpu_usados']} núcleos | RAM {result['ram_mb']} MB | VRAM {result['vram_usada_mb']} MB | errores: {result['errores_del_detector']}",
                flush=True,
            )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nResultados en {args.out}")


if __name__ == "__main__":
    main()
