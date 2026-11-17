"""Fase 8 del plan de detector compartido: capacidad de UNA instancia de visión completa.

Uso:
    python scripts/benchmark_system.py --cameras 10,25,50,100 --out ../reentreno/fase8_sistema.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", type=Path, default=ROOT / "data" / "raw_videos" / "sample_bus.mp4")
    parser.add_argument("--weights", type=Path, default=ROOT / "data" / "models" / "yolo" / "baseline_v1.pt")
    parser.add_argument("--cameras", default="10,25,50,100")
    parser.add_argument("--analysis-fps", type=float, default=6.0)
    parser.add_argument("--hw", default="none", help="Decodificación: none, any, d3d11...")
    parser.add_argument("--half", action="store_true", help="Inferencia en FP16")
    parser.add_argument("--seconds", type=float, default=30.0, help="Duración de la ventana de medición")
    parser.add_argument("--warmup", type=float, default=8.0)
    parser.add_argument("--incidents-per-minute", type=float, default=1.0, help="Incidentes simulados por cámara y minuto (0 = ninguno)")
    parser.add_argument("--incident-seconds", type=float, default=2.0, help="Duración de cada incidente simulado")
    parser.add_argument("--clip-post-seconds", type=float, default=3.0, help="Cierre del clip después del disparo")
    parser.add_argument("--clip-workers", type=int, default=None)
    parser.add_argument("--db-name", default="bench_capacity")
    parser.add_argument("--bucket", default="bench-clips")
    parser.add_argument("--no-hook", action="store_true", help="Diagnóstico: sin JPEG ni señales de salud en la captura")
    parser.add_argument("--keep", action="store_true", help="No borrar la base ni el bucket de prueba")
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


ARGS = _parse_args() if __name__ == "__main__" else None

if ARGS is not None:
    base_url = os.environ.get("MU_DATABASE_URL", "postgresql://monitoreo:monitoreo@localhost:5432/monitoreo_urbano")
    os.environ["MU_DATABASE_URL"] = base_url.rsplit("/", 1)[0] + "/" + ARGS.db_name
    os.environ["MU_S3_BUCKET_CLIPS"] = ARGS.bucket

import psycopg  # noqa: E402
from benchmark_capture_streams import PacedCameraSource, gpu_used_mb, make_test_video  # noqa: E402

from monitoreo_urbano.common.config import settings  # noqa: E402
from monitoreo_urbano.common.schemas import BoundingBox, CameraInfo, Detection  # noqa: E402
from monitoreo_urbano.vision_pipeline import batch_detector  # noqa: E402
from monitoreo_urbano.vision_pipeline.camera_stream import CameraStream  # noqa: E402
from monitoreo_urbano.vision_pipeline.detector import YOLODetector  # noqa: E402
from monitoreo_urbano.vision_pipeline.frame_encoding import make_capture_hook  # noqa: E402
from monitoreo_urbano.vision_pipeline.instance import VisionInstance  # noqa: E402
from monitoreo_urbano.vision_pipeline.instance_store import PostgresInstanceStore  # noqa: E402


def ensure_database(name: str) -> None:
    admin_url = settings.database_url.rsplit("/", 1)[0] + "/postgres"
    with psycopg.connect(admin_url, autocommit=True) as conn:
        exists = conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", (name,)).fetchone()
        if not exists:
            conn.execute(f'CREATE DATABASE "{name}"')


def drop_database(name: str) -> None:
    admin_url = settings.database_url.rsplit("/", 1)[0] + "/postgres"
    with psycopg.connect(admin_url, autocommit=True) as conn:
        conn.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')


def drop_bucket(name: str) -> None:
    from monitoreo_urbano.common.object_storage import get_minio_client

    client = get_minio_client()
    if not client.bucket_exists(name):
        return
    for obj in client.list_objects(name, recursive=True):
        client.remove_object(name, obj.object_name)
    client.remove_bucket(name)


class IncidentInjector:

    def __init__(self, camera_ids: list[str], per_minute: float, duration: float) -> None:
        self.period = 60.0 / per_minute if per_minute > 0 else None
        self.duration = duration
        self.offset = {c: (i / max(1, len(camera_ids))) * (self.period or 0) for i, c in enumerate(camera_ids)}
        self.t0 = time.monotonic()
        self.box = BoundingBox(x1=500, y1=250, x2=700, y2=450)

    def active(self, camera_id: str) -> bool:
        if self.period is None:
            return False
        return (time.monotonic() - self.t0 + self.offset[camera_id]) % self.period < self.duration

    def inject(self, result: batch_detector.DetectionResult) -> batch_detector.DetectionResult:
        if not self.active(result.camera_id):
            return result
        fire = Detection(class_name="fire", confidence=0.9, bbox=self.box)
        return batch_detector.DetectionResult(result.camera_id, result.frame, [*result.detections, fire])


def run_case(detector, video: Path, cameras: int, args: argparse.Namespace) -> dict:
    settings.clip_post_event_seconds = args.clip_post_seconds
    infos = [CameraInfo(camera_id=f"cam-{i:03d}", label=f"Cámara simulada {i:03d}", source=str(video)) for i in range(cameras)]
    hook = make_capture_hook(settings.clip_buffer_jpeg_quality)
    injector = IncidentInjector([c.camera_id for c in infos], args.incidents_per_minute, args.incident_seconds)

    def stream_factory(camera: CameraInfo) -> CameraStream:
        return CameraStream(
            camera.camera_id,
            PacedCameraSource(camera.camera_id, str(video)),
            analysis_fps=args.analysis_fps,
            queue_size=settings.camera_queue_size,
            frame_hook=None if args.no_hook else hook,
        )

    store = PostgresInstanceStore()
    store._conn.execute("TRUNCATE events")
    gpu_before = gpu_used_mb()
    instance = VisionInstance(
        infos, detector=detector, store=store, analysis_fps=args.analysis_fps, stream_factory=stream_factory,
        clip_workers=args.clip_workers, name=f"bench-{cameras}",
    )
    original = instance._on_result
    instance._hub._on_result = lambda result: original(injector.inject(result))

    started = time.perf_counter()
    instance.start()
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline and sum(c.connected for c in instance.stats().cameras) < cameras:
        time.sleep(0.5)
    connected = sum(c.connected for c in instance.stats().cameras)
    connect_seconds = time.perf_counter() - started
    if connected < cameras:
        instance.stop(clip_timeout=5)
        raise SystemExit(f"Solo se conectaron {connected} de {cameras} cámaras en 120 s")

    time.sleep(args.warmup)
    injector.t0 = time.monotonic()
    before = instance.stats()
    cpu_start, wall_start = time.process_time(), time.perf_counter()
    time.sleep(args.seconds)
    wall = time.perf_counter() - wall_start
    cpu = time.process_time() - cpu_start
    after = instance.stats()
    gpu_after = gpu_used_mb()

    ram_mb = None
    try:
        import psutil

        ram_mb = round(psutil.Process().memory_info().rss / 2**20)
    except Exception:  # noqa: BLE001
        pass

    instance.stop(clip_timeout=60)
    with psycopg.connect(settings.database_url) as conn:
        events = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]

    expected_fps = after.cameras[0].effective_fps or args.analysis_fps
    images = after.engine.images - before.engine.images
    batches = after.engine.batches - before.engine.batches
    captured = sum(a.frames_delivered - b.frames_delivered for a, b in zip(after.cameras, before.cameras, strict=True))
    dropped = sum(a.frames_dropped - b.frames_dropped for a, b in zip(after.cameras, before.cameras, strict=True))
    clips_done = after.clips.completed - before.clips.completed
    expected_incidents = cameras * args.incidents_per_minute * wall / 60.0
    return {
        "camaras": cameras,
        "segundos_en_conectar": round(connect_seconds, 1),
        "fps_de_analisis_efectivo": round(expected_fps, 2),
        "imagenes_por_segundo_pedidas": round(cameras * expected_fps, 1),
        "imagenes_por_segundo_procesadas": round(images / wall, 1),
        "cumplimiento": round(images / (cameras * expected_fps * wall), 3),
        "fotogramas_capturados": captured,
        "fotogramas_descartados": dropped,
        "porcentaje_descartado": round(100 * dropped / captured, 2) if captured else None,
        "lotes": batches,
        "tamano_medio_de_lote": round(images / batches, 1) if batches else None,
        "inferencia_por_lote_ms_mediana": after.engine.inference_ms_p50,
        "inferencia_por_lote_ms_p95": after.engine.inference_ms_p95,
        "demora_captura_a_resultado_ms_mediana": after.engine.latency_ms_p50,
        "demora_captura_a_resultado_ms_p95": after.engine.latency_ms_p95,
        "errores_del_detector": after.engine.errors,
        "nucleos_cpu_usados": round(cpu / wall, 2),
        "ram_mb": ram_mb,
        "vram_usada_mb": (gpu_after - gpu_before) if gpu_before is not None and gpu_after is not None else None,
        "incidentes_esperados_en_la_ventana": round(expected_incidents, 1),
        "clips_escritos_en_la_ventana": clips_done,
        "clips_fallidos": after.clips.failed,
        "eventos_en_la_base": events,
    }


def main() -> None:
    args = ARGS
    settings.capture_hw_acceleration = args.hw
    settings.yolo_half = args.half
    ensure_database(args.db_name)

    detector = YOLODetector(weights_path=str(args.weights), device="cuda:0")
    import numpy as np

    detector.detect_batch([np.zeros((720, 1280, 3), dtype=np.uint8)] * 4)

    report = {
        "parametros": {
            "fps_de_analisis": args.analysis_fps, "decodificacion": args.hw, "fp16": args.half,
            "incidentes_por_camara_y_minuto": args.incidents_per_minute, "cierre_de_clip_s": args.clip_post_seconds,
            "calidad_jpeg_del_buffer": settings.clip_buffer_jpeg_quality, "buffer_previo_s": settings.frame_buffer_seconds,
            "hilos_de_clips": args.clip_workers or settings.clip_writer_workers,
        },
        "resultados": [],
    }
    try:
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "test_720p30.mp4"
            make_test_video(args.video, video, seconds=int(args.seconds + args.warmup + 15))
            for cameras in [int(x) for x in args.cameras.split(",")]:
                print(f"\n== {cameras} cámaras a {args.analysis_fps:g} fps | decodificación: {args.hw} | FP16: {args.half}", flush=True)
                result = run_case(detector, video, cameras, args)
                report["resultados"].append(result)
                print(
                    f"   imágenes/s: {result['imagenes_por_segundo_procesadas']} de {result['imagenes_por_segundo_pedidas']} pedidas"
                    f" = {result['cumplimiento']:.0%} | descartados {result['porcentaje_descartado']}%\n"
                    f"   lote medio {result['tamano_medio_de_lote']}, inferencia {result['inferencia_por_lote_ms_mediana']} ms"
                    f" ({result['inferencia_por_lote_ms_p95']} p95) | demora {result['demora_captura_a_resultado_ms_mediana']} ms"
                    f" ({result['demora_captura_a_resultado_ms_p95']} p95)\n"
                    f"   CPU {result['nucleos_cpu_usados']} núcleos | RAM {result['ram_mb']} MB | VRAM {result['vram_usada_mb']} MB"
                    f" | clips {result['clips_escritos_en_la_ventana']} (fallidos {result['clips_fallidos']}) | eventos {result['eventos_en_la_base']}",
                    flush=True,
                )
    finally:
        if not args.keep:
            drop_database(args.db_name)
            drop_bucket(args.bucket)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nResultados en {args.out}")


if __name__ == "__main__":
    main()
