"""Fase 8: ¿cuánto cuesta por fotograma todo lo que hace una instancia además de la inferencia?

Uso:
    python scripts/profile_pipeline_cost.py [--frames 300]
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from monitoreo_urbano.common.config import settings  # noqa: E402
from monitoreo_urbano.vision_pipeline.camera_pipeline import CameraPipeline  # noqa: E402
from monitoreo_urbano.vision_pipeline.camera_source import Frame  # noqa: E402
from monitoreo_urbano.vision_pipeline.detector import YOLODetector  # noqa: E402
from monitoreo_urbano.vision_pipeline.frame_encoding import compute_health_signals, encode_jpeg, make_capture_hook  # noqa: E402


def timed(fn, *args, repeat: int = 1):
    started = time.perf_counter()
    for _ in range(repeat):
        out = fn(*args)
    return out, (time.perf_counter() - started) * 1000 / repeat


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, default=ROOT / "data" / "raw_videos" / "sample_bus.mp4")
    parser.add_argument("--weights", type=Path, default=ROOT / "data" / "models" / "yolo" / "baseline_v1.pt")
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--fps", type=float, default=6.0)
    args = parser.parse_args()

    cap = cv2.VideoCapture(str(args.video))
    images = []
    while len(images) < args.frames:
        ok, image = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        images.append(cv2.resize(image, (1280, 720)))
    cap.release()

    detector = YOLODetector(weights_path=str(args.weights), device="cuda:0")
    detector.detect_batch(images[:4])
    detections = []
    for start in range(0, len(images), 16):
        detections.extend(detector.detect_batch(images[start : start + 16]))
    mean_objects = statistics.mean(len(d) for d in detections)

    hook = make_capture_hook(settings.clip_buffer_jpeg_quality)
    pipeline = CameraPipeline.build(
        "cam-profile", fps=args.fps, reconnect_delay_seconds=3.0, thresholds=None,
        on_clip=lambda clip: None, on_health=lambda status: None,
    )

    jpeg_ms, health_ms, hook_ms, observe_ms, update_ms = [], [], [], [], []
    for index, (image, detection) in enumerate(zip(images, detections, strict=True)):
        _, ms = timed(encode_jpeg, image, settings.clip_buffer_jpeg_quality)
        jpeg_ms.append(ms)
        _, ms = timed(compute_health_signals, image)
        health_ms.append(ms)

        frame = Frame(image=image.copy(), timestamp=datetime.utcnow(), frame_index=index)
        _, ms = timed(hook, frame)
        hook_ms.append(ms)
        _, ms = timed(pipeline.observe, frame)
        observe_ms.append(ms)
        _, ms = timed(pipeline.update, frame, detection)
        update_ms.append(ms)

    def row(name: str, values: list[float]) -> str:
        ordered = sorted(values)
        return f"{name:<34} media {statistics.mean(values):6.2f} ms | mediana {statistics.median(values):6.2f} | p95 {ordered[int(len(ordered) * 0.95)]:6.2f}"

    print(f"{len(images)} fotogramas 1280x720, {mean_objects:.1f} detecciones por fotograma en promedio\n")
    print(row("JPEG (calidad %d)" % settings.clip_buffer_jpeg_quality, jpeg_ms))
    print(row("señales de salud (std + nitidez)", health_ms))
    print(row("gancho de captura completo", hook_ms))
    print(row("observe (buffer + salud) [motor]", observe_ms))
    print(row("update (seguimiento + reglas) [motor]", update_ms))
    per_result = statistics.mean(observe_ms) + statistics.mean(update_ms)
    print(f"\nHilo del motor: {per_result:.2f} ms por resultado -> tope de {1000 / per_result:.0f} resultados/s por instancia")
    print(f"Hilos de captura: {statistics.mean(hook_ms):.2f} ms por fotograma analizado")


if __name__ == "__main__":
    main()
