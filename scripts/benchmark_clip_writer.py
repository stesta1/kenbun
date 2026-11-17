"""Fase 8: ¿cuánto cuesta escribir un clip (decodificar el buffer JPEG, codificar el MP4 y subirlo a MinIO)?

Uso:
    python scripts/benchmark_clip_writer.py [--repeat 3]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MU_S3_BUCKET_CLIPS", "bench-clips-writer")

from monitoreo_urbano.common.config import settings  # noqa: E402
from monitoreo_urbano.vision_pipeline import clip_writer  # noqa: E402
from monitoreo_urbano.vision_pipeline.frame_encoding import EncodedFrame, encode_jpeg  # noqa: E402


def build_frames(video: Path, count: int) -> list[EncodedFrame]:
    cap = cv2.VideoCapture(str(video))
    frames: list[EncodedFrame] = []
    while len(frames) < count:
        ok, image = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        image = cv2.resize(image, (1280, 720))
        frames.append(
            EncodedFrame(
                jpeg=encode_jpeg(image, settings.clip_buffer_jpeg_quality), timestamp=datetime.utcnow(), frame_index=len(frames)
            )
        )
    cap.release()
    return frames


def measure(label: str, frames: list[EncodedFrame], fps: float, repeat: int) -> None:
    from monitoreo_urbano.common.object_storage import get_clips_client

    client = get_clips_client()
    walls, cpus = [], []
    size_mb = 0.0
    for index in range(repeat):
        wall0, cpu0 = time.perf_counter(), time.process_time()
        key = clip_writer.write_clip("bench", f"{label}-{index}", frames, fps)
        walls.append(time.perf_counter() - wall0)
        cpus.append(time.process_time() - cpu0)
        size_mb = client.stat_object(settings.s3_bucket_clips, key).size / 2**20
    print(
        f"{label:<8} {len(frames)} fotogramas a {fps:g} fps ({len(frames) / fps:.0f} s de video): "
        f"{sum(walls) / repeat:5.2f} s de reloj, {sum(cpus) / repeat:5.2f} s de CPU del proceso, {size_mb:.1f} MB"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, default=ROOT / "data" / "raw_videos" / "sample_bus.mp4")
    parser.add_argument("--fps", type=float, default=6.0)
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()

    seconds = settings.frame_buffer_seconds + settings.clip_post_event_seconds
    frames = build_frames(args.video, int(seconds * args.fps))
    print(f"Clip de {seconds:.0f} s ({settings.frame_buffer_seconds:.0f} previos + {settings.clip_post_event_seconds:.0f} posteriores)")

    settings.clip_use_nvenc = False
    measure("opencv", frames, args.fps, args.repeat)
    if clip_writer._has_ffmpeg_nvenc():
        settings.clip_use_nvenc = True
        measure("nvenc", frames, args.fps, args.repeat)
    else:
        print("nvenc    no disponible en el ffmpeg instalado")

    from monitoreo_urbano.common.object_storage import get_minio_client

    client = get_minio_client()
    for obj in client.list_objects(settings.s3_bucket_clips, recursive=True):
        client.remove_object(settings.s3_bucket_clips, obj.object_name)
    client.remove_bucket(settings.s3_bucket_clips)


if __name__ == "__main__":
    main()
