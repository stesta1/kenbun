"""Valida el camino directo integrado en `CameraPipeline` con las detecciones en caché y la deduplicación de 90 s del sistema.

Uso:
    python scripts/validate_direct_pipeline.py --weights data/models/yolo/baseline_v6.pt --fps 15 --conf 0.7 --min-seconds 0.167 --window 1.5
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import compare_models_domain as cmp  # noqa: E402
import evaluate_classifier_events as ece  # noqa: E402
import train_trajectory_classifier as ttc  # noqa: E402
from monitoreo_urbano.common.config import settings  # noqa: E402
from monitoreo_urbano.common.schemas import BoundingBox, Detection  # noqa: E402
from monitoreo_urbano.vision_pipeline.camera_health import CameraHealthMonitor  # noqa: E402
from monitoreo_urbano.vision_pipeline.camera_pipeline import CameraPipeline  # noqa: E402
from monitoreo_urbano.vision_pipeline.camera_source import Frame  # noqa: E402
from monitoreo_urbano.vision_pipeline.event_trigger import EventTriggerEngine  # noqa: E402
from monitoreo_urbano.vision_pipeline.frame_buffer import FrameRingBuffer  # noqa: E402
from monitoreo_urbano.vision_pipeline.tracker import Tracker  # noqa: E402

TOL = 3.0


class NoTracks(Tracker):
    def update(self, detections, frame):
        return []


def trigger_times(frames: list[dict], fps: float, conf: float, dedup: float, baseline: bool = False) -> list[float]:
    out: list[float] = []
    now = [0.0]
    if baseline:
        from monitoreo_urbano.vision_pipeline.tracker import ByteTrackTracker

        tracker = ByteTrackTracker(frame_rate=fps, track_buffer_seconds=settings.track_buffer_seconds)
    else:
        tracker = NoTracks()
    pipeline = CameraPipeline(
        "validacion", tracker=tracker, trigger_engine=EventTriggerEngine(fps=fps, dedup_window_seconds=dedup),
        health_monitor=CameraHealthMonitor(fps=fps, reconnect_delay_seconds=1.0, sustained_seconds=3600.0, signal_loss_sustained_seconds=3600.0),
        buffer=FrameRingBuffer(max_seconds=1, fps=fps), post_event_frames=1, on_clip=lambda c: out.append(now[0]), on_health=lambda r: None)
    black = np.zeros((4, 4, 3), dtype=np.uint8)
    base = datetime(2026, 1, 1)
    for index, item in enumerate(frames):
        now[0] = item["t"]
        frame = Frame(image=black, timestamp=base + timedelta(seconds=item["t"]), frame_index=index)
        pipeline.observe(frame)
        if baseline:
            dets = [Detection(class_name=n, confidence=c, bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2))
                    for n, c, x1, y1, x2, y2 in item["dets"]
                    if c >= settings.yolo_confidence_by_class.get(n, settings.yolo_confidence_threshold)]
        else:
            dets = [Detection(class_name="accident", confidence=c, bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2))
                    for n, c, x1, y1, x2, y2 in item["dets"] if n == "accident" and c >= conf]
        pipeline.update(frame, dets)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--conf", type=float, default=0.6)
    parser.add_argument("--min-seconds", type=float, default=0.65)
    parser.add_argument("--window", type=float, default=2.0)
    parser.add_argument("--dedup", type=float, default=90.0)
    parser.add_argument("--baseline", action="store_true", help="Camino actual del sistema (seguimiento, umbrales por clase); ignora --conf")
    args = parser.parse_args()
    settings.accident_direct_trigger = not args.baseline
    settings.incident_min_seconds = {"fire": 0.8, "accident": args.min_seconds}
    settings.incident_window_seconds = args.window
    wid = f"{args.weights.name}-{args.weights.stat().st_size}-{int(args.weights.stat().st_mtime)}"
    meta = {Path(r["path"]).stem: r for r in csv.DictReader(open(ece.SOURCE / "metadata-real.csv", encoding="utf-8"))}
    res = {}
    for split in ("val", "test"):
        n = det = falsos = 0
        for stem in sorted(ece.split_stems(split)):
            if not cmp.in_domain(meta[stem]):
                continue
            fr = json.loads(ttc.cache_path(wid, ece.SOURCE / meta[stem]["path"], 960, args.fps).read_text(encoding="utf-8"))["frames"]
            t0 = float(meta[stem]["accident_time"])
            t = trigger_times(fr, args.fps, args.conf, args.dedup, args.baseline)
            n += 1
            det += any(x >= t0 - TOL for x in t)
            falsos += any(x < t0 - TOL for x in t)
        res[split] = (det, n, falsos)
    fr7 = json.loads(ttc.cache_path(wid, ece.V7, 960, args.fps).read_text(encoding="utf-8"))["frames"]
    t7 = trigger_times(fr7, args.fps, args.conf, args.dedup, args.baseline)
    horas = fr7[-1]["t"] / 3600
    print(f"conf {args.conf}, persistencia {args.min_seconds:g} s, ventana {args.window:g} s, {args.fps:g} fps, dedup {args.dedup:g} s")
    for split, (det, n, falsos) in res.items():
        print(f"  {split}: detectados {det}/{n} ({det/n:.2f}), videos con falsa previa {falsos}")
    print(f"  V7: {len(t7)} eventos en {horas*60:.1f} min = {len(t7)/horas:.1f}/h  instantes {[round(x) for x in t7]}")


if __name__ == "__main__":
    main()
