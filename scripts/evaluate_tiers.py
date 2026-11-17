"""Niveles de certeza del choque con el código real del sistema y sus valores por defecto, sobre las detecciones en caché de v6.

Uso:
    python scripts/evaluate_tiers.py --split test --out ../reentreno/niveles_prueba.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

TOL = 3.0


class _Clock:
    now = 0.0

    @staticmethod
    def monotonic() -> float:
        return _Clock.now


def events_of(frames: list[dict], fps: float = 6.0) -> list[tuple[float, str]]:
    from monitoreo_urbano.common.config import settings
    from monitoreo_urbano.common.schemas import BoundingBox, Detection
    from monitoreo_urbano.vision_pipeline import event_trigger
    from monitoreo_urbano.vision_pipeline.camera_health import CameraHealthMonitor
    from monitoreo_urbano.vision_pipeline.camera_pipeline import CameraPipeline
    from monitoreo_urbano.vision_pipeline.camera_source import Frame
    from monitoreo_urbano.vision_pipeline.event_trigger import EventTriggerEngine
    from monitoreo_urbano.vision_pipeline.frame_buffer import FrameRingBuffer
    from monitoreo_urbano.vision_pipeline.tracker import Tracker

    class NoTracks(Tracker):
        def update(self, detections, frame):
            return []

    event_trigger.time = _Clock
    out: list[tuple[float, str]] = []
    clock = [0.0]
    floor = settings.yolo_confidence_by_class.get("accident", settings.yolo_confidence_threshold)
    pipeline = CameraPipeline(
        "niveles", tracker=NoTracks(), trigger_engine=EventTriggerEngine(fps=fps, dedup_window_seconds=settings.event_dedup_window_seconds),
        health_monitor=CameraHealthMonitor(fps=fps, reconnect_delay_seconds=1.0, sustained_seconds=1e9, signal_loss_sustained_seconds=1e9),
        buffer=FrameRingBuffer(max_seconds=1, fps=fps), post_event_frames=1,
        on_clip=lambda c: out.append((round(clock[0], 2), c.trigger_reason)), on_health=lambda r: None)
    black = np.zeros((4, 4, 3), dtype=np.uint8)
    base = datetime(2026, 1, 1)
    for index, item in enumerate(frames):
        clock[0] = _Clock.now = item["t"]
        frame = Frame(image=black, timestamp=base + timedelta(seconds=item["t"]), frame_index=index)
        pipeline.observe(frame)
        dets = [Detection(class_name="accident", confidence=c, bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2))
                for n, c, x1, y1, x2, y2 in item["dets"] if n == "accident" and c >= floor]
        pipeline.update(frame, dets)
    return out


def main() -> None:
    import compare_models_domain as cmp
    import evaluate_classifier_events as ece
    import train_trajectory_classifier as ttc

    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test")
    parser.add_argument("--weights", type=Path, default=ROOT / "data" / "models" / "yolo" / "baseline_v6.pt")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--todos", action="store_true", help="No filtrar por dominio de cámara: todos los videos de la partición")
    args = parser.parse_args()
    meta = {Path(r["path"]).stem: r for r in csv.DictReader(open(ece.SOURCE / "metadata-real.csv", encoding="utf-8"))}
    wid = f"{args.weights.name}-{args.weights.stat().st_size}-{int(args.weights.stat().st_mtime)}"
    rows = []
    for stem in sorted(ece.split_stems(args.split)):
        if not args.todos and not cmp.in_domain(meta[stem]):
            continue
        frames = json.loads(ttc.cache_path(wid, ece.SOURCE / meta[stem]["path"], 960).read_text(encoding="utf-8"))["frames"]
        t0 = float(meta[stem]["accident_time"])
        ev = events_of(frames)
        alerts = [t - t0 for t, r in ev if r == "accident"]
        reviews = [t - t0 for t, r in ev if r == "posible_choque"]
        if any(a >= -TOL for a in alerts):
            nivel = "ALERTA"
        elif alerts:
            nivel = "ALERTA TEMPRANA"
        elif reviews:
            nivel = "REVISIÓN"
        else:
            nivel = "NO REGISTRADO"
        rows.append({"video": stem, "t0": round(t0, 1), "alertas_s": [round(a, 1) for a in alerts], "revisiones_s": [round(r, 1) for r in reviews], "nivel": nivel})
    frames = json.loads(ttc.cache_path(wid, ece.V7, 960).read_text(encoding="utf-8"))["frames"]
    ev7 = events_of(frames)
    hours = frames[-1]["t"] / 3600
    n = len(rows)
    c = Counter(r["nivel"] for r in rows)
    both = sum(1 for r in rows if r["alertas_s"] and r["revisiones_s"])
    print(f"{args.split}: {n} videos del dominio")
    for k in ("ALERTA", "ALERTA TEMPRANA", "REVISIÓN", "NO REGISTRADO"):
        print(f"  {k:<16} {c[k]:>3} ({c[k] / n:.0%})")
    print(f"  Registrados (alerta o revisión): {n - c['NO REGISTRADO']}/{n} ({(n - c['NO REGISTRADO']) / n:.0%})")
    print(f"  Videos con alerta y además un pendiente de revisión: {both}")
    print(f"  Videos con más de una alerta: {sum(len(r['alertas_s']) > 1 for r in rows)} | con más de un pendiente: {sum(len(r['revisiones_s']) > 1 for r in rows)}")
    a7 = [t for t, r in ev7 if r == "accident"]
    r7 = [t for t, r in ev7 if r == "posible_choque"]
    print(f"  V7 sin choque ({hours * 60:.0f} min): alertas {len(a7)} ({len(a7) / hours:.0f}/h) en {[round(x) for x in a7]}; pendientes de revisión {len(r7)} ({len(r7) / hours:.0f}/h) en {[round(x) for x in r7]}")
    if args.out:
        args.out.write_text(json.dumps({"videos": rows, "v7": {"alertas": a7, "revisiones": r7, "horas": hours}}, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
