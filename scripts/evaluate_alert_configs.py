"""Compara configuraciones de alerta de choque sobre las detecciones en caché del detector, sin reentrenar nada.

Uso:
    python scripts/evaluate_alert_configs.py --weights data/models/yolo/baseline_v6.pt --out ../reentreno/alertas_configs_v6.json
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import statistics
import sys
from datetime import datetime, timedelta
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

TOL = 3.0
BASE_CONFIG = {"nombre": "actual", "fps": 6.0, "directo": False, "conf": 0.6, "persist_s": 0.65, "ventana_s": 2.0, "dedup_s": 90.0,
               "dedup_modo": "global", "min_muestras": 3}


def build_configs() -> list[dict]:
    configs = [dict(BASE_CONFIG)]
    for fps, conf, modo in itertools.product((6.0, 15.0), (0.4, 0.5, 0.6, 0.65, 0.7), ("global", "ubicacion")):
        configs.append({"nombre": "disparador", "fps": fps, "directo": True, "conf": conf, "persist_s": 0.0, "ventana_s": 2.0,
                        "dedup_s": 90.0, "dedup_modo": modo, "min_muestras": 1})
    for fps, conf, persist, window, modo in itertools.product((6.0, 15.0), (0.4, 0.5, 0.6), (0.33, 0.67), (1.5,), ("global", "renovar", "ubicacion")):
        configs.append({"nombre": "acumula", "fps": fps, "directo": True, "conf": conf, "persist_s": persist, "ventana_s": window,
                        "dedup_s": 90.0, "dedup_modo": modo, "min_muestras": 1})
    return configs


class _Clock:
    now = 0.0

    @staticmethod
    def monotonic() -> float:
        return _Clock.now


def event_times(frames: list[dict], cfg: dict) -> list[float]:
    from monitoreo_urbano.common.config import settings
    from monitoreo_urbano.common.schemas import BoundingBox, Detection
    from monitoreo_urbano.vision_pipeline import event_trigger
    from monitoreo_urbano.vision_pipeline.camera_health import CameraHealthMonitor
    from monitoreo_urbano.vision_pipeline.camera_pipeline import CameraPipeline
    from monitoreo_urbano.vision_pipeline.camera_source import Frame
    from monitoreo_urbano.vision_pipeline.event_trigger import EventTriggerEngine
    from monitoreo_urbano.vision_pipeline.frame_buffer import FrameRingBuffer
    from monitoreo_urbano.vision_pipeline.tracker import ByteTrackTracker, Tracker

    class NoTracks(Tracker):
        def update(self, detections, frame):
            return []

    event_trigger.time = _Clock
    fps = cfg["fps"]
    settings.accident_direct_trigger = cfg["directo"]
    settings.incident_dedup_mode = cfg["dedup_modo"]
    settings.trigger_min_samples = cfg["min_muestras"]
    settings.incident_window_seconds = cfg["ventana_s"]
    settings.incident_min_seconds = {"fire": 0.8, "accident": max(cfg["persist_s"], 0.001)}
    tracker = ByteTrackTracker(frame_rate=fps, track_buffer_seconds=settings.track_buffer_seconds) if not cfg["directo"] else NoTracks()
    out: list[float] = []
    clock = [0.0]
    pipeline = CameraPipeline(
        "alertas", tracker=tracker, trigger_engine=EventTriggerEngine(fps=fps, dedup_window_seconds=cfg["dedup_s"]),
        health_monitor=CameraHealthMonitor(fps=fps, reconnect_delay_seconds=1.0, sustained_seconds=1e9, signal_loss_sustained_seconds=1e9),
        buffer=FrameRingBuffer(max_seconds=1, fps=fps), post_event_frames=1,
        on_clip=lambda c: out.append(clock[0]) if c.trigger_reason == "accident" else None, on_health=lambda r: None)
    black = np.zeros((4, 4, 3), dtype=np.uint8)
    base = datetime(2026, 1, 1)
    for index, item in enumerate(frames):
        clock[0] = _Clock.now = item["t"]
        frame = Frame(image=black, timestamp=base + timedelta(seconds=item["t"]), frame_index=index)
        pipeline.observe(frame)
        dets = [Detection(class_name="accident", confidence=c, bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2))
                for n, c, x1, y1, x2, y2 in item["dets"] if n == "accident" and c >= cfg["conf"]]
        pipeline.update(frame, dets)
    return out


def run_config(job: tuple) -> dict:
    import compare_models_domain as cmp
    import evaluate_classifier_events as ece
    import train_trajectory_classifier as ttc

    cfg, weights, split_stems = job
    weights = Path(weights)
    wid = f"{weights.name}-{weights.stat().st_size}-{int(weights.stat().st_mtime)}"
    meta = {Path(r["path"]).stem: r for r in csv.DictReader(open(ece.SOURCE / "metadata-real.csv", encoding="utf-8"))}
    result = {"cfg": cfg, "videos": {}}
    for stem in split_stems:
        frames = json.loads(ttc.cache_path(wid, ece.SOURCE / meta[stem]["path"], 960, cfg["fps"]).read_text(encoding="utf-8"))["frames"]
        result["videos"][stem] = {"t0": float(meta[stem]["accident_time"]), "eventos": event_times(frames, cfg)}
    frames = json.loads(ttc.cache_path(wid, ece.V7, 960, cfg["fps"]).read_text(encoding="utf-8"))["frames"]
    result["v7"] = {"duracion": frames[-1]["t"], "eventos": event_times(frames, cfg)}
    return result


def summarize(res: dict, split_of: dict, split: str) -> dict:
    n = det = strict = false = 0
    latencies, per_video = [], []
    for stem, v in res["videos"].items():
        if split_of[stem] != split:
            continue
        n += 1
        ev, t0 = v["eventos"], v["t0"]
        after = [t for t in ev if t >= t0 - TOL]
        det += bool(after)
        strict += any(t >= t0 for t in ev)
        false += any(t < t0 - TOL for t in ev)
        per_video.append(len(ev))
        if after:
            latencies.append(after[0] - t0)
    latencies.sort()
    return {"n": n, "detectados": det, "estrictos": strict, "videos_con_falsa_previa": false,
            "demora_mediana_s": round(statistics.median(latencies), 2) if latencies else None,
            "demora_p90_s": round(latencies[int(0.9 * (len(latencies) - 1))], 2) if latencies else None,
            "eventos_por_video_max": max(per_video) if per_video else 0, "videos_con_mas_de_un_evento": sum(c > 1 for c in per_video)}


def main() -> None:
    import compare_models_domain as cmp
    import evaluate_classifier_events as ece

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, default=ROOT / "data" / "models" / "yolo" / "baseline_v6.pt")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--todos", action="store_true", help="No filtrar por dominio de cámara")
    parser.add_argument("--solo", default=None, help="Solo configuraciones cuyo nombre esté en esta lista separada por comas")
    args = parser.parse_args()
    meta = {Path(r["path"]).stem: r for r in csv.DictReader(open(ece.SOURCE / "metadata-real.csv", encoding="utf-8"))}
    split_of = {}
    for split in ("val", "test"):
        for stem in sorted(ece.split_stems(split)):
            if args.todos or cmp.in_domain(meta[stem]):
                split_of[stem] = split
    configs = build_configs()
    if args.solo:
        configs = [c for c in configs if c["nombre"] in args.solo.split(",")]
    jobs = [(c, str(args.weights), sorted(split_of)) for c in configs]
    with Pool(args.workers) as pool:
        results = pool.map(run_config, jobs, chunksize=1)
    rows = []
    for res in results:
        v7 = res["v7"]
        horas = v7["duracion"] / 3600
        t = sorted(v7["eventos"])
        rows.append({"cfg": res["cfg"], "val": summarize(res, split_of, "val"), "test": summarize(res, split_of, "test"),
                     "v7": {"eventos": len(t), "por_hora": round(len(t) / horas, 2), "repeticiones_lt120s": sum(1 for a, b in zip(t, t[1:]) if b - a < 120),
                            "instantes": [round(x) for x in t]}})
    args.out.write_text(json.dumps({"filas": rows}, indent=1, ensure_ascii=False), encoding="utf-8")
    print("escrito", args.out, len(rows), "configuraciones")


if __name__ == "__main__":
    main()
