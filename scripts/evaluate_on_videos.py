"""Evalúa un detector sobre videos reales con incidentes conocidos, por el seguimiento y las reglas del sistema.

Uso:
    python scripts/evaluate_on_videos.py --weights data/models/yolo/baseline_v2.pt --out ../reentreno/eval_v2.json
    python scripts/evaluate_on_videos.py --weights ... --conf fire=0.5,accident=0.6,default=0.35
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from monitoreo_urbano.common.config import settings  # noqa: E402
from monitoreo_urbano.common.schemas import BoundingBox, Detection  # noqa: E402
from monitoreo_urbano.vision_pipeline.camera_source import Frame  # noqa: E402
from monitoreo_urbano.vision_pipeline.event_trigger import EventTriggerEngine  # noqa: E402
from monitoreo_urbano.vision_pipeline.tracker import ByteTrackTracker  # noqa: E402

VIDEOS_DIR = ROOT.parent / "test_ui" / "videos"
EXPECTED = {
    "V1.mp4": "accident", "V2.mp4": "accident", "V3.mp4": "accident", "V4.mp4": "accident",
    "V5.mp4": "accident", "V6.mp4": "fire", "V8.mp4": "accident", "V7.mp4": None, "V7 uncut.mp4": None,
}
CACHE_CONFIDENCE = 0.2
IMGSZ: list[int | None] = [None]


def parse_conf(text: str | None) -> dict[str, float]:
    result: dict[str, float] = {}
    for part in (text or "").split(","):
        if part.strip():
            name, value = part.split("=")
            result[name.strip()] = float(value)
    return result


def detect_video(model, video: Path, fps: float, cache_dir: Path, weights_id: str) -> tuple[list[dict], float, float]:
    key = hashlib.md5(f"{weights_id}|{video.name}|{fps}|{CACHE_CONFIDENCE}|{IMGSZ[0]}".encode()).hexdigest()[:12]
    path = cache_dir / f"{video.stem.replace(' ', '_')}_{key}.json"
    if path.exists():
        cached = json.loads(path.read_text(encoding="utf-8"))
        return cached["frames"], cached["duration"], cached["native_fps"]

    cap = cv2.VideoCapture(str(video))
    native = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, round(native / fps))
    frames: list[dict] = []
    batch_images, batch_times = [], []
    index = 0

    def flush() -> None:
        if not batch_images:
            return
        extra = {"imgsz": IMGSZ[0]} if IMGSZ[0] else {}
        results = model.predict(batch_images, conf=CACHE_CONFIDENCE, device="cuda:0", verbose=False, **extra)
        for t, result in zip(batch_times, results, strict=True):
            dets = []
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                dets.append([result.names[int(box.cls[0])], round(float(box.conf[0]), 4), x1, y1, x2, y2])
            frames.append({"t": round(t, 3), "dets": dets})
        batch_images.clear()
        batch_times.clear()

    while True:
        ok = cap.grab()
        if not ok:
            break
        if index % stride == 0:
            ok, image = cap.retrieve()
            if ok:
                batch_images.append(image)
                batch_times.append(index / native)
                if len(batch_images) == 16:
                    flush()
        index += 1
    flush()
    cap.release()
    duration = index / native
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"frames": frames, "duration": duration, "native_fps": native}), encoding="utf-8")
    return frames, duration, native


def replay(frames: list[dict], fps: float, thresholds: dict[str, float], default: float) -> dict:
    settings.event_dedup_window_seconds = 0.0
    tracker = ByteTrackTracker(frame_rate=fps, track_buffer_seconds=settings.track_buffer_seconds)
    engine = EventTriggerEngine(fps=fps, dedup_window_seconds=0.0)
    base = datetime(2026, 1, 1)
    triggers: list[dict] = []
    max_conf: dict[str, float] = {}
    black = np.zeros((8, 8, 3), dtype=np.uint8)

    for index, item in enumerate(frames):
        detections = []
        for name, conf, x1, y1, x2, y2 in item["dets"]:
            if conf < thresholds.get(name, default):
                continue
            max_conf[name] = max(max_conf.get(name, 0.0), conf)
            detections.append(Detection(class_name=name, confidence=conf, bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)))
        frame = Frame(image=black, timestamp=base + timedelta(seconds=item["t"]), frame_index=index)
        tracked = tracker.update(detections, frame)
        for track_id in engine.update(tracked):
            triggers.append({"t": item["t"], "reason": engine.reason_for(track_id), "track_id": track_id})
    return {"triggers": triggers, "max_conf": {k: round(v, 3) for k, v in max_conf.items()}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, default=ROOT / "data" / "models" / "yolo" / "baseline_v1.pt")
    parser.add_argument("--videos", type=Path, default=VIDEOS_DIR)
    parser.add_argument("--fps", type=float, default=6.0, help="Frecuencia de análisis")
    parser.add_argument("--conf", default=None, help="Umbrales por clase, ej. fire=0.5,accident=0.6,default=0.35")
    parser.add_argument("--only", default=None, help="Nombres de video separados por coma (por defecto, todos)")
    parser.add_argument("--cache", type=Path, default=ROOT / "data" / "eval_cache")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--imgsz", type=int, default=None, help="Resolución de inferencia (por defecto, la del entrenamiento del modelo)")
    args = parser.parse_args()
    IMGSZ[0] = args.imgsz

    thresholds = parse_conf(args.conf) or dict(settings.yolo_confidence_by_class)
    default = thresholds.pop("default", settings.yolo_confidence_threshold)

    from ultralytics import YOLO

    model = YOLO(str(args.weights))
    weights_id = f"{args.weights.name}-{args.weights.stat().st_size}-{int(args.weights.stat().st_mtime)}"
    wanted = set(args.only.split(",")) if args.only else None

    print(f"Pesos {args.weights.name} | {args.fps:g} fps | umbrales {thresholds} (resto {default})\n")
    rows = []
    for name, expected in EXPECTED.items():
        video = args.videos / name
        if not video.exists() or (wanted and name not in wanted and video.stem not in wanted):
            continue
        started = time.time()
        frames, duration, native = detect_video(model, video, args.fps, args.cache, weights_id)
        result = replay(frames, args.fps, thresholds, default)
        reasons = [t["reason"] for t in result["triggers"]]
        incident = [t for t in result["triggers"] if t["reason"] in ("fire", "accident")]
        review = [t for t in result["triggers"] if t["reason"] == "posible_altercado"]
        hit = expected is not None and any(t["reason"] == expected for t in incident)
        wrong = [t for t in incident if t["reason"] != expected]
        first = next((t["t"] for t in incident if t["reason"] == expected), None)
        row = {
            "video": name, "esperado": expected, "duracion_s": round(duration, 1), "acierto": hit if expected else None,
            "primer_disparo_s": first, "disparos_de_incidente": len(incident),
            "falsos_de_incidente": len(wrong) if expected else len(incident),
            "para_revision": len(review), "max_conf": result["max_conf"],
            "disparos": result["triggers"][:12], "segundos_de_calculo": round(time.time() - started, 1),
        }
        rows.append(row)
        verdict = "—" if expected is None else ("OK" if hit else "NO DETECTA")
        fp = row["falsos_de_incidente"]
        per_hour = f" ({fp / (duration / 3600):.1f}/h)" if expected is None and duration > 600 else ""
        print(f"{name:<14} esperado {str(expected):<9} {verdict:<11} primer disparo {first}  incidentes {len(incident)}  "
              f"falsos {fp}{per_hour}  revisión {len(review)}  máx {result['max_conf']}", flush=True)

    with_incident = [r for r in rows if r["esperado"]]
    summary = {
        "videos_con_incidente": len(with_incident),
        "detectados": sum(1 for r in with_incident if r["acierto"]),
        "falsos_en_videos_sin_incidente": sum(r["falsos_de_incidente"] for r in rows if r["esperado"] is None),
        "falsos_en_videos_con_incidente": sum(r["falsos_de_incidente"] for r in with_incident),
    }
    print(f"\nResumen: {summary}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps({"pesos": args.weights.name, "fps": args.fps, "umbrales": thresholds, "resto": default,
                        "resumen": summary, "videos": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Resultados en {args.out}")


if __name__ == "__main__":
    main()
