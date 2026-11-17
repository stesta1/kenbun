"""Fase 2 del plan de detector compartido: ¿cuánto dependen las reglas de disparo de los fps de análisis?

Uso:
    python scripts/replay_rules_at_fps.py synthetic --out ../reentreno/fase2_sintetico.json
    python scripts/replay_rules_at_fps.py video --video data/raw_videos/sample_bus.mp4
"""

from __future__ import annotations

import argparse
import json
import pickle
import statistics
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from monitoreo_urbano.common.schemas import BoundingBox, Detection, TrackedObject  # noqa: E402
from monitoreo_urbano.vision_pipeline.event_trigger import (  # noqa: E402
    EventTriggerEngine,
    MinDurationRule,
    default_rules,
    frames_for_duration,
)

NATIVE_FPS = 30
SECONDS = 8
BASE_TIME = datetime(2026, 1, 1)
DEFAULT_FPS = (30, 15, 10, 6, 5)


Track = tuple[int, str, np.ndarray]


def _jitter(rng: np.random.Generator, centers: np.ndarray, sigma: float) -> np.ndarray:
    return centers + rng.normal(0.0, sigma, size=centers.shape)


def _line(start: tuple[float, float], velocity: tuple[float, float], n: int) -> np.ndarray:
    steps = np.arange(n)[:, None]
    return np.array(start) + steps * np.array(velocity)


def _wander(rng: np.random.Generator, start: tuple[float, float], n: int, step_sigma: float) -> np.ndarray:
    return np.array(start) + np.cumsum(rng.normal(0.0, step_sigma, size=(n, 2)), axis=0)


def _erratic(rng: np.random.Generator, start: tuple[float, float], n: int) -> np.ndarray:
    centers = np.zeros((n, 2))
    position = np.array(start, dtype=float)
    index = 0
    while index < n:
        hold = int(rng.integers(NATIVE_FPS // 5, NATIVE_FPS * 2 // 5 + 1))
        angle = rng.uniform(0, 2 * np.pi)
        speed = rng.uniform(4, 9)
        velocity = np.array([np.cos(angle), np.sin(angle)]) * speed
        for _ in range(hold):
            if index >= n:
                break
            position = position + velocity
            centers[index] = position
            index += 1
    return centers


def scenario_walking_apart(rng, n) -> list[Track]:
    return [(1, "person", _line((200, 400), (1.5, 0), n)), (2, "person", _line((700, 420), (1.4, 0.1), n))]


def scenario_talking(rng, n) -> list[Track]:
    return [(1, "person", _wander(rng, (400, 400), n, 0.3)), (2, "person", _wander(rng, (470, 405), n, 0.3))]


def scenario_running_together(rng, n) -> list[Track]:
    return [(1, "person", _line((100, 400), (8, 0), n)), (2, "person", _line((100, 460), (8, 0), n))]


def scenario_fight(rng, n) -> list[Track]:
    first = _erratic(rng, (500, 400), n)
    second = first + np.array([60.0, 10.0]) + _wander(rng, (0, 0), n, 1.0)
    return [(1, "person", first), (2, "person", second)]


def scenario_fire(rng, n) -> list[Track]:
    return [(1, "fire", _line((600, 300), (0, 0), n))]


SCENARIOS = {
    "caminan separados": (scenario_walking_apart, "no debe disparar"),
    "conversan quietos": (scenario_talking, "no debe disparar"),
    "corren juntos en línea recta": (scenario_running_together, "no debe disparar"),
    "pelea (movimiento brusco)": (scenario_fight, "debe disparar posible_altercado"),
    "fuego": (scenario_fire, "debe disparar"),
}


def _box(class_name: str, center: np.ndarray) -> BoundingBox:
    width, height = (60, 60) if class_name == "fire" else (40, 90)
    return BoundingBox(x1=center[0] - width / 2, y1=center[1] - height / 2, x2=center[0] + width / 2, y2=center[1] + height / 2)


def run_scenario(tracks: list[Track], fps: float, engine: EventTriggerEngine) -> float | None:
    n_native = len(tracks[0][2])
    samples = int((n_native - 1) * fps / NATIVE_FPS) + 1
    for k in range(samples):
        native_index = round(k * NATIVE_FPS / fps)
        objects = [
            TrackedObject(
                track_id=track_id,
                class_name=class_name,
                bbox=_box(class_name, centers[native_index]),
                confidence=0.9,
                frame_index=k,
                timestamp=BASE_TIME + timedelta(seconds=native_index / NATIVE_FPS),
            )
            for track_id, class_name, centers in tracks
        ]
        if engine.update(objects):
            return native_index / NATIVE_FPS
    return None


def synthetic_mode(args: argparse.Namespace) -> None:
    fps_list = [float(x) for x in args.fps.split(",")]
    n = SECONDS * NATIVE_FPS
    report: dict = {"parametros": {"semillas": args.seeds, "jitter_px": args.jitter, "segundos": SECONDS}, "resultados": []}

    print(f"Escenarios sintéticos: {args.seeds} semillas, ruido de detección de {args.jitter} px, {SECONDS} s a {NATIVE_FPS} fps")
    for name, (builder, expectation) in SCENARIOS.items():
        print(f"\n== {name}  ({expectation})")
        print(f"   {'fps':>4} | {'histórico (frames)':^26} | {'en segundos':^26}")
        print(f"   {'':>4} | {'dispara':>9} {'demora mediana':>16} | {'dispara':>9} {'demora mediana':>16}")
        for fps in fps_list:
            row = {"escenario": name, "fps": fps}
            for label, make_engine in (
                ("historico", lambda f: EventTriggerEngine()),
                ("segundos", lambda f: EventTriggerEngine(fps=f)),
            ):
                latencies: list[float] = []
                fired = 0
                for seed in range(args.seeds):
                    rng = np.random.default_rng(seed)
                    tracks = [
                        (track_id, class_name, _jitter(rng, centers, args.jitter))
                        for track_id, class_name, centers in builder(rng, n)
                    ]
                    when = run_scenario(tracks, fps, make_engine(fps))
                    if when is not None:
                        fired += 1
                        latencies.append(when)
                row[label] = {
                    "tasa_de_disparo": round(fired / args.seeds, 3),
                    "demora_mediana_s": round(statistics.median(latencies), 2) if latencies else None,
                }
            report["resultados"].append(row)
            h, s = row["historico"], row["segundos"]
            fmt = lambda r: f"{r['tasa_de_disparo']:>8.0%} {('%.2f s' % r['demora_mediana_s']) if r['demora_mediana_s'] is not None else '-':>16}"  # noqa: E731
            print(f"   {fps:>4.0f} | {fmt(h)} | {fmt(s)}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nResultados en {args.out}")


def detect_all(video: Path, weights: Path, cache: Path) -> tuple[list[list[Detection]], float]:
    if cache.exists():
        with cache.open("rb") as fh:
            return pickle.load(fh)

    import cv2
    from ultralytics import YOLO

    cap = cv2.VideoCapture(str(video))
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    model = YOLO(str(weights))
    detections: list[list[Detection]] = []
    while True:
        ok, image = cap.read()
        if not ok:
            break
        result = model.predict(source=image, device=0, conf=0.35, verbose=False)[0]
        frame_dets = []
        for box in result.boxes:
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            frame_dets.append(
                Detection(
                    class_name=result.names[int(box.cls[0].item())],
                    confidence=float(box.conf[0].item()),
                    bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                )
            )
        detections.append(frame_dets)
    cap.release()
    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.open("wb") as fh:
        pickle.dump((detections, native_fps), fh)
    return detections, native_fps


def replay_video(detections, native_fps: float, stride: int, old_track_memory: bool) -> dict:
    from monitoreo_urbano.vision_pipeline.camera_source import Frame
    from monitoreo_urbano.vision_pipeline.tracker import ByteTrackTracker

    fps = native_fps / stride
    tracker = ByteTrackTracker(frame_rate=fps, track_buffer_seconds=3.0)
    if old_track_memory:
        old_value = int(fps / 30.0 * int(3.0 * fps))
        tracker._tracker.buffer_size = old_value
        tracker._tracker.max_time_lost = old_value
    persistence = MinDurationRule(min_frames=frames_for_duration(1.0, fps, minimum=3))
    engine = EventTriggerEngine(rules=[persistence, *default_rules(fps)], fps=fps)

    blank = np.zeros((4, 4, 3), dtype=np.uint8)
    track_ids: set[int] = set()
    lengths: dict[int, int] = {}
    triggers: list[tuple[float, int, str | None]] = []
    for k, native_index in enumerate(range(0, len(detections), stride)):
        frame = Frame(image=blank, timestamp=BASE_TIME + timedelta(seconds=native_index / native_fps), frame_index=k)
        tracked = tracker.update(detections[native_index], frame)
        for obj in tracked:
            track_ids.add(obj.track_id)
            lengths[obj.track_id] = lengths.get(obj.track_id, 0) + 1
        for track_id in engine.update(tracked):
            triggers.append((round(native_index / native_fps, 2), track_id, engine.reason_for(track_id)))
    durations = [count / fps for count in lengths.values()]
    return {
        "fps": round(fps, 2),
        "pistas": len(track_ids),
        "duracion_media_pista_s": round(statistics.mean(durations), 2) if durations else 0.0,
        "disparos": len(triggers),
        "disparos_detalle": triggers,
    }


def video_mode(args: argparse.Namespace) -> None:
    cache = args.cache or Path(args.video).with_suffix(".detections.pkl")
    detections, native_fps = detect_all(args.video, args.weights, cache)
    print(f"{args.video.name}: {len(detections)} fotogramas a {native_fps:g} fps. Detecciones en caché: {cache}")

    report = []
    print(f"\n   {'fps':>5} | {'memoria de pistas':<18} | {'pistas':>6} | {'duración media':>14} | {'disparos':>8}")
    for stride in [int(x) for x in args.strides.split(",")]:
        for old in (True, False):
            result = replay_video(detections, native_fps, stride, old_track_memory=old)
            result["memoria_de_pistas"] = "anterior" if old else "corregida"
            report.append(result)
            print(
                f"   {result['fps']:>5} | {result['memoria_de_pistas']:<18} | {result['pistas']:>6} | "
                f"{result['duracion_media_pista_s']:>12.2f} s | {result['disparos']:>8}"
            )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nResultados en {args.out}")


def _occlusion_trial(fps: float, gap_seconds: float, old_track_memory: bool, seed: int) -> bool:
    from monitoreo_urbano.vision_pipeline.camera_source import Frame
    from monitoreo_urbano.vision_pipeline.tracker import ByteTrackTracker

    rng = np.random.default_rng(seed)
    tracker = ByteTrackTracker(frame_rate=fps, track_buffer_seconds=3.0)
    if old_track_memory:
        old_value = int(fps / 30.0 * int(3.0 * fps))
        tracker._tracker.buffer_size = old_value
        tracker._tracker.max_time_lost = old_value

    blank = np.zeros((4, 4, 3), dtype=np.uint8)
    visible = int(4 * fps)
    gap = int(round(gap_seconds * fps))
    speed_px_per_second = 45.0
    before: set[int] = set()
    after: set[int] = set()
    for k in range(visible + gap + visible):
        in_gap = visible <= k < visible + gap
        center = np.array([200.0 + speed_px_per_second * k / fps, 400.0]) + rng.normal(0, 1.5, 2)
        detections = (
            []
            if in_gap
            else [
                Detection(
                    class_name="person",
                    confidence=0.9,
                    bbox=BoundingBox(x1=center[0] - 20, y1=center[1] - 45, x2=center[0] + 20, y2=center[1] + 45),
                )
            ]
        )
        frame = Frame(image=blank, timestamp=BASE_TIME + timedelta(seconds=k / fps), frame_index=k)
        for obj in tracker.update(detections, frame):
            (before if k < visible else after).add(obj.track_id)
    return bool(before) and before == after


def occlusion_mode(args: argparse.Namespace) -> None:
    fps_list = [float(x) for x in args.fps.split(",")]
    gaps = [float(x) for x in args.gaps.split(",")]
    report = []
    print(f"Oclusión de una persona: ¿conserva el identificador? ({args.seeds} semillas)")
    print(f"\n   {'fps':>4} | {'memoria':<10} | " + " | ".join(f"hueco {g:g} s".rjust(10) for g in gaps))
    for fps in fps_list:
        for old in (True, False):
            rates = [
                sum(_occlusion_trial(fps, gap, old, seed) for seed in range(args.seeds)) / args.seeds for gap in gaps
            ]
            label = "anterior" if old else "corregida"
            report.append({"fps": fps, "memoria_de_pistas": label, "conserva_id": dict(zip(map(str, gaps), rates))})
            print(f"   {fps:>4.0f} | {label:<10} | " + " | ".join(f"{r:>10.0%}" for r in rates))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nResultados en {args.out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)

    occ = sub.add_parser("occlusion")
    occ.add_argument("--fps", default=",".join(str(f) for f in DEFAULT_FPS))
    occ.add_argument("--gaps", default="0.5,1,2,3")
    occ.add_argument("--seeds", type=int, default=20)
    occ.add_argument("--out", type=Path, default=None)

    syn = sub.add_parser("synthetic")
    syn.add_argument("--fps", default=",".join(str(f) for f in DEFAULT_FPS))
    syn.add_argument("--seeds", type=int, default=60)
    syn.add_argument("--jitter", type=float, default=2.0, help="Ruido de la caja del detector, en píxeles")
    syn.add_argument("--out", type=Path, default=None)

    vid = sub.add_parser("video")
    vid.add_argument("--video", type=Path, default=ROOT / "data" / "raw_videos" / "sample_bus.mp4")
    vid.add_argument("--weights", type=Path, default=ROOT / "data" / "models" / "yolo" / "baseline_v1.pt")
    vid.add_argument("--strides", default="1,2,3,5", help="Se analiza 1 de cada k fotogramas nativos")
    vid.add_argument("--cache", type=Path, default=None)
    vid.add_argument("--out", type=Path, default=None)

    args = parser.parse_args()
    {"synthetic": synthetic_mode, "video": video_mode, "occlusion": occlusion_mode}[args.mode](args)


if __name__ == "__main__":
    main()
