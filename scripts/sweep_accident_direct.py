"""Barrido de la regla de choque con las detecciones de `accident` entrando directo al motor de reglas, sin pasar por el seguimiento.

Uso:
    python scripts/sweep_accident_direct.py --weights data/models/yolo/baseline_v6.pt --out ../reentreno/barrido_directo_v6.json
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from datetime import datetime, timedelta
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

CONFS = (0.3, 0.4, 0.5, 0.6, 0.7)
NS = (1, 2, 3, 4)
WINDOWS = (1.5, 2.0, 3.0)
REFRACTORY_S = 10.0
FPS = 6.0


def one_video(job: tuple) -> dict:
    import train_trajectory_classifier as ttc
    from monitoreo_urbano.common.config import settings
    from monitoreo_urbano.common.schemas import BoundingBox, TrackedObject
    from monitoreo_urbano.vision_pipeline.event_trigger import EventTriggerEngine

    stem, video, t0, weights, fps = job
    weights = Path(weights)
    wid = f"{weights.name}-{weights.stat().st_size}-{int(weights.stat().st_mtime)}"
    path = ttc.cache_path(wid, Path(video), 960, fps)
    if not path.exists():
        return {"stem": stem, "falta": True}
    frames = json.loads(path.read_text(encoding="utf-8"))["frames"]
    duration = frames[-1]["t"] if frames else 0.0
    base = datetime(2026, 1, 1)
    out = {"stem": stem, "t0": t0, "duracion": duration, "cfg": {}}
    settings.event_dedup_window_seconds = 0.0
    for conf, n, window in itertools.product(CONFS, NS, WINDOWS):
        settings.incident_min_seconds = {"fire": 0.8, "accident": n / FPS}
        settings.incident_window_seconds = window
        engine = EventTriggerEngine(fps=fps, dedup_window_seconds=0.0)
        times, last, uid = [], -1e9, 0
        for index, item in enumerate(frames):
            objs = []
            for name, c, x1, y1, x2, y2 in item["dets"]:
                if name == "accident" and c >= conf:
                    uid += 1
                    objs.append(TrackedObject(track_id=uid, class_name="accident", bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                                              confidence=c, frame_index=index, timestamp=base + timedelta(seconds=item["t"])))
            if not objs:
                engine.update([])
                continue
            for _tid in engine.update(objs):
                if item["t"] - last >= REFRACTORY_S:
                    times.append(item["t"])
                    last = item["t"]
        out["cfg"][f"{conf}|{n}|{window}"] = times
    return out


def main() -> None:
    import compare_models_domain as cmp
    import evaluate_classifier_events as ece

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--fps", type=float, default=6.0, help="Frecuencia con que se extrajeron las detecciones")
    args = parser.parse_args()
    meta = {Path(r["path"]).stem: r for r in csv.DictReader(open(ece.SOURCE / "metadata-real.csv", encoding="utf-8"))}
    jobs, split_of = [], {}
    for split in ("val", "test"):
        for stem in sorted(ece.split_stems(split)):
            if cmp.in_domain(meta[stem]):
                jobs.append((stem, str(ece.SOURCE / meta[stem]["path"]), float(meta[stem]["accident_time"]), str(args.weights), args.fps))
                split_of[stem] = split
    jobs.append(("V7 uncut", str(ece.V7), None, str(args.weights), args.fps))
    with Pool(args.workers) as pool:
        rows = pool.map(one_video, jobs, chunksize=4)
    missing = [r["stem"] for r in rows if r.get("falta")]
    rows = [r for r in rows if not r.get("falta")]
    print("sin caché:", len(missing), flush=True)
    args.out.write_text(json.dumps({"split": split_of, "rows": rows}), encoding="utf-8")
    print("escrito", args.out, len(rows))


if __name__ == "__main__":
    main()
