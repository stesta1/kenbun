"""Detecciones crudas de v6 (960 px, 6 fps, confianza >= 0,2) sobre los videos reales de ACCIDENT y sobre V7 sin cortes.

Uso:
    python scripts/extract_detections_accident.py --shard 0 --shards 3
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import evaluate_on_videos as ev  # noqa: E402

SOURCE = ROOT / "data" / "datasets" / "_sources" / "accident"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, default=ROOT / "data" / "models" / "yolo" / "baseline_v6.pt")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--fps", type=float, default=6.0, help="Frecuencia de análisis")
    parser.add_argument("--solo-dominio", action="store_true", help="Solo los videos del dominio del requisito de cámara")
    parser.add_argument("--split", default=None, help="train, val o test: solo los videos de esa partición (V7 aparte)")
    args = parser.parse_args()
    ev.IMGSZ[0] = args.imgsz

    from ultralytics import YOLO

    model = YOLO(str(args.weights))
    weights_id = f"{args.weights.name}-{args.weights.stat().st_size}-{int(args.weights.stat().st_mtime)}"
    rows = list(csv.DictReader(open(SOURCE / "metadata-real.csv", encoding="utf-8")))
    if args.split:
        import evaluate_classifier_events as ece

        stems = ece.split_stems(args.split)
        videos = [SOURCE / r["path"] for r in rows if Path(r["path"]).stem in stems]
        if args.split == "test":
            videos.append(ev.VIDEOS_DIR / "V7 uncut.mp4")
    else:
        videos = [SOURCE / r["path"] for r in rows] + [ev.VIDEOS_DIR / "V7 uncut.mp4"]
    if args.solo_dominio:
        import compare_models_domain as cmp

        meta = {Path(r["path"]).stem: r for r in rows}
        videos = [v for v in videos if v.stem not in meta or cmp.in_domain(meta[v.stem])]
    mine = videos[args.shard::args.shards][: args.limit]
    cache = ROOT / "data" / "eval_cache"
    started = time.time()
    for i, video in enumerate(mine, 1):
        ev.detect_video(model, video, args.fps, cache, weights_id)
        if i % 20 == 0:
            print(f"fragmento {args.shard}: {i}/{len(mine)} ({time.time() - started:.0f} s)", flush=True)
    print("listo", args.shard, len(mine), f"{time.time() - started:.0f} s")


if __name__ == "__main__":
    main()
