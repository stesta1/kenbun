"""Arma un conjunto de clasificación de imagen completa (choque / tránsito normal) con los videos de ACCIDENT-real.

Uso:
    python scripts/prepare_accident_cls.py --workers 6
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "datasets" / "_sources" / "accident"
OUT = ROOT / "data" / "datasets_cls_v1"
SEED = 42
MAX_SIDE = 640


def group_of(path: str) -> str:
    return Path(path).stem.rsplit("_", 1)[0]


def read_frames(video: Path, indices: list[int]) -> dict[int, object]:
    cap = cv2.VideoCapture(str(video))
    frames: dict[int, object] = {}
    try:
        for index in indices:
            if index < 0:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if ok:
                frames[index] = frame
    finally:
        cap.release()
    return frames


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    rows = list(csv.DictReader(open(SOURCE / "metadata-real.csv", encoding="utf-8")))
    groups = sorted({group_of(r["path"]) for r in rows})
    random.Random(SEED).shuffle(groups)
    split_of = {g: ("train" if i / len(groups) < 0.8 else ("val" if i / len(groups) < 0.9 else "test")) for i, g in enumerate(groups)}
    for split in ("train", "val", "test"):
        for cls in ("accident", "normal"):
            (OUT / split / cls).mkdir(parents=True, exist_ok=True)

    def work(row: dict) -> Counter:
        counts: Counter = Counter()
        video = SOURCE / row["path"]
        if not video.exists():
            counts["falta_video"] += 1
            return counts
        total = int(float(row["no_frames"]))
        fps = total / max(float(row["duration"]), 0.1)
        impact = int(float(row["accident_frame"]))
        half, second = max(1, round(fps * 0.5)), max(1, round(fps))
        positive = [impact + k * half for k in range(0, 9) if impact + k * half < total]
        negative = [impact - j * second for j in range(2, 10) if impact - j * second >= 0]
        frames = read_frames(video, positive + negative)
        split = split_of[group_of(row["path"])]
        stem = Path(row["path"]).stem
        for cls, indices in (("accident", positive), ("normal", negative)):
            for index in indices:
                frame = frames.get(index)
                if frame is None:
                    continue
                height, width = frame.shape[:2]
                scale = MAX_SIDE / max(height, width)
                if scale < 1:
                    frame = cv2.resize(frame, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(OUT / split / cls / f"{stem}_{index}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                counts[(split, cls)] += 1
        return counts

    total: Counter = Counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, counts in enumerate(pool.map(work, rows), 1):
            total.update(counts)
            if i % 250 == 0:
                print(f"  {i}/{len(rows)} videos", flush=True)
    summary = {f"{k[0]}/{k[1]}" if isinstance(k, tuple) else k: v for k, v in total.items()}
    (ROOT.parent / "reentreno" / "dataset_cls_v1_composicion.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
