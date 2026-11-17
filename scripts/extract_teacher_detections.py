"""Detecciones de vehículos con un modelo grande (yolo11x) y recortes 2x2 sobre la ventana del impacto de cada video de ACCIDENT.

Uso:
    python scripts/extract_teacher_detections.py --split test --shard 0 --shards 1
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import torch
from torchvision.ops import batched_nms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import evaluate_classifier_events as ece  # noqa: E402

OUT = ROOT / "data" / "teacher_cache"
VEHICLES = ("car", "motorcycle", "truck", "bus")
FPS = 6.0


def tiles(frame):
    h, w = frame.shape[:2]
    th, tw = h // 2 + h // 8, w // 2 + w // 8
    boxes = [(x, y) for y in (0, h - th) for x in (0, w - tw)]
    return [(x, y, frame[y:y + th, x:x + tw]) for x, y in boxes]


def detect(model, frame, conf: float, imgsz: int) -> list[list]:
    parts = tiles(frame)
    results = model.predict([p[2] for p in parts], imgsz=imgsz, conf=conf, verbose=False)
    boxes, scores, classes = [], [], []
    for (ox, oy, _), r in zip(parts, results, strict=True):
        for b in r.boxes:
            name = r.names[int(b.cls[0])]
            if name not in VEHICLES:
                continue
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            boxes.append([x1 + ox, y1 + oy, x2 + ox, y2 + oy])
            scores.append(float(b.conf[0]))
            classes.append(VEHICLES.index(name))
    if not boxes:
        return []
    keep = batched_nms(torch.tensor(boxes), torch.tensor(scores), torch.tensor(classes), 0.5).tolist()
    return [["car" if VEHICLES[classes[k]] in ("car", "truck", "bus") else "motorcycle", round(scores[k], 4), *boxes[k]] for k in keep]


def process(model, video: Path, t_from: float, t_to: float | None, conf: float, imgsz: int) -> list[dict]:
    cap = cv2.VideoCapture(str(video))
    native = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(native / FPS))
    start = max(0, int(t_from * native))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    index, frames = start, []
    while True:
        if t_to is not None and index / native > t_to:
            break
        ok = cap.grab()
        if not ok:
            break
        if (index - start) % step == 0:
            ok, image = cap.retrieve()
            if ok:
                frames.append({"t": round(index / native, 3), "dets": detect(model, image, conf, imgsz)})
        index += 1
    cap.release()
    return frames


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", help="train, val, test o v7")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--antes", type=float, default=10.0)
    parser.add_argument("--despues", type=float, default=6.0)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=960)
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO("yolo11x.pt")
    OUT.mkdir(parents=True, exist_ok=True)
    meta = {Path(r["path"]).stem: r for r in csv.DictReader(open(ece.SOURCE / "metadata-real.csv", encoding="utf-8"))}
    if args.split == "v7":
        jobs = [("V7 uncut", ece.V7, 0.0, None)]
    else:
        stems = sorted(ece.split_stems(args.split))[args.shard::args.shards][: args.limit]
        jobs = [(s, ece.SOURCE / meta[s]["path"], max(0.0, float(meta[s]["accident_time"]) - args.antes),
                 float(meta[s]["accident_time"]) + args.despues) for s in stems]
    started = time.time()
    for i, (stem, path, t_from, t_to) in enumerate(jobs, 1):
        target = OUT / f"{stem.replace(' ', '_')}.json"
        if target.exists():
            continue
        frames = process(model, path, t_from, t_to, args.conf, args.imgsz)
        target.write_text(json.dumps({"t_from": t_from, "frames": frames}), encoding="utf-8")
        if i % 10 == 0:
            print(f"{args.split} {args.shard}: {i}/{len(jobs)} ({time.time() - started:.0f} s)", flush=True)
    print("listo", args.split, args.shard, f"{time.time() - started:.0f} s")


if __name__ == "__main__":
    main()
