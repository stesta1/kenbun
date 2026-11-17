"""Diagnóstico de por qué `accident` (o `fire`) tiene poca sensibilidad, sobre la partición de ajuste (validación).

Uso:
    python scripts/diagnose_accident.py --weights data/models/yolo/baseline_v3.pt --cls accident
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NAMES = {0: "person", 1: "car", 2: "motorcycle", 3: "fire", 4: "smoke", 5: "accident"}


def iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "datasets_v3")
    parser.add_argument("--split", default="val")
    parser.add_argument("--cls", default="accident")
    parser.add_argument("--imgsz", type=int, default=640, help="Resolución de inferencia")
    parser.add_argument("--conf", type=float, nargs="+", default=[0.2, 0.35, 0.5, 0.7])
    args = parser.parse_args()
    class_id = next(i for i, n in NAMES.items() if n == args.cls)

    from ultralytics import YOLO

    model = YOLO(str(args.weights))
    images = sorted((args.data / "images" / args.split).glob("*"))
    prefix = "accident_" if args.cls == "accident" else "dfire_"
    images = [p for p in images if p.name.startswith(prefix)]

    truth: dict[str, list[tuple]] = {}
    for image in images:
        label = args.data / "labels" / args.split / f"{image.stem}.txt"
        boxes = []
        for line in label.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if parts and int(parts[0]) == class_id:
                cx, cy, w, h = map(float, parts[1:5])
                boxes.append((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
        truth[image.name] = boxes

    predictions: dict[str, list[tuple]] = {}
    for start in range(0, len(images), 32):
        batch = images[start : start + 32]
        for image, result in zip(batch, model.predict([str(p) for p in batch], conf=0.05, classes=[class_id], verbose=False, device="cuda:0", imgsz=args.imgsz), strict=True):
            predictions[image.name] = [(*box.xyxyn[0].tolist(), float(box.conf[0])) for box in result.boxes]

    positives = [n for n, b in truth.items() if b]
    negatives = [n for n, b in truth.items() if not b]
    print(f"{args.cls} en {args.split}: {len(positives)} imágenes con recuadro, {len(negatives)} sin él, {sum(len(truth[n]) for n in positives)} recuadros")
    sizes = np.array([max(b[2] - b[0], b[3] - b[1]) for n in positives for b in truth[n]])
    print(f"tamaño del recuadro anotado (lado mayor, fracción de la imagen): p25 {np.percentile(sizes, 25):.3f}  mediana {np.median(sizes):.3f}  p75 {np.percentile(sizes, 75):.3f}")

    for conf in args.conf:
        image_hit = 0
        hit50 = hit30 = total = 0
        by_size = {"chico": [0, 0], "mediano": [0, 0], "grande": [0, 0]}
        for name in positives:
            preds = [p for p in predictions[name] if p[4] >= conf]
            if preds:
                image_hit += 1
            for box in truth[name]:
                total += 1
                best = max((iou(box, p[:4]) for p in preds), default=0.0)
                hit50 += best >= 0.5
                hit30 += best >= 0.3
                size = max(box[2] - box[0], box[3] - box[1])
                key = "chico" if size < np.percentile(sizes, 33) else ("mediano" if size < np.percentile(sizes, 66) else "grande")
                by_size[key][1] += 1
                by_size[key][0] += best >= 0.5
        false_images = sum(1 for n in negatives if any(p[4] >= conf for p in predictions[n]))
        print(
            f"conf {conf:.2f}: imagen con detección {image_hit / len(positives):.2f} | recuadro IoU≥0,5 {hit50 / total:.2f} | IoU≥0,3 {hit30 / total:.2f} | "
            f"por tamaño (IoU0,5) " + " ".join(f"{k} {v[0] / max(1, v[1]):.2f}" for k, v in by_size.items()) + f" | imágenes sin {args.cls} con falso {false_images / max(1, len(negatives)):.2f}"
        )


if __name__ == "__main__":
    main()
