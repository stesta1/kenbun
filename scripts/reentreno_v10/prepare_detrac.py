"""Prepara UA-DETRAC (conjunto de entrenamiento, 60 secuencias de tránsito sin choques) para v10.

Uso (desde la raíz del repositorio):
    .venv/Scripts/python.exe scripts/reentreno_v10/prepare_detrac.py
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import random
import shutil
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

APP = Path(__file__).resolve().parents[2]
ZIP = APP / "data" / "downloads_detrac" / "ua_detrac_training_set.zip"
OUT = APP / "data" / "detrac_v10"
EVAL = APP / "data" / "detrac_eval"
SEED = 42
STEP = 25
EVAL_SEQUENCES = 12
GRAY = (114, 114, 114)


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco-weights", type=Path, default=APP / "data" / "models" / "yolo" / "yolo11s.pt")
    parser.add_argument("--pseudo-confidence", type=float, default=0.45)
    args = parser.parse_args()

    for d in (OUT, EVAL):
        if d.exists():
            shutil.rmtree(d)
    (OUT / "images").mkdir(parents=True)
    (OUT / "labels").mkdir(parents=True)
    EVAL.mkdir(parents=True)

    z = zipfile.ZipFile(ZIP)
    names = z.namelist()
    sequences = sorted({n.split("/")[0] for n in names if n.endswith("annotations.csv")})
    rng = random.Random(SEED)
    held_out = set(rng.sample(sequences, EVAL_SEQUENCES))
    counts: Counter = Counter()

    for seq in sorted(held_out):
        frames = sorted(n for n in names if n.startswith(seq + "/") and n.endswith(".jpg"))
        writer = None
        for n in frames:
            img = cv2.imdecode(np.frombuffer(z.read(n), np.uint8), cv2.IMREAD_COLOR)
            if writer is None:
                h, w = img.shape[:2]
                writer = cv2.VideoWriter(str(EVAL / f"{seq}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 25, (w, h))
            writer.write(img)
        if writer is not None:
            writer.release()
        counts["eval_secuencias"] += 1
        counts["eval_fotogramas"] += len(frames)
        print(f"  reservada {seq}: {len(frames)} fotogramas", flush=True)

    from ultralytics import YOLO

    model = YOLO(str(args.coco_weights))
    pseudo_ids = [i for i, n in model.names.items() if n in ("person", "motorcycle")]
    pending = []
    for seq in sorted(set(sequences) - held_out):
        rows = list(csv.DictReader(io.StringIO(z.read(f"{seq}/annotations.csv").decode("utf-8"))))
        by_frame = defaultdict(list)
        for r in rows:
            by_frame[r["filename"]].append(r)
        frames = sorted(n.split("/")[1] for n in names if n.startswith(seq + "/") and n.endswith(".jpg"))
        for fname in frames[::STEP]:
            img = cv2.imdecode(np.frombuffer(z.read(f"{seq}/{fname}"), np.uint8), cv2.IMREAD_COLOR)
            h, w = img.shape[:2]
            ignored, vehicles = [], []
            for r in by_frame.get(fname, []):
                try:
                    box = [float(r["xmin"]), float(r["ymin"]), float(r["xmax"]), float(r["ymax"])]
                except (TypeError, ValueError):
                    continue
                (ignored if r["class"] == "ignored" else vehicles).append(box)
            for x1, y1, x2, y2 in ignored:
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), GRAY, thickness=-1)
            kept = [b for b in vehicles if not any(i[0] <= (b[0] + b[2]) / 2 <= i[2] and i[1] <= (b[1] + b[3]) / 2 <= i[3] for i in ignored)]
            counts["vehiculos_descartados_en_ignoradas"] += len(vehicles) - len(kept)
            stem = f"detrac_{seq}_{Path(fname).stem}"
            cv2.imwrite(str(OUT / "images" / f"{stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            lines = [f"1 {(b[0] + b[2]) / 2 / w:.6f} {(b[1] + b[3]) / 2 / h:.6f} {(b[2] - b[0]) / w:.6f} {(b[3] - b[1]) / h:.6f}" for b in kept]
            pending.append((stem, lines, kept, w, h))
            counts["entrenamiento_imagenes"] += 1
            counts["entrenamiento_vehiculos"] += len(kept)

    for start in range(0, len(pending), 64):
        batch = pending[start:start + 64]
        results = model.predict([str(OUT / "images" / f"{s}.jpg") for s, *_ in batch], classes=pseudo_ids,
                                conf=args.pseudo_confidence, device="cuda:0", verbose=False)
        for (stem, lines, kept, w, h), result in zip(batch, results, strict=True):
            for b in result.boxes:
                label = result.names[int(b.cls[0])]
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                if label == "motorcycle" and any(iou((x1, y1, x2, y2), v) > 0.3 for v in kept):
                    continue
                cx, cy, bw, bh = b.xywhn[0].tolist()
                lines.append(f"{0 if label == 'person' else 2} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                counts[f"pseudo_{label}"] += 1
            (OUT / "labels" / f"{stem}.txt").write_text("".join(f"{l}\n" for l in lines), encoding="utf-8")

    report = {"secuencias": len(sequences), "reservadas_para_evaluacion": sorted(held_out), **dict(counts)}
    (APP / "reentreno" / "detrac_v10_composicion.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    sys.exit(main())
