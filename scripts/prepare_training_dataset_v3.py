"""Conjunto de entrenamiento v3: v2 más camiones y buses como vehículos, más muestras de choque y más tránsito normal.

Uso:
    python scripts/prepare_training_dataset_v3.py --clean
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import prepare_training_dataset_v2 as v2  # noqa: E402

ROOT = v2.ROOT
SRC = ROOT / "data" / "datasets_v2"
OUT = ROOT / "data" / "datasets_v3"
v2.OUT = OUT

EXTRA_POSITIVE_OFFSETS = (4, 6, 8, 10, 12)
EXTRA_NEGATIVE_BACK = (100, 200, 250, 300)
COCO_TO_UNIFIED = {"person": 0, "car": 1, "motorcycle": 2, "bus": 1, "truck": 1}


def copy_v2() -> None:
    if OUT.exists():
        raise SystemExit(f"{OUT} ya existe: usar --clean")
    shutil.copytree(SRC, OUT)
    for split in v2.SPLITS:
        for label in (OUT / "labels" / split).glob("*.txt"):
            kept = [line for line in label.read_text(encoding="utf-8").splitlines() if line.strip() and int(line.split()[0]) >= 3]
            label.write_text("".join(f"{line}\n" for line in kept), encoding="utf-8")


def add_accident_samples() -> Counter:
    accident_dir = v2.SOURCES / "accident"
    rows = list(csv.DictReader(open(accident_dir / "metadata-real.csv", encoding="utf-8")))
    groups = sorted({v2._group(r["path"]) for r in rows})
    random.Random(v2.SEED).shuffle(groups)
    split_of = {g: ("train" if i / len(groups) < 0.8 else ("val" if i / len(groups) < 0.9 else "test")) for i, g in enumerate(groups)}

    def work(row: dict) -> Counter:
        counts: Counter = Counter()
        if split_of[v2._group(row["path"])] != "train":
            return counts
        video = accident_dir / row["path"]
        if not video.exists():
            return counts
        centre = int(float(row["accident_frame"]))
        w = float(row["x2"]) - float(row["x1"])
        h = float(row["y2"]) - float(row["y1"])
        if w <= 0 or h <= 0:
            return counts
        label = f"5 {float(row['center_x']):.6f} {float(row['center_y']):.6f} {w:.6f} {h:.6f}"
        wanted = [centre + o for o in EXTRA_POSITIVE_OFFSETS] + [centre - b for b in EXTRA_NEGATIVE_BACK]
        frames = v2._extract(video, wanted)
        stem = Path(row["path"]).stem
        for offset in EXTRA_POSITIVE_OFFSETS:
            frame = frames.get(centre + offset)
            if frame is not None:
                v2._write("train", f"accident_{stem}_p{offset + 2}", frame, [label])
                counts["accident_pos_extra"] += 1
        for back in EXTRA_NEGATIVE_BACK:
            frame = frames.get(centre - back)
            if frame is not None:
                v2._write("train", f"accident_{stem}_n{back}", frame, [])
                counts["accident_neg_extra"] += 1
        return counts

    total: Counter = Counter()
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i, counts in enumerate(pool.map(work, rows), 1):
            total.update(counts)
            if i % 250 == 0:
                print(f"  accident extra: {i}/{len(rows)} videos", flush=True)
    return total


def pseudo_label(weights: Path, confidence: float) -> Counter:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    class_ids = [i for i, n in model.names.items() if n in COCO_TO_UNIFIED]
    added: Counter = Counter()
    for split in v2.SPLITS:
        images = sorted((OUT / "images" / split).iterdir())
        for start in range(0, len(images), 64):
            batch = images[start : start + 64]
            results = model.predict([str(p) for p in batch], classes=class_ids, conf=confidence, device="cuda:0", verbose=False)
            for image_path, result in zip(batch, results, strict=True):
                lines = []
                for box in result.boxes:
                    name = result.names[int(box.cls[0])]
                    cx, cy, bw, bh = box.xywhn[0].tolist()
                    lines.append(f"{COCO_TO_UNIFIED[name]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                    added[name] += 1
                if lines:
                    with open(OUT / "labels" / split / f"{image_path.stem}.txt", "a", encoding="utf-8") as fh:
                        fh.write("".join(f"{line}\n" for line in lines))
            if (start // 64) % 40 == 0:
                print(f"  pseudo-etiquetas {split}: {start + len(batch)}/{len(images)}", flush=True)
    return added


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--coco-weights", type=Path, default=ROOT / "data" / "models" / "yolo" / "yolo11s.pt")
    parser.add_argument("--pseudo-confidence", type=float, default=0.45)
    parser.add_argument("--report", type=Path, default=ROOT.parent / "reentreno" / "dataset_v3_composicion.json")
    args = parser.parse_args()

    if args.clean and OUT.exists():
        shutil.rmtree(OUT)
    print("Copiando v2...", flush=True)
    copy_v2()
    print("Muestras extra de choque y tránsito normal...", flush=True)
    extra = add_accident_samples()
    print("Pseudo-etiquetas de person, car (con camiones y buses) y motorcycle...", flush=True)
    added = pseudo_label(args.coco_weights, args.pseudo_confidence)
    v2.write_yaml()
    (OUT / "data.yaml").write_text((OUT / "data.yaml").read_text(encoding="utf-8").replace("datasets_v2", "datasets_v3"), encoding="utf-8")

    stats = v2.report()
    stats["extra_entrenamiento"] = dict(extra)
    stats["pseudo_etiquetas_agregadas"] = dict(added)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(stats["cajas"], indent=2, ensure_ascii=False))
    print(f"Composición en {args.report}")


if __name__ == "__main__":
    main()
