"""Conjunto de entrenamiento v2: solo datos reales, con negativos y división por video.

Uso:
    python scripts/prepare_training_dataset_v2.py [--dfire-negatives 4000]
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "data" / "datasets" / "_sources"
OUT = ROOT / "data" / "datasets_v2"
SPLITS = ("train", "val", "test")
SEED = 42
MAX_SIDE = 1280
COCO_TO_UNIFIED = {"person": 0, "car": 1, "motorcycle": 2}
NAMES = {0: "person", 1: "car", 2: "motorcycle", 3: "fire", 4: "smoke", 5: "accident"}

ACCIDENT_OFFSETS = (-2, 0, 2)
NEGATIVE_BACK_FRAMES = (75, 150)


def _dirs() -> None:
    for split in SPLITS:
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)


def _write(split: str, stem: str, image, lines: list[str]) -> None:
    height, width = image.shape[:2]
    scale = MAX_SIDE / max(height, width)
    if scale < 1:
        image = cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(OUT / "images" / split / f"{stem}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    (OUT / "labels" / split / f"{stem}.txt").write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")


def prepare_dfire(negatives: int) -> Counter:
    src = SOURCES / "fire_smoke" / "data"
    remap = {0: 4, 1: 3}
    rng = random.Random(SEED)
    counts: Counter = Counter()
    for split in SPLITS:
        images = sorted((src / split / "images").iterdir())
        positives, empties = [], []
        for image_path in images:
            label_path = src / split / "labels" / f"{image_path.stem}.txt"
            lines = []
            if label_path.exists():
                for line in label_path.read_text(encoding="utf-8").splitlines():
                    parts = line.split()
                    if len(parts) < 5 or int(float(parts[0])) not in remap:
                        continue
                    coords = [float(v) for v in parts[1:5]]
                    if any(not 0.0 <= v <= 1.0 for v in coords):
                        continue
                    lines.append(" ".join([str(remap[int(float(parts[0]))]), *parts[1:5]]))
            (positives if lines else empties).append((image_path, lines))
        keep_empty = rng.sample(empties, min(len(empties), round(negatives * len(images) / 21527)))
        for image_path, lines in positives + keep_empty:
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            _write(split, f"dfire_{image_path.stem}", image, lines)
            counts[(split, "dfire_pos" if lines else "dfire_neg")] += 1
    return counts


def _group(path: str) -> str:
    stem = Path(path).stem
    return stem.rsplit("_", 1)[0]


def _extract(video: Path, indices: list[int]) -> dict[int, object]:
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


def prepare_accident() -> Counter:
    accident_dir = SOURCES / "accident"
    rows = list(csv.DictReader(open(accident_dir / "metadata-real.csv", encoding="utf-8")))

    groups = sorted({_group(r["path"]) for r in rows})
    random.Random(SEED).shuffle(groups)
    split_of = {}
    for i, group in enumerate(groups):
        fraction = i / len(groups)
        split_of[group] = "train" if fraction < 0.8 else ("val" if fraction < 0.9 else "test")

    def work(row: dict) -> Counter:
        counts: Counter = Counter()
        video = accident_dir / row["path"]
        if not video.exists():
            counts["falta_video"] += 1
            return counts
        split = split_of[_group(row["path"])]
        centre = int(float(row["accident_frame"]))
        w = float(row["x2"]) - float(row["x1"])
        h = float(row["y2"]) - float(row["y1"])
        if w <= 0 or h <= 0:
            counts["caja_invalida"] += 1
            return counts
        label = f"5 {float(row['center_x']):.6f} {float(row['center_y']):.6f} {w:.6f} {h:.6f}"
        wanted = [centre + o for o in ACCIDENT_OFFSETS] + [centre - b for b in NEGATIVE_BACK_FRAMES]
        frames = _extract(video, wanted)
        stem = Path(row["path"]).stem
        for offset in ACCIDENT_OFFSETS:
            frame = frames.get(centre + offset)
            if frame is not None:
                _write(split, f"accident_{stem}_p{offset + 2}", frame, [label])
                counts[(split, "accident_pos")] += 1
        for back in NEGATIVE_BACK_FRAMES:
            frame = frames.get(centre - back)
            if frame is not None:
                _write(split, f"accident_{stem}_n{back}", frame, [])
                counts[(split, "accident_neg")] += 1
        return counts

    total: Counter = Counter()
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i, counts in enumerate(pool.map(work, rows), 1):
            total.update(counts)
            if i % 250 == 0:
                print(f"  accident: {i}/{len(rows)} videos", flush=True)
    return total


def pseudo_label_base_classes(weights: Path, confidence: float) -> Counter:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    class_ids = [i for i, n in model.names.items() if n in COCO_TO_UNIFIED]
    added: Counter = Counter()
    for split in SPLITS:
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


def write_yaml() -> None:
    names = "\n".join(f"  {i}: {n}" for i, n in NAMES.items())
    (OUT / "data.yaml").write_text(
        f"path: data/datasets_v2\ntrain: images/train\nval: images/val\ntest: images/test\n\nnames:\n{names}\n", encoding="utf-8"
    )


def report() -> dict:
    stats: dict = {"imagenes": {}, "cajas": {}}
    per_split_prefix: Counter = Counter()
    boxes: dict = defaultdict(Counter)
    for split in SPLITS:
        for label in (OUT / "labels" / split).glob("*.txt"):
            prefix = label.stem.split("_")[0]
            lines = [l for l in label.read_text(encoding="utf-8").splitlines() if l.strip()]
            per_split_prefix[(split, prefix, "con_cajas" if lines else "negativa")] += 1
            for line in lines:
                boxes[split][NAMES[int(line.split()[0])]] += 1
    stats["imagenes"] = {f"{s}/{p}/{k}": v for (s, p, k), v in sorted(per_split_prefix.items())}
    stats["cajas"] = {s: dict(c) for s, c in boxes.items()}
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dfire-negatives", type=int, default=4000, help="Negativas de D-Fire a conservar (sobre el total del conjunto)")
    parser.add_argument("--coco-weights", type=Path, default=ROOT / "data" / "models" / "yolo" / "yolo11s.pt")
    parser.add_argument("--pseudo-confidence", type=float, default=0.45)
    parser.add_argument("--report", type=Path, default=ROOT.parent / "reentreno" / "dataset_v2_composicion.json")
    parser.add_argument("--clean", action="store_true", help="Borra data/datasets_v2 antes de armarlo")
    args = parser.parse_args()

    if args.clean and OUT.exists():
        shutil.rmtree(OUT)
    _dirs()
    print("D-Fire...", flush=True)
    dfire = prepare_dfire(args.dfire_negatives)
    print("ACCIDENT-real...", flush=True)
    accident = prepare_accident()
    print("Pseudo-etiquetas de person, car y motorcycle...", flush=True)
    added = pseudo_label_base_classes(args.coco_weights, args.pseudo_confidence)
    write_yaml()

    stats = report()
    stats["origen"] = {"dfire": {f"{k[0]}/{k[1]}": v for k, v in dfire.items()}, "accident": {str(k): v for k, v in accident.items()}}
    stats["pseudo_etiquetas_agregadas"] = dict(added)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(stats["cajas"], indent=2, ensure_ascii=False))
    print(f"Composición en {args.report}")


if __name__ == "__main__":
    sys.exit(main())
