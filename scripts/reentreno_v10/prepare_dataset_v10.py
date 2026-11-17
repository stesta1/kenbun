"""Conjunto v10 (final): v5 con particiones de ACCIDENT sin video de origen compartido, fotogramas de choque posteriores al impacto y negativos de tránsito sin choques de UA-DETRAC."""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2

APP = Path(__file__).resolve().parents[2]
PARTICION_V6 = Path(__file__).resolve().parent / "videos_accident_por_particion.csv"
SRC = APP / "data" / "datasets_v5"
OUT = APP / "data" / "datasets_v10"
ACCIDENT = APP / "data" / "datasets" / "_sources" / "accident"
SPLITS = ("train", "val", "test")
MAX_SIDE = 1280
POST_SECONDS = (0.3, 0.6, 1.0, 1.4, 2.0)
COCO_TO_UNIFIED = {"person": 0, "car": 1, "motorcycle": 2}
VEHICLE_COCO = {"car", "truck", "bus", "motorcycle"}
NAMES = {0: "person", 1: "car", 2: "motorcycle", 3: "fire", 4: "smoke", 5: "accident"}


def new_partition() -> dict[str, str]:
    part = {r["video"]: r["particion"] for r in csv.DictReader(open(PARTICION_V6, encoding="utf-8"))}
    groups = defaultdict(set)
    for video, split in part.items():
        groups[video[:11]].add(split)
    crossing = {g for g, s in groups.items() if len(s) > 1}
    return {video: ("train" if video[:11] in crossing else split) for video, split in part.items()}


def accident_clip(name: str) -> str | None:
    if not name.startswith("accident_"):
        return None
    stem = Path(name).stem[len("accident_"):]
    return stem.rsplit("_", 1)[0]


def build_base(partition: dict[str, str]) -> Counter:
    counts: Counter = Counter()
    for split in SPLITS:
        for kind in ("images", "labels"):
            (OUT / kind / split).mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        for image in (SRC / "images" / split).iterdir():
            clip = accident_clip(image.name)
            target = partition.get(clip, split) if clip is not None else split
            if clip is not None and clip not in partition:
                counts["accident_sin_manifiesto"] += 1
            dst_img = OUT / "images" / target / image.name
            if not dst_img.exists():
                os.link(image, dst_img)
            label = SRC / "labels" / split / f"{image.stem}.txt"
            if label.exists():
                shutil.copyfile(label, OUT / "labels" / target / label.name)
            counts[(split, target)] += 1
    return counts


def resize(image):
    height, width = image.shape[:2]
    scale = MAX_SIDE / max(height, width)
    if scale < 1:
        image = cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    return image


def add_post_impact(partition: dict[str, str], coco_weights: Path, filter_conf: float, pseudo_conf: float) -> Counter:
    rows = {Path(r["path"]).stem: r for r in csv.DictReader(open(ACCIDENT / "metadata-real.csv", encoding="utf-8"))}
    clips = sorted(c for c, s in partition.items() if s == "train")
    tmp = OUT / "_candidatos"
    tmp.mkdir(exist_ok=True)
    candidates = []
    missing: Counter = Counter()

    def work(clip: str):
        row = rows.get(clip)
        if row is None:
            return [], "sin_metadato"
        video = ACCIDENT / row["path"]
        if not video.exists():
            return [], "falta_video"
        total = int(float(row["no_frames"]))
        fps = total / float(row["duration"]) if float(row["duration"]) > 0 else 15.0
        centre = int(float(row["accident_frame"]))
        box = (float(row["center_x"]), float(row["center_y"]), float(row["x2"]) - float(row["x1"]), float(row["y2"]) - float(row["y1"]))
        if box[2] <= 0 or box[3] <= 0:
            return [], "caja_invalida"
        wanted = {centre + round(s * fps): s for s in POST_SECONDS}
        wanted = {i: s for i, s in wanted.items() if i < total}
        cap = cv2.VideoCapture(str(video))
        out = []
        try:
            for index in sorted(wanted):
                cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, frame = cap.read()
                if not ok:
                    continue
                name = f"accident_{clip}_t{round(wanted[index] * 10):02d}"
                path = tmp / f"{name}.jpg"
                cv2.imwrite(str(path), resize(frame), [cv2.IMWRITE_JPEG_QUALITY, 90])
                out.append((path, name, box))
        finally:
            cap.release()
        return out, None

    with ThreadPoolExecutor(max_workers=8) as pool:
        for i, (out, err) in enumerate(pool.map(work, clips), 1):
            candidates.extend(out)
            if err:
                missing[err] += 1
            if i % 200 == 0:
                print(f"  extraídos {i}/{len(clips)} clips ({len(candidates)} fotogramas)", flush=True)

    from ultralytics import YOLO

    model = YOLO(str(coco_weights))
    keep_ids = [i for i, n in model.names.items() if n in VEHICLE_COCO | set(COCO_TO_UNIFIED)]
    kept: Counter = Counter(missing)
    for start in range(0, len(candidates), 64):
        batch = candidates[start:start + 64]
        results = model.predict([str(p) for p, _, _ in batch], classes=keep_ids, conf=filter_conf, device="cuda:0", verbose=False)
        for (path, name, box), result in zip(batch, results, strict=True):
            cx, cy, w, h = box
            x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
            inside, pseudo = False, []
            for b in result.boxes:
                label = result.names[int(b.cls[0])]
                bx, by, bw, bh = b.xywhn[0].tolist()
                if label in VEHICLE_COCO and x1 <= bx <= x2 and y1 <= by <= y2:
                    inside = True
                if label in COCO_TO_UNIFIED and float(b.conf[0]) >= pseudo_conf:
                    pseudo.append(f"{COCO_TO_UNIFIED[label]} {bx:.6f} {by:.6f} {bw:.6f} {bh:.6f}")
            if not inside:
                kept["posterior_descartado_sin_vehiculo"] += 1
                path.unlink()
                continue
            shutil.move(str(path), OUT / "images" / "train" / f"{name}.jpg")
            lines = [f"5 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"] + pseudo
            (OUT / "labels" / "train" / f"{name}.txt").write_text("".join(f"{l}\n" for l in lines), encoding="utf-8")
            kept["posterior_agregado"] += 1
        if (start // 64) % 25 == 0:
            print(f"  filtrados {start + len(batch)}/{len(candidates)}", flush=True)
    shutil.rmtree(tmp, ignore_errors=True)
    return kept


def add_detrac(detrac_dir: Path) -> Counter:
    counts: Counter = Counter()
    for image in sorted((detrac_dir / "images").iterdir()):
        label = detrac_dir / "labels" / f"{image.stem}.txt"
        os.link(image, OUT / "images" / "train" / image.name)
        shutil.copyfile(label, OUT / "labels" / "train" / label.name)
        counts["detrac_agregado"] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detrac-dir", type=Path, default=None)
    parser.add_argument("--coco-weights", type=Path, default=APP / "data" / "models" / "yolo" / "yolo11s.pt")
    parser.add_argument("--filter-confidence", type=float, default=0.30)
    parser.add_argument("--pseudo-confidence", type=float, default=0.45)
    parser.add_argument("--report", type=Path, default=APP / "reentreno" / "dataset_v10_composicion.json")
    args = parser.parse_args()

    if OUT.exists():
        shutil.rmtree(OUT)
    partition = new_partition()
    print("Base de v5 con particiones corregidas...", flush=True)
    moved = build_base(partition)
    print("Fotogramas posteriores al impacto...", flush=True)
    post = add_post_impact(partition, args.coco_weights, args.filter_confidence, args.pseudo_confidence)
    detrac = add_detrac(APP / args.detrac_dir if not args.detrac_dir.is_absolute() else args.detrac_dir) if args.detrac_dir else Counter()

    names = "\n".join(f"  {i}: {n}" for i, n in NAMES.items())
    (OUT / "data.yaml").write_text(f"path: data/datasets_v10\ntrain: images/train\nval: images/val\ntest: images/test\n\nnames:\n{names}\n", encoding="utf-8")

    manifest = APP / "reentreno" / "videos_accident_por_particion_v10.csv"
    with open(manifest, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["particion", "video"])
        for video, split in sorted(partition.items(), key=lambda x: (x[1], x[0])):
            writer.writerow([split, video])

    sizes = {s: sum(1 for _ in (OUT / "images" / s).iterdir()) for s in SPLITS}
    report = {
        "videos_accident": dict(Counter(partition.values())),
        "base_v5_origen_destino": {f"{k[0]}->{k[1]}": v for k, v in moved.items() if isinstance(k, tuple)},
        "accident_sin_manifiesto": moved.get("accident_sin_manifiesto", 0),
        "posteriores": dict(post), "detrac": dict(detrac), "imagenes_por_particion": sizes,
        "segundos_posteriores": POST_SECONDS, "manifiesto_videos": str(manifest),
    }
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    sys.exit(main())
