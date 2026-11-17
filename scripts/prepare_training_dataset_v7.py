"""Arma el conjunto v7: el v5 más recortes de choque a resolución nativa y más imágenes de fuego y humo de FASDD_CV.

Uso:
    python scripts/prepare_training_dataset_v7.py
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shutil
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import prepare_training_dataset_v5 as v5  # noqa: E402

V5 = ROOT / "data" / "datasets_v5"
OUT = ROOT / "data" / "datasets_v7"
SOURCE = ROOT / "data" / "datasets" / "_sources" / "accident"
CROP = 640
OFFSETS = (-2, 0, 2)
BACK = (75, 150)
CLS_BASE = {0: 0, 2: 1, 3: 2}


def train_stems() -> set[str]:
    out = set()
    for p in (V5 / "images" / "train").iterdir():
        m = re.match(r"^accident_(.+)_[pn]\d+\.", p.name)
        if m:
            out.add(m.group(1))
    return out


def crops_for_video(job: tuple) -> dict:
    stem, row, seed = job
    rng = random.Random(f"{seed}-{stem}")
    video = SOURCE / row["path"]
    width, height = int(row["width"]), int(row["height"])
    if min(width, height) < CROP:
        return {"stem": stem, "skip": "chico"}
    bx1, by1, bx2, by2 = (float(row[k]) for k in ("x1", "y1", "x2", "y2"))
    px1, px2, py1, py2 = bx1 * width, bx2 * width, by1 * height, by2 * height
    if px2 - px1 > CROP - 40 or py2 - py1 > CROP - 40:
        return {"stem": stem, "skip": "caja grande"}
    centre = int(float(row["accident_frame"]))
    cap = cv2.VideoCapture(str(video))

    def window() -> tuple[int, int]:
        x_lo, x_hi = max(0, int(px2) - CROP + 10), min(int(px1) - 10, width - CROP)
        y_lo, y_hi = max(0, int(py2) - CROP + 10), min(int(py1) - 10, height - CROP)
        x0 = rng.randint(x_lo, x_hi) if x_hi >= x_lo else max(0, min(width - CROP, int(px1) - CROP // 2))
        y0 = rng.randint(y_lo, y_hi) if y_hi >= y_lo else max(0, min(height - CROP, int(py1) - CROP // 2))
        return x0, y0

    def frame(index: int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, image = cap.read()
        return image if ok and image is not None and image.shape[0] == height and image.shape[1] == width else None

    saved = []
    first_window = None
    jobs = [(0, 0), (0, 1), (-2, 0), (2, 0)]
    for offset, rep in jobs:
        image = frame(centre + offset)
        if image is None:
            continue
        x0, y0 = window()
        if first_window is None:
            first_window = (x0, y0)
        name = f"crop_{stem}_p{offset + 2}{'b' if rep else ''}"
        label = f"5 {((px1 + px2) / 2 - x0) / CROP:.6f} {((py1 + py2) / 2 - y0) / CROP:.6f} {(px2 - px1) / CROP:.6f} {(py2 - py1) / CROP:.6f}"
        cv2.imwrite(str(OUT / "images" / "train" / f"{name}.jpg"), image[y0:y0 + CROP, x0:x0 + CROP], [cv2.IMWRITE_JPEG_QUALITY, 92])
        (OUT / "labels" / "train" / f"{name}.txt").write_text(label + "\n", encoding="utf-8")
        saved.append(name)
    if first_window is not None:
        x0, y0 = first_window
        for back in BACK:
            if centre - back < 0:
                continue
            image = frame(centre - back)
            if image is None:
                continue
            name = f"crop_{stem}_n{back}"
            cv2.imwrite(str(OUT / "images" / "train" / f"{name}.jpg"), image[y0:y0 + CROP, x0:x0 + CROP], [cv2.IMWRITE_JPEG_QUALITY, 92])
            (OUT / "labels" / "train" / f"{name}.txt").write_text("", encoding="utf-8")
            saved.append(name)
    cap.release()
    return {"stem": stem, "saved": saved}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fire-only", type=int, default=3000)
    parser.add_argument("--both", type=int, default=3000)
    parser.add_argument("--smoke-only", type=int, default=2000)
    parser.add_argument("--null", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    rng = random.Random(args.seed)

    for sub in ("images", "labels"):
        for split in ("train", "val", "test"):
            (OUT / sub / split).mkdir(parents=True, exist_ok=True)

    print("1/5 Copiando v5 (enlaces duros)...", flush=True)
    for sub in ("images", "labels"):
        for split in ("train", "val", "test"):
            for path in (V5 / sub / split).iterdir():
                v5.link(path, OUT / sub / split / path.name)
    text = (V5 / "data.yaml").read_text(encoding="utf-8").replace("datasets_v5", "datasets_v7")
    (OUT / "data.yaml").write_text(text, encoding="utf-8")
    n_v5 = len(list((OUT / "images" / "train").iterdir()))

    print("2/5 Recortes de choque a resolución nativa...", flush=True)
    meta = {Path(r["path"]).stem: r for r in csv.DictReader(open(SOURCE / "metadata-real.csv", encoding="utf-8"))}
    jobs = [(s, meta[s], args.seed) for s in sorted(train_stems())]
    with Pool(args.workers) as pool:
        results = pool.map(crops_for_video, jobs, chunksize=8)
    skipped = {}
    crop_names = []
    for r in results:
        if "skip" in r:
            skipped[r["skip"]] = skipped.get(r["skip"], 0) + 1
        else:
            crop_names += r["saved"]
    print(f"   {len(crop_names)} recortes de {len(results) - sum(skipped.values())} videos; omitidos {skipped}", flush=True)

    print("3/5 Seudoetiquetas de clases base en los recortes (COCO yolo11n, 0,45)...", flush=True)
    from ultralytics import YOLO

    coco = YOLO(str(ROOT / "data" / "models" / "yolo" / "yolo11n_coco.pt"))
    base_boxes = 0
    for i in range(0, len(crop_names), 64):
        names = crop_names[i:i + 64]
        preds = coco.predict([str(OUT / "images" / "train" / f"{n}.jpg") for n in names], conf=0.45, classes=list(CLS_BASE), verbose=False, device=0)
        for n, pred in zip(names, preds, strict=True):
            lines = []
            for b in pred.boxes:
                cx, cy, w, h = b.xywhn[0].tolist()
                lines.append(f"{CLS_BASE[int(b.cls[0])]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            if lines:
                with open(OUT / "labels" / "train" / f"{n}.txt", "a", encoding="utf-8") as fh:
                    fh.write("".join(f"{line}\n" for line in lines))
                base_boxes += len(lines)
    print(f"   {base_boxes} cajas base agregadas", flush=True)

    print("4/5 FASDD_CV adicional...", flush=True)
    ref_files = [p for split in ("val", "test") for p in (OUT / "images" / split).iterdir()]
    with Pool(args.workers) as pool:
        reference = np.array([h for h in pool.map(v5.hash_file, ref_files, chunksize=64) if h >= 0], dtype=np.uint64)
    existing = {p.name for p in (OUT / "images" / "train").iterdir()}
    pools: dict[str, list[Path]] = {"fire": [], "both": [], "smoke": [], "null": []}
    for image in sorted((v5.FASDD / "train" / "images").iterdir()):
        if f"fasdd_{image.stem[:60]}.jpg" in existing:
            continue
        key = {"fire": "fire", "bothFireAndSmoke": "both", "smoke": "smoke", "neitherFireNorSmoke": "null"}.get(image.name.split("_")[0])
        if key:
            pools[key].append(image)
    quota = {"fire": args.fire_only, "both": args.both, "smoke": args.smoke_only, "null": args.null}
    chosen = []
    for key, items in pools.items():
        rng.shuffle(items)
        chosen += [(key, p) for p in items[: quota[key]]]
    with Pool(args.workers) as pool:
        hashes = pool.map(v5.hash_file, [p for _, p in chosen], chunksize=64)
    valid = [(k, p, h) for (k, p), h in zip(chosen, hashes, strict=True) if h >= 0]
    near = v5.hamming_min(np.array([h for _, _, h in valid], dtype=np.uint64), reference) <= v5.HAMMING
    summary = {"fire": 0, "both": 0, "smoke": 0, "null": 0}
    boxes = {"fire": 0, "smoke": 0}
    dropped = 0
    for (key, image, _), is_near in zip(valid, near, strict=True):
        if is_near:
            dropped += 1
            continue
        label = v5.FASDD / "train" / "labels" / f"{image.stem}.txt"
        lines = []
        if label.exists():
            for line in label.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) >= 5 and parts[0] in ("0", "1"):
                    cls = 3 if parts[0] == "0" else 4
                    boxes["fire" if cls == 3 else "smoke"] += 1
                    lines.append(f"{cls} {' '.join(parts[1:5])}")
        name = f"fasdd_{image.stem[:60]}"
        v5.link(image, OUT / "images" / "train" / f"{name}.jpg")
        (OUT / "labels" / "train" / f"{name}.txt").write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
        summary[key] += 1
    print(f"   FASDD_CV adicional: {sum(summary.values())} imágenes ({dropped} descartadas por parecido) {summary} cajas {boxes}", flush=True)

    print("5/5 Resumen...", flush=True)
    result = {
        "base": "datasets_v5 (sin cambios en validación ni prueba)",
        "recortes_choque": {"imagenes": len(crop_names), "videos": len(results) - sum(skipped.values()), "omitidos": skipped,
                            "cajas_base_seudoetiquetadas": base_boxes, "recorte_px": CROP},
        "fasdd_cv_adicional": {"imagenes": summary, "cajas": boxes, "descartadas_por_parecido": dropped},
        "entrenamiento_v5": n_v5,
        "entrenamiento_total": len(list((OUT / "images" / "train").iterdir())),
    }
    (ROOT.parent / "reentreno" / "dataset_v7_composicion.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
