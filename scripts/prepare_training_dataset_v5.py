"""Arma el conjunto v5: el v4 más choques de CCTV (justjuu, CC0) y fuego/humo reales (FASDD_CV, CC BY 4.0).

Uso:
    python scripts/prepare_training_dataset_v5.py --fire-only 6000 --both 6000 --smoke-only 3000 --null 4000
"""

from __future__ import annotations

import argparse
import io
import json
import os
import random
import shutil
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
V4 = ROOT / "data" / "datasets_v4"
OUT = ROOT / "data" / "datasets_v5"
JUSTJUU = ROOT / "data" / "downloads_v5" / "justjuu__traffic-accident-cctv-object-detection" / "data"
FASDD = ROOT / "data" / "downloads_v5" / "fasdd_cv"
HAMMING = 5


def dhash(source) -> int:
    image = Image.open(source)
    image.draft("L", (64, 64))
    pixels = np.asarray(image.convert("L").resize((9, 8), Image.BILINEAR), dtype=np.int16)
    bits = (pixels[:, 1:] > pixels[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def hash_file(path: Path) -> int:
    try:
        return dhash(path)
    except Exception:  # noqa: BLE001
        return -1


def hamming_min(hashes: np.ndarray, reference: np.ndarray) -> np.ndarray:
    out = np.full(len(hashes), 64, dtype=np.int16)
    for ref in reference:
        x = np.bitwise_xor(hashes, ref)
        counts = np.zeros(len(x), dtype=np.int16)
        for shift in range(0, 64, 8):
            counts += POPCOUNT[((x >> np.uint64(shift)) & np.uint64(255)).astype(np.uint8)]
        out = np.minimum(out, counts)
    return out


POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.int16)


def link(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fire-only", type=int, default=6000)
    parser.add_argument("--both", type=int, default=6000)
    parser.add_argument("--smoke-only", type=int, default=3000)
    parser.add_argument("--null", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    rng = random.Random(args.seed)

    for sub in ("images", "labels"):
        for split in ("train", "val", "test"):
            (OUT / sub / split).mkdir(parents=True, exist_ok=True)

    print("1/5 Copiando v4 (enlaces duros)...", flush=True)
    counts = {"v4": 0}
    for sub in ("images", "labels"):
        for split in ("train", "val", "test"):
            for path in (V4 / sub / split).iterdir():
                link(path, OUT / sub / split / path.name)
                counts["v4"] += 1
    shutil.copy2(V4 / "data.yaml", OUT / "data.yaml")
    text = (OUT / "data.yaml").read_text(encoding="utf-8").replace("datasets_v4", "datasets_v5")
    (OUT / "data.yaml").write_text(text, encoding="utf-8")

    print("2/5 Hash de referencia (validación y prueba de v4)...", flush=True)
    ref_files = [p for split in ("val", "test") for p in (V4 / "images" / split).iterdir()]
    with Pool(args.workers) as pool:
        reference = [h for h in pool.map(hash_file, ref_files, chunksize=64) if h >= 0]
    reference = np.array(reference, dtype=np.uint64)
    print(f"   referencias: {len(reference)}", flush=True)

    print("3/5 justjuu (choques CCTV)...", flush=True)
    rows = []
    for parquet in sorted(JUSTJUU.glob("*.parquet")):
        frame = pd.read_parquet(parquet)
        for index, row in frame.iterrows():
            rows.append((parquet.stem, index, row["image"]["bytes"], row["objects"]))
    hashes = np.array([dhash(io.BytesIO(r[2])) for r in rows], dtype=np.uint64)
    near = hamming_min(hashes, reference) <= HAMMING
    kept_jj = 0
    boxes_jj = {"accident": 0, "car": 0}
    for (stem, index, data, objects), is_near in zip(rows, near, strict=True):
        if is_near:
            continue
        name = f"jj_{stem}_{index}"
        (OUT / "images" / "train" / f"{name}.png").write_bytes(data)
        lines = []
        for (x, y, w, h), category in zip(objects["bbox"], objects["category"], strict=True):
            cls = 5 if int(category) == 0 else 1
            boxes_jj["accident" if cls == 5 else "car"] += 1
            lines.append(f"{cls} {(x + w / 2) / 640:.6f} {(y + h / 2) / 640:.6f} {w / 640:.6f} {h / 640:.6f}")
        (OUT / "labels" / "train" / f"{name}.txt").write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
        kept_jj += 1
    print(f"   justjuu: {kept_jj} de {len(rows)} imágenes ({int(near.sum())} descartadas por parecerse a validación/prueba)", flush=True)

    print("4/5 FASDD_CV...", flush=True)
    pools: dict[str, list[Path]] = {"fire": [], "both": [], "smoke": [], "null": []}
    for image in sorted((FASDD / "train" / "images").iterdir()):
        prefix = image.name.split("_")[0]
        key = {"fire": "fire", "bothFireAndSmoke": "both", "smoke": "smoke", "neitherFireNorSmoke": "null"}.get(prefix)
        if key:
            pools[key].append(image)
    quota = {"fire": args.fire_only, "both": args.both, "smoke": args.smoke_only, "null": args.null}
    chosen: list[tuple[str, Path]] = []
    for key, items in pools.items():
        rng.shuffle(items)
        chosen += [(key, p) for p in items[: quota[key]]]
    with Pool(args.workers) as pool:
        fasdd_hashes = pool.map(hash_file, [p for _, p in chosen], chunksize=64)
    valid = [(k, p, h) for (k, p), h in zip(chosen, fasdd_hashes, strict=True) if h >= 0]
    near = hamming_min(np.array([h for _, _, h in valid], dtype=np.uint64), reference) <= HAMMING
    summary = {"fire": 0, "both": 0, "smoke": 0, "null": 0}
    boxes = {"fire": 0, "smoke": 0}
    dropped = 0
    for (key, image, _), is_near in zip(valid, near, strict=True):
        if is_near:
            dropped += 1
            continue
        label = FASDD / "train" / "labels" / f"{image.stem}.txt"
        lines = []
        if label.exists():
            for line in label.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) >= 5 and parts[0] in ("0", "1"):
                    cls = 3 if parts[0] == "0" else 4
                    boxes["fire" if cls == 3 else "smoke"] += 1
                    lines.append(f"{cls} {' '.join(parts[1:5])}")
        name = f"fasdd_{image.stem[:60]}"
        link(image, OUT / "images" / "train" / f"{name}.jpg")
        (OUT / "labels" / "train" / f"{name}.txt").write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
        summary[key] += 1
    print(f"   FASDD_CV: {sum(summary.values())} imágenes ({dropped} descartadas por parecerse a validación/prueba) {summary} cajas {boxes}", flush=True)

    print("5/5 Resumen...", flush=True)
    result = {
        "base": "datasets_v4 (sin cambios en validación ni prueba)",
        "justjuu": {"imagenes": kept_jj, "cajas": boxes_jj, "descartadas_por_parecido": int(len(rows) - kept_jj)},
        "fasdd_cv": {"imagenes": summary, "cajas": boxes, "descartadas_por_parecido": dropped},
        "umbral_hamming": HAMMING,
        "entrenamiento_total": len(list((OUT / "images" / "train").iterdir())),
    }
    (ROOT.parent / "reentreno" / "dataset_v5_composicion.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
