"""Suma al entrenamiento v3 una muestra de COCO train2017 con etiquetas humanas de person, car y motorcycle.

Uso:
    python scripts/prepare_coco_train.py --motorcycle 2400 --vehicle 2000 --person 2000
"""

from __future__ import annotations

import argparse
import json
import random
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
ZIP = ROOT / "data" / "eval_coco" / "_zips" / "annotations_trainval2017.zip"
OUT = ROOT / "data" / "datasets_v3"
IDS = {1: 0, 3: 1, 4: 2, 6: 1, 8: 1}


def download(item: tuple[str, Path]) -> bool:
    url, path = item
    if path.exists() and path.stat().st_size > 0:
        return True
    for _ in range(3):
        try:
            with urlopen(url, timeout=30) as response:
                path.write_bytes(response.read())
            return True
        except Exception:  # noqa: BLE001
            continue
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--motorcycle", type=int, default=2400)
    parser.add_argument("--vehicle", type=int, default=2000)
    parser.add_argument("--person", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    print("Leyendo anotaciones de train2017...", flush=True)
    with zipfile.ZipFile(ZIP) as z:
        data = json.loads(z.read("annotations/instances_train2017.json"))
    images = {i["id"]: i for i in data["images"]}
    per_image: dict[int, list[dict]] = {}
    for ann in data["annotations"]:
        if ann["category_id"] in IDS and not ann.get("iscrowd", 0) and ann["bbox"][2] > 2 and ann["bbox"][3] > 2:
            per_image.setdefault(ann["image_id"], []).append(ann)
    del data

    has = {k: {a["category_id"] for a in v} for k, v in per_image.items()}
    rng = random.Random(args.seed)
    moto = sorted(i for i, c in has.items() if 4 in c)
    heavy = sorted(i for i, c in has.items() if (6 in c or 8 in c) and 4 not in c)
    cars = sorted(i for i, c in has.items() if 3 in c and 6 not in c and 8 not in c and 4 not in c)
    people = sorted(i for i, c in has.items() if 1 in c and not (c & {3, 4, 6, 8}))
    for pool in (moto, heavy, cars, people):
        rng.shuffle(pool)
    chosen = (
        moto[: args.motorcycle]
        + heavy[: args.vehicle // 2]
        + cars[: args.vehicle - args.vehicle // 2]
        + people[: args.person]
    )
    print(f"Imágenes elegidas: {len(chosen)} (motos {min(len(moto), args.motorcycle)}, camiones/buses {min(len(heavy), args.vehicle // 2)}, autos {min(len(cars), args.vehicle - args.vehicle // 2)}, personas {min(len(people), args.person)})", flush=True)

    (OUT / "images" / "train").mkdir(parents=True, exist_ok=True)
    (OUT / "labels" / "train").mkdir(parents=True, exist_ok=True)
    jobs = [(images[i]["coco_url"], OUT / "images" / "train" / f"coco_{Path(images[i]['file_name']).stem}.jpg") for i in chosen]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        ok = list(pool.map(download, jobs))
    print(f"Descargadas: {sum(ok)} de {len(jobs)}", flush=True)

    counts = {"person": 0, "car": 0, "motorcycle": 0}
    names = {0: "person", 1: "car", 2: "motorcycle"}
    written = 0
    for image_id, success in zip(chosen, ok, strict=True):
        if not success:
            continue
        info = images[image_id]
        stem = f"coco_{Path(info['file_name']).stem}"
        lines = []
        for ann in per_image[image_id]:
            x, y, w, h = ann["bbox"]
            cls = IDS[ann["category_id"]]
            lines.append(f"{cls} {(x + w / 2) / info['width']:.6f} {(y + h / 2) / info['height']:.6f} {w / info['width']:.6f} {h / info['height']:.6f}")
            counts[names[cls]] += 1
        (OUT / "labels" / "train" / f"{stem}.txt").write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
        written += 1
    summary = {"imagenes": written, "cajas": counts, "fuente": "COCO train2017, etiquetas humanas, camiones y buses como car"}
    (ROOT.parent / "reentreno" / "dataset_v3_coco.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
