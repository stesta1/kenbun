"""Muestra anotada de COCO val2017 para evaluar `person`, `car` y `motorcycle` con etiquetas humanas.

Uso:
    python scripts/prepare_coco_eval.py --n 500
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data" / "eval_coco"
ZIPS = BASE / "_zips"

COCO_IDS = {1: "person", 3: "car", 4: "motorcycle", 6: "bus", 8: "truck"}
UNIFIED = {"person": 0, "car": 1, "motorcycle": 2}
NAMES = {0: "person", 1: "car", 2: "motorcycle", 3: "fire", 4: "smoke", 5: "accident"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with zipfile.ZipFile(ZIPS / "annotations_trainval2017.zip") as z:
        annotations = json.loads(z.read("annotations/instances_val2017.json"))
    images = {i["id"]: i for i in annotations["images"]}
    per_image: dict[int, list[dict]] = {}
    for ann in annotations["annotations"]:
        if ann["category_id"] in COCO_IDS and not ann.get("iscrowd", 0):
            per_image.setdefault(ann["image_id"], []).append(ann)

    ids = sorted(per_image)
    random.Random(args.seed).shuffle(ids)
    chosen = ids[: args.n]

    for variant in ("strict", "vehicles"):
        out = BASE / variant
        if out.exists():
            shutil.rmtree(out)
        (out / "images" / "val").mkdir(parents=True)
        (out / "labels" / "val").mkdir(parents=True)

    counts = {"strict": {}, "vehicles": {}}
    with zipfile.ZipFile(ZIPS / "val2017.zip") as z:
        for image_id in chosen:
            info = images[image_id]
            data = z.read(f"val2017/{info['file_name']}")
            width, height = info["width"], info["height"]
            for variant in ("strict", "vehicles"):
                lines = []
                for ann in per_image[image_id]:
                    name = COCO_IDS[ann["category_id"]]
                    if name in ("bus", "truck"):
                        if variant == "strict":
                            continue
                        name = "car"
                    x, y, w, h = ann["bbox"]
                    if w <= 1 or h <= 1:
                        continue
                    lines.append(f"{UNIFIED[name]} {(x + w / 2) / width:.6f} {(y + h / 2) / height:.6f} {w / width:.6f} {h / height:.6f}")
                    counts[variant][name] = counts[variant].get(name, 0) + 1
                stem = Path(info["file_name"]).stem
                (BASE / variant / "images" / "val" / f"{stem}.jpg").write_bytes(data)
                (BASE / variant / "labels" / "val" / f"{stem}.txt").write_text("".join(f"{l}\n" for l in lines), encoding="utf-8")

    for variant in ("strict", "vehicles"):
        names = "\n".join(f"  {i}: {n}" for i, n in NAMES.items())
        (BASE / variant / "data.yaml").write_text(
            f"path: data/eval_coco/{variant}\ntrain: images/val\nval: images/val\ntest: images/val\n\nnames:\n{names}\n", encoding="utf-8"
        )
    (BASE / "composicion.json").write_text(
        json.dumps({"imagenes": len(chosen), "semilla": args.seed, "instancias": counts, "fuente": "COCO val2017 (instances_val2017.json), sin multitudes (iscrowd)"}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
