"""Arma el conjunto v8: el v7 restringido al dominio que fija el requisito de cámara.

Uso:
    python scripts/prepare_training_dataset_v8.py --min-alto 40 --min-height 480 --excluir-calidad Very_Poor --excluir-noche
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V7 = ROOT / "data" / "datasets_v7"
OUT = ROOT / "data" / "datasets_v8"
SOURCE = ROOT / "data" / "datasets" / "_sources" / "accident"


def alto_aparente(r: dict) -> float:
    return (float(r["y2"]) - float(r["y1"])) * int(r["height"]) * 960 / int(r["width"])


def in_domain(r: dict, a: argparse.Namespace) -> bool:
    if alto_aparente(r) < a.min_alto or int(r["height"]) < a.min_height:
        return False
    if r["quality"] in a.excluir_calidad.split(","):
        return False
    if a.excluir_noche and r["day_time"] == "night":
        return False
    if a.escenas and r["scene_layout"] not in a.escenas.split(","):
        return False
    return True


def link(src: Path, dst: Path) -> None:
    import os
    import shutil

    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-alto", type=float, default=40.0)
    parser.add_argument("--min-height", type=int, default=480)
    parser.add_argument("--excluir-calidad", default="Very_Poor")
    parser.add_argument("--excluir-noche", action="store_true")
    parser.add_argument("--escenas", default=None, help="Tipos de escena permitidos, separados por coma (por defecto, todos)")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    meta = {Path(r["path"]).stem: r for r in csv.DictReader(open(SOURCE / "metadata-real.csv", encoding="utf-8"))}
    keep_stem = {s: in_domain(r, args) for s, r in meta.items()}
    pattern = re.compile(r"^(?:accident|crop)_(.+?)_(?:[pn]\d+b?)\.")

    stats = {"train": {"mantenidas": 0, "quitadas_accident": 0, "quitadas_justjuu": 0},
             "val": {"mantenidas": 0, "quitadas_accident": 0, "quitadas_justjuu": 0}, "test": {"mantenidas": 0}}
    for sub in ("images", "labels"):
        for split in ("train", "val", "test"):
            (args.out / sub / split).mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        for img in sorted((V7 / "images" / split).iterdir()):
            label = V7 / "labels" / split / f"{img.stem}.txt"
            drop = None
            if split != "test":
                m = pattern.match(img.name)
                if m and m.group(1) in keep_stem and not keep_stem[m.group(1)]:
                    drop = "quitadas_accident"
                elif img.name.startswith("jj_") and label.exists():
                    for line in label.read_text(encoding="utf-8").splitlines():
                        p = line.split()
                        if len(p) == 5 and p[0] == "5" and float(p[4]) * 640 * 960 / 640 < args.min_alto:
                            drop = "quitadas_justjuu"
                            break
            if drop:
                stats[split][drop] += 1
                continue
            link(img, args.out / "images" / split / img.name)
            if label.exists():
                link(label, args.out / "labels" / split / label.name)
            stats[split]["mantenidas"] += 1
    text = (V7 / "data.yaml").read_text(encoding="utf-8").replace("datasets_v7", args.out.name)
    (args.out / "data.yaml").write_text(text, encoding="utf-8")

    by_split = {}
    for split, name in (("train", "train"), ("val", "val"), ("test", "test")):
        stems = {m.group(1) for img in (args.out / "images" / name).iterdir() if (m := pattern.match(img.name))}
        by_split[split] = len(stems)
    summary = {"criterio": {k: (v if not isinstance(v, Path) else str(v)) for k, v in vars(args).items() if k != "out"},
               "videos_en_dominio_total": sum(keep_stem.values()), "videos_totales": len(keep_stem),
               "imagenes": stats, "videos_con_imagenes_por_particion": by_split}
    Path(ROOT.parent / "reentreno" / "dataset_v8_composicion.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
