"""Hojas de contacto de una muestra al azar por fuente de datos, para revisar a ojo si las imágenes son CCTV elevado.

Uso:
    python scripts/make_contact_sheets.py --per-sheet 48
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data" / "datasets_v5" / "images" / "train"
OUT = ROOT.parent / "reentreno" / "muestras_fuentes"
SOURCES = ("dfire_", "accident_", "jj_", "fasdd_", "coco_")
THUMB = (200, 150)
COLUMNS, ROWS = 8, 6


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-sheet", type=int, default=COLUMNS * ROWS)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    files = sorted(TRAIN.iterdir())
    for prefix in SOURCES:
        pool = [p for p in files if p.name.startswith(prefix) and p.suffix.lower() in (".jpg", ".jpeg", ".png")]
        if not pool:
            continue
        picked = rng.sample(pool, min(args.per_sheet, len(pool)))
        sheet = Image.new("RGB", (COLUMNS * THUMB[0], ROWS * (THUMB[1] + 14)), "white")
        draw = ImageDraw.Draw(sheet)
        for index, path in enumerate(picked):
            image = Image.open(path)
            image.draft("RGB", THUMB)
            image = image.convert("RGB").resize(THUMB)
            x, y = (index % COLUMNS) * THUMB[0], (index // COLUMNS) * (THUMB[1] + 14)
            sheet.paste(image, (x, y))
            draw.text((x + 2, y + THUMB[1] + 1), path.stem[-20:], fill="black")
        target = OUT / f"{prefix.rstrip('_')}.jpg"
        sheet.save(target, quality=85)
        print(f"{prefix.rstrip('_'):<10} {len(pool):>6} imágenes en entrenamiento -> {target}")


if __name__ == "__main__":
    main()
