"""Empaqueta un conjunto de datos YOLO en un zip sin compresión (las imágenes ya están comprimidas) para subirlo a Drive.

Uso:
    python scripts/package_for_colab.py --dataset data/datasets_v5 --out colab/datasets_v5.zip
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=ROOT / "data" / "datasets_v5")
    parser.add_argument("--out", type=Path, default=ROOT / "colab" / "datasets_v5.zip")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    yaml_text = (
        "path: .\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n\n"
        "names:\n"
        "  0: person\n  1: car\n  2: motorcycle\n  3: fire\n  4: smoke\n  5: accident\n"
    )
    total = 0
    with zipfile.ZipFile(args.out, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        archive.writestr("data.yaml", yaml_text)
        for sub in ("images", "labels"):
            for split in ("train", "val", "test"):
                for path in sorted((args.dataset / sub / split).iterdir()):
                    archive.write(path, f"{sub}/{split}/{path.name}")
                    total += 1
    print(f"{total} archivos, {args.out.stat().st_size / 1e9:.2f} GB -> {args.out}")


if __name__ == "__main__":
    main()
