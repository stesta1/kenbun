"""Entrena el detector v2 (6 clases, solo datos reales) a partir de los pesos preentrenados en COCO.

Uso:
    python scripts/train_detector_v2.py [--epochs 60] [--batch 16]
"""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=ROOT / "data" / "models" / "yolo" / "yolo11s.pt")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "datasets_v2" / "data.yaml")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=15, help="Épocas sin mejora antes de cortar")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--name", default="baseline_v2_training")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    from ultralytics import YOLO

    project = ROOT / "data" / "models" / "yolo"
    if args.resume:
        YOLO(str(project / args.name / "weights" / "last.pt")).train(resume=True)
        return

    model = YOLO(str(args.base))
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        patience=args.patience,
        batch=args.batch,
        imgsz=args.imgsz,
        workers=args.workers,
        device=0,
        project=str(project),
        name=args.name,
        exist_ok=True,
        seed=0,
        close_mosaic=10,
        plots=True,
    )


if __name__ == "__main__":
    main()
