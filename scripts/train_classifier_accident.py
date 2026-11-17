"""Entrena desde cero el clasificador de choque (imagen completa: choque / tránsito normal).

Uso:
    python scripts/train_classifier_accident.py [--epochs 30] [--imgsz 320]
"""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", default="yolo11s-cls.yaml")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "datasets_cls_v1")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--name", default="accident_cls_v1_training")
    parser.add_argument("--optimizer", default="auto")
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--cos-lr", action="store_true")
    parser.add_argument("--pretrained", action="store_true", help="Parte de los pesos preentrenados de --arch (por defecto, desde cero)")
    args = parser.parse_args()

    from ultralytics import YOLO

    YOLO(args.arch).train(
        data=str(args.data),
        pretrained=args.pretrained,
        epochs=args.epochs,
        patience=args.patience,
        batch=args.batch,
        imgsz=args.imgsz,
        workers=args.workers,
        device=0,
        project=str(ROOT / "data" / "models" / "yolo"),
        name=args.name,
        exist_ok=True,
        seed=0,
        optimizer=args.optimizer,
        lr0=args.lr0,
        cos_lr=args.cos_lr,
        plots=True,
    )


if __name__ == "__main__":
    main()
