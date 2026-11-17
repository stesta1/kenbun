"""Entrena el detector v7 (6 clases, solo datos reales) con YOLO26s a partir de los pesos preentrenados en COCO.

Uso:
    python scripts/train_detector_v7.py --time 7.5
"""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=ROOT / "data" / "models" / "yolo" / "yolo26s.pt")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "datasets_v7" / "data.yaml")
    parser.add_argument("--time", type=float, default=7.5, help="Horas de entrenamiento (reemplaza a --epochs)")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch", type=int, default=12)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--name", default="baseline_v7_training")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    from ultralytics import YOLO

    project = ROOT / "data" / "models" / "yolo"
    if args.resume:
        YOLO(str(project / args.name / "weights" / "last.pt")).train(resume=True)
        return

    YOLO(str(args.base)).train(
        data=str(args.data),
        time=args.time,
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
