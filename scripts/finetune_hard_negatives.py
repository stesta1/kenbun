"""Ajuste fino de un detector con negativos difíciles (ver `prepare_hard_negatives.py`).

Uso:
    python scripts/finetune_hard_negatives.py --base data/models/yolo/baseline_v6.pt --name baseline_v6hn
"""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "datasets_hn" / "data.yaml")
    parser.add_argument("--name", required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr0", type=float, default=0.002)
    args = parser.parse_args()

    from ultralytics import YOLO

    YOLO(str(args.base)).train(
        data=str(args.data), epochs=args.epochs, imgsz=960, batch=args.batch, workers=args.workers, device=0,
        optimizer="SGD", lr0=args.lr0, lrf=0.1, warmup_epochs=0, close_mosaic=3, val=False, plots=False,
        project=str(ROOT / "data" / "models" / "yolo"), name=args.name, exist_ok=True, seed=0,
    )


if __name__ == "__main__":
    main()
