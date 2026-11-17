"""Entrena v10 con exactamente la configuración registrada de v6 (args.yaml), cambiando solo el conjunto y el nombre.

Uso (desde la raíz del repositorio):
    .venv/Scripts/python.exe scripts/reentreno_v10/train_v10.py
    .venv/Scripts/python.exe scripts/reentreno_v10/train_v10.py --resume
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

APP = Path(__file__).resolve().parents[2]
PROJECT = APP / "data" / "models" / "yolo"
NAME = "baseline_v10_training"

SKIP = {"task", "mode", "model", "data", "name", "project", "save_dir", "resume", "source", "cfg", "tracker"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    from ultralytics import YOLO

    if args.resume:
        YOLO(str(PROJECT / NAME / "weights" / "last.pt")).train(resume=True)
        return

    v6 = yaml.safe_load((PROJECT / "baseline_v6_training" / "args.yaml").read_text(encoding="utf-8"))
    train_args = {k: v for k, v in v6.items() if k not in SKIP}
    train_args.update(data=str(APP / "data" / "datasets_v10" / "data.yaml"), project=str(PROJECT), name=NAME, exist_ok=True)
    print({k: train_args[k] for k in ("epochs", "batch", "imgsz", "patience", "seed", "deterministic", "close_mosaic", "optimizer")})
    YOLO(str(PROJECT / "yolo11s.pt")).train(**train_args)


if __name__ == "__main__":
    main()
