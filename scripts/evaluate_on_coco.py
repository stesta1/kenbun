"""Evalúa `person`, `car` y `motorcycle` de un detector sobre la muestra anotada de COCO val2017.

Uso:
    python scripts/evaluate_on_coco.py --weights data/models/yolo/baseline_v3.pt --out ../reentreno/coco_v3.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_CLASSES = ("person", "car", "motorcycle")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--conf", type=float, default=0.001, help="Umbral de la evaluación (para la curva completa)")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--imgsz", type=int, default=640, help="Resolución de inferencia")
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(str(args.weights))
    composition = json.loads((ROOT / "data" / "eval_coco" / "composicion.json").read_text(encoding="utf-8"))
    report = {"pesos": args.weights.name, "imagenes": composition["imagenes"], "fuente": composition["fuente"], "variantes": {}}
    for variant in ("strict", "vehicles"):
        data = ROOT / "data" / "eval_coco" / variant / "data.yaml"
        result = model.val(data=str(data), split="val", imgsz=args.imgsz, batch=32, conf=args.conf, verbose=False, plots=False, workers=0)
        rows = {}
        for j, class_id in enumerate(result.box.ap_class_index):
            name = result.names[int(class_id)]
            if name in BASE_CLASSES:
                rows[name] = {
                    "n": composition["instancias"][variant].get(name),
                    "P": round(float(result.box.p[j]), 3),
                    "R": round(float(result.box.r[j]), 3),
                    "mAP50": round(float(result.box.ap50[j]), 3),
                    "mAP50-95": round(float(result.box.ap[j]), 3),
                }
        report["variantes"][variant] = rows
        print(variant, json.dumps(rows, ensure_ascii=False))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Resultados en {args.out}")


if __name__ == "__main__":
    main()
