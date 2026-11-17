"""Elige la confianza mínima de cada clase con la partición de ajuste (validación) y la aplica una sola vez a la de prueba.

Uso:
    python scripts/select_thresholds.py --weights data/models/yolo/baseline_v3.pt --out ../reentreno/umbrales_v3.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ALERT_CLASSES = ("fire", "accident")


def curves(model, data: Path, split: str, imgsz: int = 640):
    result = model.val(data=str(data), split=split, imgsz=imgsz, batch=32, conf=0.001, verbose=False, plots=False, workers=0)
    box = result.box
    return result, np.asarray(box.px), np.asarray(box.p_curve), np.asarray(box.r_curve), np.asarray(box.f1_curve)


def choose(px, p, r, f1, objective: str, target: float) -> tuple[float, float, float]:
    if objective == "precision":
        ok = np.where(p >= target)[0]
        index = int(ok[0]) if len(ok) else int(np.argmax(f1))
    else:
        index = int(np.argmax(f1))
    return float(px[index]), float(p[index]), float(r[index])


def evaluate_with(model, data: Path, split: str, thresholds: dict[str, float], names: dict[int, str], imgsz: int = 640) -> dict:
    result = model.val(data=str(data), split=split, imgsz=imgsz, batch=32, conf=0.001, verbose=False, plots=False, workers=0)
    box = result.box
    px = np.asarray(box.px)
    rows = {}
    for j, class_id in enumerate(box.ap_class_index):
        name = names[int(class_id)]
        index = int(np.argmin(np.abs(px - thresholds[name])))
        rows[name] = {
            "umbral": round(thresholds[name], 3),
            "P": round(float(box.p_curve[j][index]), 3),
            "R": round(float(box.r_curve[j][index]), 3),
            "mAP50": round(float(box.ap50[j]), 3),
            "mAP50-95": round(float(box.ap[j]), 3),
        }
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "datasets_v3" / "data.yaml")
    parser.add_argument("--objective", choices=("f1", "precision"), default="f1")
    parser.add_argument("--target", type=float, default=0.8, help="Precisión pedida a las clases que alertan (objetivo precision)")
    parser.add_argument("--general", type=float, default=0.35, help="Umbral de las clases sin criterio propio si no hay curva")
    parser.add_argument("--skip-test", action="store_true", help="Solo elegir con la validación; no mide la prueba")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--imgsz", type=int, default=640, help="Resolución de inferencia")
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(str(args.weights))
    names = model.names

    print("Paso 1: elegir con la partición de ajuste (val)")
    result, px, p, r, f1 = curves(model, args.data, "val", args.imgsz)
    chosen: dict[str, dict] = {}
    for j, class_id in enumerate(result.box.ap_class_index):
        name = names[int(class_id)]
        objective = args.objective if name in ALERT_CLASSES else "f1"
        threshold, precision, recall = choose(px, p[j], r[j], f1[j], objective, args.target)
        chosen[name] = {"umbral": round(threshold, 3), "objetivo": objective, "P_ajuste": round(precision, 3), "R_ajuste": round(recall, 3)}
        print(f"  {name:<11} umbral {threshold:.3f}  ({objective})  P {precision:.3f}  R {recall:.3f}")
    thresholds = {name: row["umbral"] for name, row in chosen.items()}

    test: dict = {}
    if not args.skip_test:
        print("Paso 2: medir una sola vez con esos umbrales en la partición de prueba (test)")
        test = evaluate_with(model, args.data, "test", thresholds, names, args.imgsz)
    for name, row in test.items():
        print(f"  {name:<11} umbral {row['umbral']:.3f}  P {row['P']:.3f}  R {row['R']:.3f}  mAP50 {row['mAP50']:.3f}")
    mean_map = float(np.mean([row["mAP50"] for row in test.values()])) if test else None
    if test:
        print(f"  mAP50 media de {len(test)} clases: {mean_map:.3f}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps({"pesos": args.weights.name, "objetivo": args.objective, "ajuste": chosen, "prueba": test, "mAP50_media_prueba": round(mean_map, 3) if mean_map else None}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"Resultados en {args.out}")


if __name__ == "__main__":
    main()
