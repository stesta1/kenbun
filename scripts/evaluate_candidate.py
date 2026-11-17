"""Evalúa de punta a punta un detector candidato y deja los resultados en reentreno/.

Uso:
    python scripts/evaluate_candidate.py --tag v5 --data data/datasets_v5/data.yaml
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT.parent / "reentreno"
PY = sys.executable


def run(args: list[str], log: Path | None = None) -> str:
    result = subprocess.run([PY, "-X", "utf8", *args], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    text = result.stdout + result.stderr
    if log:
        log.write_text(text, encoding="utf-8")
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True, help="Ej. v5; busca data/models/yolo/baseline_<tag>_training/weights/best.pt")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--sweep", type=float, nargs="+", default=[0.5, 0.6, 0.7, 0.8])
    parser.add_argument("--imgsz", type=int, default=640, help="Resolución de inferencia (la del entrenamiento)")
    args = parser.parse_args()
    size = str(args.imgsz)

    weights = ROOT / "data" / "models" / "yolo" / f"baseline_{args.tag}.pt"
    best = ROOT / "data" / "models" / "yolo" / f"baseline_{args.tag}_training" / "weights" / "best.pt"
    if not weights.exists():
        shutil.copy2(best, weights)

    print("1/4 umbrales (val) y prueba (test)", flush=True)
    thresholds_file = OUT / f"umbrales_{args.tag}_precision.json"
    run(["scripts/select_thresholds.py", "--weights", str(weights), "--data", str(args.data), "--objective", "precision", "--target", "0.8", "--imgsz", size, "--out", str(thresholds_file)], OUT / f"umbrales_{args.tag}.log")
    thresholds = json.loads(thresholds_file.read_text(encoding="utf-8"))
    fire = thresholds["ajuste"]["fire"]["umbral"]
    accident = thresholds["ajuste"]["accident"]["umbral"]
    print(json.dumps(thresholds["prueba"], ensure_ascii=False), "mAP medio", thresholds["mAP50_media_prueba"], flush=True)

    print("2/4 COCO val (clases base)", flush=True)
    run(["scripts/evaluate_on_coco.py", "--weights", str(weights), "--imgsz", size, "--out", str(OUT / f"coco_baseline_{args.tag}.json")], OUT / f"coco_{args.tag}.log")

    print("3/4 diagnóstico de accident", flush=True)
    datadir = args.data.parent
    run(["scripts/diagnose_accident.py", "--weights", str(weights), "--data", str(datadir), "--cls", "accident", "--imgsz", size], OUT / f"diagnostico_accident_{args.tag}.log")

    print("4/4 videos de test_ui", flush=True)
    summary = {}
    for label, acc in [("val", accident)] + [(str(s), s) for s in args.sweep]:
        out = OUT / f"eval_{args.tag}_acc{label}.json"
        text = run(["scripts/evaluate_on_videos.py", "--weights", str(weights), "--conf", f"fire={fire},accident={acc},default=0.35", "--imgsz", size, "--out", str(out)], OUT / f"eval_{args.tag}_acc{label}.log")
        line = next((ln for ln in text.splitlines() if ln.startswith("Resumen")), "")
        summary[label] = {"accident_conf": acc, "resumen": line}
        print(f"  accident {acc}: {line}", flush=True)
    (OUT / f"evaluacion_{args.tag}.json").write_text(json.dumps({"fire_conf": fire, "umbrales": thresholds, "videos": summary}, indent=2, ensure_ascii=False), encoding="utf-8")
    print("listo", flush=True)


if __name__ == "__main__":
    main()
