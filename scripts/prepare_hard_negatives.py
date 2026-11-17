"""Conjunto de ajuste fino con negativos difíciles de tránsito denso, para bajar los falsos positivos de `accident`.

Uso:
    python scripts/prepare_hard_negatives.py --corte 1023
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import evaluate_classifier_events as ece  # noqa: E402
import train_trajectory_classifier as ttc  # noqa: E402

OUT = ROOT / "data" / "datasets_hn"
V8 = ROOT / "data" / "datasets_v8"
COCO = {0: 0, 2: 1, 3: 2}


def frames_of(weights: Path) -> list[dict]:
    wid = f"{weights.name}-{weights.stat().st_size}-{int(weights.stat().st_mtime)}"
    return json.loads(ttc.cache_path(wid, ece.V7, 960).read_text(encoding="utf-8"))["frames"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corte", type=float, default=1023.0)
    parser.add_argument("--umbral", type=float, default=0.25)
    parser.add_argument("--comunes", type=int, default=600)
    parser.add_argument("--repos-choque", type=int, default=3000)
    parser.add_argument("--repos-otras", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    yolo = ROOT / "data" / "models" / "yolo"
    hard: dict[float, float] = {}
    for w in (yolo / "baseline_v6.pt", yolo / "baseline_v8_training" / "weights" / "best.pt"):
        for f in frames_of(w):
            if f["t"] < args.corte:
                conf = max((d[1] for d in f["dets"] if d[0] == "accident"), default=0.0)
                if conf >= args.umbral:
                    hard[round(f["t"], 2)] = max(hard.get(round(f["t"], 2), 0.0), conf)
    times_hard = sorted(hard)
    allt = [round(f["t"], 2) for f in frames_of(yolo / "baseline_v6.pt") if f["t"] < args.corte]
    pool = [t for t in allt if t not in hard]
    common = sorted(rng.sample(pool, min(args.comunes, len(pool))))
    print(f"negativos difíciles: {len(times_hard)}; comunes: {len(common)}", flush=True)

    for sub in ("images", "labels"):
        (OUT / sub / "train").mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(ece.V7))
    fps = cap.get(cv2.CAP_PROP_FPS)
    names = []
    for kind, ts in (("dificil", times_hard), ("comun", common)):
        for t in ts:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
            ok, im = cap.read()
            if not ok:
                continue
            name = f"hn_{kind}_{int(t * 100):06d}"
            cv2.imwrite(str(OUT / "images" / "train" / f"{name}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 92])
            (OUT / "labels" / "train" / f"{name}.txt").write_text("", encoding="utf-8")
            names.append(name)
    cap.release()

    from ultralytics import YOLO

    coco = YOLO(str(yolo / "yolo11n_coco.pt"))
    for i in range(0, len(names), 32):
        batch = names[i:i + 32]
        preds = coco.predict([str(OUT / "images" / "train" / f"{n}.jpg") for n in batch], conf=0.45, classes=list(COCO), verbose=False, device=0)
        for n, p in zip(batch, preds, strict=True):
            lines = [f"{COCO[int(b.cls[0])]} " + " ".join(f"{v:.6f}" for v in b.xywhn[0].tolist()) for b in p.boxes]
            (OUT / "labels" / "train" / f"{n}.txt").write_text("".join(f"{x}\n" for x in lines), encoding="utf-8")

    listed = [str(p.resolve()) for p in (OUT / "images" / "train").iterdir()]
    v8_train = sorted((V8 / "images" / "train").iterdir())
    choque, otras = [], []
    for img in v8_train:
        label = V8 / "labels" / "train" / f"{img.stem}.txt"
        has = label.exists() and any(line.startswith("5 ") for line in label.read_text(encoding="utf-8").splitlines())
        (choque if has else otras).append(img)
    rng.shuffle(choque)
    rng.shuffle(otras)
    replay = [str(p.resolve()) for p in choque[:args.repos_choque] + otras[:args.repos_otras]]
    train_list = replay + listed + [p for p in listed if "/hn_dificil_" in p.replace("\\", "/")] * 2
    (OUT / "train.txt").write_text("\n".join(train_list), encoding="utf-8")
    names_block = (V8 / "data.yaml").read_text(encoding="utf-8").split("names:")[1]
    (OUT / "data.yaml").write_text(f"path: {OUT.resolve()}\ntrain: train.txt\nval: {(V8 / 'images' / 'val').resolve()}\nnames:{names_block}", encoding="utf-8")
    print(json.dumps({"dificiles": len(times_hard), "comunes": len(common), "reposicion_choque": min(args.repos_choque, len(choque)),
                      "reposicion_otras": min(args.repos_otras, len(otras)), "total_lista": len(train_list), "corte_s": args.corte}, indent=1))


if __name__ == "__main__":
    main()
