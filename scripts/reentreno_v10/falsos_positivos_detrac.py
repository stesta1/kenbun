"""Falsos positivos de v6 en tránsito normal: secuencias de UA-DETRAC (Wen et al., arXiv 1511.04136), sin choques.

Uso (desde la raíz del repositorio):
    .venv/Scripts/python.exe scripts/reentreno_v10/falsos_positivos_detrac.py --out-dir reentreno/corrida_noche/<fecha>/detrac
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np

APP = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(APP / "scripts"))
sys.path.insert(0, str(APP / "src"))
ZIP = APP / "data" / "downloads_detrac" / "ua_detrac_training_set.zip"
EVAL = APP / "data" / "detrac_eval"
EXTRA = 12
SEED = 42
FPS = 6.0
V7 = {"minutos": 34.1, "alertas_por_h": 8.8, "pendientes_por_h": 5.3}


def to_video(z: zipfile.ZipFile, names: list[str], seq: str, out: Path) -> None:
    frames = sorted(n for n in names if n.startswith(seq + "/") and n.endswith(".jpg"))
    writer = None
    for n in frames:
        img = cv2.imdecode(np.frombuffer(z.read(n), np.uint8), cv2.IMREAD_COLOR)
        if writer is None:
            h, w = img.shape[:2]
            writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), 25, (w, h))
        writer.write(img)
    if writer is not None:
        writer.release()


def snapshot(video: Path, t: float, dets: list, out: Path) -> None:
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, img = cap.read()
    cap.release()
    if not ok:
        return
    for name, conf, x1, y1, x2, y2 in dets:
        if name == "accident":
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            cv2.putText(img, f"choque {conf:.2f}", (int(x1), max(15, int(y1) - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.imwrite(str(out), img)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=APP / "data" / "models" / "yolo" / "baseline_v6.pt")
    args = parser.parse_args()
    logging.disable(logging.CRITICAL)
    import evaluate_on_videos as ev
    from evaluate_tiers import events_of
    from ultralytics import YOLO

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "fotogramas").mkdir(exist_ok=True)

    z = zipfile.ZipFile(ZIP)
    names = z.namelist()
    sequences = sorted({n.split("/")[0] for n in names if n.endswith("annotations.csv")})
    done = sorted(p.stem for p in EVAL.glob("*.mp4"))
    extra = random.Random(SEED).sample([s for s in sequences if s not in done], EXTRA)
    for seq in extra:
        if not (EVAL / f"{seq}.mp4").exists():
            print("convirtiendo", seq, flush=True)
            to_video(z, names, seq, EVAL / f"{seq}.mp4")
    videos = [EVAL / f"{s}.mp4" for s in sorted(done + extra)]

    model = YOLO(str(args.weights))
    wid = f"{args.weights.name}-{args.weights.stat().st_size}-{int(args.weights.stat().st_mtime)}"
    ev.IMGSZ[0] = 960
    cache = APP / "data" / "eval_cache"
    cache.mkdir(exist_ok=True)

    detalle, total_s, n_alert, n_pend = [], 0.0, 0, 0
    for video in videos:
        frames = ev.detect_video(model, video, FPS, cache, wid)[0]
        eventos = events_of(frames)
        secs = frames[-1]["t"] if frames else 0.0
        total_s += secs
        filas = []
        for t, motivo in eventos:
            nearest = min(frames, key=lambda f: abs(f["t"] - t))
            conf = max((d[1] for d in nearest["dets"] if d[0] == "accident"), default=None)
            foto = args.out_dir / "fotogramas" / f"{video.stem}_{t:06.1f}_{motivo}.jpg"
            snapshot(video, nearest["t"], nearest["dets"], foto)
            filas.append({"t_s": t, "motivo": motivo, "confianza_choque": conf, "fotograma": foto.name})
        a = sum(f["motivo"] == "accident" for f in filas)
        p = sum(f["motivo"] == "posible_choque" for f in filas)
        n_alert, n_pend = n_alert + a, n_pend + p
        detalle.append({"secuencia": video.stem, "segundos": round(secs, 1), "alertas": a, "pendientes": p, "eventos": filas})
        print(f"{video.stem}: {secs:.0f} s, {a} alertas, {p} pendientes", flush=True)

    h = total_s / 3600
    resumen = {"secuencias": len(videos), "minutos": round(total_s / 60, 1), "alertas": n_alert, "alertas_por_h": round(n_alert / h, 1),
               "pendientes": n_pend, "pendientes_por_h": round(n_pend / h, 1),
               "secuencias_con_alerta": sum(d["alertas"] > 0 for d in detalle), "v7_referencia": V7,
               "pesos": args.weights.name, "fps_analisis": FPS, "imgsz": 960, "semilla_seleccion": SEED,
               "seleccion": {"ya_convertidas": done, "agregadas": sorted(extra)}}
    print(json.dumps(resumen, indent=2, ensure_ascii=False))
    (args.out_dir / "falsos_positivos_detrac.json").write_text(json.dumps({"resumen": resumen, "detalle": detalle}, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
