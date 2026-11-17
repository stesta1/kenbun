"""Compara v6 y v10 con la misma vara: particiones de v10 (sin video de origen compartido) y los mismos scripts que v6.

Uso (desde la raíz del repositorio):
    .venv/Scripts/python.exe scripts/reentreno_v10/evaluate_v10.py
"""
from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

APP = Path(__file__).resolve().parents[2]
OUT = APP / "reentreno"
sys.path.insert(0, str(APP / "scripts"))
sys.path.insert(0, str(APP / "src"))
YOLO_DIR = APP / "data" / "models" / "yolo"
DATA = APP / "data" / "datasets_v10" / "data.yaml"
MANIFEST = OUT / "videos_accident_por_particion_v10.csv"
DETRAC = APP / "data" / "detrac_eval"
TOL = 3.0
FPS = 6.0


def per_image(tag: str, weights: Path) -> dict:
    out = OUT / f"umbrales_{tag}_particion_v10.json"
    if not out.exists():
        subprocess.run([sys.executable, "-X", "utf8", "scripts/select_thresholds.py", "--weights", str(weights), "--data", str(DATA),
                        "--objective", "precision", "--target", "0.8", "--imgsz", "960", "--out", str(out)], cwd=APP, check=True)
    return json.loads(out.read_text(encoding="utf-8"))


def tiers(rows: list[tuple[str, Path, float]], frames_of) -> dict:
    from evaluate_tiers import events_of

    videos, good, early = [], 0, 0
    for stem, video, t0 in rows:
        ev = events_of(frames_of(video))
        alerts = [t - t0 for t, r in ev if r == "accident"]
        reviews = [t - t0 for t, r in ev if r == "posible_choque"]
        good += sum(a >= -TOL for a in alerts)
        early += sum(a < -TOL for a in alerts)
        nivel = ("ALERTA" if any(a >= -TOL for a in alerts) else "ALERTA TEMPRANA" if alerts else "REVISIÓN" if reviews else "NO REGISTRADO")
        videos.append({"video": stem, "t0": round(t0, 1), "alertas_s": [round(a, 1) for a in alerts], "revisiones_s": [round(r, 1) for r in reviews], "nivel": nivel})
    c = Counter(v["nivel"] for v in videos)
    n = len(videos)
    return {"n": n, **{k: c[k] for k in ("ALERTA", "ALERTA TEMPRANA", "REVISIÓN", "NO REGISTRADO")},
            "sensibilidad_alerta": round(c["ALERTA"] / n, 3), "registrados": n - c["NO REGISTRADO"],
            "alertas_correctas": good, "alertas_tempranas": early, "precision_alerta": round(good / (good + early), 3) if good + early else None,
            "videos": videos}


def false_events(videos: list[Path], frames_of) -> dict:
    from evaluate_tiers import events_of

    a = r = 0
    seconds = 0.0
    detail = {}
    for video in videos:
        frames = frames_of(video)
        ev = events_of(frames)
        na, nr = sum(x == "accident" for _, x in ev), sum(x == "posible_choque" for _, x in ev)
        a, r, seconds = a + na, r + nr, seconds + frames[-1]["t"]
        detail[video.stem] = {"alertas": [t for t, x in ev if x == "accident"], "pendientes": [t for t, x in ev if x == "posible_choque"]}
    h = seconds / 3600
    return {"minutos": round(seconds / 60, 1), "alertas": a, "alertas_por_h": round(a / h, 1), "pendientes": r, "pendientes_por_h": round(r / h, 1), "detalle": detail}


def main() -> None:
    import evaluate_on_videos as ev
    from ultralytics import YOLO

    v10 = YOLO_DIR / "baseline_v10.pt"
    if not v10.exists():
        shutil.copy2(YOLO_DIR / "baseline_v10_training" / "weights" / "best.pt", v10)
    models = {"v6": YOLO_DIR / "baseline_v6.pt", "v10": v10}

    meta = {Path(r["path"]).stem: r for r in csv.DictReader(open(APP / "data" / "datasets" / "_sources" / "accident" / "metadata-real.csv", encoding="utf-8"))}
    part = {r["video"]: r["particion"] for r in csv.DictReader(open(MANIFEST, encoding="utf-8"))}
    source = APP / "data" / "datasets" / "_sources" / "accident"
    split_rows = {s: [(v, source / meta[v]["path"], float(meta[v]["accident_time"])) for v in sorted(part) if part[v] == s] for s in ("test", "val")}
    v7 = ev.VIDEOS_DIR / "V7 uncut.mp4"
    detrac = sorted(DETRAC.glob("*.mp4"))
    ev.IMGSZ[0] = 960

    report = {}
    for tag, weights in models.items():
        print(f"== {tag}: por imagen", flush=True)
        img = per_image(tag, weights)
        model = YOLO(str(weights))
        wid = f"{weights.name}-{weights.stat().st_size}-{int(weights.stat().st_mtime)}"
        cache = APP / "data" / "eval_cache"
        cache.mkdir(exist_ok=True)

        def frames_of(video: Path, model=model, wid=wid, cache=cache):
            return ev.detect_video(model, video, FPS, cache, wid)[0]

        print(f"== {tag}: videos", flush=True)
        report[tag] = {
            "por_imagen": {"mAP50_media_prueba": img["mAP50_media_prueba"], "prueba": img["prueba"], "umbrales": img["ajuste"]},
            "test": tiers(split_rows["test"], frames_of),
            "val": tiers(split_rows["val"], frames_of),
            "v7": false_events([v7], frames_of),
            "detrac": false_events(detrac, frames_of),
        }
        r = report[tag]
        print(f"   test: alerta {r['test']['ALERTA']}/{r['test']['n']}, precisión {r['test']['precision_alerta']}, registrados {r['test']['registrados']}; "
              f"V7 {r['v7']['alertas']} alertas/{r['v7']['pendientes']} pend.; DETRAC {r['detrac']['alertas']} alertas/{r['detrac']['pendientes']} pend.", flush=True)
        (OUT / "evaluacion_v6_v10.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
