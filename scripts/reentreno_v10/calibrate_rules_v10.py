"""Recalibra para v10 los umbrales de la regla de alerta de choque, con el mismo motor del sistema y solo la validación de v10.

Uso (desde la raíz del repositorio):
    .venv/Scripts/python.exe scripts/reentreno_v10/calibrate_rules_v10.py
"""
from __future__ import annotations

import csv
import itertools
import json
import logging
import sys
from pathlib import Path

APP = Path(__file__).resolve().parents[2]
OUT = APP / "reentreno"
sys.path.insert(0, str(APP / "scripts"))
sys.path.insert(0, str(APP / "src"))
sys.path.insert(0, str(Path(__file__).parent))

ALERT = (0.30, 0.35, 0.40, 0.45, 0.50)
STRONG = (0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75)
SECONDS = (0.35, 0.50, 0.65, 0.85, 1.00)


def main() -> None:
    logging.disable(logging.CRITICAL)
    import evaluate_on_videos as ev
    import evaluate_v10 as e10
    from monitoreo_urbano.common.config import settings

    weights = e10.YOLO_DIR / "baseline_v10.pt"
    wid = f"{weights.name}-{weights.stat().st_size}-{int(weights.stat().st_mtime)}"
    ev.IMGSZ[0] = 960
    cache = APP / "data" / "eval_cache"

    def frames_of(video: Path, memo={}):
        if video not in memo:
            memo[video] = ev.detect_video(None, video, e10.FPS, cache, wid)[0]
        return memo[video]

    meta = {Path(r["path"]).stem: r for r in csv.DictReader(open(APP / "data" / "datasets" / "_sources" / "accident" / "metadata-real.csv", encoding="utf-8"))}
    part = {r["video"]: r["particion"] for r in csv.DictReader(open(e10.MANIFEST, encoding="utf-8"))}
    source = APP / "data" / "datasets" / "_sources" / "accident"
    rows = {s: [(v, source / meta[v]["path"], float(meta[v]["accident_time"])) for v in sorted(part) if part[v] == s] for s in ("test", "val")}

    def apply(a: float, s: float, m: float) -> None:
        settings.incident_alert_confidence = {"accident": a}
        settings.incident_strong_confidence = {"accident": s}
        settings.incident_min_seconds = {**settings.incident_min_seconds, "accident": m}

    grid = []
    for a, s, m in itertools.product(ALERT, STRONG, SECONDS):
        if s < a:
            continue
        apply(a, s, m)
        r = e10.tiers(rows["val"], frames_of)
        grid.append({"alerta": a, "fuerte": s, "segundos": m, "oportunas": r["ALERTA"], "precision": r["precision_alerta"], "registrados": r["registrados"]})
        print(grid[-1], flush=True)

    ok = [g for g in grid if g["precision"] is not None and g["precision"] >= 0.90]
    best = max(ok, key=lambda g: (g["oportunas"], g["precision"], g["alerta"], g["fuerte"], g["segundos"]))
    print("ELEGIDA (validación):", best, flush=True)

    apply(best["alerta"], best["fuerte"], best["segundos"])
    test = e10.tiers(rows["test"], frames_of)
    v7 = e10.false_events([ev.VIDEOS_DIR / "V7 uncut.mp4"], frames_of)
    detrac = e10.false_events(sorted(e10.DETRAC.glob("*.mp4")), frames_of)
    adopt = test["ALERTA"] > 94 and (test["precision_alerta"] or 0) >= 0.90 and v7["alertas"] <= 5 and detrac["alertas"] <= 3
    summary = {k: test[k] for k in ("n", "ALERTA", "ALERTA TEMPRANA", "REVISIÓN", "NO REGISTRADO", "registrados", "precision_alerta")}
    print("PRUEBA:", summary, "| V7", v7["alertas"], "alertas", v7["pendientes"], "pend. | DETRAC", detrac["alertas"], "alertas", detrac["pendientes"], "pend.")
    print("SE ADOPTA" if adopt else "NO SE ADOPTA")
    (OUT / "calibracion_reglas_v10.json").write_text(json.dumps({"grilla_val": grid, "elegida": best, "prueba": test, "v7": v7, "detrac": detrac, "adopta": adopt},
                                                                indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
