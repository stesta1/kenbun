"""Evalúa el clasificador de choque por evento, sobre video, con la partición por video del detector.

Uso:
    python scripts/evaluate_classifier_events.py probs --weights data/models/yolo/accident_cls_v1_training/weights/best.pt
    python scripts/evaluate_classifier_events.py sweep --out ../reentreno/cls_v1_eventos.json
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "datasets" / "_sources" / "accident"
DETECTOR_SET = ROOT / "data" / "datasets_v5" / "images"
V7 = ROOT.parent / "test_ui" / "videos" / "V7 uncut.mp4"
PROBS = ROOT.parent / "reentreno" / "cls_v1_probs.json"
FPS = 6.0
REFRACTORY_S = 10.0
FALSOS_POR_HORA_OBJETIVO = 3.5


def split_stems(split: str) -> set[str]:
    stems = set()
    for p in (DETECTOR_SET / split).iterdir():
        m = re.match(r"^accident_(.+)_[pn]\d+\.", p.name)
        if m:
            stems.add(m.group(1))
    return stems


def frame_probs(model, video: Path, accident_idx: int) -> tuple[list[float], float]:
    cap = cv2.VideoCapture(str(video))
    native = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(native / FPS))
    batch, probs, i = [], [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            batch.append(frame)
            if len(batch) == 64:
                probs += [float(r.probs.data[accident_idx]) for r in model.predict(batch, imgsz=320, verbose=False)]
                batch = []
        i += 1
    if batch:
        probs += [float(r.probs.data[accident_idx]) for r in model.predict(batch, imgsz=320, verbose=False)]
    cap.release()
    return probs, native / step


def cmd_probs(args) -> None:
    from ultralytics import YOLO

    model = YOLO(str(args.weights))
    accident_idx = [k for k, v in model.names.items() if v == "accident"][0]
    meta = {Path(r["path"]).stem: r for r in csv.DictReader(open(SOURCE / "metadata-real.csv", encoding="utf-8"))}
    out = {"weights": str(args.weights), "videos": {}}
    if PROBS.exists() and not args.rehacer:
        out = json.loads(PROBS.read_text(encoding="utf-8"))
    jobs = [(s, stem) for s in ("val", "test") for stem in sorted(split_stems(s))] + [("normal", "V7 uncut")]
    for n, (split, stem) in enumerate(jobs, 1):
        if stem in out["videos"]:
            continue
        path = V7 if split == "normal" else SOURCE / meta[stem]["path"]
        probs, fps = frame_probs(model, path, accident_idx)
        row = {"split": split, "fps": fps, "p": [round(x, 4) for x in probs]}
        if split != "normal":
            m = meta[stem]
            row.update({"accident_time": float(m["accident_time"]), "quality": m["quality"], "duration": float(m["duration"]),
                        "height": int(m["height"]), "scene": m["scene_layout"], "weather": m["weather"], "day_time": m["day_time"]})
        out["videos"][stem] = row
        if n % 20 == 0:
            print(f"{n}/{len(jobs)}", flush=True)
            PROBS.write_text(json.dumps(out), encoding="utf-8")
    PROBS.write_text(json.dumps(out), encoding="utf-8")
    print("listo", len(out["videos"]))


def alerts(p: list[float], fps: float, thr: float, n: int, m: int, offset: float = 0.0) -> list[float]:
    times, last = [], -1e9
    for i in range(len(p)):
        window = p[max(0, i - m + 1): i + 1]
        t = offset + i / fps
        if len(window) >= m and sum(x >= thr for x in window) >= n and t - last >= REFRACTORY_S:
            times.append(t)
            last = t
    return times


def measure(videos: dict, thr: float, n: int, m: int) -> dict:
    det = early = total = 0
    delays, pre_hours, pre_false, miss = [], 0.0, 0, []
    for stem, v in videos.items():
        a = alerts(v["p"], v["fps"], thr, n, m, v.get("t_from", 0.0))
        t0 = v["accident_time"]
        total += 1
        pre_hours += (t0 - v.get("t_from", 0.0)) / 3600
        pre_false += sum(t < t0 for t in a)
        after = [t for t in a if t >= t0]
        early += any(t < t0 for t in a)
        if after:
            det += 1
            delays.append(after[0] - t0)
        else:
            miss.append(stem)
    delays.sort()
    return {"videos": total, "detectados": det, "tasa": round(det / total, 3), "con_alerta_temprana": early,
            "falsos_previos": pre_false, "falsos_por_hora_previos": round(pre_false / pre_hours, 2) if pre_hours else None,
            "demora_mediana_s": round(delays[len(delays) // 2], 2) if delays else None,
            "demora_p95_s": round(delays[int(0.95 * (len(delays) - 1))], 2) if delays else None, "sin_deteccion": miss}


def subset144(v: dict) -> bool:
    return v["quality"] in ("Good", "Fine", "Excellent", "Poor") and 6 <= v["duration"] <= 30 and v["height"] >= 480


def cmd_sweep(args) -> None:
    data = json.loads(Path(args.probs).read_text(encoding="utf-8"))["videos"]
    val = {k: v for k, v in data.items() if v["split"] == "val"}
    test = {k: v for k, v in data.items() if v["split"] == "test"}
    v7 = data["V7 uncut"]
    grid = [(t, n, m) for t in (0.5, 0.7, 0.8, 0.9, 0.95, 0.98) for (n, m) in ((1, 1), (2, 3), (3, 6), (4, 6), (6, 6))]
    table = []
    for t, n, m in grid:
        r = measure(val, t, n, m)
        table.append({"umbral": t, "N": n, "M": m, "val_tasa": r["tasa"], "val_falsos_h": r["falsos_por_hora_previos"], "val_demora_med": r["demora_mediana_s"]})
    ok = [r for r in table if r["val_falsos_h"] is not None and r["val_falsos_h"] <= FALSOS_POR_HORA_OBJETIVO]
    best = max(ok, key=lambda r: (r["val_tasa"], -r["val_falsos_h"])) if ok else None
    result = {"validacion": table, "elegida": best}
    if best:
        t, n, m = best["umbral"], best["N"], best["M"]
        res = {"todos": measure(test, t, n, m), "subconjunto_144": measure({k: v for k, v in test.items() if subset144(v)}, t, n, m)}
        for r in res.values():
            r["sin_deteccion"] = r["sin_deteccion"][:5] + ["..."] if len(r["sin_deteccion"]) > 5 else r["sin_deteccion"]
        fv7 = alerts(v7["p"], v7["fps"], t, n, m)
        res["v7_sin_cortes"] = {"minutos": round(len(v7["p"]) / v7["fps"] / 60, 1), "alertas": len(fv7), "instantes_s": [round(x, 1) for x in fv7]}
        result["prueba"] = res
    print(json.dumps(result, indent=1, ensure_ascii=False))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("probs")
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--rehacer", action="store_true")
    p.set_defaults(fn=cmd_probs)
    s = sub.add_parser("sweep")
    s.add_argument("--probs", type=Path, default=PROBS)
    s.add_argument("--out", type=Path, default=None)
    s.set_defaults(fn=cmd_sweep)
    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
