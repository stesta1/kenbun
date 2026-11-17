"""Barrido de la confianza y la persistencia de `accident` con las reglas reales del sistema, por video y por evento.

Uso:
    python scripts/sweep_accident_rules.py --out ../reentreno/barrido_reglas_accident.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

CONFS = (0.3, 0.4, 0.5, 0.6, 0.7)
PERSIST = (0.33, 0.5, 0.65, 1.0)
REFRACTORY_S = 10.0
OBJETIVO_FALSOS_H = 3.5
WEIGHTS = ROOT / "data" / "models" / "yolo" / "baseline_v6.pt"


def one_video(job: tuple) -> dict:
    import evaluate_on_videos as ev
    import train_trajectory_classifier as ttc
    from monitoreo_urbano.common.config import settings

    stem, video, t0, weights, imgsz = job
    weights = Path(weights)
    wid = f"{weights.name}-{weights.stat().st_size}-{int(weights.stat().st_mtime)}"
    path = ttc.cache_path(wid, Path(video), imgsz)
    if not path.exists():
        return {"stem": stem, "falta": True}
    frames = json.loads(path.read_text(encoding="utf-8"))["frames"]
    duration = frames[-1]["t"] if frames else 0.0
    out = {"stem": stem, "t0": t0, "duracion": duration, "cfg": {}}
    for conf in CONFS:
        for persist in PERSIST:
            settings.incident_min_seconds = {"fire": 0.8, "accident": persist}
            result = ev.replay(frames, 6.0, {"accident": conf, "fire": 0.403}, 0.35)
            times, last = [], -1e9
            for tr in result["triggers"]:
                if tr["reason"] == "accident" and tr["t"] - last >= REFRACTORY_S:
                    times.append(tr["t"])
                    last = tr["t"]
            out["cfg"][f"{conf}|{persist}"] = times
    return out


def main() -> None:
    import evaluate_classifier_events as ece

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT.parent / "reentreno" / "barrido_reglas_accident.json")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--weights", type=Path, default=WEIGHTS)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--dump-rows", type=Path, default=None, help="Guarda los disparos de cada video y configuración")
    parser.add_argument("--reuse", type=Path, default=None, help="Volcado previo: no se recalculan los videos que ya tienen resultado")
    args = parser.parse_args()
    meta = {Path(r["path"]).stem: r for r in csv.DictReader(open(ece.SOURCE / "metadata-real.csv", encoding="utf-8"))}
    jobs, split_of = [], {}
    for split in ("val", "test"):
        for stem in sorted(ece.split_stems(split)):
            jobs.append((stem, str(ece.SOURCE / meta[stem]["path"]), float(meta[stem]["accident_time"]), str(args.weights), args.imgsz))
            split_of[stem] = split
    jobs.append(("V7 uncut", str(ece.V7), None, str(args.weights), args.imgsz))
    previous = {}
    if args.reuse and args.reuse.exists():
        previous = {r["stem"]: r for r in json.loads(args.reuse.read_text(encoding="utf-8"))["rows"] if not r.get("falta")}
    todo = [j for j in jobs if j[0] not in previous]
    with Pool(args.workers) as pool:
        rows = list(previous[j[0]] for j in jobs if j[0] in previous) + pool.map(one_video, todo, chunksize=4)
    print(f"reusados {len(previous)}, calculados {len(todo)}")
    if args.dump_rows:
        args.dump_rows.write_text(json.dumps({"split": split_of, "rows": rows}), encoding="utf-8")
    missing = [r["stem"] for r in rows if r.get("falta")]
    rows = [r for r in rows if not r.get("falta")]
    print("sin caché:", len(missing))

    def aggregate(split: str, key: str) -> dict:
        det = total = false = 0
        hours = 0.0
        for r in rows:
            if r["stem"] == "V7 uncut" or split_of[r["stem"]] != split:
                continue
            times = r["cfg"][key]
            t0 = r["t0"]
            total += 1
            hours += t0 / 3600
            false += sum(t < t0 for t in times)
            det += any(t >= t0 for t in times)
        return {"videos": total, "detectados": det, "tasa": round(det / total, 3) if total else None,
                "falsos_por_hora": round(false / hours, 2) if hours else None}

    v7 = next((r for r in rows if r["stem"] == "V7 uncut"), None)
    table = []
    for conf in CONFS:
        for persist in PERSIST:
            key = f"{conf}|{persist}"
            row = {"confianza": conf, "persistencia_s": persist, "val": aggregate("val", key), "test": aggregate("test", key)}
            if v7:
                row["v7_falsos_por_hora"] = round(len(v7["cfg"][key]) / (v7["duracion"] / 3600), 2)
            table.append(row)
    ok = [r for r in table if r["val"]["falsos_por_hora"] <= OBJETIVO_FALSOS_H]
    best = max(ok, key=lambda r: (r["val"]["tasa"], -r["val"]["falsos_por_hora"])) if ok else None
    result = {"tabla": table, "elegida_en_validacion": best, "sin_cache": missing}
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    for r in table:
        print(f"conf {r['confianza']} pers {r['persistencia_s']}: val {r['val']['tasa']} ({r['val']['falsos_por_hora']}/h)  "
              f"test {r['test']['tasa']} ({r['test']['falsos_por_hora']}/h)  V7 {r.get('v7_falsos_por_hora')}/h")
    print("elegida:", best and (best["confianza"], best["persistencia_s"]))


if __name__ == "__main__":
    main()
