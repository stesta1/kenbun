"""Prueba offline de un filtro por movimiento sobre las alertas de choque de la regla directa.

Uso:
    python scripts/veto_motion_analysis.py --weights data/models/yolo/baseline_v6.pt --fps 15 --rows ../reentreno/barrido_directo_v6_15fps.json
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import evaluate_classifier_events as ece  # noqa: E402
import train_trajectory_classifier as ttc  # noqa: E402

TOL = 3.0
VEH = ("car", "motorcycle")


def centre(b):
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def size(b):
    return math.sqrt(max((b[2] - b[0]) * (b[3] - b[1]), 1.0))


def motion_feature(frames: list[dict], times: list[float], t: float, delta: float = 1.0):
    i = bisect.bisect_left(times, t)
    near = [f for f in frames if abs(f["t"] - t) <= 0.5]
    best = None
    for f in near:
        for d in f["dets"]:
            if d[0] == "accident" and d[1] >= 0.3 and (best is None or d[1] > best[1]):
                best = d
    if best is None:
        return None
    box = best[2:6]
    cx, cy = centre(box)
    w, h = (box[2] - box[0]) * 1.2, (box[3] - box[1]) * 1.2
    ia = min(max(i, 0), len(frames) - 1)
    ib = min(bisect.bisect_left(times, frames[ia]["t"] + delta), len(frames) - 1)
    if ib == ia:
        return None
    va = [d[2:6] for d in frames[ia]["dets"] if d[0] in VEH and d[1] >= 0.3 and abs(centre(d[2:6])[0] - cx) <= w / 2 and abs(centre(d[2:6])[1] - cy) <= h / 2]
    vb = [d[2:6] for d in frames[ib]["dets"] if d[0] in VEH and d[1] >= 0.3]
    dt = frames[ib]["t"] - frames[ia]["t"]
    speeds = []
    for a in va:
        ca = centre(a)
        cand = [(math.hypot(centre(b)[0] - ca[0], centre(b)[1] - ca[1]), b) for b in vb]
        if not cand:
            continue
        dist, _ = min(cand, key=lambda x: x[0])
        if dist <= 1.5 * size(a):
            speeds.append(dist / dt / size(a))
    return statistics.median(speeds) if speeds else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--cfgs", nargs="+", default=["0.7|1|1.5", "0.6|2|1.5", "0.6|1|3.0", "0.5|2|1.5"])
    parser.add_argument("--umbrales", type=float, nargs="+", default=[0.5, 1.0, 1.5, 2.0, 3.0])
    args = parser.parse_args()
    data = json.loads(args.rows.read_text(encoding="utf-8"))
    meta = {Path(r["path"]).stem: r for r in csv.DictReader(open(ece.SOURCE / "metadata-real.csv", encoding="utf-8"))}
    wid = f"{args.weights.name}-{args.weights.stat().st_size}-{int(args.weights.stat().st_mtime)}"
    rows = {r["stem"]: r for r in data["rows"]}
    cache = {}

    def frames_of(stem: str):
        if stem not in cache:
            video = ece.V7 if stem == "V7 uncut" else ece.SOURCE / meta[stem]["path"]
            fr = json.loads(ttc.cache_path(wid, video, 960, args.fps).read_text(encoding="utf-8"))["frames"]
            cache[stem] = (fr, [f["t"] for f in fr])
        return cache[stem]

    feats: dict[tuple[str, str], list[tuple[float, float | None]]] = {}
    for cfg in args.cfgs:
        for stem, r in rows.items():
            fr, tm = frames_of(stem)
            feats[(cfg, stem)] = [(t, motion_feature(fr, tm, t)) for t in r["cfg"][cfg]]
    v7 = rows["V7 uncut"]
    print("Velocidad de los vehículos dentro de la alerta (largos de cuerpo por segundo, mediana):")
    for cfg in args.cfgs[:1]:
        f7 = [x[1] for x in feats[(cfg, "V7 uncut")]]
        dom = [x[1] for s, r in rows.items() if s != "V7 uncut" and data["split"][s] == "test" for x in feats[(cfg, s)] if x[0] >= r["t0"] - TOL]
        fmt = lambda v: [round(x, 2) if x is not None else None for x in v]
        print(f"  {cfg} V7 (falsos): {fmt(f7)}")
        d = [x for x in dom if x is not None]
        print(f"  {cfg} alertas verdaderas en prueba: n={len(dom)}, medibles {len(d)}, mediana {statistics.median(d) if d else None:.2f}, p75 {sorted(d)[int(.75*len(d))] if d else None:.2f}, p90 {sorted(d)[int(.9*len(d))] if d else None:.2f}")
    for cfg in args.cfgs:
        print(f"\n== {cfg} (dominio, prueba n=74; detección con tolerancia 3 s)")
        for thr in [None] + args.umbrales:
            det = vfalse = 0
            n = 0
            for s, r in rows.items():
                if s == "V7 uncut" or data["split"][s] != "test":
                    continue
                n += 1
                kept = [t for t, f in feats[(cfg, s)] if thr is None or f is None or f < thr]
                det += any(t >= r["t0"] - TOL for t in kept)
                vfalse += any(t < r["t0"] - TOL for t in kept)
            v7k = [t for t, f in feats[(cfg, "V7 uncut")] if thr is None or f is None or f < thr]
            print(f"  veto {'sin filtro' if thr is None else f'>= {thr:g} cuerpos/s':<18} detectados {det}/{n} ({det/n:.2f}) | videos con falsa previa {vfalse} | V7 {len(v7k) / (v7['duracion'] / 3600):.1f}/h")


if __name__ == "__main__":
    main()
