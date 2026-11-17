"""Cuántos videos de choque reciben algún aviso, y cuándo: antes (más de 3 s), cerca del impacto, o más tarde."""
import csv
import sys
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import compare_models_domain as cmp  # noqa: E402
import evaluate_alert_configs as e  # noqa: E402
import evaluate_classifier_events as ece  # noqa: E402

if __name__ == "__main__":
    meta = {Path(r["path"]).stem: r for r in csv.DictReader(open(ece.SOURCE / "metadata-real.csv", encoding="utf-8"))}
    stems = [s for s in sorted(ece.split_stems("test")) if cmp.in_domain(meta[s])]
    cfgs = [{"nombre": "x", "fps": 6.0, "directo": True, "conf": c, "persist_s": n / 6, "ventana_s": 2.0, "dedup_s": 90.0, "dedup_modo": "renovar", "min_muestras": 1}
            for c, n in ((0.25, 1), (0.4, 1), (0.5, 1), (0.5, 2), (0.5, 4))]
    w = str(ROOT / "data" / "models" / "yolo" / "baseline_v6.pt")
    with Pool(4) as pool:
        res = pool.map(e.run_config, [(c, w, stems) for c in cfgs])
    print("umbral | detecciones | con algún aviso en el video | primer aviso: >3 s antes / de 3 s antes a 0 / 0 a 3 s después / más tarde | sin aviso")
    for c, r in zip(cfgs, res):
        a = b = d = f = n0 = 0
        for s, v in r["videos"].items():
            ev, t0 = v["eventos"], v["t0"]
            if not ev:
                n0 += 1
                continue
            t = ev[0] - t0
            if t < -3: a += 1
            elif t < 0: b += 1
            elif t <= 3: d += 1
            else: f += 1
        n = len(stems)
        print(f"  {c['conf']:.2f} | {round(c['persist_s'] * 6)} | {n - n0}/{n} ({(n - n0) / n:.0%}) | {a} / {b} / {d} / {f} | {n0}")
