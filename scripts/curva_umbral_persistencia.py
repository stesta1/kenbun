"""Cuánto cuesta avisar en más choques: barre el umbral de confianza y las detecciones mínimas de la regla directa (6 fps, deduplicación `renovar`)."""
import csv
import itertools
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
    split_of = {s: sp for sp in ("val", "test") for s in sorted(ece.split_stems(sp)) if cmp.in_domain(meta[s])}
    cfgs = [{"nombre": "x", "fps": 6.0, "directo": True, "conf": conf, "persist_s": n / 6, "ventana_s": 2.0, "dedup_s": 90.0,
             "dedup_modo": modo, "min_muestras": 1} for conf, n, modo in itertools.product((0.25, 0.4, 0.5), (1, 2), ("global", "renovar", "ubicacion"))]
    w = str(ROOT / "data" / "models" / "yolo" / "baseline_v6.pt")
    with Pool(4) as pool:
        res = pool.map(e.run_config, [(c, w, sorted(split_of)) for c in cfgs])
    print("umbral | detecciones mínimas | deduplicación | avisan (de 74) | eventos falsos en V7 (34 min) | videos con falsa previa (>3 s antes) | demora mediana s")
    for c, r in zip(cfgs, res):
        t = e.summarize(r, split_of, "test")
        v7 = len(r["v7"]["eventos"])
        print(f"  {c['conf']:.2f} | {round(c['persist_s'] * 6)} | {c['dedup_modo']:<9} | {t['detectados']:>2} ({t['detectados'] / 74:.0%}) | {v7} ({v7 / (r['v7']['duracion'] / 3600):.0f}/h) | {t['videos_con_falsa_previa']} | {t['demora_mediana_s']} | más de 1 evento: {t['videos_con_mas_de_un_evento']}")
