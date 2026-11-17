"""Tabla por video de prueba del dominio: qué hizo el detector, si avisó el sistema, si quedaría para revisión o si no se detectó."""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import compare_models_domain as cmp  # noqa: E402
import evaluate_alert_configs as e  # noqa: E402
import evaluate_classifier_events as ece  # noqa: E402
import train_trajectory_classifier as ttc  # noqa: E402

REVISION = 0.25
TIPOS = {"t-bone": "choque en T", "rear-end": "alcance", "single": "un solo vehículo", "sideswipe": "roce lateral", "head-on": "frontal"}

if __name__ == "__main__":
    meta = {Path(r["path"]).stem: r for r in csv.DictReader(open(ece.SOURCE / "metadata-real.csv", encoding="utf-8"))}
    stems = [s for s in sorted(ece.split_stems("test")) if cmp.in_domain(meta[s])]
    w = ROOT / "data" / "models" / "yolo" / "baseline_v6.pt"
    wid = f"{w.name}-{w.stat().st_size}-{int(w.stat().st_mtime)}"
    cfg = {"nombre": "defaults", "fps": 6.0, "directo": True, "conf": 0.5, "persist_s": 0.65, "ventana_s": 2.0, "dedup_s": 90.0, "dedup_modo": "renovar", "min_muestras": 3}
    eventos = e.run_config((cfg, str(w), stems))["videos"]
    rows = []
    for s in stems:
        m = meta[s]
        t0 = float(m["accident_time"])
        fr = json.loads(ttc.cache_path(wid, ece.SOURCE / m["path"], 960).read_text(encoding="utf-8"))["frames"]
        window = [f for f in fr if f["t"] >= t0 - 3]
        confs = [(f["t"], max((d[1] for d in f["dets"] if d[0] == "accident"), default=0.0)) for f in window]
        mx = max((c for _, c in confs), default=0.0)
        n25 = sum(c >= REVISION for _, c in confs)
        n50 = sum(c >= 0.5 for _, c in confs)
        ev = eventos[s]["eventos"]
        alerta = [t - t0 for t in ev if t >= t0 - 3]
        temprana = [t - t0 for t in ev if t < t0 - 3]
        if alerta:
            nivel = "ALERTA"
        elif temprana:
            nivel = "ALERTA TEMPRANA"
        elif n25:
            nivel = "REVISIÓN"
        else:
            nivel = "NO DETECTADO"
        rows.append({"video": s, "choque": "sí", "tipo": TIPOS.get(m["type"], m["type"]), "escena": m["scene_layout"], "impacto_s": round(t0, 1),
                     "detector_dispara": "sí" if n25 else "no", "conf_max": round(mx, 2), "detecciones_ge_0_25": n25, "detecciones_ge_0_5": n50,
                     "primer_aviso_s_vs_impacto": round(alerta[0], 1) if alerta else (round(temprana[0], 1) if temprana else None), "nivel": nivel})
    out = ROOT.parent / "reentreno" / "tabla_74_videos_niveles.json"
    out.write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    with open(ROOT.parent / "reentreno" / "tabla_74_videos_niveles.csv", "w", encoding="utf-8-sig", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    from collections import Counter
    print(Counter(r["nivel"] for r in rows), "| el detector dispara (>=0,25) en", sum(r["detector_dispara"] == "sí" for r in rows), "de", len(rows))
