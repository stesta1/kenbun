"""Regla de coincidencia entre dos detectores: se alerta solo si ambos disparan `accident` con menos de `--ventana` s de diferencia.

Uso:
    python scripts/ensemble_rule_models.py --a ../reentreno/barrido_reglas_por_video.json --b ../reentreno/barrido_v8_por_video.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import compare_models_domain as cmp  # noqa: E402
import evaluate_classifier_events as ece  # noqa: E402


def agree(times_b: list[float], times_a: list[float], window: float) -> list[float]:
    return [t for t in times_b if any(abs(t - u) <= window for u in times_a)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", type=Path, required=True)
    parser.add_argument("--b", type=Path, required=True)
    parser.add_argument("--ventana", type=float, default=3.0)
    parser.add_argument("--max-v7", type=float, nargs="+", default=[3.6, 7.1])
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    A = json.loads(args.a.read_text(encoding="utf-8"))
    B = json.loads(args.b.read_text(encoding="utf-8"))
    meta = {Path(r["path"]).stem: r for r in csv.DictReader(open(ece.SOURCE / "metadata-real.csv", encoding="utf-8"))}
    ra = {r["stem"]: r for r in A["rows"]}
    rb = {r["stem"]: r for r in B["rows"]}
    cfgs_a, cfgs_b = list(A["rows"][0]["cfg"]), list(B["rows"][0]["cfg"])
    split_of = A["split"]
    stems = {s: [x for x in sorted(ece.split_stems(s)) if cmp.in_domain(meta[x])] for s in ("val", "test")}
    v7a, v7b = ra["V7 uncut"], rb["V7 uncut"]

    def evaluate(split: str, ca: str, cb: str) -> dict:
        det = false = 0
        hours = 0.0
        for s in stems[split]:
            t0 = ra[s]["t0"]
            both = agree(rb[s]["cfg"][cb], ra[s]["cfg"][ca], args.ventana)
            hours += t0 / 3600
            false += sum(t < t0 for t in both)
            det += any(t >= t0 for t in both)
        n = len(stems[split])
        return {"n": n, "det": det, "tasa": det / n, "falsas": false, "precision": det / (det + false) if det + false else None}

    table = []
    for ca in cfgs_a:
        for cb in cfgs_b:
            v7_events = len(agree(v7b["cfg"][cb], v7a["cfg"][ca], args.ventana))
            row = {"v6": ca, "v8": cb, "v7_falsos_h": v7_events / (v7b["duracion"] / 3600), "val": evaluate("val", ca, cb), "test": evaluate("test", ca, cb)}
            table.append(row)
    result = {"ventana_s": args.ventana, "elegidas": {}}
    for budget in args.max_v7:
        ok = [r for r in table if r["v7_falsos_h"] <= budget]
        best = max(ok, key=lambda r: (r["val"]["tasa"], r["val"]["precision"] or 0)) if ok else None
        result["elegidas"][str(budget)] = best
        if best:
            t = best["test"]
            print(f"Presupuesto V7 <= {budget}/h -> v6 {best['v6']} + v8 {best['v8']}: validación {best['val']['det']}/{best['val']['n']} = {best['val']['tasa']:.2f}; "
                  f"PRUEBA {t['det']}/{t['n']} = {t['tasa']:.2f}, precisión {t['precision']:.2f}, V7 {best['v7_falsos_h']:.1f}/h")
    print("\nFrontera en prueba (mejor detección con cada tope de falsos por hora en V7, sin elegir en validación):")
    for budget in (1.8, 3.6, 5.3, 7.1, 10.6, 15.9):
        ok = [r for r in table if r["v7_falsos_h"] <= budget]
        b = max(ok, key=lambda r: (r["test"]["tasa"], r["test"]["precision"] or 0)) if ok else None
        if b:
            t = b["test"]
            print(f"  V7 <= {budget}/h: {t['det']}/{t['n']} = {t['tasa']:.2f}, precisión {t['precision']:.2f} (v6 {b['v6']} + v8 {b['v8']}, V7 {b['v7_falsos_h']:.1f}/h)")
    result["tabla"] = table
    if args.out:
        args.out.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
