"""Resume el barrido directo de la regla de choque (`sweep_accident_direct.py`): detección y falsos por configuración, en dominio.

Uso:
    python scripts/analyze_sweep_direct.py --a v6=../reentreno/barrido_directo_v6.json --b v8=../reentreno/barrido_directo_v8.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

TOL = 3.0


def load(spec: str):
    name, path = spec.split("=", 1)
    return name, json.loads(Path(path).read_text(encoding="utf-8"))


def stats(data: dict, split: str, cfg: str) -> dict:
    n = strict = tol = false = 0
    for r in data["rows"]:
        if r["stem"] == "V7 uncut" or data["split"][r["stem"]] != split:
            continue
        times, t0 = r["cfg"][cfg], r["t0"]
        n += 1
        strict += any(t >= t0 for t in times)
        tol += any(t >= t0 - TOL for t in times)
        false += sum(t < t0 - TOL for t in times)
    return {"n": n, "estricta": strict, "tolerancia": tol, "falsas": false}


def v7_rate(data: dict, cfg: str) -> float:
    v7 = next(r for r in data["rows"] if r["stem"] == "V7 uncut")
    return len(v7["cfg"][cfg]) / (v7["duracion"] / 3600)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True)
    parser.add_argument("--b", default=None)
    parser.add_argument("--max-v7", type=float, nargs="+", default=[3.6, 7.1, 10.6])
    parser.add_argument("--max-previos", type=float, default=0.15, help="Falsas previas por video admitidas en validación")
    args = parser.parse_args()
    for spec in [s for s in (args.a, args.b) if s]:
        name, data = load(spec)
        cfgs = list(data["rows"][0]["cfg"])
        table = {c: {"val": stats(data, "val", c), "test": stats(data, "test", c), "v7": v7_rate(data, c)} for c in cfgs}
        print(f"\n=== {name}: regla directa (confianza | cuadros | ventana s) ===")
        for budget in args.max_v7:
            ok = [c for c in cfgs if table[c]["v7"] <= budget and table[c]["val"]["falsas"] / max(table[c]["val"]["n"], 1) <= args.max_previos]
            if not ok:
                print(f"  V7 <= {budget}/h: ninguna configuración")
                continue
            best = max(ok, key=lambda c: (table[c]["val"]["tolerancia"], -table[c]["v7"]))
            v, t = table[best]["val"], table[best]["test"]
            print(f"  V7 <= {budget}/h -> {best}: validación {v['tolerancia']}/{v['n']} ({v['tolerancia']/v['n']:.2f}); "
                  f"PRUEBA {t['tolerancia']}/{t['n']} = {t['tolerancia']/t['n']:.2f} (estricta {t['estricta']}/{t['n']}), "
                  f"falsas previas {t['falsas']}, V7 {table[best]['v7']:.1f}/h")
        print("  frontera en prueba (mejor con tolerancia por tope de V7, sin elegir en validación):")
        for budget in (1.8, 3.6, 5.3, 7.1, 10.6, 15.9):
            ok = [c for c in cfgs if table[c]["v7"] <= budget]
            b = max(ok, key=lambda c: (table[c]["test"]["tolerancia"], -table[c]["test"]["falsas"])) if ok else None
            if b:
                t = table[b]["test"]
                print(f"    V7 <= {budget}/h: {t['tolerancia']}/{t['n']} = {t['tolerancia']/t['n']:.2f} (estricta {t['estricta']}), falsas previas {t['falsas']}  [{b}, V7 {table[b]['v7']:.1f}/h]")


if __name__ == "__main__":
    main()
