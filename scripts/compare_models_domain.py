"""Compara dos detectores por evento, dentro y fuera del dominio del requisito de cámara.

Uso:
    python scripts/compare_models_domain.py --a v6=../reentreno/barrido_reglas_por_video.json --b v8=../reentreno/barrido_v8_por_video.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import evaluate_classifier_events as ece  # noqa: E402


def alto(m: dict) -> float:
    return (float(m["y2"]) - float(m["y1"])) * int(m["height"]) * 960 / int(m["width"])


def in_domain(m: dict) -> bool:
    return alto(m) >= 40 and int(m["height"]) >= 480 and m["quality"] != "Very_Poor" and m["day_time"] == "day"


def load(spec: str) -> tuple[str, dict]:
    name, path = spec.split("=", 1)
    return name, json.loads(Path(path).read_text(encoding="utf-8"))


def stats(data: dict, meta: dict, split: str, cfg: str, domain: bool | None) -> dict:
    det = total = false = 0
    hours = 0.0
    for r in data["rows"]:
        if r["stem"] == "V7 uncut" or data["split"][r["stem"]] != split:
            continue
        m = meta[r["stem"]]
        if domain is True and not in_domain(m):
            continue
        times, t0 = r["cfg"][cfg], r["t0"]
        total += 1
        hours += t0 / 3600
        false += sum(t < t0 for t in times)
        det += any(t >= t0 for t in times)
    return {"n": total, "det": det, "tasa": det / total if total else 0.0, "falsas": false,
            "falsos_h": false / hours if hours else 0.0, "precision": det / (det + false) if det + false else None}


def v7_rate(data: dict, cfg: str) -> float:
    v7 = next(r for r in data["rows"] if r["stem"] == "V7 uncut")
    return len(v7["cfg"][cfg]) / (v7["duracion"] / 3600)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True)
    parser.add_argument("--b", required=True)
    parser.add_argument("--max-falsos-v7", type=float, default=3.5, help="Falsos por hora admitidos en V7 al elegir la configuración")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    meta = {Path(r["path"]).stem: r for r in csv.DictReader(open(ece.SOURCE / "metadata-real.csv", encoding="utf-8"))}
    models = [load(args.a), load(args.b)]
    cfgs = list(models[0][1]["rows"][0]["cfg"].keys())
    result: dict = {"configuraciones": {}, "elegidas": {}}
    for name, data in models:
        print(f"\n=== {name}: configuración -> dominio (prueba) detectados / precisión / falsos por hora (V7) ===")
        table = {}
        for cfg in cfgs:
            val = stats(data, meta, "val", cfg, True)
            test = stats(data, meta, "test", cfg, True)
            full = stats(data, meta, "test", cfg, None)
            table[cfg] = {"val_dominio": val, "test_dominio": test, "test_completo": full, "v7_falsos_h": v7_rate(data, cfg)}
            p = test["precision"]
            print(f"  {cfg:<9} dominio {test['det']:>2}/{test['n']} ({test['tasa']:.2f})  precisión {('%.2f' % p) if p is not None else '-':>4}  "
                  f"V7 {table[cfg]['v7_falsos_h']:>5.1f}/h   completo {full['det']}/{full['n']} ({full['tasa']:.2f})")
        ok = [c for c in cfgs if table[c]["v7_falsos_h"] <= args.max_falsos_v7]
        pick = max(ok, key=lambda c: (table[c]["val_dominio"]["tasa"], -table[c]["v7_falsos_h"])) if ok else None
        result["configuraciones"][name] = table
        result["elegidas"][name] = pick
        if pick:
            t = table[pick]["test_dominio"]
            print(f"  -> elegida en validación (V7 <= {args.max_falsos_v7}/h): {pick}; prueba en dominio {t['det']}/{t['n']} = {t['tasa']:.2f}, "
                  f"precisión {t['precision']}, V7 {table[pick]['v7_falsos_h']:.1f}/h")
        else:
            print(f"  -> ninguna configuración cumple V7 <= {args.max_falsos_v7}/h")
    if args.out:
        args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
