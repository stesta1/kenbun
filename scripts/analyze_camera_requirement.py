"""Relaciona la detección por evento de las reglas con propiedades observables de la cámara, para fijar un requisito de instalación.

Uso:
    python scripts/analyze_camera_requirement.py --rows ../reentreno/barrido_reglas_por_video.json --cfg 0.6\|0.65 --min-alto 40
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

BINS = [(0, 30), (30, 40), (40, 60), (60, 90), (90, 10_000)]


def alto_aparente(r: dict) -> float:
    return (float(r["y2"]) - float(r["y1"])) * int(r["height"]) * 960 / int(r["width"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--cfg", default="0.6|0.65")
    parser.add_argument("--min-alto", type=float, default=0.0)
    parser.add_argument("--escenas", default=None, help="Tipos de escena separados por coma (por defecto, todas)")
    args = parser.parse_args()
    data = json.loads(args.rows.read_text(encoding="utf-8"))
    split_of = data["split"]
    meta = {Path(r["path"]).stem: r for r in csv.DictReader(open(ece.SOURCE / "metadata-real.csv", encoding="utf-8"))}
    escenas = set(args.escenas.split(",")) if args.escenas else None
    v7 = next((r for r in data["rows"] if r["stem"] == "V7 uncut"), None)

    def stats(split: str, keep) -> dict:
        det = total = false = 0
        hours = 0.0
        for r in data["rows"]:
            if r["stem"] == "V7 uncut" or split_of[r["stem"]] not in split:
                continue
            m = meta[r["stem"]]
            if not keep(m):
                continue
            times, t0 = r["cfg"][args.cfg], r["t0"]
            total += 1
            hours += t0 / 3600
            false += sum(t < t0 for t in times)
            det += any(t >= t0 for t in times)
        return {"videos": total, "detectados": det, "tasa": round(det / total, 3) if total else None,
                "falsos_por_hora": round(false / hours, 1) if hours else None}

    print(f"Configuración (confianza|persistencia) {args.cfg}\n")
    print("Detección por tamaño aparente del choque (alto del recuadro a 960 px de ancho), validación + prueba:")
    for lo, hi in BINS:
        s = stats(("val", "test"), lambda m, lo=lo, hi=hi: lo <= alto_aparente(m) < hi)
        print(f"  {lo:>3}-{min(hi, 999):<3} px: {s['detectados']:>3}/{s['videos']:<3} = {s['tasa']}")
    print("\nDetección por tipo de escena, validación + prueba:")
    kinds = sorted({m["scene_layout"] for m in meta.values()})
    for k in kinds:
        s = stats(("val", "test"), lambda m, k=k: m["scene_layout"] == k)
        if s["videos"] >= 5:
            print(f"  {k:<30} {s['detectados']:>3}/{s['videos']:<3} = {s['tasa']}")

    def keep(m: dict) -> bool:
        return alto_aparente(m) >= args.min_alto and (escenas is None or m["scene_layout"] in escenas)

    print(f"\nRequisito: alto aparente >= {args.min_alto:g} px" + (f", escenas {sorted(escenas)}" if escenas else ""))
    for split, name in (("val", "validación"), ("test", "prueba")):
        s = stats((split,), keep)
        print(f"  {name}: {s['detectados']}/{s['videos']} = {s['tasa']}   falsos por hora previos al impacto: {s['falsos_por_hora']}")
    if v7:
        print(f"  V7 (tránsito normal): {len(v7['cfg'][args.cfg]) / (v7['duracion'] / 3600):.1f} falsos por hora")


if __name__ == "__main__":
    main()
