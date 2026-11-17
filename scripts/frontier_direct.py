"""Frontera de detección de la regla directa por número de videos con alguna alerta falsa previa al impacto (escenas independientes) y falsos en V7.

Uso: python scripts/frontier_direct.py nombre=ruta.json [nombre=ruta.json ...]
Detección con tolerancia de 3 s (alerta temprana). Dominio, prueba. 'Falsa previa' = alerta más de 3 s antes del impacto anotado.
"""
import json
import sys

TOL = 3.0


def main() -> None:
    for spec in sys.argv[1:]:
        name, path = spec.split("=", 1)
        d = json.load(open(path, encoding="utf-8"))
        cfgs = list(d["rows"][0]["cfg"])
        v7 = next(r for r in d["rows"] if r["stem"] == "V7 uncut")
        rows = []
        for c in cfgs:
            n = det = vids = 0
            for r in d["rows"]:
                if r["stem"] == "V7 uncut" or d["split"][r["stem"]] != "test":
                    continue
                t, t0 = r["cfg"][c], r["t0"]
                n += 1
                det += any(x >= t0 - TOL for x in t)
                vids += any(x < t0 - TOL for x in t)
            rows.append((c, det, vids, len(v7["cfg"][c]) / (v7["duracion"] / 3600), n))
        print(f"== {name} (prueba, dominio, n={rows[0][4]})")
        for budget in (0, 1, 2, 4):
            ok = [r for r in rows if r[2] <= budget]
            b = max(ok, key=lambda r: (r[1], -r[3])) if ok else None
            print(f"  videos con falsa previa <= {budget}: " + (f"{b[1]}/{b[4]} = {b[1]/b[4]:.2f} [{b[0]}], V7 {b[3]:.1f}/h" if b else "ninguna"))
        for vb in (3.6, 7.1, 10.6):
            ok = [r for r in rows if r[3] <= vb and r[2] <= 2]
            b = max(ok, key=lambda r: (r[1], -r[3])) if ok else None
            print(f"  V7 <= {vb}/h y <= 2 videos con falsa previa: " + (f"{b[1]}/{b[4]} = {b[1]/b[4]:.2f} [{b[0]}]" if b else "ninguna"))


if __name__ == "__main__":
    main()
