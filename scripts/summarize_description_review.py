"""Resume la revisión a ciegas de descripciones: veredictos humanos por variante y métricas automáticas."""
from __future__ import annotations

import argparse
import itertools
import json
import math
import re
import statistics
from pathlib import Path

CRITERIA = ["fiel", "vehiculos", "tipo_choque", "consecuencias", "util"]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * p)


def words(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


def mean_similarity(texts: list[str]) -> float:
    sets = [words(t) for t in texts]
    pairs = [len(a & b) / len(a | b) for a, b in itertools.combinations(sets, 2) if a | b]
    return statistics.mean(pairs) if pairs else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+", type=Path, required=True)
    parser.add_argument("--clave", type=Path)
    parser.add_argument("--veredictos", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    variants = {}
    for path in args.variants:
        data = json.loads(path.read_text(encoding="utf-8"))
        variants[data["etiqueta"]] = data
    summary: dict[str, dict] = {}

    print("## Métricas automáticas\n")
    print("| Variante | Descripciones | Segundos (media) | Caracteres (media) | «de frente» | Parecido entre descripciones |")
    print("|---|---|---|---|---|---|")
    for label, data in variants.items():
        rows = data["resultados"]
        texts = [r["descripcion"] for r in rows]
        auto = {
            "n": len(rows),
            "segundos": round(statistics.mean(r["segundos"] for r in rows), 2),
            "caracteres": round(statistics.mean(len(t) for t in texts)),
            "de_frente": sum("de frente" in t.lower() for t in texts),
            "parecido": round(mean_similarity(texts), 3),
        }
        summary[label] = {"automaticas": auto}
        print(f"| {label} | {auto['n']} | {auto['segundos']} | {auto['caracteres']} | {auto['de_frente']} de {auto['n']} | {auto['parecido']} |")

    if args.clave and args.veredictos:
        clave = json.loads(args.clave.read_text(encoding="utf-8"))
        verdicts = json.loads(args.veredictos.read_text(encoding="utf-8"))
        marks: dict[str, dict[str, dict]] = {label: {} for label in variants}
        for key, mark in verdicts.items():
            if not mark.get("revisada"):
                continue
            event_id, pos = key.rsplit("_", 1)
            marks[clave[event_id][pos]][event_id] = mark
        print("\n## Veredictos humanos (revisión a ciegas)\n")
        print("| Variante | Revisadas | " + " | ".join(CRITERIA) + " |")
        print("|---|---|" + "---|" * len(CRITERIA))
        for label in variants:
            rows = marks[label]
            n = len(rows)
            cells, human = [], {}
            for c in CRITERIA:
                k = sum(1 for m in rows.values() if m.get(c))
                lo, hi = wilson(k, n)
                human[c] = {"k": k, "n": n, "wilson": [round(lo, 3), round(hi, 3)]}
                cells.append(f"{k}/{n} ({lo:.2f}–{hi:.2f})")
            summary[label]["humanos"] = human
            print(f"| {label} | {n} | " + " | ".join(cells) + " |")
        base = next(iter(variants))
        print(f"\n## Fidelidad pareada contra «{base}» (McNemar exacto)\n")
        print("| Variante | Fiel solo en la variante | Fiel solo en la base | p |")
        print("|---|---|---|---|")
        for label in list(variants)[1:]:
            ids = set(marks[base]) & set(marks[label])
            b = sum(1 for i in ids if marks[label][i].get("fiel") and not marks[base][i].get("fiel"))
            c = sum(1 for i in ids if marks[base][i].get("fiel") and not marks[label][i].get("fiel"))
            summary[label]["mcnemar"] = {"solo_variante": b, "solo_base": c, "p": round(mcnemar_exact(b, c), 4)}
            print(f"| {label} | {b} | {c} | {mcnemar_exact(b, c):.3f} |")
    if args.out:
        args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
