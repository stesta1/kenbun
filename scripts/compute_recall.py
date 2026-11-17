"""Calcula Recall@5 y la respuesta correcta de las preguntas sin evidencia, a partir de una corrida de `measure_query.py`.

Uso:
    python scripts/compute_recall.py --consulta ../reentreno/medicion/consulta_choques_1.json --preguntas ../reentreno/medicion/preguntas_choques.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--consulta", type=Path, required=True)
    parser.add_argument("--preguntas", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    questions = {q["id"]: q for q in json.loads(args.preguntas.read_text(encoding="utf-8"))["preguntas"]}
    rows = json.loads(args.consulta.read_text(encoding="utf-8"))["filas"]
    result: dict = {"por_repeticion": {}, "por_pregunta": {}}
    for rep in sorted({r["rep"] for r in rows}):
        by_type: dict[str, list[float]] = {"contenido": [], "restringida": []}
        absent = 0
        for r in (r for r in rows if r["rep"] == rep):
            q = questions[r["id"]]
            if q["tipo"] == "sin_evidencia":
                absent += 0 if r["eventos"] else 1
                continue
            rel = set(q["relevantes"])
            hits = len(set(r["eventos"][:5]) & rel)
            by_type[q["tipo"]].append(hits / min(5, len(rel)) if rel else 0.0)
        allv = by_type["contenido"] + by_type["restringida"]
        result["por_repeticion"][rep] = {
            "recall5_total": round(statistics.mean(allv), 3), "recall5_contenido": round(statistics.mean(by_type["contenido"]), 3),
            "recall5_restringida": round(statistics.mean(by_type["restringida"]), 3), "ausencia_correcta": f"{absent}/10",
        }
    for qid, q in questions.items():
        if q["tipo"] == "sin_evidencia":
            continue
        vals = []
        for r in (r for r in rows if r["id"] == qid):
            rel = set(q["relevantes"])
            vals.append(len(set(r["eventos"][:5]) & rel) / min(5, len(rel)) if rel else 0.0)
        result["por_pregunta"][qid] = {"relevantes": len(q["relevantes"]), "recall5_medio": round(statistics.mean(vals), 2)}
    print(json.dumps(result["por_repeticion"], indent=1, ensure_ascii=False))
    if args.out:
        args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
