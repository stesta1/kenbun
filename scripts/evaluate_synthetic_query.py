"""Mide la exactitud del consultor sobre el conjunto sintético (apartado 5.6 de la tesis).

Uso:
    python scripts/evaluate_synthetic_query.py --url http://127.0.0.1:8001 --out ../reentreno/medicion/consulta_sintetica/resultados.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
DIR_POR_DEFECTO = ROOT.parent / "reentreno" / "medicion" / "consulta_sintetica"
TIPOS = ("contenido", "restringida")


def recall(relevantes: set[str], devueltos: list[str], normalizado: bool) -> float:
    aciertos = len(relevantes & set(devueltos[:5]))
    return aciertos / (min(5, len(relevantes)) if normalizado else len(relevantes))


def media(valores: list[float]) -> float:
    return round(statistics.mean(valores), 3) if valores else 0.0


def preguntar(client: httpx.Client, pregunta: dict) -> tuple[float, dict | None, str | None]:
    cuerpo = {"query": pregunta["pregunta"]}
    for destino, origen in (("camera_id", "camera_id"), ("start_ts", "desde"), ("end_ts", "hasta")):
        if pregunta.get(origen):
            cuerpo[destino] = pregunta[origen]
    t0 = time.perf_counter()
    try:
        respuesta = client.post("/chat", json=cuerpo)
        respuesta.raise_for_status()
        return time.perf_counter() - t0, respuesta.json(), None
    except Exception as exc:  # noqa: BLE001
        return time.perf_counter() - t0, None, repr(exc)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8001")
    parser.add_argument("--db", default="monitoreo_sintetico")
    parser.add_argument("--preguntas", type=Path, default=DIR_POR_DEFECTO / "preguntas.json")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--out", type=Path, default=DIR_POR_DEFECTO / "resultados.json")
    args = parser.parse_args()

    from monitoreo_urbano.common.config import settings
    from urllib.parse import urlsplit, urlunsplit

    settings.database_url = urlunsplit(urlsplit(settings.database_url)._replace(path="/" + args.db))
    os.environ["MU_DATABASE_URL"] = settings.database_url
    from monitoreo_urbano.event_store.hybrid_search import EventStore

    preguntas = json.loads(args.preguntas.read_text(encoding="utf-8"))["preguntas"]
    store = EventStore()

    ranking: dict[str, list[str]] = {}
    for p in preguntas:
        resultados = store.hybrid_search(
            p["pregunta"], camera_id=p.get("camera_id") or None,
            start_ts=datetime.fromisoformat(p["desde"]) if p.get("desde") else None,
            end_ts=datetime.fromisoformat(p["hasta"]) if p.get("hasta") else None,
        )
        ranking[p["id"]] = [r.event.event_id for r in resultados]

    filas: list[dict] = []
    with httpx.Client(base_url=args.url, headers={"x-api-key": settings.api_key}, timeout=300) as client:
        preguntar(client, {"pregunta": "¿Hubo algún choque?"})
        for rep in range(1, args.reps + 1):
            for p in preguntas:
                segundos, datos, error = preguntar(client, p)
                filas.append({
                    "id": p["id"], "tipo": p["tipo"], "rep": rep, "latencia_s": round(segundos, 2), "error": error,
                    "respuesta": datos["answer"] if datos else None,
                    "eventos": [e["event_id"] for e in datos["events"]] if datos else [],
                })
                print(p["id"], rep, round(segundos, 1), error or "", flush=True)
    store.close()

    por_id = {p["id"]: p for p in preguntas}
    resumen: dict = {"por_repeticion": {}, "busqueda_sola": {}, "por_pregunta": {}}
    for rep in range(1, args.reps + 1):
        filas_rep = [f for f in filas if f["rep"] == rep]
        fila = {}
        for tipo in TIPOS:
            fila[f"recall5_{tipo}"] = media([recall(set(por_id[f["id"]]["relevantes"]), f["eventos"], False) for f in filas_rep if f["tipo"] == tipo])
        fila["recall5_total"] = media([recall(set(por_id[f["id"]]["relevantes"]), f["eventos"], False) for f in filas_rep if f["tipo"] in TIPOS])
        con_respuesta = [f for f in filas_rep if f["tipo"] in TIPOS]
        fila["precision_total"] = media([len(set(por_id[f["id"]]["relevantes"]) & set(f["eventos"])) / len(f["eventos"]) if f["eventos"] else 0.0 for f in con_respuesta])
        fila["eventos_citados_medio"] = media([float(len(f["eventos"])) for f in con_respuesta])
        ausentes = [f for f in filas_rep if f["tipo"] == "sin_evidencia"]
        fila["ausencia_correcta"] = f"{sum(1 for f in ausentes if not f['eventos'])}/{len(ausentes)}"
        resumen["por_repeticion"][rep] = fila
    for tipo in TIPOS:
        reales = [p for p in preguntas if p["tipo"] == tipo]
        resumen["busqueda_sola"][f"recall5_{tipo}"] = media([recall(set(p["relevantes"]), ranking[p["id"]], False) for p in reales])
    todas = [p for p in preguntas if p["tipo"] in TIPOS]
    resumen["busqueda_sola"]["recall5_total"] = media([recall(set(p["relevantes"]), ranking[p["id"]], False) for p in todas])
    resumen["busqueda_sola"]["recall5_total_normalizado"] = media([recall(set(p["relevantes"]), ranking[p["id"]], True) for p in todas])
    for p in todas:
        vals = [recall(set(p["relevantes"]), f["eventos"], False) for f in filas if f["id"] == p["id"]]
        resumen["por_pregunta"][p["id"]] = {"relevantes": len(p["relevantes"]), "recall5_chat_medio": round(statistics.mean(vals), 2),
                                            "recall5_busqueda": round(recall(set(p["relevantes"]), ranking[p["id"]], False), 2)}
    latencias = [f["latencia_s"] for f in filas if not f["error"]]
    resumen["latencia_s"] = {"mediana": round(statistics.median(latencias), 2), "p95": round(sorted(latencias)[max(0, -(-95 * len(latencias) // 100) - 1)], 2), "maximo": max(latencias)}
    resumen["errores"] = sum(1 for f in filas if f["error"])

    print(json.dumps(resumen, indent=1, ensure_ascii=False))
    args.out.write_text(json.dumps({
        "fecha": datetime.now().isoformat(timespec="seconds"), "repeticiones": args.reps, "resumen": resumen,
        "ranking_busqueda": ranking, "filas": filas,
    }, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
