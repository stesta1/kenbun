"""Demora del sistema: desde el disparo (alta del evento en la base) hasta la recepción de la alerta en el cliente.

Uso:
    python scripts/analyze_latency_db.py --json ../reentreno/latencia_alerta_1_v4.json --db monitoreo_latencia
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def events(db: str) -> list[tuple[str, str, float]]:
    out = subprocess.run(
        ["docker", "exec", "monitoreo_urbano_app-postgres-1", "psql", "-U", "monitoreo", "-d", db, "-At", "-F", "|", "-c",
         "select camera_id, trigger_reason, extract(epoch from created_at at time zone 'utc') from events order by created_at"],
        capture_output=True, text=True, check=True).stdout
    return [(c, r, float(t)) for c, r, t in (line.split("|") for line in out.splitlines() if line)]


def p95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--db", default="monitoreo_latencia")
    args = parser.parse_args()
    data = json.loads(args.json.read_text(encoding="utf-8"))
    evs = events(args.db)
    sistema = []
    for row in data["filas"]:
        if row["t_recv_epoch"] is None:
            row["latencia_sistema_s"] = None
            continue
        cands = [t for c, r, t in evs if c == row["camara"] and r == row["motivo"] and row["t_launch_epoch"] <= t <= row["t_recv_epoch"] + 0.5]
        row["latencia_sistema_s"] = round(row["t_recv_epoch"] - min(cands), 2) if cands else None
        if row["latencia_sistema_s"] is not None:
            sistema.append(row["latencia_sistema_s"])
    n = len(data["filas"])
    data["resumen"]["desde_disparo_sistema"] = {
        "n": n, "con_alerta": len(sistema), "mediana_s": round(statistics.median(sistema), 2) if sistema else None,
        "p95_s": round(p95(sistema), 2) if sistema else None, "maximo_s": max(sistema) if sistema else None, "minimo_s": min(sistema) if sistema else None,
    }
    args.json.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(data["resumen"], indent=1, ensure_ascii=False))
    print([(r["video"], r["latencia_s"], r["latencia_sistema_s"]) for r in data["filas"]])


if __name__ == "__main__":
    main()
