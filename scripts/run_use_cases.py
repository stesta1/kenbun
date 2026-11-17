"""Ejecuta los siete casos de uso contra el sistema en marcha y registra entrada, esperado, observado y resultado.

Uso:
    python scripts/run_use_cases.py --out ../reentreno/medicion/casos_de_uso_flujo_vigente.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import httpx
import websockets

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from monitoreo_urbano.common.auth import with_stream_credentials  # noqa: E402
from monitoreo_urbano.common.config import settings  # noqa: E402
from monitoreo_urbano.common.db import get_connection  # noqa: E402
from monitoreo_urbano.common.repository import get_by_id, insert_described_event  # noqa: E402

CAMERA = "cinco-n03"
VIDEO = ROOT.parent / "test_ui" / "videos" / "choques_niveles" / "N03.mp4"
RESULTS: list[dict] = []


def record(cu: str, caso: str, entrada: str, esperado: str, observado: str, ok: bool) -> None:
    RESULTS.append({"cu": cu, "caso": caso, "entrada": entrada, "esperado": esperado, "observado": observado, "resultado": "Cumple" if ok else "Falla"})
    print(cu, caso, "->", "Cumple" if ok else "Falla", "|", observado, flush=True)


def naive_utc(ts: datetime) -> datetime:
    return ts.astimezone(timezone.utc).replace(tzinfo=None) if ts.tzinfo else ts


def status_of(conn, event_id: str) -> tuple[str, str]:
    row = conn.execute("SELECT status, review_status FROM events WHERE event_id = %s", (event_id,)).fetchone()
    return row["status"], row["review_status"]


def wait_status(conn, event_id: str, wanted: str, timeout: float = 180) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        if status_of(conn, event_id)[0] == wanted:
            return True
        time.sleep(1)
    return False


async def listen_alerts(duration: float, bucket: list[dict]) -> None:
    uri = f"ws://127.0.0.1:{settings.api_port}/alerts"
    async with websockets.connect(uri, additional_headers={"x-api-key": settings.api_key}) as ws:
        end = time.time() + duration
        while time.time() < end:
            try:
                bucket.append(json.loads(await asyncio.wait_for(ws.recv(), timeout=0.5)))
            except asyncio.TimeoutError:
                continue


def publish(lead: float, tail: float) -> None:
    w, h, fps = 960, 720, 15
    fit = f"scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={fps},setpts=PTS-STARTPTS"

    def black(seconds: float) -> list[str]:
        return ["-re", "-f", "lavfi", "-t", str(seconds), "-i", f"color=c=black:s={w}x{h}:r={fps}"]

    graph = f"[0:v]setsar=1,setpts=PTS-STARTPTS[a];[1:v]{fit}[b];[2:v]setsar=1,setpts=PTS-STARTPTS[c];[a][b][c]concat=n=3:v=1:a=0[o]"
    cmd = [
        "ffmpeg", "-loglevel", "error", *black(lead), "-re", "-i", str(VIDEO), *black(tail), "-filter_complex", graph, "-map", "[o]",
        "-c:v", "libx264", "-preset", "veryfast", "-tune", "zerolatency", "-pix_fmt", "yuv420p", "-f", "rtsp", "-rtsp_transport", "tcp",
        with_stream_credentials(f"rtsp://localhost:8554/{CAMERA}"),
    ]
    subprocess.run(cmd, check=True)


def clone_event(conn, source_id: str, reason: str, description: str | None) -> str:
    event = get_by_id(conn, source_id)
    event = event.model_copy(update={
        "event_id": str(uuid.uuid4()), "trigger_reason": reason, "description_nl": description, "vehicles": [],
        "review_status": event.review_status.__class__("pending"), "created_at": datetime.utcnow(),
    })
    insert_described_event(conn, event)
    if description is None:
        conn.execute("UPDATE events SET status = 'processing', description_nl = NULL, described_at = NULL WHERE event_id = %s", (event.event_id,))
        conn.commit()
    return event.event_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    conn = get_connection()
    api = httpx.Client(base_url=f"http://127.0.0.1:{settings.api_port}", headers={"x-api-key": settings.api_key}, timeout=180)
    since = datetime.utcnow() - timedelta(seconds=5)

    alerts: list[dict] = []
    thread = threading.Thread(target=lambda: asyncio.run(listen_alerts(110, alerts)), daemon=True)
    thread.start()
    seen: list[str] = []
    stop = threading.Event()

    def poll() -> None:
        c = get_connection()
        last = None
        while not stop.is_set():
            row = c.execute("SELECT status FROM events WHERE camera_id = %s AND created_at >= %s ORDER BY created_at DESC LIMIT 1", (CAMERA, since)).fetchone()
            if row and row["status"] != last:
                last = row["status"]
                seen.append(last)
            time.sleep(0.25)

    poller = threading.Thread(target=poll, daemon=True)
    poller.start()
    publish(lead=10, tail=30)
    rows = conn.execute("SELECT event_id, trigger_reason FROM events WHERE camera_id = %s AND created_at >= %s ORDER BY created_at", (CAMERA, since)).fetchall()
    alert_ids = [r["event_id"] for r in rows if r["trigger_reason"] == "accident"]
    new = alert_ids[-1] if alert_ids else (rows[-1]["event_id"] if rows else None)
    if new:
        wait_status(conn, new, "described", 240)
    time.sleep(6)
    stop.set()
    thread.join(5)
    mine = [a for a in alerts if a.get("camera_id") == CAMERA and a.get("event_id") == new]
    final = status_of(conn, new) if new else ("-", "-")
    chat_ids = [e["event_id"] for e in api.post("/chat", json={"query": "¿Hubo un choque?"}).json()["events"]]
    pending_ids = [e["event_id"] for e in api.get("/events/pending-review").json()]
    ok = len(mine) == 1 and bool(new) and final == ("described", "pending") and new not in chat_ids and new in pending_ids
    record("CU1", "Flujo principal", f"Video N03 (choque) por la cámara {CAMERA}",
           "Una alerta de choque por /alerts; el evento recorre sus estados, queda pendiente de revisión y no es consultable",
           f"{len(mine)} alerta(s), motivo {mine[0].get('trigger_reason') if mine else '-'}; estados: {' → '.join(seen)}; final {final}; en la cola: {new in pending_ids}; consultable: {new in chat_ids}", ok)

    source = new or conn.execute("SELECT event_id FROM events ORDER BY created_at DESC LIMIT 1").fetchone()["event_id"]
    weak_alerts: list[dict] = []
    weak = [clone_event(conn, source, "posible_altercado", f"Dos personas se mueven de forma agitada junto a un vehículo (evento de prueba {i}).") for i in (1, 2)]
    asyncio.run(listen_alerts(4, weak_alerts))
    time.sleep(4)
    stays = all(status_of(conn, w) == ("described", "pending") for w in weak)
    emitted = [a for a in weak_alerts if a.get("event_id") in weak]
    record("CU1", "Flujo alternativo (señal débil, evento de prueba)", "Evento de prueba con motivo posible_altercado",
           "El evento se registra sin emitir alerta y queda pendiente", f"{len(emitted)} alerta(s); estados {[status_of(conn, w) for w in weak]}", stays and not emitted)

    r_noveh = api.post(f"/events/{new}/details", json={"description_nl": "Choque entre dos vehículos en un cruce.", "vehicles": []}) if new else None
    record("CU4", "Confirmar un choque sin vehículos", "Confirmar la alerta sin cargar vehículos", "Rechazo (422): un choque necesita al menos un vehículo",
           f"HTTP {r_noveh.status_code if r_noveh is not None else '-'}", r_noveh is not None and r_noveh.status_code == 422)
    r_old = api.post(f"/events/{new}/review", json={"review_status": "confirmed"}) if new else None
    record("CU4", "Confirmar por la vía directa", "POST /events/{id}/review con «confirmed»", "Rechazo (400): confirmar exige la revisión completa",
           f"HTTP {r_old.status_code if r_old is not None else '-'}", r_old is not None and r_old.status_code == 400)
    text_ok = "Choque entre un auto blanco y una camioneta gris en un cruce, confirmado por el operador."
    r_ok = api.post(f"/events/{new}/details", json={"description_nl": text_ok, "vehicles": [{"vehicle_type": "auto", "color": "blanco"}, {"vehicle_type": "camioneta", "color": "gris"}]}) if new else None
    indexed = bool(new) and wait_status(conn, new, "indexed", 120)
    record("CU4", "Confirmar la alerta con descripción y vehículos", "Confirmar la alerta con el texto y dos vehículos", "El evento queda confirmado e indexado",
           f"HTTP {r_ok.status_code if r_ok is not None else '-'}; estado {status_of(conn, new) if new else '-'}", bool(r_ok is not None and r_ok.status_code == 200 and indexed))
    alert_like = clone_event(conn, source, "accident", "Choque entre dos vehículos (alerta de prueba que se descarta).")
    r_dis = api.post(f"/events/{alert_like}/review", json={"review_status": "dismissed"})
    time.sleep(4)
    listed = [e["event_id"] for e in api.get("/events/pending-review").json()]
    found = [e["event_id"] for e in api.post("/chat", json={"query": "alerta de prueba que se descarta"}).json()["events"]]
    record("CU4", "Descartar una alerta", "Descartar una alerta de prueba", "No vuelve a la cola, no se indexa y no es consultable",
           f"HTTP {r_dis.status_code}; en la cola: {alert_like in listed}; estado {status_of(conn, alert_like)}; consultable: {alert_like in found}",
           r_dis.status_code == 200 and alert_like not in listed and status_of(conn, alert_like)[0] == "described" and alert_like not in found)

    resp = api.post("/chat", json={"query": "¿Hubo un choque entre un auto blanco y una camioneta gris?"}).json()
    ids = [e["event_id"] for e in resp["events"]]
    record("CU2", "Flujo principal", "«¿Hubo un choque entre un auto blanco y una camioneta gris?»", "Respuesta que cita el evento confirmado",
           f"{len(ids)} evento(s) citado(s); confirmado incluido: {new in ids}; respuesta: {resp['answer'][:110]!r}", bool(new) and new in ids)
    resp2 = api.post("/chat", json={"query": "¿Hubo una pelea entre personas?", "camera_id": "cinco-glkd"}).json()
    record("CU2", "Flujo alternativo (sin eventos)", "«¿Hubo una pelea entre personas?» en una cámara sin eventos", "Informa que no hay resultados, sin inventar",
           f"{len(resp2['events'])} evento(s); respuesta: {resp2['answer'][:110]!r}", not resp2["events"])
    start = naive_utc(get_by_id(conn, new).start_ts) if new else datetime.utcnow()
    lo, hi = start - timedelta(minutes=10), start + timedelta(minutes=10)
    resp3 = api.post("/chat", json={"query": "¿Qué incidentes hubo?", "camera_id": CAMERA, "start_ts": lo.isoformat(), "end_ts": hi.isoformat()}).json()
    cams = sorted({e["camera_id"] for e in resp3["events"]})
    stamps = [naive_utc(datetime.fromisoformat(e["start_ts"])) for e in resp3["events"]]
    inside = bool(stamps) and all(lo <= t <= hi for t in stamps)
    record("CU2", "Acotada por cámara y período (RF8)", f"«¿Qué incidentes hubo?» en {CAMERA}, ±10 min del evento",
           "Solo eventos de esa cámara y con inicio dentro del período",
           f"{len(resp3['events'])} evento(s), cámaras {cams}; inicios {[t.strftime('%H:%M:%S') for t in stamps]} dentro de "
           f"[{lo:%H:%M:%S}, {hi:%H:%M:%S}] UTC: {inside}", bool(resp3["events"]) and cams == [CAMERA] and inside)
    lo_n, hi_n = start - timedelta(hours=2), start - timedelta(hours=1)
    resp4 = api.post("/chat", json={"query": "¿Qué incidentes hubo?", "camera_id": CAMERA, "start_ts": lo_n.isoformat(), "end_ts": hi_n.isoformat()}).json()
    record("CU2", "Acotada a un período sin eventos (RF8)", f"«¿Qué incidentes hubo?» en {CAMERA}, de 2 h a 1 h antes del evento",
           "Ningún evento: el del período adyacente queda excluido",
           f"{len(resp4['events'])} evento(s); evento real incluido: {new in [e['event_id'] for e in resp4['events']]}", not resp4["events"])

    cited = next((e for e in resp["events"] if e["event_id"] == new), resp["events"][0] if resp["events"] else None)
    if cited:
        url = api.get(f"/clip-url/{cited['clip_path']}").json()["url"]
        full = httpx.get(url, timeout=120)
        decoded, fps, declared = 0, 0.0, 0
        if full.status_code == 200 and full.content:
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                tmp.write(full.content)
            cap = cv2.VideoCapture(tmp.name)
            fps, declared = cap.get(cv2.CAP_PROP_FPS) or 0.0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            while cap.read()[0]:
                decoded += 1
            cap.release()
            Path(tmp.name).unlink(missing_ok=True)
        seconds = decoded / fps if fps else 0.0
        span = (naive_utc(datetime.fromisoformat(cited["end_ts"])) - naive_utc(datetime.fromisoformat(cited["start_ts"]))).total_seconds()
        record("CU3", "Flujo principal", f"Clip del evento citado {cited['event_id'][:8]}",
               "La URL firmada entrega el clip completo y se decodifican todos sus fotogramas",
               f"HTTP {full.status_code}, {len(full.content)} bytes; {decoded} de {declared} fotogramas decodificados a {fps:.1f} fps "
               f"({seconds:.1f} s; el evento dura {span:.1f} s)",
               full.status_code == 200 and decoded > 0 and decoded >= declared - 1 and seconds >= 1.0)
    missing = api.get(f"/clip-url/{CAMERA}/no-existe.mp4").json()["url"]
    miss = httpx.get(missing, timeout=60)
    record("CU3", "Clip inexistente (limitación documentada)", "Ruta de clip que no existe", "El sistema entrega el enlace sin verificar; la reproducción falla del lado del cliente",
           f"Enlace entregado; HTTP {miss.status_code} al reproducirlo", miss.status_code >= 400)

    cams_list = api.get("/cameras").json()
    health = api.get("/cameras/health").json()
    record("CU5", "Flujo principal", "GET /cameras y /cameras/health", "Catálogo con el estado de salud de cada cámara",
           f"{len(cams_list)} cámara(s); estados {sorted({h['status'] for h in health})}", bool(cams_list) and len(health) == len(cams_list))

    target = clone_event(conn, source, "accident", "Dos autos chocan en una esquina (descripción automática de prueba).")
    original = conn.execute("SELECT description_nl FROM events WHERE event_id = %s", (target,)).fetchone()["description_nl"]
    corrected = "Choque entre un auto naranja y una camioneta verde en una rotonda, corregido por el operador."
    r6 = api.post(f"/events/{target}/details", json={"description_nl": corrected, "vehicles": [{"vehicle_type": "auto", "color": "naranja"}]})
    reindexed = wait_status(conn, target, "indexed", 120)
    row = conn.execute("SELECT description_nl, description_nl_original, description_edited FROM events WHERE event_id = %s", (target,)).fetchone()
    record("CU6", "Corrección conservando el original", "Corregir la descripción de un evento durante su revisión",
           "Texto vigente corregido, original conservado, marcado como intervenido y reindexado",
           f"HTTP {r6.status_code}; original conservado: {row['description_nl_original'] == original}; editado: {row['description_edited']}; reindexado: {reindexed}",
           r6.status_code == 200 and row["description_nl"] == corrected and row["description_nl_original"] == original and bool(row["description_edited"]) and reindexed)
    found = [e["event_id"] for e in api.post("/chat", json={"query": "auto naranja camioneta verde rotonda corregido por el operador"}).json()["events"]]
    record("CU6", "Recuperación sin filtros", "Consulta con las palabras de la corrección, sin filtros", "El evento corregido es recuperable por CU2",
           f"recuperado: {target in found}", target in found)
    nodesc = clone_event(conn, source, "accident", None)
    r2 = api.post(f"/events/{nodesc}/details", json={"description_nl": "x", "vehicles": [{"vehicle_type": "auto", "color": "rojo"}]})
    record("CU6", "Evento sin descripción generada", "Revisar un evento que todavía no tiene descripción", "Rechazo (409)", f"HTTP {r2.status_code}", r2.status_code == 409)

    listed = [e["event_id"] for e in api.get("/events/pending-review").json()]
    record("CU7", "Listado de eventos sin confirmar", "GET /events/pending-review", "Aparecen los dos eventos de prueba",
           f"{len(listed)} evento(s) listados; ambos presentes: {all(w in listed for w in weak)}", all(w in listed for w in weak))
    r7 = api.post(f"/events/{weak[0]}/details", json={"description_nl": "Dos personas discuten y se empujan junto a un vehículo, confirmado por el operador.", "vehicles": []})
    indexed7 = wait_status(conn, weak[0], "indexed", 90)
    record("CU7", "Flujo principal (confirmar)", "Confirmar el primer evento de prueba", "Pasa a indexarse", f"HTTP {r7.status_code}; indexado: {indexed7}", r7.status_code == 200 and indexed7)
    api.post(f"/events/{weak[1]}/review", json={"review_status": "dismissed"})
    time.sleep(6)
    after = [e["event_id"] for e in api.get("/events/pending-review").json()]
    st = status_of(conn, weak[1])
    record("CU7", "Flujo alternativo (descartar)", "Descartar el segundo evento de prueba", "No vuelve a aparecer en el listado ni se indexa",
           f"en el listado: {weak[1] in after}; estado {st}", weak[1] not in after and st[0] == "described")

    args.out.write_text(json.dumps({
        "fecha": datetime.now().isoformat(timespec="seconds"), "casos": RESULTS,
        "eventos_de_prueba": weak + [alert_like, target, nodesc], "evento_real": new,
    }, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
