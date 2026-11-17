"""Arma la planilla de revisión humana de la fidelidad de las respuestas de consulta (apartado 5.6 de la tesis).

Uso (con MU_DATABASE_URL apuntando a la base de la medición):
    python scripts/prepare_query_review.py --consulta ../reentreno/consulta_1.json --preguntas ../reentreno/medicion/preguntas.json --out ../reentreno/medicion/revision_consulta
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from monitoreo_urbano.common.db import get_connection  # noqa: E402

PAGE = """<!doctype html><html lang="es"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Revisión de respuestas</title>
<style>body{{font:15px/1.4 system-ui,sans-serif;max-width:900px;margin:0 auto;padding:16px;background:#fff;color:#111}}
.item{{border:1px solid #bbb;border-radius:8px;padding:12px;margin:14px 0}}pre{{white-space:pre-wrap;background:#f4f4f4;padding:8px;border-radius:6px}}
label{{margin-right:16px}}textarea{{width:100%}}button{{padding:8px 14px}}.done{{background:#e8f5e9}}.auto{{font-size:13px;color:#555}}
@media (prefers-color-scheme:dark){{body{{background:#161616;color:#eee}}pre{{background:#262626}}.auto{{color:#aaa}}}}</style>
<h1>Revisión de respuestas: {n} preguntas</h1>
<p>Una respuesta es <b>fiel</b> si sus afirmaciones están respaldadas por los eventos citados (su descripción) y respeta la
cámara y el período pedidos. En las preguntas sin eventos, es fiel si informa que no hay resultados y no inventa. Los veredictos
se guardan en este navegador; al terminar, usar «Exportar».</p>
<p><button onclick="exportar()">Exportar veredictos (JSON)</button> <span id="cont"></span></p>
{items}
<script>
const KEY='fidelidad_consulta_choques_v1';let datos=JSON.parse(localStorage.getItem(KEY)||'{{}}');
function guardar(id){{const v=document.querySelector('input[name="v_'+id+'"]:checked');datos[id]={{veredicto:v?v.value:null,nota:document.getElementById('n_'+id).value}};
localStorage.setItem(KEY,JSON.stringify(datos));pintar();}}
function pintar(){{let c=0;for(const id in datos){{const e=document.getElementById('i_'+id);if(!e)continue;const ok=datos[id].veredicto;e.classList.toggle('done',!!ok);if(ok)c++;}}document.getElementById('cont').textContent=c+' de {n} revisadas';}}
for(const id in datos){{const d=datos[id];if(d.veredicto){{const r=document.querySelector('input[name="v_'+id+'"][value="'+d.veredicto+'"]');if(r)r.checked=true;}}const t=document.getElementById('n_'+id);if(t)t.value=d.nota||'';}}
pintar();
function exportar(){{const b=new Blob([JSON.stringify(datos,null,2)],{{type:'application/json'}});const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='veredictos_consulta_choques.json';a.click();}}
</script></html>"""

ITEM = """<div class="item" id="i_{id}"><h3>{id} · {tipo}</h3>
<p><b>Pregunta:</b> {pregunta}</p><p class="auto">Filtros: {filtros}</p>
<p><b>Respuesta del sistema:</b></p><pre>{respuesta}</pre>
<p><b>Eventos citados ({k}):</b></p><pre>{eventos}</pre>
<p class="auto">Verificación automática: {auto}</p>
<label><input type="radio" name="v_{id}" value="fiel" onchange="guardar('{id}')"> fiel</label>
<label><input type="radio" name="v_{id}" value="no_fiel" onchange="guardar('{id}')"> no fiel</label>
<textarea id="n_{id}" rows="2" placeholder="Nota" oninput="guardar('{id}')"></textarea></div>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--consulta", type=Path, required=True)
    parser.add_argument("--preguntas", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    questions = json.loads(args.preguntas.read_text(encoding="utf-8"))["preguntas"]
    rows = json.loads(args.consulta.read_text(encoding="utf-8"))["filas"]
    conn = get_connection()
    items: list[str] = []
    for q in questions:
        row = next(r for r in rows if r["id"] == q["id"] and r["rep"] == 1)
        lines, bad = [], 0
        for event_id in row["eventos"]:
            e = conn.execute("SELECT camera_id, trigger_reason, start_ts, end_ts, description_nl FROM events WHERE event_id = %s", (event_id,)).fetchone()
            ok_cam = not q.get("camera_id") or e["camera_id"] == q["camera_id"]
            ok_time = True
            if q.get("desde"):
                ok_time = e["start_ts"] >= datetime.fromisoformat(q["desde"]) and e["end_ts"] <= datetime.fromisoformat(q["hasta"])
            bad += 0 if (ok_cam and ok_time) else 1
            lines.append(f"- {e['camera_id']} · {e['trigger_reason']} · {e['start_ts']:%H:%M:%S} · {(e['description_nl'] or '').splitlines()[0][:160]}")
        filtros = ", ".join(f"{k}={q[k]}" for k in ("camera_id", "desde", "hasta") if q.get(k)) or "ninguno"
        auto = f"{len(row['eventos'])} eventos citados, {bad} fuera de la cámara o del período pedidos"
        items.append(ITEM.format(id=q["id"], tipo=q["tipo"], pregunta=html.escape(q["pregunta"]), filtros=filtros, respuesta=html.escape(row["respuesta"] or ""),
                                 k=len(row["eventos"]), eventos=html.escape("\n".join(lines) or "(ninguno)"), auto=auto))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "revision_consulta.html").write_text(PAGE.format(n=len(questions), items="\n".join(items)), encoding="utf-8")
    print(f"{len(questions)} preguntas en {args.out}")


if __name__ == "__main__":
    main()
