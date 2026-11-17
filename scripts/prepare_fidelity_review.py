"""Arma la planilla de revisión humana de la fidelidad de las descripciones (apartado 5.5 de la tesis).

Uso (con MU_DATABASE_URL apuntando a la base de la medición):
    python scripts/prepare_fidelity_review.py --n 30 --out ../reentreno/medicion/revision_fidelidad
    python scripts/prepare_fidelity_review.py --n 30 --eventos ../reentreno/medicion/eventos_choques.json --out ../reentreno/medicion/revision_fidelidad_choques
"""

from __future__ import annotations

import argparse
import html
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from monitoreo_urbano.common.config import settings  # noqa: E402
from monitoreo_urbano.common.db import get_connection  # noqa: E402
from monitoreo_urbano.common.object_storage import get_clips_client  # noqa: E402

PAGE = """<!doctype html><html lang="es"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Revisión de fidelidad</title>
<style>body{{font:15px/1.4 system-ui,sans-serif;max-width:900px;margin:0 auto;padding:16px;background:#fff;color:#111}}
.item{{border:1px solid #bbb;border-radius:8px;padding:12px;margin:14px 0}}video{{width:100%;max-height:420px;background:#000}}
pre{{white-space:pre-wrap;background:#f4f4f4;padding:8px;border-radius:6px}}label{{margin-right:16px}}textarea{{width:100%}}
button{{padding:8px 14px}}.done{{background:#e8f5e9}}@media (prefers-color-scheme:dark){{body{{background:#161616;color:#eee}}pre{{background:#262626}}}}</style>
<h1>Revisión de fidelidad: {n} clips</h1>
<p>Cada clip contiene un choque real (nombre del video de origen en el título) y puede incluir, antes del choque, tránsito normal
o pantalla negra. Mirar el video y contrastarlo con la descripción. Marcar <b>respaldada</b> si todas sus afirmaciones se ven
en el video y no agrega atributos inventados; en caso contrario, <b>no respaldada</b>, y anotar qué falla. Los veredictos se
guardan en este navegador; al terminar, usar «Exportar».</p>
<p><button onclick="exportar()">Exportar veredictos (JSON)</button> <span id="cont"></span></p>
{items}
<script>
const KEY='fidelidad_choques_v1';let datos=JSON.parse(localStorage.getItem(KEY)||'{{}}');
function guardar(id){{const v=document.querySelector('input[name="v_'+id+'"]:checked');datos[id]={{veredicto:v?v.value:null,nota:document.getElementById('n_'+id).value}};
localStorage.setItem(KEY,JSON.stringify(datos));pintar();}}
function pintar(){{let c=0;for(const id in datos){{const e=document.getElementById('i_'+id);if(!e)continue;const ok=datos[id].veredicto;e.classList.toggle('done',!!ok);if(ok)c++;}}document.getElementById('cont').textContent=c+' de {n} revisados';}}
for(const id in datos){{const d=datos[id];if(d.veredicto){{const r=document.querySelector('input[name="v_'+id+'"][value="'+d.veredicto+'"]');if(r)r.checked=true;}}const t=document.getElementById('n_'+id);if(t)t.value=d.nota||'';}}
pintar();
function exportar(){{const b=new Blob([JSON.stringify(datos,null,2)],{{type:'application/json'}});const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='veredictos_fidelidad_choques.json';a.click();}}
</script></html>"""

ITEM = """<div class="item" id="i_{eid}"><h3>{k}. {video}{camera} · {reason} · {ts}</h3>
<video controls preload="metadata" src="clips/{eid}.mp4"></video>
<pre>{desc}</pre>
<label><input type="radio" name="v_{eid}" value="respaldada" onchange="guardar('{eid}')"> respaldada</label>
<label><input type="radio" name="v_{eid}" value="no_respaldada" onchange="guardar('{eid}')"> no respaldada</label>
<textarea id="n_{eid}" rows="2" placeholder="Nota (qué no se ve en el video, qué se inventó)" oninput="guardar('{eid}')"></textarea></div>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--eventos", type=Path, default=None, help="Mapa evento → video; un evento por video de choque")
    args = parser.parse_args()

    conn = get_connection()
    mapa = json.loads(args.eventos.read_text(encoding="utf-8"))["eventos"] if args.eventos else None
    rows = conn.execute(
        "SELECT event_id, camera_id, trigger_reason, created_at, description_nl, clip_path FROM events "
        "WHERE description_nl IS NOT NULL AND status IN ('described','indexed') ORDER BY event_id"
    ).fetchall()
    if mapa:
        first: dict[str, dict] = {}
        for row in sorted(rows, key=lambda r: r["created_at"]):
            info = mapa.get(row["event_id"])
            if info and row["trigger_reason"] == "accident" and info["video"] not in first:
                first[info["video"]] = row
        rows = [first[v] for v in sorted(first)]
    rng = random.Random(args.seed)
    chosen = rng.sample(rows, min(args.n, len(rows)))
    (args.out / "clips").mkdir(parents=True, exist_ok=True)
    client = get_clips_client()
    items, manifest = [], []
    for k, row in enumerate(chosen, 1):
        client.fget_object(settings.s3_bucket_clips, row["clip_path"], str(args.out / "clips" / f"{row['event_id']}.mp4"))
        items.append(ITEM.format(
            eid=row["event_id"], k=k, video=(mapa[row["event_id"]]["video"] + " · ") if mapa else "", camera=row["camera_id"], reason=row["trigger_reason"],
            ts=row["created_at"].strftime("%H:%M:%S"), desc=html.escape(row["description_nl"]),
        ))
        manifest.append({"orden": k, "event_id": row["event_id"], "camera_id": row["camera_id"], "motivo": row["trigger_reason"],
                         "video": mapa[row["event_id"]]["video"] if mapa else None})
    (args.out / "revision.html").write_text(PAGE.format(n=len(chosen), items="\n".join(items)), encoding="utf-8")
    (args.out / "manifiesto.json").write_text(json.dumps({"semilla": args.seed, "eventos": manifest}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{len(chosen)} clips reservados de {len(rows)} eventos descritos en {args.out}")


if __name__ == "__main__":
    main()
