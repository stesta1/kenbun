"""Arma la planilla de revisión humana a ciegas de las descripciones de varias variantes sobre los mismos clips."""
from __future__ import annotations

import argparse
import html
import json
import random
from pathlib import Path

CRITERIA = [
    ("fiel", "Todo lo que afirma se ve en el video y no inventa nada"),
    ("vehiculos", "Dice bien cuántos vehículos hay y de qué tipo y color"),
    ("tipo_choque", "Dice bien cómo fue el choque (o dice que no se distingue)"),
    ("consecuencias", "Dice bien las consecuencias que se ven (o que no se ven)"),
    ("util", "Le serviría a un operador para decidir qué hacer"),
]
PAGE = """<!doctype html><html lang="es"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Revisión a ciegas de descripciones</title>
<style>body{{font:15px/1.4 system-ui,sans-serif;max-width:980px;margin:0 auto;padding:16px;background:#fff;color:#111}}
.item{{border:1px solid #bbb;border-radius:8px;padding:12px;margin:14px 0}}video{{width:100%;max-height:420px;background:#000}}
pre{{white-space:pre-wrap;background:#f4f4f4;padding:8px;border-radius:6px;margin:6px 0}}.d{{border-top:1px dashed #bbb;margin-top:10px;padding-top:6px}}
label{{display:block;margin:2px 0}}textarea{{width:100%}}button{{padding:8px 14px}}.done{{background:#e8f5e9}}
@media (prefers-color-scheme:dark){{body{{background:#161616;color:#eee}}pre{{background:#262626}}}}</style>
<h1>Revisión a ciegas: {n} clips</h1>
<p>Cada clip tiene un choque real (puede haber antes tránsito normal o pantalla negra). Mirar el video y juzgar cada
descripción por separado, sin comparar con las otras. Las descripciones están en orden aleatorio y no se indica cuál es de cuál
variante. Marcar los criterios que se cumplen. Los veredictos se guardan en este navegador; al terminar, usar «Exportar».</p>
<p><button onclick="exportar()">Exportar veredictos (JSON)</button> <span id="cont"></span></p>
{items}
<script>
const KEY='{key}';let datos=JSON.parse(localStorage.getItem(KEY)||'{{}}');
const CRIT={crit};
function guardar(id,pos){{const k=id+'_'+pos;const o={{}};for(const c of CRIT){{o[c]=document.getElementById(c+'_'+k).checked;}}
o.nota=document.getElementById('n_'+k).value;o.revisada=true;datos[k]=o;localStorage.setItem(KEY,JSON.stringify(datos));pintar();}}
function pintar(){{let c=0;for(const k in datos){{if(datos[k].revisada)c++;}}document.getElementById('cont').textContent=c+' de {total} descripciones revisadas';}}
for(const k in datos){{for(const c of CRIT){{const e=document.getElementById(c+'_'+k);if(e)e.checked=!!datos[k][c];}}const t=document.getElementById('n_'+k);if(t)t.value=datos[k].nota||'';}}
pintar();
function exportar(){{const b=new Blob([JSON.stringify(datos,null,1)],{{type:'application/json'}});const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='{filename}';a.click();}}
</script></html>"""


def crop_block(out: Path, video: str) -> str:
    if not (out / "recortes" / f"{video}.mp4").exists():
        return ""
    return (
        '<p><b>Lo que vio el modelo</b> (zona del choque que ubicó el detector):</p>'
        f'<video controls preload="none" src="recortes/{video}.mp4"></video>'
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+", type=Path, required=True)
    parser.add_argument("--clips-rel", default="../../../test_ui/videos/choques")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--key", default="descripciones_ciegas_v1", help="clave de localStorage: distinta en cada planilla para que no se pisen")
    parser.add_argument("--filename", default="veredictos_descripciones.json", help="nombre del archivo que descarga «Exportar»")
    args = parser.parse_args()

    variants = [json.loads(p.read_text(encoding="utf-8")) for p in args.variants]
    by_event = {v["etiqueta"]: {r["video"]: r for r in v["resultados"]} for v in variants}
    first = variants[0]["resultados"]
    rng = random.Random(args.seed)
    clave: dict[str, dict[str, str]] = {}
    items = []
    for clip in first:
        event_id = clip["video"]
        labels = [v["etiqueta"] for v in variants]
        rng.shuffle(labels)
        clave[event_id] = {str(i + 1): label for i, label in enumerate(labels)}
        blocks = []
        for pos, label in enumerate(labels, start=1):
            text = html.escape(by_event[label][event_id]["descripcion"])
            key = f"{event_id}_{pos}"
            checks = "".join(
                f'<label><input type="checkbox" id="{c}_{key}" onchange="guardar(\'{event_id}\',{pos})"> {html.escape(t)}</label>'
                for c, t in CRITERIA
            )
            blocks.append(
                f'<div class="d"><b>Descripción {pos}</b><pre>{text}</pre>{checks}'
                f'<textarea id="n_{key}" rows="2" placeholder="Nota: qué falla o qué inventa" oninput="guardar(\'{event_id}\',{pos})"></textarea></div>'
            )
        items.append(
            f'<div class="item" id="i_{event_id}"><h3>Clip {clip["orden"]} ({html.escape(clip["video"])})</h3>'
            f'<video controls preload="none" src="{args.clips_rel}/{event_id}.mp4"></video>{crop_block(args.out, event_id)}{"".join(blocks)}</div>'
        )
    total = len(first) * len(variants)
    args.out.mkdir(parents=True, exist_ok=True)
    page = PAGE.format(
        n=len(first), items="\n".join(items), key=args.key, filename=args.filename, total=total, crit=json.dumps([c for c, _ in CRITERIA])
    )
    (args.out / "index.html").write_text(page, encoding="utf-8")
    (args.out / "clave.json").write_text(json.dumps(clave, indent=1), encoding="utf-8")
    print(f"{len(first)} clips, {len(variants)} variantes, {total} descripciones -> {args.out / 'index.html'}")


if __name__ == "__main__":
    main()
