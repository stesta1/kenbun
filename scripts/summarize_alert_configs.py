"""Tabla de `evaluate_alert_configs.py`: detección, demora, falsas previas, duplicados y falsos en V7 por configuración."""
import json
import sys

d = json.load(open(sys.argv[1], encoding="utf-8"))["filas"]
filtro = sys.argv[2] if len(sys.argv) > 2 else None
print("grupo | fps | conf | persist s | ventana s | dedup | VAL det | TEST det (estr) | demora med/p90 s | falsas previas (val/test) | >1 evento (test) | V7 ev (por hora, repet.)")
for r in d:
    c, v, t, p = r["cfg"], r["val"], r["test"], r["v7"]
    if filtro and c["nombre"] != filtro:
        continue
    print(f"{c['nombre']:<10}| {c['fps']:>4g} | {c['conf']:.2f} | {c['persist_s']:.2f} | {c['ventana_s']:.1f} | {c['dedup_modo']:<9} | "
          f"{v['detectados']:>2}/{v['n']} | {t['detectados']:>2}/{t['n']} ({t['estrictos']:>2}) | {t['demora_mediana_s']}/{t['demora_p90_s']} | "
          f"{v['videos_con_falsa_previa']}/{t['videos_con_falsa_previa']} | {t['videos_con_mas_de_un_evento']} | {p['eventos']} ({p['por_hora']}/h, {p['repeticiones_lt120s']} rep.)")
