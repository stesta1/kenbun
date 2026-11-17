"""Genera el conjunto sintético de PRUEBA (reservado) de la exactitud del consultor, más amplio y variado que el de desarrollo.

Uso:
    python scripts/generate_synthetic_query_set_test.py --out ../reentreno/medicion/consulta_sintetica_prueba
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import uuid
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_synthetic_query_set import (  # noqa: E402
    CAMARAS, COLORES, ESCENAS, MAPA, OBJETOS, RASGOS_CHOQUE, RASGOS_INCENDIO, TIPOS,
    con_articulo, descripcion_choque, descripcion_incendio, lista,
)

SEMILLA = 20261105
INICIO = datetime(2026, 9, 7)
DIAS = 28
CONTEOS = {"choque": 60, "incendio": 20, "humo": 5, "pelea": 15}

FAMILIAS = {
    "choque": {"accident", "posible_choque"}, "incendio": {"fire"}, "humo": {"smoke"}, "pelea": {"fight", "posible_altercado"},
}
PERSONAS = ["dos personas", "un grupo de personas", "tres jóvenes", "dos hombres"]
LUGARES_PELEA = {
    "vereda": ("en la vereda", "sobre una vereda"),
    "plaza": ("en una plaza", "en una plaza"),
    "estacion_subte": ("en una estación de subte", "en una estación de subte"),
    "bar": ("a la salida de un bar", "cerca de un bar"),
    "supermercado": ("en la entrada de un supermercado", "en un supermercado"),
}
RASGOS_PELEA = {
    "golpes": ("Se golpean entre sí.", "con golpes entre las personas"),
    "objetos": ("Uno de ellos arroja objetos.", "con objetos arrojados"),
    "separan": ("Otras personas intentan separarlos.", "con gente tratando de separarlos"),
    "patrullero": ("Llega un patrullero.", "con un patrullero en el lugar"),
}
OBJETOS_HUMO = {
    "edificio": ("un edificio", "un edificio"),
    "auto": ("el capó de un auto", "un auto"),
    "basural": ("un basural", "un basural"),
    "fabrica": ("la chimenea de una fábrica", "una fábrica"),
    "pastizal": ("un pastizal", "un pastizal"),
}
RASGOS_HUMO = {
    "negro": ("El humo es negro y espeso.", "con humo negro"),
    "blanco": ("El humo es blanco.", "con humo blanco"),
    "calle": ("El humo se extiende sobre la calle.", "que se extiende sobre la calle"),
    "bomberos": ("Hay bomberos en el lugar.", "con bomberos en el lugar"),
}
PALABRAS_CATEGORIA = {
    "choque": ("choques", "choque"), "incendio": ("incendios", "incendio"), "humo": ("humo", "humo"), "pelea": ("peleas", "pelea"),
}
SIN_EVIDENCIA = [
    ("¿Se estrelló un avión?", ["avión", "avion"]), ("¿Hubo un choque con un helicóptero?", ["helic"]),
    ("¿Hubo una inundación en la calle?", ["inunda"]), ("¿Se derrumbó un edificio?", ["derrumb"]),
    ("¿Hubo un tiroteo?", ["tiroteo", "disparo"]), ("¿Hubo un robo a un local?", ["robo"]),
    ("¿Se cayó un árbol sobre un auto?", ["árbol", "arbol"]), ("¿Hubo un choque con un animal?", ["animal"]),
    ("¿Hubo un choque con un tractor?", ["tractor"]), ("¿Se incendió un barco?", ["barco"]),
    ("¿Hubo un accidente con un monopatín?", ["monopat"]), ("¿Hubo un corte de luz en la zona?", [" luz"]),
]


def capitalizar(texto: str) -> str:
    return texto[0].upper() + texto[1:]


def descripcion_pelea(e: dict, rng: random.Random) -> str:
    cola = RASGOS_PELEA[e["rasgo"]][0] if e["rasgo"] else ""
    plantilla = rng.choice(["{p} se pelean {l}. {r}", "Pelea entre {p} {l}. {r}", "{p} discuten y terminan peleando {l}. {r}"])
    return capitalizar(plantilla.format(p=e["personas"], l=LUGARES_PELEA[e["escena"]][0], r=cola).strip())


def descripcion_humo(e: dict, rng: random.Random) -> str:
    cola = RASGOS_HUMO[e["rasgo"]][0]
    plantilla = rng.choice(["Sale humo de {o}. {r}", "Se ve humo en {o}. {r}"])
    return capitalizar(plantilla.format(o=OBJETOS_HUMO[e["objeto"]][0], r=cola).strip())


def generar_eventos(rng: random.Random, clips: list[str]) -> list[dict]:
    eventos: list[dict] = []
    familias = [f for f, n in CONTEOS.items() for _ in range(n)]
    rng.shuffle(familias)
    for i, familia in enumerate(familias):
        camara = rng.choice(CAMARAS)
        inicio = INICIO + timedelta(days=rng.randrange(DIAS), seconds=rng.randrange(86400))
        e = {
            "event_id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"sintetico-prueba-{SEMILLA}-{i}")), "familia": familia,
            "camera_id": camara[0], "camera_label": camara[1], "ubicacion": camara[2],
            "start_ts": inicio.replace(microsecond=0), "end_ts": (inicio + timedelta(seconds=rng.randrange(8, 25))).replace(microsecond=0),
            "clip_path": "sintetico/" + clips[i % len(clips)], "vehiculos": [], "escena": "", "rasgo": "", "objeto": "", "personas": "",
        }
        if familia == "choque":
            e["trigger_reason"] = "accident" if rng.random() < 0.75 else "posible_choque"
            pares: list[tuple[str, str]] = []
            while len(pares) < (3 if rng.random() < 0.12 else 2):
                par = (rng.choice(list(TIPOS)), rng.choice(list(COLORES)))
                if par not in pares:
                    pares.append(par)
            e["vehiculos"] = pares
            e["escena"] = rng.choice(list(ESCENAS))
            e["rasgo"] = rng.choice(list(RASGOS_CHOQUE) + [""] * 3)
            e["description_nl"] = descripcion_choque(e, rng)
        elif familia == "incendio":
            e["trigger_reason"] = "fire"
            e["objeto"], e["rasgo"] = rng.choice(list(OBJETOS)), rng.choice(list(RASGOS_INCENDIO))
            e["description_nl"] = descripcion_incendio(e, rng)
        elif familia == "humo":
            e["trigger_reason"] = "smoke"
            e["objeto"], e["rasgo"] = rng.choice(list(OBJETOS_HUMO)), rng.choice(list(RASGOS_HUMO))
            e["description_nl"] = descripcion_humo(e, rng)
        else:
            e["trigger_reason"] = "fight" if rng.random() < 0.55 else "posible_altercado"
            e["personas"] = rng.choice(PERSONAS)
            e["escena"] = rng.choice(list(LUGARES_PELEA))
            e["rasgo"] = rng.choice(list(RASGOS_PELEA) + [""])
            e["description_nl"] = descripcion_pelea(e, rng)
        eventos.append(e)
    return eventos


def coincide(e: dict, *, familia="", vehiculos=(), escena="", rasgo="", objeto="") -> bool:
    return (
        (not familia or e["familia"] == familia) and all(v in e["vehiculos"] for v in vehiculos)
        and (not escena or e["escena"] == escena) and (not rasgo or e["rasgo"] == rasgo) and (not objeto or e["objeto"] == objeto)
    )


def ids(eventos: list[dict], **filtros) -> list[str]:
    return [e["event_id"] for e in eventos if coincide(e, **filtros)]


def preguntas_contenido(eventos: list[dict], rng: random.Random) -> list[dict]:
    preguntas: list[dict] = []
    conjuntos: set[frozenset[str]] = set()

    def agregar(texto: str, criterio: str, relevantes: list[str]) -> bool:
        clave = frozenset(relevantes)
        if not 1 <= len(relevantes) <= 5 or clave in conjuntos or any(p["pregunta"] == texto for p in preguntas):
            return False
        conjuntos.add(clave)
        preguntas.append({"tipo": "contenido", "pregunta": texto, "camera_id": "", "desde": "", "hasta": "", "relevantes": relevantes, "criterio": criterio})
        return True

    choques = [e for e in eventos if e["familia"] == "choque"]

    def hacer(cantidad: int, marca: str, generar) -> None:
        intentos = 0
        while sum(1 for p in preguntas if p["criterio"].startswith(marca)) < cantidad and intentos < 600:
            intentos += 1
            generar()

    def dos_vehiculos() -> None:
        e = rng.choice(choques)
        (t1, c1), (t2, c2) = e["vehiculos"][:2]
        texto = rng.choice(["¿Hubo algún choque entre {a} y {b}?", "Necesito el choque de {a} contra {b}.", "Mostrame el choque donde participaron {a} y {b}."]).format(
            a=con_articulo(t1, c1, rng), b=con_articulo(t2, c2, rng))
        agregar(texto, f"choque, dos vehículos: {t1} {c1} y {t2} {c2}", ids(eventos, familia="choque", vehiculos=[(t1, c1), (t2, c2)]))

    def un_vehiculo() -> None:
        t, c = rng.choice(list(TIPOS)), rng.choice(list(COLORES))
        texto = rng.choice(["¿Qué choques hubo con {a}?", "Buscá los choques en los que estuvo involucrado {a}."]).format(a=con_articulo(t, c, rng))
        agregar(texto, f"choque, un vehículo: {t} {c}", ids(eventos, familia="choque", vehiculos=[(t, c)]))

    def escena_choque() -> None:
        escena, rasgo = rng.choice(list(ESCENAS)), rng.choice(list(RASGOS_CHOQUE))
        agregar(f"¿Hubo un choque {ESCENAS[escena][1]} {RASGOS_CHOQUE[rasgo][1]}?", f"choque, escena {escena} y rasgo {rasgo}", ids(eventos, familia="choque", escena=escena, rasgo=rasgo))

    def incendio() -> None:
        objeto, rasgo = rng.choice(list(OBJETOS)), rng.choice(list(RASGOS_INCENDIO))
        agregar(f"¿Hubo algún incendio en {OBJETOS[objeto][1]} {RASGOS_INCENDIO[rasgo][1]}?", f"incendio, objeto {objeto} y rasgo {rasgo}", ids(eventos, familia="incendio", objeto=objeto, rasgo=rasgo))

    def humo() -> None:
        objeto, rasgo = rng.choice(list(OBJETOS_HUMO)), rng.choice(list(RASGOS_HUMO))
        agregar(f"¿Se vio humo saliendo de {OBJETOS_HUMO[objeto][1]} {RASGOS_HUMO[rasgo][1]}?", f"humo, objeto {objeto} y rasgo {rasgo}", ids(eventos, familia="humo", objeto=objeto, rasgo=rasgo))

    def pelea() -> None:
        lugar, rasgo = rng.choice(list(LUGARES_PELEA)), rng.choice(list(RASGOS_PELEA))
        agregar(f"¿Hubo una pelea {LUGARES_PELEA[lugar][1]} {RASGOS_PELEA[rasgo][1]}?", f"pelea, lugar {lugar} y rasgo {rasgo}", ids(eventos, familia="pelea", escena=lugar, rasgo=rasgo))

    hacer(3, "choque, dos vehículos", dos_vehiculos)
    hacer(3, "choque, un vehículo", un_vehiculo)
    hacer(3, "choque, escena", escena_choque)
    hacer(3, "incendio", incendio)
    hacer(2, "humo, objeto", humo)
    hacer(3, "pelea", pelea)
    for familia in ("humo",):
        relevantes = ids(eventos, familia=familia)
        agregar("¿Hubo humo en algún lado?", f"categoría {familia}", relevantes)
    intentos = 0
    while len(preguntas) < 20 and intentos < 600:
        intentos += 1
        escena_choque()
    for numero, p in enumerate(preguntas, start=1):
        p["id"] = f"C{numero:02d}"
    return preguntas


def en_periodo(e: dict, camara: str, desde: datetime, hasta: datetime) -> bool:
    return e["camera_id"] == camara and e["start_ts"] >= desde and e["end_ts"] <= hasta


def preguntas_restringidas(eventos: list[dict], rng: random.Random) -> tuple[list[dict], list[dict]]:
    simples: list[dict] = []
    con_categoria: list[dict] = []
    vacias: list[dict] = []
    vistos: set[tuple] = set()
    frases = ["¿Qué incidentes hubo en esta cámara en ese período?", "Contame qué pasó en esta cámara en ese período.", "¿Hubo algún evento en esta cámara en ese período?"]
    intentos = 0
    while (len(simples) < 10 or len(con_categoria) < 10 or len(vacias) < 8) and intentos < 20000:
        intentos += 1
        camara = rng.choice(CAMARAS)[0]
        desde = INICIO + timedelta(days=rng.randrange(DIAS - 4), hours=rng.randrange(24))
        hasta = desde + timedelta(days=rng.choice([2, 3, 4]))
        familia = rng.choice(list(FAMILIAS))
        clave = (camara, desde, hasta, familia)
        if clave in vistos:
            continue
        vistos.add(clave)
        en_ventana = [e for e in eventos if en_periodo(e, camara, desde, hasta)]
        de_familia = [e["event_id"] for e in en_ventana if e["familia"] == familia]
        base = {"camera_id": camara, "desde": desde.isoformat(), "hasta": hasta.isoformat()}
        if 1 <= len(en_ventana) <= 5 and len(simples) < 10:
            simples.append({"tipo": "restringida", "pregunta": rng.choice(frases), **base, "relevantes": [e["event_id"] for e in en_ventana],
                            "criterio": f"eventos de {camara} dentro del período"})
        elif 1 <= len(de_familia) <= 5 and len(en_ventana) > len(de_familia) and len(con_categoria) < 10:
            plural, singular = PALABRAS_CATEGORIA[familia]
            texto = rng.choice([f"¿Hubo {plural} en esta cámara en ese período?", f"¿Qué {plural} se registraron en esta cámara en ese período?"]) if familia != "humo" else "¿Hubo humo en esta cámara en ese período?"
            con_categoria.append({"tipo": "restringida", "pregunta": texto, **base, "relevantes": de_familia,
                                  "criterio": f"eventos de {camara} dentro del período, de la categoría {familia}"})
        elif en_ventana and not de_familia and len(vacias) < 4:
            vacias.append({**base, "familia": familia, "con_otros": True})
        elif not en_ventana and 4 <= len(vacias) < 8:
            vacias.append({**base, "familia": familia, "con_otros": False})
    restringidas = simples + con_categoria
    for numero, p in enumerate(restringidas, start=1):
        p["id"] = f"R{numero:02d}"
    return restringidas, vacias


def preguntas_sin_evidencia(eventos: list[dict], ventanas: list[dict]) -> list[dict]:
    textos = " ".join(e["description_nl"].lower() for e in eventos)
    preguntas: list[dict] = []
    for texto, claves in SIN_EVIDENCIA[:12]:
        for clave in claves:
            assert clave not in textos, f"«{clave}» aparece en una descripción: la consulta sí tendría evidencia"
        preguntas.append({"tipo": "sin_evidencia", "pregunta": texto, "camera_id": "", "desde": "", "hasta": "", "relevantes": [], "criterio": "ningún evento menciona lo pedido"})
    for v in ventanas[:8]:
        desde, hasta = datetime.fromisoformat(v["desde"]), datetime.fromisoformat(v["hasta"])
        assert not any(en_periodo(e, v["camera_id"], desde, hasta) and e["familia"] == v["familia"] for e in eventos)
        plural, singular = PALABRAS_CATEGORIA[v["familia"]]
        texto = "¿Hubo humo en esta cámara en ese período?" if v["familia"] == "humo" else f"¿Hubo {plural} en esta cámara en ese período?"
        preguntas.append({"tipo": "sin_evidencia", "pregunta": texto, "camera_id": v["camera_id"], "desde": v["desde"], "hasta": v["hasta"], "relevantes": [],
                          "criterio": "la cámara tiene eventos en el período, pero de otras categorías" if v["con_otros"] else "la cámara no tiene eventos en el período"})
    for numero, p in enumerate(preguntas, start=1):
        p["id"] = f"S{numero:02d}"
    return preguntas


LEEME = """# Conjunto sintético de PRUEBA de la consulta

Generado por `scripts/generate_synthetic_query_set_test.py` con la semilla {semilla}. Es el conjunto reservado: no se usa para ajustar el
consultor. El de desarrollo está en `consulta_sintetica`.

**Los eventos son inventados.** {n_eventos} eventos confirmados de cuatro familias ({conteos}), con cámara, fecha, descripción redactada como la haría un
operador, vehículos (en los choques) y un clip cualquiera del corpus como referencia. La prueba mide solo si el consultor devuelve el evento
cuya descripción corresponde a lo pedido.

- `preguntas.csv` / `preguntas.json`: {n_preguntas} consultas (20 de contenido, 20 acotadas por cámara y período, 20 sin evidencia). Los relevantes se definen por las etiquetas
  de generación, no por el texto, y cada consulta con respuesta tiene entre 1 y 5.
"""


def main() -> None:
    global SEMILLA
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--semilla", type=int, default=SEMILLA)
    args = parser.parse_args()
    SEMILLA = args.semilla
    rng = random.Random(SEMILLA)
    clips = [item["archivo"] for item in json.loads(MAPA.read_text(encoding="utf-8"))]
    eventos = generar_eventos(rng, clips)
    contenido = preguntas_contenido(eventos, rng)
    restringidas, vacias = preguntas_restringidas(eventos, rng)
    ausentes = preguntas_sin_evidencia(eventos, vacias)
    preguntas = contenido + restringidas + ausentes
    assert (len(contenido), len(restringidas), len(ausentes)) == (20, 20, 20), (len(contenido), len(restringidas), len(ausentes))

    args.out.mkdir(parents=True, exist_ok=True)
    campos = ["event_id", "camera_id", "camera_label", "ubicacion", "start_ts", "end_ts", "trigger_reason", "description_nl",
              "vehiculos", "escena", "rasgo", "objeto", "clip_path", "review_status", "reviewed_by", "sintetico"]
    with (args.out / "eventos.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()
        for e in eventos:
            writer.writerow({
                **{k: e[k] for k in ("event_id", "camera_id", "camera_label", "ubicacion", "trigger_reason", "description_nl", "escena", "rasgo", "objeto", "clip_path")},
                "start_ts": e["start_ts"].isoformat(), "end_ts": e["end_ts"].isoformat(), "vehiculos": ";".join(f"{t}:{c}" for t, c in e["vehiculos"]),
                "review_status": "confirmed", "reviewed_by": "operador-sintetico", "sintetico": "true",
            })
    with (args.out / "camaras.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["camera_id", "label", "ubicacion", "comuna"])
        writer.writerows(CAMARAS)
    with (args.out / "preguntas.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "tipo", "pregunta", "camera_id", "desde", "hasta", "relevantes", "criterio"])
        writer.writeheader()
        for p in preguntas:
            writer.writerow({**{k: p[k] for k in ("id", "tipo", "pregunta", "camera_id", "desde", "hasta", "criterio")}, "relevantes": ";".join(p["relevantes"])})
    (args.out / "preguntas.json").write_text(json.dumps({"semilla": SEMILLA, "preguntas": preguntas}, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.out / "LEEME.md").write_text(LEEME.format(
        semilla=SEMILLA, n_eventos=len(eventos), conteos=", ".join(f"{n} de {f}" for f, n in CONTEOS.items()), n_preguntas=len(preguntas)), encoding="utf-8")
    cantidades = [len(p["relevantes"]) for p in preguntas if p["tipo"] != "sin_evidencia"]
    print(f"{len(eventos)} eventos y {len(preguntas)} consultas en {args.out}; relevantes por consulta: mín {min(cantidades)}, máx {max(cantidades)}")


if __name__ == "__main__":
    main()
