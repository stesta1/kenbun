"""Genera el conjunto SINTÉTICO de la prueba de exactitud del consultor (apartado 5.6 de la tesis).

Uso:
    python scripts/generate_synthetic_query_set.py --out ../reentreno/medicion/consulta_sintetica
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import uuid
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAPA = ROOT.parent / "test_ui" / "videos" / "choques" / "mapa_videos.json"

SEMILLA = 20261020
INICIO = datetime(2026, 9, 21)
DIAS = 14
N_CHOQUES = 36
N_INCENDIOS = 12

CAMARAS = [
    ("cam-01", "Av. Independencia y Sarandí", "Av. Independencia y Sarandí", "1"),
    ("cam-02", "Av. Rivadavia y Boyacá", "Av. Rivadavia y Boyacá", "11"),
    ("cam-03", "Av. Corrientes y Callao", "Av. Corrientes y Callao", "3"),
    ("cam-04", "Av. Cabildo y Juramento", "Av. Cabildo y Juramento", "13"),
    ("cam-05", "Av. Santa Fe y Pueyrredón", "Av. Santa Fe y Pueyrredón", "2"),
    ("cam-06", "Av. San Juan y Boedo", "Av. San Juan y Boedo", "5"),
]

TIPOS = {
    "auto": (["auto", "coche"], "m"),
    "moto": (["moto"], "f"),
    "camion": (["camión"], "m"),
    "camioneta": (["camioneta"], "f"),
    "pickup": (["pickup"], "f"),
}
COLORES = {
    "blanco": ("blanco", "blanca"), "negro": ("negro", "negra"), "rojo": ("rojo", "roja"), "azul": ("azul", "azul"),
    "gris": ("gris", "gris"), "naranja": ("naranja", "naranja"), "verde": ("verde", "verde"), "amarillo": ("amarillo", "amarilla"),
}
ESCENAS = {
    "avenida": ("sobre la avenida", "sobre una avenida"),
    "esquina": ("en una esquina con semáforo", "en una esquina con semáforo"),
    "rotonda": ("en una rotonda", "en una rotonda"),
    "autopista": ("en la autopista", "en una autopista"),
    "cruce_peatonal": ("sobre un cruce peatonal", "en un cruce peatonal"),
    "estacionamiento": ("en un estacionamiento", "en un estacionamiento"),
    "tunel": ("dentro de un túnel", "en un túnel"),
}
RASGOS_CHOQUE = {
    "peaton": ("Hay una persona parada cerca del lugar.", "con una persona cerca"),
    "humo": ("Sale humo del motor de uno de los vehículos.", "con humo saliendo de un vehículo"),
    "ambulancia": ("Llega una ambulancia al lugar.", "con una ambulancia en el lugar"),
    "transito_cortado": ("Se corta el tránsito en dos carriles.", "con el tránsito cortado"),
    "lluvia": ("La calzada está mojada por la lluvia.", "con lluvia"),
    "noche": ("El hecho ocurre de noche.", "de noche"),
    "grua": ("Se acerca una grúa.", "con una grúa en el lugar"),
}
OBJETOS = {
    "edificio": ("un edificio de viviendas", "un edificio de viviendas"),
    "local": ("un local comercial", "un local comercial"),
    "vehiculo": ("un vehículo estacionado", "un vehículo estacionado"),
    "contenedor": ("un contenedor de residuos", "un contenedor de residuos"),
    "galpon": ("un galpón", "un galpón"),
    "pastizal": ("un pastizal junto a la autopista", "un pastizal"),
}
RASGOS_INCENDIO = {
    "humo_negro": ("Sale una columna de humo negro.", "con una columna de humo negro"),
    "bomberos": ("Hay bomberos trabajando en el lugar.", "con bomberos trabajando"),
    "llamas": ("Se ven llamas desde la calle.", "con llamas visibles desde la calle"),
    "evacuacion": ("Se evacúa a los vecinos.", "con evacuación de vecinos"),
}

SIN_EVIDENCIA = [
    ("¿Hubo un choque con un tren?", ["tren"]),
    ("¿Hubo un incendio en un barco?", ["barco"]),
    ("¿Se volcó un camión cisterna?", ["cisterna", "volc"]),
    ("¿Hubo un choque con un colectivo?", ["colectivo"]),
    ("¿Se registró un choque con una bicicleta?", ["bicicleta"]),
    ("¿Hubo una explosión en una estación de servicio?", ["explosi", "estación de servicio"]),
]


def con_articulo(tipo: str, color: str, rng: random.Random) -> str:
    nombres, genero = TIPOS[tipo]
    nombre = rng.choice(nombres) if nombres[0] != "auto" else rng.choice(["auto", "auto", "coche"])
    if nombre == "coche":
        genero = "m"
    colores = COLORES[color]
    adjetivo = colores[1] if genero == "f" else colores[0]
    articulo = "una" if genero == "f" else "un"
    return f"{articulo} {nombre} {adjetivo}"


def lista(frases: list[str]) -> str:
    return frases[0] if len(frases) == 1 else ", ".join(frases[:-1]) + " y " + frases[-1]


def descripcion_choque(evento: dict, rng: random.Random) -> str:
    vehiculos = lista([con_articulo(t, c, rng) for t, c in evento["vehiculos"]])
    escena = ESCENAS[evento["escena"]][0]
    cola = RASGOS_CHOQUE[evento["rasgo"]][0] if evento["rasgo"] else ""
    plantilla = rng.choice([
        "{v} chocan {e}. {r}",
        "Choque entre {v} {e}. {r}",
        "Colisión {e} entre {v}. {r}",
    ])
    texto = plantilla.format(v=vehiculos, e=escena, r=cola).strip()
    return texto[0].upper() + texto[1:]


def descripcion_incendio(evento: dict, rng: random.Random) -> str:
    objeto = OBJETOS[evento["objeto"]][0]
    cola = RASGOS_INCENDIO[evento["rasgo"]][0] if evento["rasgo"] else ""
    plantilla = rng.choice(["Incendio en {o}. {r}", "Se incendia {o}. {r}", "Fuego en {o}. {r}"])
    texto = plantilla.format(o=objeto, r=cola).strip()
    return texto[0].upper() + texto[1:]


def generar_eventos(rng: random.Random, clips: list[str]) -> list[dict]:
    eventos: list[dict] = []
    for i in range(N_CHOQUES + N_INCENDIOS):
        choque = i < N_CHOQUES
        camara = rng.choice(CAMARAS)
        inicio = INICIO + timedelta(days=rng.randrange(DIAS), seconds=rng.randrange(86400))
        evento = {
            "event_id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"sintetico-{SEMILLA}-{i}")),
            "camera_id": camara[0], "camera_label": camara[1], "ubicacion": camara[2],
            "start_ts": inicio.replace(microsecond=0), "end_ts": (inicio + timedelta(seconds=rng.randrange(8, 25))).replace(microsecond=0),
            "clip_path": "sintetico/" + clips[i % len(clips)],
            "vehiculos": [], "escena": "", "rasgo": "", "objeto": "",
        }
        if choque:
            evento["trigger_reason"] = "accident" if i < N_CHOQUES - 6 else "posible_choque"
            cantidad = 3 if i % 9 == 8 else 2
            pares: list[tuple[str, str]] = []
            while len(pares) < cantidad:
                par = (rng.choice(list(TIPOS)), rng.choice(list(COLORES)))
                if par not in pares:
                    pares.append(par)
            evento["vehiculos"] = pares
            evento["escena"] = rng.choice(list(ESCENAS))
            evento["rasgo"] = rng.choice(list(RASGOS_CHOQUE) + [""] * 3)
            evento["description_nl"] = descripcion_choque(evento, rng)
        else:
            evento["trigger_reason"] = "fire"
            evento["objeto"] = rng.choice(list(OBJETOS))
            evento["rasgo"] = rng.choice(list(RASGOS_INCENDIO))
            evento["description_nl"] = descripcion_incendio(evento, rng)
        eventos.append(evento)
    return eventos


def relevantes_contenido(eventos: list[dict], *, razon=None, vehiculos=(), escena="", rasgo="", objeto="") -> list[str]:
    salida = []
    for e in eventos:
        if razon == "choque" and e["trigger_reason"] not in ("accident", "posible_choque"):
            continue
        if razon == "incendio" and e["trigger_reason"] != "fire":
            continue
        if any(v not in e["vehiculos"] for v in vehiculos):
            continue
        if escena and e["escena"] != escena:
            continue
        if rasgo and e["rasgo"] != rasgo:
            continue
        if objeto and e["objeto"] != objeto:
            continue
        salida.append(e["event_id"])
    return salida


def preguntas_contenido(eventos: list[dict], rng: random.Random) -> list[dict]:
    preguntas: list[dict] = []
    usados: set[str] = set()
    conjuntos: set[frozenset[str]] = set()

    def agregar(texto: str, criterio: str, relevantes: list[str]) -> bool:
        if not 1 <= len(relevantes) <= 5 or texto in usados or frozenset(relevantes) in conjuntos:
            return False
        usados.add(texto)
        conjuntos.add(frozenset(relevantes))
        preguntas.append({"tipo": "contenido", "pregunta": texto, "camera_id": "", "desde": "", "hasta": "", "relevantes": relevantes, "criterio": criterio})
        return True

    choques = [e for e in eventos if e["vehiculos"]]
    incendios = [e for e in eventos if e["trigger_reason"] == "fire"]

    plantillas_par = ["¿Hubo algún choque entre {a} y {b}?", "Necesito el choque de {a} contra {b}.", "Mostrame el choque donde participaron {a} y {b}."]
    intentos = 0
    while sum(1 for p in preguntas if "dos vehículos" in p["criterio"]) < 3 and intentos < 200:
        intentos += 1
        e = rng.choice(choques)
        (t1, c1), (t2, c2) = e["vehiculos"][:2]
        rel = relevantes_contenido(eventos, razon="choque", vehiculos=[(t1, c1), (t2, c2)])
        texto = rng.choice(plantillas_par).format(a=con_articulo(t1, c1, rng), b=con_articulo(t2, c2, rng))
        agregar(texto, f"choque con dos vehículos: {t1} {c1} y {t2} {c2}", rel)

    plantillas_uno = ["¿Qué choques hubo con {a}?", "Buscá los choques en los que estuvo involucrado {a}."]
    intentos = 0
    while sum(1 for p in preguntas if "un vehículo" in p["criterio"]) < 2 and intentos < 200:
        intentos += 1
        t, c = rng.choice(rng.choice(choques)["vehiculos"][:1] + [(rng.choice(list(TIPOS)), rng.choice(list(COLORES)))])
        rel = relevantes_contenido(eventos, razon="choque", vehiculos=[(t, c)])
        agregar(rng.choice(plantillas_uno).format(a=con_articulo(t, c, rng)), f"choque con un vehículo: {t} {c}", rel)

    intentos = 0
    while sum(1 for p in preguntas if "escena" in p["criterio"]) < 3 and intentos < 400:
        intentos += 1
        escena, rasgo = rng.choice(list(ESCENAS)), rng.choice(list(RASGOS_CHOQUE))
        rel = relevantes_contenido(eventos, razon="choque", escena=escena, rasgo=rasgo)
        texto = f"¿Hubo un choque {ESCENAS[escena][1]} {RASGOS_CHOQUE[rasgo][1]}?"
        agregar(texto, f"choque, escena {escena} y rasgo {rasgo}", rel)

    intentos = 0
    while sum(1 for p in preguntas if "incendio" in p["criterio"]) < 2 and intentos < 400:
        intentos += 1
        objeto, rasgo = rng.choice(list(OBJETOS)), rng.choice(list(RASGOS_INCENDIO))
        rel = relevantes_contenido(eventos, razon="incendio", objeto=objeto, rasgo=rasgo)
        texto = f"¿Hubo algún incendio en {OBJETOS[objeto][1]} {RASGOS_INCENDIO[rasgo][1]}?"
        agregar(texto, f"incendio, objeto {objeto} y rasgo {rasgo}", rel)

    for numero, p in enumerate(preguntas, start=1):
        p["id"] = f"C{numero:02d}"
    return preguntas


def en_periodo(e: dict, camara: str, desde: datetime, hasta: datetime) -> bool:
    return e["camera_id"] == camara and e["start_ts"] >= desde and e["end_ts"] <= hasta


def preguntas_restringidas(eventos: list[dict], rng: random.Random) -> tuple[list[dict], list[dict]]:
    con_eventos: list[dict] = []
    vacias: list[tuple[str, datetime, datetime]] = []
    vistos: set[tuple] = set()
    frases = ["¿Qué incidentes hubo en esta cámara en ese período?", "Contame qué pasó en esta cámara en ese período.", "¿Hubo algún evento en esta cámara en ese período?"]
    intentos = 0
    while (len(con_eventos) < 10 or len(vacias) < 2) and intentos < 5000:
        intentos += 1
        camara = rng.choice(CAMARAS)[0]
        desde = INICIO + timedelta(days=rng.randrange(DIAS - 2), hours=rng.randrange(24))
        hasta = desde + timedelta(days=rng.choice([2, 3, 4]))
        clave = (camara, desde, hasta)
        if clave in vistos:
            continue
        vistos.add(clave)
        rel = [e["event_id"] for e in eventos if en_periodo(e, camara, desde, hasta)]
        if 1 <= len(rel) <= 5 and len(con_eventos) < 10:
            con_eventos.append({
                "tipo": "restringida", "pregunta": rng.choice(frases), "camera_id": camara,
                "desde": desde.isoformat(), "hasta": hasta.isoformat(), "relevantes": rel,
                "criterio": f"eventos de {camara} con inicio y fin dentro del período",
            })
        elif not rel and len(vacias) < 2:
            vacias.append((camara, desde, hasta))
    for numero, p in enumerate(con_eventos, start=1):
        p["id"] = f"R{numero:02d}"
    return con_eventos, [{"camera_id": c, "desde": d, "hasta": h} for c, d, h in vacias]


def preguntas_sin_evidencia(eventos: list[dict], ventanas_vacias: list[dict]) -> list[dict]:
    textos = " ".join(e["description_nl"].lower() for e in eventos)
    preguntas: list[dict] = []
    for texto, claves in SIN_EVIDENCIA:
        for clave in claves:
            assert clave not in textos, f"La palabra clave «{clave}» aparece en una descripción: la consulta sí tendría evidencia"
        preguntas.append({"tipo": "sin_evidencia", "pregunta": texto, "camera_id": "", "desde": "", "hasta": "", "relevantes": [], "criterio": "ningún evento menciona lo pedido"})
    fuera = [
        {"camera_id": "cam-03", "desde": datetime(2026, 8, 1), "hasta": datetime(2026, 8, 3)},
        {"camera_id": "cam-05", "desde": datetime(2026, 8, 10), "hasta": datetime(2026, 8, 14)},
    ]
    for ventana in (ventanas_vacias + fuera)[:4]:
        assert not any(en_periodo(e, ventana["camera_id"], ventana["desde"], ventana["hasta"]) for e in eventos)
        preguntas.append({
            "tipo": "sin_evidencia", "pregunta": "¿Hubo un choque en esta cámara en ese período?", "camera_id": ventana["camera_id"],
            "desde": ventana["desde"].isoformat(), "hasta": ventana["hasta"].isoformat(), "relevantes": [],
            "criterio": "la cámara no tiene eventos en el período",
        })
    for numero, p in enumerate(preguntas, start=1):
        p["id"] = f"S{numero:02d}"
    return preguntas


LEEME = """# Conjunto sintético de la prueba de consulta

Generado por `scripts/generate_synthetic_query_set.py` con la semilla {semilla}.

**Los eventos son inventados.** Texto, vehículos, cámara y fecha se generaron al azar; cada evento referencia un clip cualquiera
del corpus de choques, que no tiene por qué coincidir con la descripción. La prueba mide solo si el consultor devuelve el evento
cuya descripción corresponde a lo pedido, aislado de la calidad de las descripciones del VLM.

- `eventos.csv`: {n_eventos} eventos confirmados ({n_choques} choques y {n_incendios} incendios) con la descripción que redactaría el operador,
  los vehículos cargados en la revisión y las etiquetas con las que se generó cada uno.
- `camaras.csv`: las seis cámaras del catálogo sintético.
- `preguntas.csv` / `preguntas.json`: {n_preguntas} consultas (10 de contenido, 10 acotadas por cámara y período, 10 sin evidencia) con los eventos
  relevantes, definidos por las etiquetas y no por el texto de la descripción. Las consultas de contenido tienen entre 1 y 5 relevantes.

Cargar en una base separada: `python scripts/load_synthetic_events.py --reset`.
Medir: `python scripts/evaluate_synthetic_query.py`.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rng = random.Random(SEMILLA)
    clips = [item["archivo"] for item in json.loads(MAPA.read_text(encoding="utf-8"))]
    eventos = generar_eventos(rng, clips)

    contenido = preguntas_contenido(eventos, rng)
    restringidas, vacias = preguntas_restringidas(eventos, rng)
    ausentes = preguntas_sin_evidencia(eventos, vacias)
    preguntas = contenido + restringidas + ausentes
    assert len(contenido) == 10 and len(restringidas) == 10 and len(ausentes) == 10, (len(contenido), len(restringidas), len(ausentes))

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "eventos.csv").open("w", encoding="utf-8", newline="") as f:
        campos = ["event_id", "camera_id", "camera_label", "ubicacion", "start_ts", "end_ts", "trigger_reason", "description_nl",
                  "vehiculos", "escena", "rasgo", "objeto", "clip_path", "review_status", "reviewed_by", "sintetico"]
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()
        for e in eventos:
            writer.writerow({
                **{k: e[k] for k in ("event_id", "camera_id", "camera_label", "ubicacion", "trigger_reason", "description_nl", "escena", "rasgo", "objeto", "clip_path")},
                "start_ts": e["start_ts"].isoformat(), "end_ts": e["end_ts"].isoformat(),
                "vehiculos": ";".join(f"{t}:{c}" for t, c in e["vehiculos"]),
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
        semilla=SEMILLA, n_eventos=len(eventos), n_choques=N_CHOQUES, n_incendios=N_INCENDIOS, n_preguntas=len(preguntas)), encoding="utf-8")
    cantidades = [len(p["relevantes"]) for p in preguntas if p["tipo"] != "sin_evidencia"]
    print(f"{len(eventos)} eventos y {len(preguntas)} consultas en {args.out}; relevantes por consulta: mín {min(cantidades)}, máx {max(cantidades)}")


if __name__ == "__main__":
    main()
