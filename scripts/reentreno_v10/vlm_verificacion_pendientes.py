"""Experimento B: verificación de los pendientes de posible choque con el VLM (SÍ/NO) para escalarlos a alerta.

Uso (desde la raíz del repositorio):
    .venv/Scripts/python.exe scripts/reentreno_v10/vlm_verificacion_pendientes.py --split val --variantes p1 p2 p3 --out reentreno/vlm_pendientes/val.json
    .venv/Scripts/python.exe scripts/reentreno_v10/vlm_verificacion_pendientes.py --split test --variantes p2 --out reentreno/vlm_pendientes/test.json
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import cv2

APP = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(APP / "scripts"))
sys.path.insert(0, str(APP / "src"))
TOL = 3.0
FPS = 6.0
N_FRAMES = 8
VLM = "Qwen/Qwen2-VL-2B-Instruct"
DETRAC = APP / "data" / "detrac_eval"

PROMPTS = {
    "p1": "¿Se ve en estas imágenes un choque entre vehículos, o de un vehículo contra un objeto? Respondé solo SÍ o NO.",
    "p2": ("Son cuadros sucesivos de una cámara de tránsito. ¿Ocurre un accidente de tránsito: un choque, una colisión o un vuelco? "
           "Si los vehículos solo circulan, frenan o están detenidos cerca unos de otros, respondé NO. Respondé solo SÍ o NO."),
    "p3": "These are consecutive frames from a traffic camera. Is there a traffic accident (a collision between vehicles, or a vehicle crashing into an object)? Answer only YES or NO.",
}


class Verifier:

    def __init__(self) -> None:
        import torch
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(VLM, min_pixels=128 * 28 * 28, max_pixels=320 * 28 * 28)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(VLM, torch_dtype=torch.float16, device_map="cuda:0")
        self.torch = torch

    def ask(self, images: list, prompt: str) -> tuple[str, float]:
        content = [{"type": "image", "image": im} for im in images] + [{"type": "text", "text": prompt}]
        text = self.processor.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=images, return_tensors="pt").to(self.model.device)
        self.torch.cuda.synchronize()
        start = time.perf_counter()
        out = self.model.generate(**inputs, max_new_tokens=4, do_sample=False)
        self.torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        raw = self.processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
        return raw, elapsed


def is_yes(raw: str) -> bool:
    word = raw.strip().lstrip("¡¿\"'*").upper()
    return word.startswith(("SÍ", "SI", "YES"))


def window(video: Path, t: float, before: float, after: float) -> list:
    from PIL import Image

    cap = cv2.VideoCapture(str(video))
    start, end = max(0.0, t - before), t + after
    images = []
    for i in range(N_FRAMES):
        ms = (start + (end - start) * i / (N_FRAMES - 1)) * 1000
        cap.set(cv2.CAP_PROP_POS_MSEC, ms)
        ok, frame = cap.read()
        if ok:
            images.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    return images


def nivel(alerts: list[float], reviews: list[float]) -> str:
    if any(a >= -TOL for a in alerts):
        return "ALERTA"
    if alerts:
        return "ALERTA TEMPRANA"
    return "REVISIÓN" if reviews else "NO REGISTRADO"


def summarize(rows: list[dict], key_alerts: str, key_reviews: str) -> dict:
    c = Counter(nivel(r[key_alerts], r[key_reviews]) for r in rows)
    good = sum(a >= -TOL for r in rows for a in r[key_alerts])
    early = sum(a < -TOL for r in rows for a in r[key_alerts])
    n = len(rows)
    return {"n": n, **{k: c[k] for k in ("ALERTA", "ALERTA TEMPRANA", "REVISIÓN", "NO REGISTRADO")},
            "sensibilidad_alerta": round(c["ALERTA"] / n, 3) if n else None, "alertas_correctas": good, "alertas_tempranas": early,
            "precision_alerta": round(good / (good + early), 3) if good + early else None}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--variantes", nargs="+", default=list(PROMPTS))
    parser.add_argument("--antes", type=float, default=4.0, help="segundos del tramo antes del instante del pendiente")
    parser.add_argument("--despues", type=float, default=0.0, help="segundos después (0: no espera fotogramas posteriores)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    logging.disable(logging.CRITICAL)
    import evaluate_classifier_events as ece
    import evaluate_on_videos as ev
    from evaluate_tiers import events_of
    from ultralytics import YOLO

    weights = APP / "data" / "models" / "yolo" / "baseline_v6.pt"
    wid = f"{weights.name}-{weights.stat().st_size}-{int(weights.stat().st_mtime)}"
    ev.IMGSZ[0] = 960
    detector = YOLO(str(weights))
    cache = APP / "data" / "eval_cache"

    def frames_of(video: Path) -> list[dict]:
        return ev.detect_video(detector, video, FPS, cache, wid)[0]

    meta = {Path(r["path"]).stem: r for r in csv.DictReader(open(ece.SOURCE / "metadata-real.csv", encoding="utf-8"))}
    stems = sorted(ece.split_stems(args.split))
    choques = []
    for stem in stems:
        video = ece.SOURCE / meta[stem]["path"]
        t0 = float(meta[stem]["accident_time"])
        evs = events_of(frames_of(video))
        choques.append({"video": stem, "ruta": str(video), "t0": t0,
                        "alertas_abs": [t for t, r in evs if r == "accident"], "pendientes_abs": [t for t, r in evs if r == "posible_choque"]})
    controles = []
    if args.split == "test":
        for video in [ece.V7] + sorted(DETRAC.glob("*.mp4")):
            frames = frames_of(video)
            evs = events_of(frames)
            controles.append({"video": video.stem, "ruta": str(video), "segundos": frames[-1]["t"] if frames else 0.0,
                              "grupo": "V7" if video == ece.V7 else "UA-DETRAC",
                              "alertas_abs": [t for t, r in evs if r == "accident"], "pendientes_abs": [t for t, r in evs if r == "posible_choque"]})
    n_pend = sum(len(v["pendientes_abs"]) for v in choques + controles)
    print(f"{args.split}: {len(choques)} videos de choque, {len(controles)} de control, {n_pend} pendientes a verificar", flush=True)

    verifier = Verifier()
    for _ in range(2):
        verifier.ask(window(Path(choques[0]["ruta"]), 5.0, args.antes, args.despues), PROMPTS[args.variantes[0]])

    report = {"split": args.split, "modelo": VLM, "cuadros": N_FRAMES, "antes_s": args.antes, "despues_s": args.despues,
              "prompts": {k: PROMPTS[k] for k in args.variantes}, "variantes": {}}
    for var in args.variantes:
        lat = []
        for item in choques + controles:
            item.setdefault("respuestas", {})[var] = []
            for t in item["pendientes_abs"]:
                raw, secs = verifier.ask(window(Path(item["ruta"]), t, args.antes, args.despues), PROMPTS[var])
                lat.append(secs)
                item["respuestas"][var].append({"t": t, "respuesta": raw, "si": is_yes(raw), "segundos": round(secs, 3)})
        rows = []
        for v in choques:
            yes = [a["t"] for a in v["respuestas"][var] if a["si"]]
            rows.append({"base_alertas": [a - v["t0"] for a in v["alertas_abs"]], "base_pend": [p - v["t0"] for p in v["pendientes_abs"]],
                         "vlm_alertas": [a - v["t0"] for a in v["alertas_abs"] + yes],
                         "vlm_pend": [p - v["t0"] for p in v["pendientes_abs"] if p not in yes]})
        resumen = {"base": summarize(rows, "base_alertas", "base_pend"), "con_vlm": summarize(rows, "vlm_alertas", "vlm_pend"),
                   "si_en_pendientes_de_choque": sum(a["si"] for v in choques for a in v["respuestas"][var]),
                   "pendientes_de_choque": sum(len(v["pendientes_abs"]) for v in choques),
                   "demora_verificacion_s": {"mediana": round(statistics.median(lat), 3), "p95": round(sorted(lat)[int(0.95 * (len(lat) - 1))], 3),
                                             "maximo": round(max(lat), 3)} if lat else None}
        for grupo in ("V7", "UA-DETRAC"):
            items = [c for c in controles if c["grupo"] == grupo]
            if not items:
                continue
            h = sum(c["segundos"] for c in items) / 3600
            a0 = sum(len(c["alertas_abs"]) for c in items)
            p0 = sum(len(c["pendientes_abs"]) for c in items)
            esc = sum(a["si"] for c in items for a in c["respuestas"][var])
            resumen[grupo] = {"minutos": round(h * 60, 1), "alertas_base": a0, "alertas_con_vlm": a0 + esc, "alertas_por_h_base": round(a0 / h, 1),
                              "alertas_por_h_con_vlm": round((a0 + esc) / h, 1), "pendientes_base": p0, "pendientes_con_vlm": p0 - esc}
        report["variantes"][var] = resumen
        print(var, json.dumps(resumen, ensure_ascii=False), flush=True)
    report["videos"] = [{k: v for k, v in item.items() if k != "ruta"} for item in choques + controles]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
