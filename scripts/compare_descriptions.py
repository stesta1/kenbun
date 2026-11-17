"""Genera las descripciones de los videos de choque de prueba con una variante de prompts, modelo y entrada."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from monitoreo_urbano.common.config import settings  # noqa: E402
from monitoreo_urbano.vlm_describer.describer import (  # noqa: E402
    Qwen2VLDescriber,
    _sample_frames,
    describe_without_physical_traits,
)


def read_frames(path: Path):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


CROP_MARGIN = 4.0
CROP_MIN_FRACTION = 0.4
WINDOW_BEFORE, WINDOW_AFTER = 3.0, 4.0


def detector_crop(frames, fps, detector, accident_id):
    stride = max(1, round(fps / 5))
    sampled = list(range(0, len(frames), stride))
    best = (0.0, None, None)
    for start in range(0, len(sampled), 16):
        batch = [frames[i] for i in sampled[start : start + 16]]
        for offset, res in enumerate(detector.predict(batch, imgsz=960, conf=0.25, verbose=False)):
            for conf, cls, box in zip(res.boxes.conf, res.boxes.cls, res.boxes.xyxy):
                if int(cls) == accident_id and float(conf) > best[0]:
                    best = (float(conf), sampled[start + offset], [float(x) for x in box])
    conf, peak, box = best
    if peak is None:
        return None
    height, width = frames[0].shape[:2]
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    crop_w = min(width, max((box[2] - box[0]) * CROP_MARGIN, width * CROP_MIN_FRACTION))
    crop_h = min(height, max((box[3] - box[1]) * CROP_MARGIN, height * CROP_MIN_FRACTION))
    x1 = int(min(max(cx - crop_w / 2, 0), width - crop_w))
    y1 = int(min(max(cy - crop_h / 2, 0), height - crop_h))
    first = max(0, peak - int(WINDOW_BEFORE * fps))
    last = min(len(frames), peak + int(WINDOW_AFTER * fps))
    window = frames[first:last]
    return [f[y1 : y1 + int(crop_h), x1 : x1 + int(crop_w)] for f in window], {"confianza": round(conf, 2), "pico_s": round(peak / fps, 1), "recorte": [x1, y1, int(crop_w), int(crop_h)]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--videos-dir", type=Path, default=ROOT.parent / "test_ui/videos/choques")
    parser.add_argument("--manifest", type=Path, default=ROOT.parent / "reentreno/medicion/revision_fidelidad_choques/manifiesto.json")
    parser.add_argument("--prompts-dir", type=Path, default=settings.vlm_prompts_dir)
    parser.add_argument("--model", default=settings.vlm_model_name)
    parser.add_argument("--quantization", default="", help="bnb4 para cargar el 7B sin cuantizar en 4 bits")
    parser.add_argument("--input-mode", choices=["images", "video"], default="images")
    parser.add_argument("--max-frames", type=int, default=8, help="cuadros repartidos que ve el modelo")
    parser.add_argument("--max-pixels", type=int, default=320 * 28 * 28, help="píxeles máximos por cuadro")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--detector-crop", action="store_true", help="recortar alrededor del choque que ubica el detector v6")
    parser.add_argument("--only", help="videos separados por coma")
    parser.add_argument("--from-dir", action="store_true", help="tomar los videos de la carpeta en lugar del manifiesto")
    parser.add_argument("--exclude", nargs="*", default=[], help="JSON de resultados cuyos videos no se repiten")
    parser.add_argument("--include", nargs="*", default=[], help="videos a sumar aunque estén en --exclude")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    settings.vlm_prompts_dir = args.prompts_dir
    settings.vlm_quantization = args.quantization
    settings.vlm_input_mode = args.input_mode
    settings.vlm_max_pixels = args.max_pixels
    settings.vlm_max_frames = args.max_frames
    if args.from_dir:
        used = set()
        for path in args.exclude:
            used |= {r["video"] for r in json.loads(Path(path).read_text(encoding="utf-8"))["resultados"]}
        names = sorted(p.stem for p in args.videos_dir.glob("*.mp4") if p.stem not in used)
        names += [n for n in args.include if n not in names]
        events = [{"orden": i + 1, "video": n, "camera_id": "cam", "motivo": "accident"} for i, n in enumerate(names)]
    else:
        events = json.loads(args.manifest.read_text(encoding="utf-8"))["eventos"]
    if args.only:
        events = [e for e in events if e["video"] in args.only.split(",")]
    events = events[: args.limit]
    detector = accident_id = None
    if args.detector_crop:
        from ultralytics import YOLO

        detector = YOLO(str(settings.yolo_weights_path))
        accident_id = next(i for i, n in detector.names.items() if n == "accident")

    describer = Qwen2VLDescriber(model_name=args.model)
    results = []
    for item in events:
        frames = read_frames(args.videos_dir / f"{item['video']}.mp4")
        extra = {}
        if detector is not None:
            fps = cv2.VideoCapture(str(args.videos_dir / f"{item['video']}.mp4")).get(cv2.CAP_PROP_FPS) or 25.0
            located = detector_crop(frames, fps, detector, accident_id)
            if located:
                frames, extra = located
        chosen = _sample_frames(frames, args.max_frames)
        started = time.perf_counter()
        result = describe_without_physical_traits(
            describer, chosen, camera_label=item["camera_id"], object_classes=["car", "accident"], trigger_reason=item["motivo"]
        )
        seconds = time.perf_counter() - started
        results.append(
            {
                "orden": item["orden"],
                "video": item["video"],
                "descripcion": result.description_nl,
                "atributos": result.attributes.model_dump(),
                "segundos": round(seconds, 2),
                "cuadros": len(chosen),
                "deteccion": extra,
            }
        )
        print(item["orden"], f"{seconds:.1f}s", result.description_nl[:110].replace("\n", " | "), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "etiqueta": args.label,
                "modelo": args.model,
                "entrada": args.input_mode,
                "cuadros": args.max_frames,
                "pixeles": args.max_pixels,
                "prompts": str(args.prompts_dir),
                "resultados": results,
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
