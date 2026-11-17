"""Fase 0 del plan de detector compartido: cuántas cámaras puede sostener una instancia.

Uso:
    python scripts/benchmark_shared_detector.py --out ../reentreno/fase0_resultados.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

TARGET_SIZE = (1280, 720)
NATIVE_FPS = 30
ANALYSIS_FPS = (5, 6, 8, 10, 15, 30)


def load_source_frames(video: Path, count: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"No se pudo abrir {video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, max(total - 1, 0), count).astype(int)
    frames: list[np.ndarray] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        frames.append(_to_720p(frame))
    cap.release()
    if not frames:
        raise SystemExit("No se pudo leer ningún fotograma del video de entrada")
    return frames


def _to_720p(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    target_ratio = TARGET_SIZE[0] / TARGET_SIZE[1]
    if width / height < target_ratio:
        new_height = round(width / target_ratio)
        top = (height - new_height) // 2
        frame = frame[top : top + new_height, :]
    elif width / height > target_ratio:
        new_width = round(height * target_ratio)
        left = (width - new_width) // 2
        frame = frame[:, left : left + new_width]
    return cv2.resize(frame, TARGET_SIZE, interpolation=cv2.INTER_AREA)


def bench_detector(
    weights: Path,
    frames: list[np.ndarray],
    batch_sizes: list[int],
    iterations: int,
    warmup: int,
    imgsz: int,
    conf: float,
    half: bool,
    device: str,
    label: str,
) -> list[dict]:
    import torch
    from ultralytics import YOLO

    model = YOLO(str(weights))
    rows: list[dict] = []

    for batch_size in batch_sizes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        def make_batch(offset: int) -> list[np.ndarray]:
            return [frames[(offset + k) % len(frames)] for k in range(batch_size)]

        for step in range(warmup):
            model.predict(make_batch(step), imgsz=imgsz, conf=conf, half=half, device=device, verbose=False)

        wall_times: list[float] = []
        speeds: list[dict] = []
        detections = 0
        cpu_start = time.process_time()
        wall_start = time.perf_counter()
        for step in range(iterations):
            batch = make_batch(step * batch_size)
            torch.cuda.synchronize()
            started = time.perf_counter()
            results = model.predict(batch, imgsz=imgsz, conf=conf, half=half, device=device, verbose=False)
            torch.cuda.synchronize()
            wall_times.append(time.perf_counter() - started)
            speeds.append(results[0].speed)
            detections += sum(len(r.boxes) for r in results)
        wall_total = time.perf_counter() - wall_start
        cpu_total = time.process_time() - cpu_start

        images = iterations * batch_size
        batch_ms = float(np.median(wall_times) * 1000)
        row = {
            "variante": label,
            "lote": batch_size,
            "ms_por_lote_mediana": round(batch_ms, 1),
            "ms_por_lote_p95": round(float(np.percentile(wall_times, 95) * 1000), 1),
            "imagenes_por_segundo": round(images / wall_total, 1),
            "ms_por_imagen_preprocesado": round(float(np.mean([s["preprocess"] for s in speeds])), 2),
            "ms_por_imagen_inferencia": round(float(np.mean([s["inference"] for s in speeds])), 2),
            "ms_por_imagen_posprocesado": round(float(np.mean([s["postprocess"] for s in speeds])), 2),
            "nucleos_cpu_del_hilo_de_inferencia": round(cpu_total / wall_total, 2),
            "vram_pico_asignada_mb": round(torch.cuda.max_memory_allocated() / 2**20),
            "vram_pico_reservada_mb": round(torch.cuda.max_memory_reserved() / 2**20),
            "detecciones_por_imagen": round(detections / images, 2),
        }
        rows.append(row)
        print(
            f"  [{label}] lote {batch_size:>3}: {row['imagenes_por_segundo']:>7.1f} img/s | "
            f"lote {row['ms_por_lote_mediana']:>6.1f} ms | pre {row['ms_por_imagen_preprocesado']:.2f} "
            f"inf {row['ms_por_imagen_inferencia']:.2f} pos {row['ms_por_imagen_posprocesado']:.2f} ms/img | "
            f"VRAM {row['vram_pico_reservada_mb']} MB",
            flush=True,
        )
    return rows


def make_test_video(source: Path, out_path: Path, seconds: int) -> None:
    command = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-stream_loop", "-1", "-i", str(source),
        "-t", str(seconds),
        "-vf", "crop='min(iw,ih*16/9)':'min(ih,iw*9/16)',scale=1280:720,fps=30,noise=alls=10:allf=t",
        "-c:v", "libx264", "-preset", "veryfast", "-g", "30",
        "-b:v", "2500k", "-maxrate", "2500k", "-bufsize", "5000k",
        "-pix_fmt", "yuv420p", "-an", str(out_path),
    ]
    subprocess.run(command, check=True)


def _decode_worker(path: Path, frames_to_decode: int, mode: str, counter: list[int], index: int) -> None:
    cap = cv2.VideoCapture(str(path))
    done = 0
    while done < frames_to_decode:
        ok = cap.read()[0] if mode == "read" else cap.grab()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        done += 1
    cap.release()
    counter[index] = done


def bench_decode(path: Path, thread_counts: list[int], frames_per_thread: int) -> list[dict]:
    rows: list[dict] = []
    for mode in ("read", "grab"):
        for threads in thread_counts:
            counter = [0] * threads
            workers = [
                threading.Thread(target=_decode_worker, args=(path, frames_per_thread, mode, counter, i))
                for i in range(threads)
            ]
            cpu_start = time.process_time()
            wall_start = time.perf_counter()
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join()
            wall = time.perf_counter() - wall_start
            cpu = time.process_time() - cpu_start
            total = sum(counter)
            row = {
                "modo": "decodificar y convertir a BGR" if mode == "read" else "solo decodificar (grab)",
                "hilos": threads,
                "fotogramas_por_segundo_total": round(total / wall, 1),
                "fotogramas_por_segundo_por_hilo": round(total / wall / threads, 1),
                "nucleos_cpu_usados": round(cpu / wall, 2),
                "camaras_a_30fps_que_sostiene": round(total / wall / NATIVE_FPS, 1),
            }
            rows.append(row)
            print(
                f"  [{row['modo']}] {threads:>2} hilos: {row['fotogramas_por_segundo_total']:>7.1f} fps | "
                f"{row['nucleos_cpu_usados']:.2f} núcleos | = {row['camaras_a_30fps_que_sostiene']:.1f} cámaras a 30 fps",
                flush=True,
            )
    return rows


def capacity_table(detector_rows: list[dict], decode_rows: list[dict], variant: str) -> list[dict]:
    candidates = [r for r in detector_rows if r["variante"] == variant]
    best = max(candidates, key=lambda r: r["imagenes_por_segundo"])
    grab = [r for r in decode_rows if r["modo"].startswith("solo")]
    best_decode = max(grab, key=lambda r: r["fotogramas_por_segundo_total"]) if grab else None
    rows = []
    for fps in ANALYSIS_FPS:
        by_gpu = best["imagenes_por_segundo"] / fps
        by_cpu = best_decode["camaras_a_30fps_que_sostiene"] if best_decode else None
        rows.append(
            {
                "fps_de_analisis": fps,
                "camaras_por_gpu": round(by_gpu, 1),
                "camaras_por_cpu_decodificando": by_cpu,
                "camaras_por_instancia": round(min(by_gpu, by_cpu), 1) if by_cpu else round(by_gpu, 1),
                "mejor_lote": best["lote"],
                "imagenes_por_segundo_gpu": best["imagenes_por_segundo"],
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, default=ROOT / "data" / "models" / "yolo" / "baseline_v1.pt")
    parser.add_argument("--video", type=Path, default=ROOT / "data" / "raw_videos" / "sample_bus.mp4")
    parser.add_argument("--batch-sizes", default="1,2,4,8,16,32")
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--device", default="0")
    parser.add_argument("--no-half", action="store_true", help="Inferencia en FP32 en lugar de FP16")
    parser.add_argument("--skip-detector", action="store_true")
    parser.add_argument("--skip-decode", action="store_true")
    parser.add_argument("--decode-threads", default="1,2,4,8,12,16")
    parser.add_argument("--decode-frames-per-thread", type=int, default=600)
    parser.add_argument("--out", type=Path, default=None, help="Ruta del JSON de resultados")
    args = parser.parse_args()

    report: dict = {"entorno": {}, "detector": [], "decodificacion": [], "capacidad": {}}

    import platform

    report["entorno"] = {
        "python": platform.python_version(),
        "sistema": platform.platform(),
        "pesos": str(args.weights),
        "imgsz": args.imgsz,
        "fp16": not args.no_half,
        "entrada": "1280x720 (video de ejemplo adaptado)",
    }
    try:
        import torch

        report["entorno"]["torch"] = torch.__version__
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            report["entorno"]["gpu"] = f"{props.name} ({props.total_memory / 2**30:.0f} GB)"
        import ultralytics

        report["entorno"]["ultralytics"] = ultralytics.__version__
    except Exception as error:  # noqa: BLE001
        print(f"Aviso: no se pudo leer el entorno de GPU: {error}")

    if not args.skip_detector:
        print("== Detección por lotes (YOLO de producción)", flush=True)
        frames = load_source_frames(args.video, count=64)
        small = [cv2.resize(f, (640, 360), interpolation=cv2.INTER_AREA) for f in frames]
        batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
        common = dict(
            weights=args.weights, batch_sizes=batch_sizes, iterations=args.iterations, warmup=args.warmup,
            imgsz=args.imgsz, conf=args.conf, half=not args.no_half, device=args.device,
        )
        report["detector"] += bench_detector(frames=frames, label="entrada 1280x720", **common)
        report["detector"] += bench_detector(frames=small, label="entrada reducida a 640x360 en la captura", **common)

    if not args.skip_decode:
        print("== Decodificación de video 720p a 30 fps", flush=True)
        with tempfile.TemporaryDirectory() as tmp:
            test_video = Path(tmp) / "test_720p30.mp4"
            make_test_video(args.video, test_video, seconds=20)
            threads = [int(x) for x in args.decode_threads.split(",")]
            report["decodificacion"] = bench_decode(test_video, threads, args.decode_frames_per_thread)

    if report["detector"]:
        for variant in sorted({r["variante"] for r in report["detector"]}):
            report["capacidad"][variant] = capacity_table(report["detector"], report["decodificacion"], variant)
        print("\n== Capacidad estimada por instancia (cámaras a 720p/30 fps)")
        for variant, rows in report["capacidad"].items():
            print(f"-- {variant}")
            for row in rows:
                print(
                    f"   análisis a {row['fps_de_analisis']:>2} fps: GPU {row['camaras_por_gpu']:>6.1f} | "
                    f"CPU {row['camaras_por_cpu_decodificando']} | instancia {row['camaras_por_instancia']}"
                )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nResultados en {args.out}")


if __name__ == "__main__":
    main()
