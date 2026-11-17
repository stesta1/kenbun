"""Genera el corpus de la prueba de fidelidad y consulta con un incidente real por video (apartados 5.5 y 5.6 de la tesis).

Uso:
    python scripts/generate_corpus_choques.py --out ../reentreno/medicion/corrida_choques.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from monitoreo_urbano.common.auth import with_stream_credentials  # noqa: E402

VIDEOS = ROOT.parent / "test_ui" / "videos"
CHOQUES = VIDEOS / "choques"
LEAD_IN = 8.0
W, H, FPS = 960, 720, 15
CAMERAS = ["lat-cam-1", "lat-cam-2"]


def ffmpeg_cmd(camera: str, video: Path, hold: float, sin_transito: bool = False) -> list[str]:
    fit = f"scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={FPS},setpts=PTS-STARTPTS"
    fc = (
        f"[0:v]setsar=1,setpts=PTS-STARTPTS[a];[1:v]{fit}[b];[2:v]{fit}[c];[a][b][c]concat=n=3:v=1:a=0[o]"
    )
    inputs = ["-re", "-i", str(video), "-re", "-stream_loop", "-1", "-t", str(hold), "-i", str(VIDEOS / "V7.mp4")]
    if sin_transito:
        inputs = ["-re", "-i", str(video)]
        fc = f"[0:v]setsar=1,setpts=PTS-STARTPTS[a];[1:v]{fit}[b];[a][b]concat=n=2:v=1:a=0[o]"
    return [
        "ffmpeg", "-loglevel", "error",
        "-re", "-f", "lavfi", "-t", str(LEAD_IN), "-i", f"color=c=black:s={W}x{H}:r={FPS}",
        *inputs,
        "-filter_complex", fc, "-map", "[o]",
        "-c:v", "libx264", "-preset", "veryfast", "-tune", "zerolatency", "-pix_fmt", "yuv420p",
        "-f", "rtsp", "-rtsp_transport", "tcp", with_stream_credentials(f"rtsp://localhost:8554/{camera}"),
    ]


def duration(path: Path) -> float:
    cap = cv2.VideoCapture(str(path))
    value = cap.get(cv2.CAP_PROP_FRAME_COUNT) / (cap.get(cv2.CAP_PROP_FPS) or 30.0)
    cap.release()
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip-post", type=float, default=30.0, help="Debe coincidir con MU_CLIP_POST_EVENT_SECONDS del módulo 1")
    parser.add_argument("--gap", type=float, default=8.0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0, help="Solo los primeros N videos (prueba de humo)")
    parser.add_argument("--skip", type=int, default=0, help="Omite los primeros N videos (corridas de complemento)")
    parser.add_argument("--no-fire", action="store_true", help="No publica el incendio al final")
    parser.add_argument("--carpeta", type=Path, default=CHOQUES, help="Carpeta con los videos de choque")
    parser.add_argument("--sin-transito", action="store_true", help="No agrega tránsito normal (V7) después del video; la señal se corta al terminar")
    parser.add_argument("--patron", default="C*.mp4", help="Patrón de los videos dentro de la carpeta")
    args = parser.parse_args()

    choques = sorted(p for p in args.carpeta.glob(args.patron))
    choques = choques[args.skip:]
    if args.limit:
        choques = choques[: args.limit]
    plan = [tuple(choques[i:i + 2]) for i in range(0, len(choques), 2)]
    if not args.no_fire:
        plan.append((VIDEOS / "V6.mp4",))
    hold = args.clip_post + 15
    if args.sin_transito:
        hold = 14.0
    runs: list[dict] = []
    for n, combo in enumerate(plan, 1):
        t_launch = time.time()
        procs = [subprocess.Popen(ffmpeg_cmd(cam, video, hold, args.sin_transito)) for cam, video in zip(CAMERAS, combo)]
        longest = max(duration(v) for v in combo)
        for cam, video in zip(CAMERAS, combo):
            runs.append({
                "corrida": n, "camara": cam, "video": video.stem,
                "t_launch_utc": datetime.fromtimestamp(t_launch, timezone.utc).isoformat(timespec="seconds"),
                "t_fin_utc": datetime.fromtimestamp(t_launch + LEAD_IN + longest + hold + 4, timezone.utc).isoformat(timespec="seconds"),
            })
        print(f"corrida {n}/{len(plan)}: {[v.stem for v in combo]}", flush=True)
        time.sleep(max(0, t_launch + LEAD_IN + longest + hold + 4 - time.time()))
        for proc in procs:
            proc.terminate()
        time.sleep(args.gap)
        args.out.write_text(json.dumps({"fecha": datetime.now().isoformat(timespec="seconds"), "clip_post_s": args.clip_post, "corridas": runs}, indent=2, ensure_ascii=False), encoding="utf-8")
    print("listo", flush=True)


if __name__ == "__main__":
    main()
