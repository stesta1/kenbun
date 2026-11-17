from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from monitoreo_urbano.common.auth import with_stream_credentials  # noqa: E402
from monitoreo_urbano.common.config import settings  # noqa: E402

_RESTART_DELAY_SECONDS = 3.0


def _build_command(source_file: Path, camera_id: str, mediamtx_host: str) -> list[str]:
    return [
        "ffmpeg",
        "-re",
        "-stream_loop", "-1",
        "-i", str(source_file),
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-pix_fmt", "yuv420p",
        "-force_key_frames", "expr:gte(t,n_forced*1)",
        "-f", "rtsp",
        "-rtsp_transport", "tcp",
        with_stream_credentials(f"rtsp://{mediamtx_host}/{camera_id}"),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simula una cámara IP republicando un video en loop hacia MediaMTX por RTSP"
    )
    parser.add_argument("--camera-id", default="cam-demo", help="Path en MediaMTX — debe coincidir con cameras.yaml")
    parser.add_argument(
        "--source-file",
        default=str(settings.raw_videos_dir / "sample_bus.mp4"),
        help="Video a republicar en loop (default: el de scripts/make_sample_video.py)",
    )
    parser.add_argument("--mediamtx-host", default="localhost:8554", help="host:puerto RTSP de MediaMTX")
    args = parser.parse_args()

    source_file = Path(args.source_file)
    if not source_file.exists():
        raise FileNotFoundError(f"No se encontró {source_file} — generalo con scripts/make_sample_video.py")

    cmd = _build_command(source_file, args.camera_id, args.mediamtx_host)
    print(f"Publicando {source_file} como cámara IP simulada en rtsp://{args.mediamtx_host}/{args.camera_id}")
    print("Ctrl+C para cortar.")

    try:
        while True:
            process = subprocess.run(cmd)
            print(f"ffmpeg terminó (code={process.returncode}), reintentando en {_RESTART_DELAY_SECONDS}s...")
            time.sleep(_RESTART_DELAY_SECONDS)
    except KeyboardInterrupt:
        print("Corte manual — cámara simulada detenida.")


if __name__ == "__main__":
    main()
