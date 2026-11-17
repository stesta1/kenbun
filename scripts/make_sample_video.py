from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from monitoreo_urbano.common.config import settings  # noqa: E402

_SOURCE_IMAGE = Path(__file__).resolve().parents[1] / ".venv/Lib/site-packages/ultralytics/assets/bus.jpg"
_OUTPUT_PATH = settings.raw_videos_dir / "sample_bus.mp4"
_DURATION_SECONDS = 30
_FPS = 15
_OUTPUT_SIZE = (960, 720)


def main() -> None:
    if not _SOURCE_IMAGE.exists():
        raise FileNotFoundError(f"No se encontró {_SOURCE_IMAGE} — ¿está instalado ultralytics en este venv?")

    image = cv2.imread(str(_SOURCE_IMAGE))
    h, w = image.shape[:2]

    settings.raw_videos_dir.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(_OUTPUT_PATH), fourcc, _FPS, _OUTPUT_SIZE)

    total_frames = _DURATION_SECONDS * _FPS
    min_zoom, max_zoom = 1.0, 1.35

    for i in range(total_frames):
        phase = (i / total_frames) * 2 * np.pi
        zoom = min_zoom + (max_zoom - min_zoom) * (0.5 - 0.5 * np.cos(phase))

        crop_w, crop_h = int(w / zoom), int(h / zoom)
        x0 = (w - crop_w) // 2 + int(0.05 * w * np.sin(phase))
        y0 = (h - crop_h) // 2
        x0, y0 = max(0, min(x0, w - crop_w)), max(0, min(y0, h - crop_h))

        crop = image[y0 : y0 + crop_h, x0 : x0 + crop_w]
        frame = cv2.resize(crop, _OUTPUT_SIZE)
        writer.write(frame)

    writer.release()
    print(f"Video de ejemplo escrito en {_OUTPUT_PATH} ({_DURATION_SECONDS}s a {_FPS} fps)")


if __name__ == "__main__":
    main()
