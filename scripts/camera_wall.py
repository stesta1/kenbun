from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2  # noqa: E402

from monitoreo_urbano.common.cameras import load_camera_catalog  # noqa: E402
from monitoreo_urbano.vision_pipeline.visualizer import load_preview, tile_images  # noqa: E402

_REFRESH_SECONDS = 1 / 10
_WINDOW_NAME = "Cámaras"


def main() -> None:
    cameras = load_camera_catalog()
    if not cameras:
        print("No hay cámaras en config/cameras.yaml — nada para mostrar.")
        return

    print(f"Mostrando {len(cameras)} cámara(s): {', '.join(cameras)}. 'q' en la ventana para salir.")
    try:
        while True:
            try:
                tiles = [
                    (f"{cam.camera_id} — {cam.label}", load_preview(cam.camera_id)) for cam in cameras.values()
                ]
                grid = tile_images(tiles)
                cv2.imshow(_WINDOW_NAME, grid)
            except Exception as exc:
                print(f"Aviso: no se pudo actualizar la grilla ({exc})")

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            time.sleep(_REFRESH_SECONDS)
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
