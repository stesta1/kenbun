from __future__ import annotations

import yaml

from monitoreo_urbano.common.config import settings
from monitoreo_urbano.common.schemas import CameraInfo


def load_camera_catalog(path=None) -> dict[str, CameraInfo]:
    catalog_path = path or settings.cameras_config_path
    if not catalog_path.exists():
        raise FileNotFoundError(
            f"No existe el catálogo de cámaras en {catalog_path}. Copiá config/cameras.example.yaml."
        )

    with open(catalog_path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    cameras: dict[str, CameraInfo] = {}
    for entry in raw.get("cameras", []):
        info = CameraInfo(**entry)
        cameras[info.camera_id] = info
    return cameras


def get_camera(camera_id: str, path=None) -> CameraInfo:
    catalog = load_camera_catalog(path)
    if camera_id not in catalog:
        raise KeyError(f"Cámara '{camera_id}' no está en el catálogo ({path or settings.cameras_config_path})")
    return catalog[camera_id]
