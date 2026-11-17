from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from monitoreo_urbano.common.config import settings

ALL_CAMERAS = "*"
DEFAULT_INSTANCE_NAME = "vision-1"


class InstanceConfig(BaseModel):

    name: str
    cameras: list[str] = Field(min_length=1)
    device: str | None = None
    analysis_fps: float | None = Field(default=None, gt=0)


def load_instances(path: Path | None = None, catalog: list[str] | None = None) -> list[InstanceConfig]:
    """Lee `config/instances.yaml` y valida la asignación de cámaras.

    Si el archivo no existe, devuelve una sola instancia con todas las cámaras del catálogo. `catalog` son los
    `camera_id` existentes; si no se pasa, se lee de `config/cameras.yaml`.
    """
    instances_path = path or settings.instances_config_path
    known = catalog if catalog is not None else _catalog_ids()

    if not instances_path.exists():
        if not known:
            raise ValueError("No hay cámaras en el catálogo para armar una instancia")
        return [InstanceConfig(name=DEFAULT_INSTANCE_NAME, cameras=list(known))]

    with open(instances_path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    instances: list[InstanceConfig] = []
    for entry in raw.get("instances", []):
        entry = dict(entry)
        if entry.get("cameras") == ALL_CAMERAS:
            entry["cameras"] = list(known)
        elif isinstance(entry.get("cameras"), str):
            entry["cameras"] = [c.strip() for c in entry["cameras"].split(",") if c.strip()]
        instances.append(InstanceConfig(**entry))

    _validate(instances, known, instances_path)
    return instances


def get_instance(name: str, path: Path | None = None, catalog: list[str] | None = None) -> InstanceConfig:
    instances = load_instances(path, catalog)
    for instance in instances:
        if instance.name == name:
            return instance
    names = ", ".join(i.name for i in instances)
    raise KeyError(f"Instancia '{name}' no está definida. Disponibles: {names}")


def _validate(instances: list[InstanceConfig], known: list[str], path: Path) -> None:
    if not instances:
        raise ValueError(f"{path} no define ninguna instancia")

    names = [i.name for i in instances]
    duplicated_names = sorted({n for n in names if names.count(n) > 1})
    if duplicated_names:
        raise ValueError(f"Nombres de instancia repetidos en {path}: {', '.join(duplicated_names)}")

    owners: dict[str, str] = {}
    for instance in instances:
        for camera_id in instance.cameras:
            if camera_id not in known:
                raise ValueError(f"La instancia '{instance.name}' usa la cámara '{camera_id}', que no está en el catálogo")
            if camera_id in owners:
                raise ValueError(
                    f"La cámara '{camera_id}' está asignada a '{owners[camera_id]}' y a '{instance.name}'; "
                    "cada cámara debe tener una sola instancia"
                )
            owners[camera_id] = instance.name


def _catalog_ids() -> list[str]:
    from monitoreo_urbano.common.cameras import load_camera_catalog

    return list(load_camera_catalog())


if __name__ == "__main__":
    # Lo usa scripts/start_system.ps1: imprime un nombre de instancia por línea.
    for configured in load_instances():
        print(configured.name)
