from __future__ import annotations

import hmac
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote, urlsplit, urlunsplit

import yaml

from monitoreo_urbano.common.config import settings


SERVICE_OPERATOR_NAME = "servicio"
OPEN_OPERATOR_NAME = "sin-autenticar"


@dataclass(frozen=True)
class Operator:
    name: str
    cameras: frozenset[str] | None = None

    def can_see(self, camera_id: str) -> bool:
        return self.cameras is None or camera_id in self.cameras

    def restrict(self, requested: frozenset[str] | None) -> frozenset[str] | None:
        if self.cameras is None:
            return requested
        return self.cameras if requested is None else self.cameras & requested


def load_operators(path: Path | None = None) -> list[tuple[str, Operator]]:
    config_path = path or settings.operators_config_path
    if not config_path.exists():
        return []
    with open(config_path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    operators: list[tuple[str, Operator]] = []
    names: set[str] = set()
    for entry in raw.get("operators") or []:
        name = str(entry.get("name", "")).strip()
        key = str(entry.get("key", ""))
        if not name or not key:
            raise ValueError(f"Cada operador de {config_path} necesita 'name' y 'key': {entry!r}")
        if name in names or name in (SERVICE_OPERATOR_NAME, OPEN_OPERATOR_NAME):
            raise ValueError(f"Nombre de operador repetido o reservado en {config_path}: {name!r}")
        names.add(name)
        cameras = entry.get("cameras")
        operators.append((key, Operator(name, None if cameras is None else frozenset(map(str, cameras)))))
    return operators


def auth_enabled() -> bool:
    return bool(settings.api_key) or bool(load_operators())


def identify(candidate: str | None) -> Operator | None:
    if not auth_enabled():
        return Operator(OPEN_OPERATOR_NAME)
    if not candidate:
        return None

    encoded = candidate.encode("utf-8")
    found: Operator | None = None
    for key, operator in load_operators():
        if hmac.compare_digest(encoded, key.encode("utf-8")):
            found = operator
    if settings.api_key and hmac.compare_digest(encoded, settings.api_key.encode("utf-8")):
        found = Operator(SERVICE_OPERATOR_NAME)
    return found


def is_valid_key(candidate: str | None) -> bool:
    return identify(candidate) is not None


def with_stream_credentials(url: str) -> str:
    """Agrega usuario y clave de servicio a una URL RTSP para presentarse ante MediaMTX.

    Se hace al abrir la conexión y no en el catálogo de cámaras, así la clave no queda guardada
    en la base de datos ni viaja en GET /cameras. No toca URLs que ya traen credenciales ni
    fuentes que no son RTSP.
    """
    if not settings.api_key:
        return url
    parts = urlsplit(url)
    if parts.scheme not in ("rtsp", "rtsps") or "@" in parts.netloc:
        return url
    credentials = f"{quote(settings.stream_user, safe='')}:{quote(settings.api_key, safe='')}"
    return urlunsplit(parts._replace(netloc=f"{credentials}@{parts.netloc}"))
