from __future__ import annotations

from fastapi import Header, HTTPException
from pydantic import BaseModel, ConfigDict

from monitoreo_urbano.common.auth import Operator, identify


def current_operator(x_api_key: str | None = Header(default=None)) -> Operator:
    operator = identify(x_api_key)
    if operator is None:
        raise HTTPException(status_code=401, detail="Clave de acceso inválida o ausente")
    return operator


class MediaMTXAuthRequest(BaseModel):

    model_config = ConfigDict(extra="ignore")

    user: str | None = None
    password: str | None = None
    token: str | None = None
    ip: str | None = None
    action: str | None = None
    path: str | None = None
    protocol: str | None = None


def check_mediamtx_request(request: MediaMTXAuthRequest) -> None:
    """MediaMTX acepta la conexión si esto responde 2xx.

    La clave llega como contraseña (RTSP y HLS con autenticación básica) o como token. Además de
    identificar al operador, se aplica su alcance: solo puede leer los streams de sus cámaras (el
    path de MediaMTX es el camera_id) y publicar solo la clave de servicio.
    """
    operator = identify(request.password) or identify(request.token)
    if operator is None:
        raise HTTPException(status_code=401, detail="Clave de acceso inválida o ausente")

    if operator.cameras is None:
        return
    if request.action == "publish":
        raise HTTPException(status_code=403, detail="Este operador no puede publicar video")
    camera_id = (request.path or "").split("/", 1)[0]
    if not operator.can_see(camera_id):
        raise HTTPException(status_code=403, detail="Este operador no tiene acceso a esa cámara")
