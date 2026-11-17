from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from datetime import timedelta

from fastapi import APIRouter, Depends, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from monitoreo_urbano.common.auth import Operator, identify
from monitoreo_urbano.common.config import PROJECT_ROOT, settings
from monitoreo_urbano.common.db import get_pool
from monitoreo_urbano.common.logging import get_logger
from monitoreo_urbano.common.object_storage import get_clips_client
from monitoreo_urbano.common.repository import (
    correct_description,
    set_category,
    delete_runtime_override,
    get_all_cameras,
    get_camera_health,
    get_by_id,
    get_events_pending_review,
    get_runtime_overrides,
    set_event_details,
    set_review,
    set_runtime_override,
)
from monitoreo_urbano.common.runtime_config import ADJUSTABLE_THRESHOLD_KEYS
from monitoreo_urbano.common.schemas import (
    CameraHealth,
    CameraInfo,
    CategoryUpdate,
    SELECTABLE_CATEGORIES,
    VEHICLE_REASONS,
    ChatRequest,
    ChatResponse,
    DescriptionCorrection,
    EventDetails,
    EventRecord,
    EventStatus,
    ReviewStatus,
    ReviewUpdate,
    RuntimeConfigResponse,
    RuntimeConfigUpdate,
    RuntimeConfigValue,
)
from monitoreo_urbano.event_store.hybrid_search import EventStore
from monitoreo_urbano.query_service.alert_relay import ConnectionManager, run_alert_poll_loop
from monitoreo_urbano.query_service.auth import MediaMTXAuthRequest, check_mediamtx_request, current_operator
from monitoreo_urbano.query_service.health_relay import run_health_poll_loop
from monitoreo_urbano.query_service.llm_client import OpenAICompatibleLLMClient
from monitoreo_urbano.query_service.rag_pipeline import RAGPipeline

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pool = get_pool()
    app.state.store = EventStore(pool=app.state.pool)
    app.state.llm_client = OpenAICompatibleLLMClient()
    app.state.rag_pipeline = RAGPipeline(app.state.store, app.state.llm_client)
    app.state.connection_manager = ConnectionManager()
    app.state.alert_task = asyncio.create_task(run_alert_poll_loop(app.state.connection_manager))
    app.state.health_manager = ConnectionManager()
    app.state.health_task = asyncio.create_task(run_health_poll_loop(app.state.health_manager))

    logger.info("query_service arriba en %s:%s", settings.api_host, settings.api_port)
    yield

    app.state.alert_task.cancel()
    app.state.health_task.cancel()
    app.state.store.close()
    app.state.llm_client.close()


app = FastAPI(title="Monitoreo Urbano — Módulo 3b (consulta + alertas)", lifespan=lifespan)

protected = APIRouter(dependencies=[Depends(current_operator)])

_WS_AUTH_TIMEOUT_SECONDS = 5.0


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/ui/")


def _visible_event(conn, event_id: str, operator: Operator) -> EventRecord:
    event = get_by_id(conn, event_id)
    if event is None or not operator.can_see(event.camera_id):
        raise HTTPException(status_code=404, detail="Evento no encontrado")
    return event


def _require_clip_access(clip_path: str, operator: Operator) -> None:
    if not operator.can_see(clip_path.split("/", 1)[0]):
        raise HTTPException(status_code=403, detail="Este operador no tiene acceso a esa cámara")


def _require_full_access(operator: Operator) -> None:
    if operator.cameras is not None:
        raise HTTPException(status_code=403, detail="Solo un operador con acceso a todas las cámaras puede hacer esto")


def _presigned_clip_url(clip_path: str) -> str:
    client = get_clips_client()
    return client.presigned_get_object(
        settings.s3_bucket_clips, clip_path, expires=timedelta(minutes=settings.s3_presigned_url_expiry_minutes)
    )


@protected.get("/clips/{clip_path:path}")
def get_clip(clip_path: str, operator: Operator = Depends(current_operator)) -> RedirectResponse:
    _require_clip_access(clip_path, operator)
    return RedirectResponse(url=_presigned_clip_url(clip_path))


@protected.get("/clip-url/{clip_path:path}")
def get_clip_url(clip_path: str, operator: Operator = Depends(current_operator)) -> dict[str, str]:
    _require_clip_access(clip_path, operator)
    return {"url": _presigned_clip_url(clip_path)}


@protected.get("/auth/check")
def auth_check(operator: Operator = Depends(current_operator)) -> dict:
    return {"ok": True, "operator": operator.name, "cameras": None if operator.cameras is None else sorted(operator.cameras)}


@app.post("/auth/mediamtx", include_in_schema=False)
def mediamtx_auth(request: MediaMTXAuthRequest) -> dict[str, bool]:
    check_mediamtx_request(request)
    return {"ok": True}


@protected.get("/cameras", response_model=list[CameraInfo])
def get_cameras(operator: Operator = Depends(current_operator)) -> list[CameraInfo]:
    with app.state.pool.connection() as conn:
        return [camera for camera in get_all_cameras(conn) if operator.can_see(camera.camera_id)]


@protected.get("/cameras/health", response_model=list[CameraHealth])
def get_cameras_health(operator: Operator = Depends(current_operator)) -> list[CameraHealth]:
    with app.state.pool.connection() as conn:
        return [health for health in get_camera_health(conn) if operator.can_see(health.camera_id)]


@protected.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, operator: Operator = Depends(current_operator)) -> ChatResponse:
    return app.state.rag_pipeline.answer(
        query=request.query,
        camera_id=request.camera_id,
        start_ts=request.start_ts,
        end_ts=request.end_ts,
        camera_ids=operator.cameras,
        history=request.history,
        excluded_event_ids=frozenset(request.excluded_event_ids),
    )


@protected.get("/events/pending-review", response_model=list[EventRecord])  # antes de /events/{event_id}: si no, "pending-review" matchea ahí
def get_pending_review_events(operator: Operator = Depends(current_operator)) -> list[EventRecord]:
    with app.state.pool.connection() as conn:
        events = get_events_pending_review(conn)
    return [event for event in events if operator.can_see(event.camera_id)]


@protected.get("/events/{event_id}", response_model=EventRecord)
def get_event(event_id: str, operator: Operator = Depends(current_operator)) -> EventRecord:
    with app.state.pool.connection() as conn:
        return _visible_event(conn, event_id, operator)


@protected.post("/events/{event_id}/review", response_model=EventRecord)
def review_event(event_id: str, update: ReviewUpdate, operator: Operator = Depends(current_operator)) -> EventRecord:
    # Confirmar exige revisar el evento (descripción y, en un choque, los vehículos): va por /details. Acá solo se descarta.
    if update.review_status == ReviewStatus.CONFIRMED:
        raise HTTPException(status_code=400, detail="Un evento se confirma desde su revisión (POST /events/{id}/details)")
    with app.state.pool.connection() as conn:
        _visible_event(conn, event_id, operator)
        set_review(conn, event_id, update.review_status, operator.name)
        return get_by_id(conn, event_id)


@protected.post("/events/{event_id}/category", response_model=EventRecord)
def change_event_category(
    event_id: str, update: CategoryUpdate, operator: Operator = Depends(current_operator)
) -> EventRecord:
    if update.trigger_reason not in SELECTABLE_CATEGORIES:
        raise HTTPException(status_code=400, detail=f"Categoría inválida. Opciones: {', '.join(SELECTABLE_CATEGORIES)}")
    with app.state.pool.connection() as conn:
        event = _visible_event(conn, event_id, operator)
        if event.trigger_reason != update.trigger_reason:
            set_category(conn, event_id, update.trigger_reason, operator.name)
        return get_by_id(conn, event_id)


@protected.post("/events/{event_id}/description", response_model=EventRecord)
def correct_event_description(
    event_id: str, correction: DescriptionCorrection, operator: Operator = Depends(current_operator)
) -> EventRecord:
    with app.state.pool.connection() as conn:
        event = _visible_event(conn, event_id, operator)
        if event.description_nl is None:
            raise HTTPException(status_code=409, detail="El evento todavía no tiene una descripción para corregir")
        correct_description(conn, event_id, correction.description_nl, operator.name)
        return get_by_id(conn, event_id)


@protected.post("/events/{event_id}/details", response_model=EventRecord)
def review_event_details(
    event_id: str, details: EventDetails, operator: Operator = Depends(current_operator)
) -> EventRecord:
    """Revisión completa de un evento (alerta o posible evento): descripción final, vehículos de un choque y confirmación.

    Es la única forma de confirmar un evento; descartar sigue siendo POST /events/{id}/review. Los vehículos solo
    corresponden a las categorías de choque, donde se exige al menos uno; en las demás se ignoran.
    """
    reason = details.trigger_reason or None
    if reason is not None and reason not in SELECTABLE_CATEGORIES:
        raise HTTPException(status_code=400, detail=f"Categoría inválida. Opciones: {', '.join(SELECTABLE_CATEGORIES)}")
    description = details.description_nl.strip()
    if not description:
        raise HTTPException(status_code=422, detail="La descripción no puede estar vacía")
    with app.state.pool.connection() as conn:
        event = _visible_event(conn, event_id, operator)
        if event.review_status == ReviewStatus.DISMISSED:
            raise HTTPException(status_code=409, detail="El evento fue descartado y no se puede confirmar")
        if event.description_nl is None and event.status != EventStatus.FAILED:
            raise HTTPException(status_code=409, detail="El evento todavía no tiene una descripción para revisar")
        final_reason = reason or event.trigger_reason
        vehicles = details.vehicles if final_reason in VEHICLE_REASONS else []
        if final_reason in VEHICLE_REASONS and not vehicles:
            raise HTTPException(status_code=422, detail="Un choque necesita al menos un vehículo involucrado")
        if final_reason != event.trigger_reason:
            set_category(conn, event_id, final_reason, operator.name)
        set_event_details(conn, event_id, description, vehicles, operator.name)
        return get_by_id(conn, event_id)


@protected.get("/config/runtime", response_model=RuntimeConfigResponse)
def get_runtime_config(operator: Operator = Depends(current_operator)) -> RuntimeConfigResponse:
    _require_full_access(operator)
    with app.state.pool.connection() as conn:
        overrides = get_runtime_overrides(conn)
    values = [
        RuntimeConfigValue(
            key=key,
            value=float(overrides[key]) if key in overrides else getattr(settings, key),
            default=getattr(settings, key),
            is_override=key in overrides,
        )
        for key in ADJUSTABLE_THRESHOLD_KEYS
    ]
    return RuntimeConfigResponse(values=values)


@protected.post("/config/runtime", response_model=RuntimeConfigValue)
def set_runtime_config(
    update: RuntimeConfigUpdate, operator: Operator = Depends(current_operator)
) -> RuntimeConfigValue:
    _require_full_access(operator)
    if update.key not in ADJUSTABLE_THRESHOLD_KEYS:
        raise HTTPException(status_code=400, detail=f"'{update.key}' no es un umbral ajustable")
    with app.state.pool.connection() as conn:
        set_runtime_override(conn, update.key, update.value)
    return RuntimeConfigValue(
        key=update.key, value=update.value, default=getattr(settings, update.key), is_override=True
    )


@protected.delete("/config/runtime/{key}", response_model=RuntimeConfigValue)
def reset_runtime_config(key: str, operator: Operator = Depends(current_operator)) -> RuntimeConfigValue:
    _require_full_access(operator)
    if key not in ADJUSTABLE_THRESHOLD_KEYS:
        raise HTTPException(status_code=400, detail=f"'{key}' no es un umbral ajustable")
    with app.state.pool.connection() as conn:
        delete_runtime_override(conn, key)
    default = getattr(settings, key)
    return RuntimeConfigValue(key=key, value=default, default=default, is_override=False)


async def _authenticate_websocket(websocket: WebSocket) -> Operator | None:
    operator = identify(websocket.headers.get("x-api-key"))
    if operator is not None:
        return operator
    try:
        raw = await asyncio.wait_for(websocket.receive_text(), timeout=_WS_AUTH_TIMEOUT_SECONDS)
        candidate = json.loads(raw).get("api_key")
    except (asyncio.TimeoutError, WebSocketDisconnect, ValueError, AttributeError):
        return None
    return identify(candidate)


def _health_snapshot(allowed: frozenset[str] | None) -> list[CameraHealth]:
    with app.state.pool.connection() as conn:
        return [h for h in get_camera_health(conn) if allowed is None or h.camera_id in allowed]


async def _serve_channel(websocket: WebSocket, cameras: str | None, manager: ConnectionManager, snapshot=None) -> None:
    requested = None if cameras is None else frozenset(c for c in cameras.split(",") if c)
    await websocket.accept()
    operator = await _authenticate_websocket(websocket)
    if operator is None:
        await websocket.close(code=4401)
        return
    allowed = operator.restrict(requested)
    manager.register(websocket, allowed)
    try:
        if snapshot is not None:
            for message in await asyncio.to_thread(snapshot, allowed):
                await websocket.send_json(message.model_dump(mode="json"))
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)


@app.websocket("/alerts")
async def alerts_websocket(websocket: WebSocket, cameras: str | None = None) -> None:
    await _serve_channel(websocket, cameras, app.state.connection_manager)


@app.websocket("/camera-status")
async def camera_status_websocket(websocket: WebSocket, cameras: str | None = None) -> None:
    await _serve_channel(websocket, cameras, app.state.health_manager, snapshot=_health_snapshot)


app.include_router(protected)


class InterfazEstatica(StaticFiles):
    """Sirve la interfaz pidiendo al navegador que revalide cada archivo, para que un cambio se vea sin vaciar la caché."""

    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response


# La interfaz de operador se sirve desde el mismo origen que la API: así no hace falta CORS.
# Va al final para que ninguna ruta de la API quede tapada por el montaje.
app.mount("/ui", InterfazEstatica(directory=PROJECT_ROOT / "frontend", html=True), name="ui")
