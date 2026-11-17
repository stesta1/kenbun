from __future__ import annotations

import re
from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator


INCIDENT_REASONS = frozenset({"fire", "accident", "fight"})
CAMERA_HEALTH_REASONS = frozenset({"corte_de_señal", "tapada"})
HEALTH_OK = "ok"
HEALTH_NO_ANALYSIS = "sin_analisis"  # no la escribe nadie: se deriva del latido vencido
ALERTABLE_REASONS = INCIDENT_REASONS
SELECTABLE_CATEGORIES = ("posible_altercado", "posible_choque", "fight", "accident", "fire")
VEHICLE_TYPES = {"auto": "auto", "moto": "moto", "camion": "camión", "camioneta": "camioneta", "pickup": "pickup"}
VEHICLE_REASONS = frozenset({"accident", "posible_choque"})  # categorías de choque: exigen al menos un vehículo al confirmar
MAX_VEHICLES = 12
CATEGORY_LABELS = {"fire": "incendio", "accident": "choque", "fight": "pelea", "posible_altercado": "posible altercado", "posible_choque": "posible choque", "smoke": "humo"}


class EventStatus(str, Enum):
    RECORDING = "recording"  # el aviso ya salió; el clip sigue grabándose hasta cerrar el tramo posterior
    PENDING_DESCRIPTION = "pending_description"
    PROCESSING = "processing"
    DESCRIBED = "described"
    INDEXED = "indexed"
    FAILED = "failed"


class ReviewStatus(str, Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    DISMISSED = "dismissed"


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class Detection(BaseModel):
    class_name: str
    confidence: float
    bbox: BoundingBox


class TrackedObject(BaseModel):
    track_id: int
    class_name: str
    bbox: BoundingBox
    confidence: float
    frame_index: int
    timestamp: datetime


class CameraInfo(BaseModel):
    camera_id: str
    label: str
    source: str
    comuna: str | None = None
    lat: float | None = None
    lon: float | None = None
    ubicacion: str | None = None  # calles de la cámara, ej. «Av. Santa Fe y Callao»: la usa la consulta para ubicar el hecho


class CameraHealth(BaseModel):
    camera_id: str
    camera_label: str
    status: str
    since: datetime
    updated_at: datetime


class Vehicle(BaseModel):

    vehicle_type: Literal["auto", "moto", "camion", "camioneta", "pickup"]
    color: str = Field(default="", max_length=40)

    @field_validator("color")
    @classmethod
    def _normalize_color(cls, value: str) -> str:
        return re.sub(r"\s+", " ", value).strip().lower()

    @property
    def label(self) -> str:
        return f"{VEHICLE_TYPES[self.vehicle_type]} {self.color}".strip()


class EventAttributes(BaseModel):
    vehicle_colors: list[str] = Field(default_factory=list)
    clothing_colors: list[str] = Field(default_factory=list)
    direction: str | None = None
    behavior_tags: list[str] = Field(default_factory=list)


class EventRecord(BaseModel):
    event_id: str
    camera_id: str
    camera_label: str
    track_id: int
    object_classes: list[str]
    trigger_reason: str | None = None

    start_ts: datetime
    end_ts: datetime

    clip_path: str

    status: EventStatus = EventStatus.PENDING_DESCRIPTION
    description_nl: str | None = None
    attributes: EventAttributes = Field(default_factory=EventAttributes)
    confidence: float | None = None

    review_status: ReviewStatus = ReviewStatus.PENDING
    reviewed_by: str | None = None

    description_nl_original: str | None = None
    description_edited: bool = False
    trigger_reason_original: str | None = None

    vehicles: list[Vehicle] = Field(default_factory=list)

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    described_at: datetime | None = None


class SearchResultItem(BaseModel):
    event: EventRecord
    vector_score: float | None = None
    fts_score: float | None = None
    combined_score: float


class ReviewUpdate(BaseModel):
    review_status: ReviewStatus
    reviewed_by: str | None = None


class CategoryUpdate(BaseModel):
    trigger_reason: str


class EventDetails(BaseModel):

    description_nl: str = Field(min_length=1, max_length=2000)
    trigger_reason: str | None = None
    vehicles: list[Vehicle] = Field(default_factory=list, max_length=MAX_VEHICLES)


class DescriptionCorrection(BaseModel):
    description_nl: str
    reviewed_by: str | None = None


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=4000)
    event_ids: list[str] = Field(default_factory=list, max_length=12)


class ChatRequest(BaseModel):
    query: str = Field(max_length=2000)
    camera_id: str | None = None
    start_ts: datetime | None = None
    end_ts: datetime | None = None
    history: list[ChatTurn] = Field(default_factory=list, max_length=40)
    excluded_event_ids: list[str] = Field(default_factory=list, max_length=200)


class ChatResponse(BaseModel):
    answer: str
    events: list[EventRecord]
    search_query: str | None = None
    excluded_event_ids: list[str] = Field(default_factory=list)


class RuntimeConfigValue(BaseModel):
    key: str
    value: float
    default: float
    is_override: bool


class RuntimeConfigResponse(BaseModel):
    values: list[RuntimeConfigValue]


class RuntimeConfigUpdate(BaseModel):
    key: str
    value: float


class AlertMessage(BaseModel):
    event_id: str
    camera_id: str
    camera_label: str
    timestamp: datetime
    status: EventStatus
    trigger_reason: str | None = None
    description_nl: str | None = None
