from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass

from monitoreo_urbano.common.config import settings
from monitoreo_urbano.common.runtime_config import RuntimeThresholds
from monitoreo_urbano.common.schemas import INCIDENT_REASONS, BoundingBox, TrackedObject

# Clases que el detector dispara por sí solas. `fight` ya no está: se quitó de los modelos por falta de datos
# reales con recuadros; una pelea llega al operador como `posible_altercado` y él la puede reclasificar.
INCIDENT_CLASSES = frozenset({"fire", "accident"})
REVIEW_REASON = "posible_choque"
_INCIDENT_DEDUP_KEYS = INCIDENT_CLASSES | {REVIEW_REASON}
VEHICLE_CLASSES = frozenset({"car", "motorcycle"})


def frames_for_duration(seconds: float, fps: float, minimum: int = 1) -> int:
    return max(minimum, round(seconds * fps))


def _bbox_center(bbox: BoundingBox) -> tuple[float, float]:
    return (bbox.x1 + bbox.x2) / 2, (bbox.y1 + bbox.y2) / 2


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _tortuosity(history: deque[TrackedObject]) -> float:
    centers = [_bbox_center(obj.bbox) for obj in history]
    if len(centers) < 2:
        return 1.0

    path_length = sum(_distance(centers[i], centers[i + 1]) for i in range(len(centers) - 1))
    net_displacement = _distance(centers[0], centers[-1])
    if net_displacement < 1.0:
        return path_length
    return path_length / net_displacement


class EventRule(ABC):
    tick: int = 0  # fotogramas procesados por el motor; lo actualiza el motor antes de cada fotograma

    @abstractmethod
    def evaluate(
        self, obj: TrackedObject, history: deque[TrackedObject], frame_tracks: list[TrackedObject]
    ) -> str | None:
        raise NotImplementedError

    def flush(self) -> list[tuple[TrackedObject, str]]:
        return []


class MinDurationRule(EventRule):

    def __init__(self, min_frames: int = 8) -> None:
        self.min_frames = min_frames

    def evaluate(
        self, obj: TrackedObject, history: deque[TrackedObject], frame_tracks: list[TrackedObject]
    ) -> str | None:
        return obj.class_name if len(history) >= self.min_frames else None


class DirectIncidentRule(EventRule):
    """Dispara un incidente (fuego, choque) cuando se acumulan suficientes detecciones suyas.

    Sin `window_frames` cuenta las observaciones del mismo seguimiento. Con `window_frames` para una clase cuenta, en
    esa ventana, los fotogramas con una detección de la clase cerca de la actual, sea cual sea el identificador de
    seguimiento: un choque aparece en ráfagas cortas que ByteTrack reparte en varios identificadores, y ninguno llega
    solo a la persistencia pedida.
    """

    def __init__(
        self,
        classes: frozenset[str] = INCIDENT_CLASSES,
        min_frames: int = 8,
        min_frames_by_class: dict[str, int] | None = None,
        window_frames: dict[str, int] | None = None,
        cluster_radius: float = 3.0,
        min_confidence_by_class: dict[str, float] | None = None,
        strong_confidence_by_class: dict[str, float] | None = None,
    ) -> None:
        self.min_confidence_by_class = min_confidence_by_class or {}
        self.strong_confidence_by_class = strong_confidence_by_class or {}
        self.classes = classes
        self.min_frames = min_frames
        self.min_frames_by_class = min_frames_by_class or {}
        self.window_frames = window_frames or {}
        self.cluster_radius = cluster_radius
        self._hits: dict[str, deque[tuple[int, float, float, float]]] = defaultdict(deque)

    def evaluate(
        self, obj: TrackedObject, history: deque[TrackedObject], frame_tracks: list[TrackedObject]
    ) -> str | None:
        if obj.class_name not in self.classes:
            return None
        # Solo cuentan para la alerta las detecciones con la confianza que ella exige; las más débiles las toma la regla de revisión.
        if obj.confidence < self.min_confidence_by_class.get(obj.class_name, 0.0):
            return None
        if obj.confidence >= self.strong_confidence_by_class.get(obj.class_name, 2.0):
            return obj.class_name
        needed = self.min_frames_by_class.get(obj.class_name, self.min_frames)
        window = self.window_frames.get(obj.class_name)
        if window is None:
            return obj.class_name if len(history) >= needed else None

        centre = _bbox_center(obj.bbox)
        size = max(obj.bbox.x2 - obj.bbox.x1, obj.bbox.y2 - obj.bbox.y1)
        hits = self._hits[obj.class_name]
        hits.append((obj.frame_index, centre[0], centre[1], size))
        while hits and hits[0][0] < obj.frame_index - window:
            hits.popleft()

        frames_with_hit = {
            frame
            for frame, x, y, other_size in hits
            if _distance((x, y), centre) <= self.cluster_radius * max(size, other_size)
        }
        return obj.class_name if len(frames_with_hit) >= needed else None


class ReviewIncidentRule(EventRule):
    """Choque con evidencia débil: `min_hits` detecciones con confianza >= `min_confidence` cerca del mismo lugar en la ventana.

    Dispara `posible_choque`, que no alerta: queda para revisión del operador. La regla de alerta, que va antes, ya tomó los casos con
    detecciones fuertes; acá llegan los que no alcanzaron su confianza o su persistencia. Como la de alerta, cuenta detecciones
    cercanas sin importar el identificador de seguimiento.
    """

    def __init__(
        self,
        class_name: str = "accident",
        reason: str = REVIEW_REASON,
        min_confidence: float = 0.25,
        min_hits: int = 2,
        window_frames: int = 12,
        cluster_radius: float = 3.0,
        hold_frames: int = 9,
    ) -> None:
        self.class_name = class_name
        self.reason = reason
        self.min_confidence = min_confidence
        self.min_hits = min_hits
        self.window_frames = window_frames
        self.cluster_radius = cluster_radius
        self.hold_frames = hold_frames
        self._hits: deque[tuple[int, float, float, float]] = deque()
        self._candidate: tuple[int, TrackedObject] | None = None  # (fotograma en que empezó la espera, última detección)

    def evaluate(
        self, obj: TrackedObject, history: deque[TrackedObject], frame_tracks: list[TrackedObject]
    ) -> str | None:
        if obj.class_name != self.class_name or obj.confidence < self.min_confidence:
            return None
        centre = _bbox_center(obj.bbox)
        size = max(obj.bbox.x2 - obj.bbox.x1, obj.bbox.y2 - obj.bbox.y1)
        self._hits.append((obj.frame_index, centre[0], centre[1], size))
        while self._hits and self._hits[0][0] < obj.frame_index - self.window_frames:
            self._hits.popleft()
        frames_with_hit = {
            frame
            for frame, x, y, other_size in self._hits
            if _distance((x, y), centre) <= self.cluster_radius * max(size, other_size)
        }
        if len(frames_with_hit) >= self.min_hits:
            # No se dispara en el momento: se espera `hold_frames` por si la evidencia crece y el incidente sale como alerta,
            # así un mismo choque no queda como alerta y además como pendiente de revisión.
            self._candidate = (self._candidate[0] if self._candidate else self.tick, obj)
        return None

    def flush(self) -> list[tuple[TrackedObject, str]]:
        if self._candidate is not None and self.tick - self._candidate[0] >= self.hold_frames:
            obj = self._candidate[1]
            self._candidate = None
            return [(obj, self.reason)]
        return []


class PersonVehicleProximityRule(EventRule):

    def __init__(self, radius_px: float = 150.0, min_frames: int = 8) -> None:
        self.radius_px = radius_px
        self.min_frames = min_frames

    def evaluate(
        self, obj: TrackedObject, history: deque[TrackedObject], frame_tracks: list[TrackedObject]
    ) -> str | None:
        if obj.class_name != "person" or len(history) < self.min_frames:
            return None

        center = _bbox_center(obj.bbox)
        for other in frame_tracks:
            if other.track_id == obj.track_id or other.class_name not in VEHICLE_CLASSES:
                continue
            if _distance(center, _bbox_center(other.bbox)) <= self.radius_px:
                return "persona_junto_a_vehiculo"
        return None


class VehicleConvergenceRule(EventRule):

    def __init__(self, radius_px: float = 100.0, min_frames: int = 8) -> None:
        self.radius_px = radius_px
        self.min_frames = min_frames

    def evaluate(
        self, obj: TrackedObject, history: deque[TrackedObject], frame_tracks: list[TrackedObject]
    ) -> str | None:
        if obj.class_name not in VEHICLE_CLASSES or len(history) < self.min_frames:
            return None

        center = _bbox_center(obj.bbox)
        for other in frame_tracks:
            if other.track_id == obj.track_id or other.class_name not in VEHICLE_CLASSES:
                continue
            if _distance(center, _bbox_center(other.bbox)) <= self.radius_px:
                return "vehiculos_muy_cerca"
        return None


class PersonAltercationRule(EventRule):
    """Dos personas cercanas y agitadas durante varios segundos seguidos. Solo para revisión del operador.

    Una persona está *agitada* si en su historial recorre mucho (en alturas de persona por segundo) y cambia
    seguido de dirección, ignorando el temblor de la caja (pasos menores a `noise_floor` alturas). Dos personas
    agitadas y a menos de `proximity_heights` alturas una de otra, de forma sostenida durante `min_pair_seconds`,
    disparan `posible_altercado`. Una persona quieta, que camina o corre derecho, o un cruce breve no disparan.

    Las distancias se miden en alturas de persona (altura de la caja) y no en píxeles, así que no dependen de la
    resolución ni de qué tan lejos esté la cámara. Los tiempos salen de los fps del motor.
    """

    MIN_HISTORY_SECONDS = 1.5

    def __init__(
        self,
        fps: float | None = None,
        proximity_heights: float | None = None,
        min_speed: float | None = None,
        tortuosity_threshold: float | None = None,
        min_pair_seconds: float | None = None,
        noise_floor: float = 0.05,
        gap_seconds: float = 1.0,
        min_frames: int = 8,
    ) -> None:
        self.fps = fps or 30.0
        self.proximity_heights = proximity_heights if proximity_heights is not None else settings.altercation_proximity_heights
        self.min_speed = min_speed if min_speed is not None else settings.altercation_min_speed
        self.tortuosity_threshold = (
            tortuosity_threshold if tortuosity_threshold is not None else settings.altercation_tortuosity
        )
        self.min_pair_seconds = min_pair_seconds if min_pair_seconds is not None else settings.altercation_min_seconds
        self.noise_floor = noise_floor
        self.gap_frames = max(1, round(gap_seconds * self.fps))
        self.min_frames = min_frames
        self._agitated_at: dict[int, int] = {}
        self._pairs: dict[tuple[int, int], list[int]] = {}

    def _agitated(self, history: deque[TrackedObject]) -> bool:
        if len(history) < max(2, self.min_frames):
            return False
        elapsed = (history[-1].frame_index - history[0].frame_index) / self.fps
        if elapsed < self.MIN_HISTORY_SECONDS:
            return False

        heights = sorted(obj.bbox.y2 - obj.bbox.y1 for obj in history)
        height = heights[len(heights) // 2]
        if height <= 0:
            return False

        centers = [_bbox_center(obj.bbox) for obj in history]
        path = 0.0
        for a, b in zip(centers, centers[1:]):
            step = _distance(a, b)
            if step >= self.noise_floor * height:
                path += step
        speed = path / height / elapsed
        net = _distance(centers[0], centers[-1])
        tortuosity = path / max(net, 0.25 * height)
        return speed >= self.min_speed and tortuosity >= self.tortuosity_threshold

    def evaluate(
        self, obj: TrackedObject, history: deque[TrackedObject], frame_tracks: list[TrackedObject]
    ) -> str | None:
        if obj.class_name != "person":
            return None

        frame = obj.frame_index
        if not self._agitated(history):
            self._agitated_at.pop(obj.track_id, None)
            return None
        self._agitated_at[obj.track_id] = frame

        center = _bbox_center(obj.bbox)
        height = obj.bbox.y2 - obj.bbox.y1
        for other in frame_tracks:
            if other.track_id == obj.track_id or other.class_name != "person":
                continue
            if frame - self._agitated_at.get(other.track_id, -10**9) > self.gap_frames:
                continue
            reference = max(height, other.bbox.y2 - other.bbox.y1)
            if _distance(center, _bbox_center(other.bbox)) > self.proximity_heights * reference:
                continue

            key = (min(obj.track_id, other.track_id), max(obj.track_id, other.track_id))
            span = self._pairs.get(key)
            if span is None or frame - span[1] > self.gap_frames:
                span = [frame, frame]
                self._pairs[key] = span
            span[1] = frame
            if (span[1] - span[0]) / self.fps >= self.min_pair_seconds:
                return "posible_altercado"

        self._forget_old(frame)
        return None

    def _forget_old(self, frame: int) -> None:
        horizon = frame - 10 * self.gap_frames
        for key in [k for k, span in self._pairs.items() if span[1] < horizon]:
            del self._pairs[key]
        for track_id in [t for t, seen in self._agitated_at.items() if seen < horizon]:
            del self._agitated_at[track_id]


def default_rules(fps: float | None = None) -> list[EventRule]:
    if fps is None:
        return [
            DirectIncidentRule(),
            PersonAltercationRule(),
        ]

    min_frames = frames_for_duration(settings.trigger_min_seconds, fps, minimum=settings.trigger_min_samples)
    by_class = {
        name: frames_for_duration(seconds, fps, minimum=settings.trigger_min_samples)
        for name, seconds in settings.incident_min_seconds.items()
    }
    window = frames_for_duration(settings.incident_window_seconds, fps)
    windows = {name: max(window, 3 * frames) for name, frames in by_class.items()}
    rules: list[EventRule] = [
        DirectIncidentRule(
            min_frames=min_frames,
            min_frames_by_class=by_class,
            window_frames=windows,
            cluster_radius=settings.incident_cluster_radius,
            min_confidence_by_class=settings.incident_alert_confidence,
            strong_confidence_by_class=settings.incident_strong_confidence,
        )
    ]
    if settings.accident_review_enabled:
        rules.append(
            ReviewIncidentRule(
                min_confidence=settings.incident_review_confidence,
                min_hits=settings.incident_review_min_hits,
                window_frames=windows.get("accident", window),
                cluster_radius=settings.incident_cluster_radius,
                hold_frames=frames_for_duration(settings.incident_review_hold_seconds, fps),
            )
        )
    rules.append(PersonAltercationRule(fps=fps, min_frames=min_frames))
    return rules


@dataclass
class _RecentTrigger:
    class_name: str
    center_x: float
    center_y: float
    monotonic_time: float
    size: float = 0.0


class EventTriggerEngine:
    def __init__(
        self,
        rules: list[EventRule] | None = None,
        history_size: int | None = None,
        stale_after_frames: int | None = None,
        dedup_radius_px: float = 250.0,
        dedup_window_seconds: float = 90.0,
        thresholds: RuntimeThresholds | None = None,
        fps: float | None = None,
    ) -> None:
        """`fps` es la frecuencia a la que el motor recibe fotogramas. Si se indica, el historial, la
        ausencia tolerada y la persistencia de las reglas por defecto salen de segundos
        (`settings.trigger_*_seconds`); sin `fps` valen los 150 y 60 fotogramas históricos. Un valor
        explícito de `history_size` o `stale_after_frames` siempre manda."""
        self.rules = rules if rules is not None else default_rules(fps)
        if history_size is None:
            history_size = (
                frames_for_duration(settings.trigger_history_seconds, fps, minimum=settings.trigger_min_samples)
                if fps is not None
                else 150
            )
        if stale_after_frames is None:
            stale_after_frames = (
                frames_for_duration(settings.trigger_stale_after_seconds, fps) if fps is not None else 60
            )
        self.history_size = history_size
        self.stale_after_frames = stale_after_frames
        self.thresholds = thresholds
        self._static_dedup_radius_px = dedup_radius_px
        self._static_dedup_window_seconds = dedup_window_seconds

        self._history: dict[int, deque[TrackedObject]] = defaultdict(lambda: deque(maxlen=self.history_size))
        self._already_triggered: set[int] = set()
        self._missing_streak: dict[int, int] = defaultdict(int)
        self._recent_triggers: list[_RecentTrigger] = []
        self._trigger_reasons: dict[int, str] = {}
        self._tick = 0

    @property
    def dedup_radius_px(self) -> float:
        return self.thresholds.event_dedup_radius_px if self.thresholds else self._static_dedup_radius_px

    @property
    def dedup_window_seconds(self) -> float:
        return self.thresholds.event_dedup_window_seconds if self.thresholds else self._static_dedup_window_seconds

    def update(self, tracked_objects: list[TrackedObject]) -> list[int]:
        triggered: list[int] = []
        seen_ids = {obj.track_id for obj in tracked_objects}
        self._tick += 1
        for rule in self.rules:
            rule.tick = self._tick

        for obj in tracked_objects:
            self._missing_streak[obj.track_id] = 0
            self._history[obj.track_id].append(obj)

            if obj.track_id in self._already_triggered:
                continue

            reason = self._first_matching_reason(obj, self._history[obj.track_id], tracked_objects)
            if reason is None:
                continue
            self._register(obj, reason, triggered)

        for rule in self.rules:
            for obj, reason in rule.flush():
                if obj.track_id not in self._already_triggered:
                    self._register(obj, reason, triggered)

        for track_id in list(self._history):
            if track_id in seen_ids:
                continue
            self._missing_streak[track_id] += 1
            if self._missing_streak[track_id] >= self.stale_after_frames:
                self._history.pop(track_id, None)
                self._already_triggered.discard(track_id)
                self._missing_streak.pop(track_id, None)
                self._trigger_reasons.pop(track_id, None)

        return triggered

    def _register(self, obj: TrackedObject, reason: str, triggered: list[int]) -> None:
        self._already_triggered.add(obj.track_id)

        now = time.monotonic()
        self._forget_expired_triggers(now)
        center_x, center_y = _bbox_center(obj.bbox)

        size = max(obj.bbox.x2 - obj.bbox.x1, obj.bbox.y2 - obj.bbox.y1)
        # El aviso de revisión se deduplica aparte de la alerta, para que uno débil no tape a uno fuerte; y no sale si el
        # mismo incidente ya salió como alerta.
        key = REVIEW_REASON if reason == REVIEW_REASON else obj.class_name
        if self._has_recent_trigger_nearby(key, center_x, center_y, now, size):
            return
        if reason == REVIEW_REASON and self._has_recent_trigger_nearby(obj.class_name, center_x, center_y, now, size):
            return

        self._recent_triggers.append(
            _RecentTrigger(class_name=key, center_x=center_x, center_y=center_y, monotonic_time=now, size=size)
        )
        self._trigger_reasons[obj.track_id] = reason
        triggered.append(obj.track_id)

    def _first_matching_reason(
        self, obj: TrackedObject, history: deque[TrackedObject], frame_tracks: list[TrackedObject]
    ) -> str | None:
        for rule in self.rules:
            reason = rule.evaluate(obj, history, frame_tracks)
            if reason is not None:
                return reason
        return None

    def _forget_expired_triggers(self, now: float) -> None:
        self._recent_triggers = [
            t for t in self._recent_triggers if (now - t.monotonic_time) < self.dedup_window_seconds
        ]

    def _has_recent_trigger_nearby(
        self, class_name: str, center_x: float, center_y: float, now: float | None = None, size: float = 0.0
    ) -> bool:
        for t in self._recent_triggers:
            if t.class_name != class_name:
                continue
            if class_name in _INCIDENT_DEDUP_KEYS:
                mode = settings.incident_dedup_mode
                if mode == "ubicacion":
                    reach = settings.incident_cluster_radius * max(size, t.size, 1.0)
                    if _distance((center_x, center_y), (t.center_x, t.center_y)) > reach:
                        continue
                if mode != "global" and now is not None:
                    t.monotonic_time = now  # la escena sigue mostrando el incidente: se renueva la espera
                return True
            distance = math.hypot(center_x - t.center_x, center_y - t.center_y)
            if distance <= self.dedup_radius_px:
                return True
        return False

    def history_for(self, track_id: int) -> deque[TrackedObject]:
        return self._history[track_id]

    def reason_for(self, track_id: int) -> str | None:
        return self._trigger_reasons.get(track_id)
