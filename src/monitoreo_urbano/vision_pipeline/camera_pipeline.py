from __future__ import annotations

from collections.abc import Callable
import itertools
import uuid
from dataclasses import dataclass, field

from monitoreo_urbano.common.config import settings
from monitoreo_urbano.common.logging import get_logger
from monitoreo_urbano.common.runtime_config import RuntimeThresholds
from monitoreo_urbano.common.schemas import Detection, TrackedObject
from monitoreo_urbano.vision_pipeline.camera_health import CameraHealthMonitor
from monitoreo_urbano.vision_pipeline.camera_source import Frame
from monitoreo_urbano.vision_pipeline.event_trigger import EventTriggerEngine
from monitoreo_urbano.vision_pipeline.frame_buffer import FrameRingBuffer
from monitoreo_urbano.vision_pipeline.frame_encoding import EncodedFrame, stored_form
from monitoreo_urbano.vision_pipeline.tracker import ByteTrackTracker, Tracker

logger = get_logger(__name__)


@dataclass
class OpenClip:
    track_id: int
    object_classes: set[str]
    trigger_reason: str | None
    frames: list[Frame | EncodedFrame]
    frames_remaining_after_trigger: int
    confidences: list[float] = field(default_factory=list)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))


class CameraPipeline:
    """Estado y lógica de una cámara: buffer previo, salud, seguimiento, reglas de disparo y clips abiertos.

    No sabe de dónde vienen los fotogramas ni quién detecta: recibe cada fotograma con sus
    detecciones ya hechas. Tampoco toca la base de datos ni el almacenamiento: avisa por callbacks
    cuando se dispara un evento (`on_open`, para avisar sin esperar al clip), cuando un clip se cierra (`on_clip`) y cuando cambia la salud de la cámara (`on_health`).
    Esa separación permite que una misma instancia de detección mantenga una `CameraPipeline` por
    cámara y que el detector se comparta entre todas.

    Orden de uso por fotograma: `observe(frame)`, después la detección, después
    `update(frame, detecciones)`. Si la cámara no entrega señal, `on_signal_loss()`.
    """

    def __init__(
        self,
        camera_id: str,
        *,
        tracker: Tracker,
        trigger_engine: EventTriggerEngine,
        health_monitor: CameraHealthMonitor,
        buffer: FrameRingBuffer,
        post_event_frames: int,
        on_clip: Callable[[OpenClip], None],
        on_health: Callable[[str], None],
        on_open: Callable[[OpenClip], None] | None = None,
    ) -> None:
        self.camera_id = camera_id
        self._tracker = tracker
        self._trigger_engine = trigger_engine
        self._health_monitor = health_monitor
        self._buffer = buffer
        self._post_event_frames = post_event_frames
        self._on_clip = on_clip
        self._on_open = on_open
        self._on_health = on_health
        self._open_clips: dict[int, OpenClip] = {}
        self._last_observed: tuple[Frame, Frame | EncodedFrame] | None = None
        self._direct_ids = itertools.count(1_000_000_000)  # lejos de los identificadores del seguimiento

    @classmethod
    def build(
        cls,
        camera_id: str,
        *,
        fps: float,
        reconnect_delay_seconds: float,
        thresholds: RuntimeThresholds | None,
        on_clip: Callable[[OpenClip], None],
        on_health: Callable[[str], None],
        on_open: Callable[[OpenClip], None] | None = None,
    ) -> CameraPipeline:
        buffer = FrameRingBuffer(max_seconds=settings.frame_buffer_seconds, fps=fps)
        tracker = ByteTrackTracker(frame_rate=fps, track_buffer_seconds=settings.track_buffer_seconds)
        trigger_engine = EventTriggerEngine(thresholds=thresholds, fps=fps)
        health_monitor = CameraHealthMonitor(
            fps=fps,
            reconnect_delay_seconds=reconnect_delay_seconds,
            thresholds=thresholds,
        )
        return cls(
            camera_id,
            tracker=tracker,
            trigger_engine=trigger_engine,
            health_monitor=health_monitor,
            buffer=buffer,
            post_event_frames=max(1, int(settings.clip_post_event_seconds * fps)),
            on_clip=on_clip,
            on_health=on_health,
            on_open=on_open,
        )

    @property
    def recording_track_ids(self) -> set[int]:
        return set(self._open_clips)

    def on_signal_loss(self) -> None:
        # Sin señal no llegarán más fotogramas: los clips abiertos se cierran con lo que tienen para que el evento no quede grabando.
        for track_id, clip in list(self._open_clips.items()):
            self._on_clip(clip)
            del self._open_clips[track_id]
        reason = self._health_monitor.on_signal_loss()
        if reason:
            self._on_health(reason)

    def _stored(self, frame: Frame) -> Frame | EncodedFrame:
        if self._last_observed is not None and self._last_observed[0] is frame:
            return self._last_observed[1]
        return stored_form(frame)

    def observe(self, frame: Frame) -> None:
        stored = stored_form(frame)
        self._last_observed = (frame, stored)
        self._buffer.append(stored)
        reason = self._health_monitor.check(frame)
        if reason:
            self._on_health(reason)

    def _direct_objects(self, detections: list[Detection], frame: Frame) -> list[TrackedObject]:
        return [
            TrackedObject(
                track_id=next(self._direct_ids),
                class_name=d.class_name,
                bbox=d.bbox,
                confidence=d.confidence,
                frame_index=frame.frame_index,
                timestamp=frame.timestamp,
            )
            for d in detections
        ]

    def update(self, frame: Frame, detections: list[Detection]) -> list[TrackedObject]:
        if settings.accident_direct_trigger:
            direct = [d for d in detections if d.class_name == "accident"]
            tracked_objects = self._tracker.update([d for d in detections if d.class_name != "accident"], frame)
            tracked_objects = tracked_objects + self._direct_objects(direct, frame)
        else:
            tracked_objects = self._tracker.update(detections, frame)
        triggered_ids = self._trigger_engine.update(tracked_objects)

        by_track_id = {obj.track_id: obj for obj in tracked_objects}

        for track_id in triggered_ids:
            logger.info("Evento disparado en %s: track_id=%s", self.camera_id, track_id)
            history = self._trigger_engine.history_for(track_id)
            self._open_clips[track_id] = OpenClip(
                track_id=track_id,
                object_classes={obj.class_name for obj in history},
                trigger_reason=self._trigger_engine.reason_for(track_id),
                frames=self._buffer.snapshot(),
                frames_remaining_after_trigger=self._post_event_frames,
                confidences=[obj.confidence for obj in history],
            )
            if self._on_open is not None:
                self._on_open(self._open_clips[track_id])

        for track_id, clip in list(self._open_clips.items()):
            # El fotograma que dispara el evento ya está en el buffer, y por lo tanto en el clip que se
            # acaba de abrir: no se vuelve a agregar, pero sí cuenta como el primero del tramo posterior.
            if track_id not in triggered_ids:
                clip.frames.append(self._stored(frame))
            clip.frames_remaining_after_trigger -= 1
            if track_id in by_track_id:
                clip.confidences.append(by_track_id[track_id].confidence)

            if clip.frames_remaining_after_trigger <= 0:
                self._on_clip(clip)
                del self._open_clips[track_id]

        return tracked_objects
