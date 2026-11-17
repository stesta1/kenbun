from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from monitoreo_urbano.common.schemas import BoundingBox, Detection, TrackedObject
from monitoreo_urbano.vision_pipeline.camera_source import Frame


class Tracker(ABC):
    @abstractmethod
    def update(self, detections: list[Detection], frame: Frame) -> list[TrackedObject]:
        raise NotImplementedError


class ByteTrackTracker(Tracker):
    def __init__(self, frame_rate: float = 30.0, track_buffer_seconds: float = 3.0) -> None:
        from boxmot.trackers.bbox import ByteTrack

        # ByteTrack ya escala el buffer por frame_rate / 30 (buffer_size = frame_rate / 30 * track_buffer),
        # así que se le pasa el valor correspondiente a 30 fps. Pasarle `segundos * frame_rate` aplicaba el
        # fps dos veces: a 6 fps una pista perdida se recordaba 0,6 s en lugar de los 3 s configurados.
        track_buffer = max(1, round(track_buffer_seconds * 30))
        self._tracker = ByteTrack(frame_rate=int(round(frame_rate)), track_buffer=track_buffer)
        self._class_names: dict[int, str] = {}
        self._name_to_id: dict[str, int] = {}

    def _class_id(self, class_name: str) -> int:
        if class_name not in self._name_to_id:
            new_id = len(self._name_to_id)
            self._name_to_id[class_name] = new_id
            self._class_names[new_id] = class_name
        return self._name_to_id[class_name]

    def update(self, detections: list[Detection], frame: Frame) -> list[TrackedObject]:
        if not detections:
            dets_array = np.empty((0, 6))
        else:
            dets_array = np.array(
                [
                    [d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2, d.confidence, self._class_id(d.class_name)]
                    for d in detections
                ]
            )

        tracks = self._tracker.update(dets_array, frame.image)

        tracked_objects: list[TrackedObject] = []
        boxes, ids, confs, classes = tracks.xyxy, tracks.id, tracks.conf, tracks.cls
        for i in range(len(tracks)):
            x1, y1, x2, y2 = boxes[i]
            tracked_objects.append(
                TrackedObject(
                    track_id=int(ids[i]),
                    class_name=self._class_names.get(int(classes[i]), str(int(classes[i]))),
                    bbox=BoundingBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2)),
                    confidence=float(confs[i]),
                    frame_index=frame.frame_index,
                    timestamp=frame.timestamp,
                )
            )
        return tracked_objects
