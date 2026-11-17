from __future__ import annotations

from datetime import datetime

import numpy as np

from monitoreo_urbano.common.schemas import HEALTH_OK, BoundingBox, TrackedObject
from monitoreo_urbano.vision_pipeline import camera_pipeline
from monitoreo_urbano.vision_pipeline.camera_health import CameraHealthMonitor
from monitoreo_urbano.vision_pipeline.camera_pipeline import CameraPipeline, OpenClip
from monitoreo_urbano.vision_pipeline.camera_source import Frame
from monitoreo_urbano.vision_pipeline.event_trigger import EventTriggerEngine, MinDurationRule
from monitoreo_urbano.vision_pipeline.frame_buffer import FrameRingBuffer
from monitoreo_urbano.vision_pipeline.tracker import Tracker


class ScriptedTracker(Tracker):

    def __init__(self, objects_by_frame: dict[int, list[TrackedObject]] | None = None) -> None:
        self._objects_by_frame = objects_by_frame or {}

    def update(self, detections, frame):
        return self._objects_by_frame.get(frame.frame_index, [])


def _noisy_image(seed: int, size: int = 20) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(size, size, 3), dtype=np.uint8)


def _black_image(size: int = 20) -> np.ndarray:
    return np.zeros((size, size, 3), dtype=np.uint8)


def _frame(index: int, image: np.ndarray | None = None) -> Frame:
    return Frame(image=image if image is not None else _noisy_image(index), timestamp=datetime.utcnow(), frame_index=index)


def _tracked(track_id: int, frame_index: int, class_name: str = "car", confidence: float = 0.9) -> TrackedObject:
    return TrackedObject(
        track_id=track_id,
        class_name=class_name,
        bbox=BoundingBox(x1=0, y1=0, x2=10, y2=10),
        confidence=confidence,
        frame_index=frame_index,
        timestamp=datetime.utcnow(),
    )


def _pipeline(
    tracker: Tracker | None = None,
    *,
    min_frames: int = 2,
    post_event_frames: int = 2,
    clips: list[OpenClip] | None = None,
    health: list[str] | None = None,
    camera_id: str = "cam-test",
) -> CameraPipeline:
    return CameraPipeline(
        camera_id,
        tracker=tracker or ScriptedTracker(),
        trigger_engine=EventTriggerEngine(rules=[MinDurationRule(min_frames=min_frames)]),
        health_monitor=CameraHealthMonitor(
            fps=1.0, reconnect_delay_seconds=1.0, sustained_seconds=2.0, signal_loss_sustained_seconds=2.0
        ),
        buffer=FrameRingBuffer(max_seconds=10, fps=1.0),
        post_event_frames=post_event_frames,
        on_clip=(clips if clips is not None else []).append,
        on_health=(health if health is not None else []).append,
    )


def _run(pipeline: CameraPipeline, frame_count: int) -> None:
    for index in range(frame_count):
        frame = _frame(index)
        pipeline.observe(frame)
        pipeline.update(frame, [])


def test_update_returns_what_the_tracker_reports():
    tracked = [_tracked(track_id=1, frame_index=0)]
    pipeline = _pipeline(ScriptedTracker({0: tracked}))
    frame = _frame(0)
    pipeline.observe(frame)
    assert pipeline.update(frame, []) == tracked


def test_delivers_the_clip_once_after_the_post_event_frames():
    clips: list[OpenClip] = []
    tracker = ScriptedTracker({i: [_tracked(track_id=7, frame_index=i)] for i in range(10)})
    pipeline = _pipeline(tracker, clips=clips, min_frames=2, post_event_frames=2)

    _run(pipeline, frame_count=6)

    assert len(clips) == 1
    clip = clips[0]
    assert clip.track_id == 7
    assert clip.trigger_reason == "car"
    assert clip.object_classes == {"car"}
    assert clip.frames[0].frame_index == 0
    assert clip.frames[-1].frame_index == 2


def test_clip_carries_the_context_buffered_before_the_trigger():
    clips: list[OpenClip] = []
    tracker = ScriptedTracker({i: [_tracked(track_id=1, frame_index=i)] for i in range(10)})
    pipeline = _pipeline(tracker, clips=clips, min_frames=4, post_event_frames=1)

    _run(pipeline, frame_count=5)

    assert [f.frame_index for f in clips[0].frames][:4] == [0, 1, 2, 3]


def test_the_triggering_frame_appears_only_once_in_the_clip():
    clips: list[OpenClip] = []
    tracker = ScriptedTracker({i: [_tracked(track_id=1, frame_index=i)] for i in range(10)})
    pipeline = _pipeline(tracker, clips=clips, min_frames=3, post_event_frames=3)

    _run(pipeline, frame_count=8)

    indices = [f.frame_index for f in clips[0].frames]
    assert indices == sorted(set(indices))
    assert indices == [0, 1, 2, 3, 4]


def test_reports_the_tracks_that_have_an_open_clip():
    tracker = ScriptedTracker({i: [_tracked(track_id=3, frame_index=i)] for i in range(10)})
    pipeline = _pipeline(tracker, min_frames=2, post_event_frames=3)

    assert pipeline.recording_track_ids == set()
    _run(pipeline, frame_count=2)
    assert pipeline.recording_track_ids == {3}
    for index in range(2, 6):
        frame = _frame(index)
        pipeline.observe(frame)
        pipeline.update(frame, [])
    assert pipeline.recording_track_ids == set()


def test_averages_confidence_of_the_triggering_track():
    clips: list[OpenClip] = []
    objects = {i: [_tracked(track_id=1, frame_index=i, confidence=0.5 + i / 10)] for i in range(10)}
    pipeline = _pipeline(ScriptedTracker(objects), clips=clips, min_frames=2, post_event_frames=1)

    _run(pipeline, frame_count=4)

    assert clips and len(clips[0].confidences) >= 2


def test_signal_loss_is_reported_once_after_the_sustained_attempts():
    health: list[str] = []
    pipeline = _pipeline(health=health)

    pipeline.on_signal_loss()
    assert health == []
    pipeline.on_signal_loss()
    pipeline.on_signal_loss()
    assert health == ["corte_de_señal"]


def test_covered_camera_is_reported_and_its_recovery_too():
    health: list[str] = []
    pipeline = _pipeline(health=health)

    for index in range(3):
        pipeline.observe(_frame(index, _black_image()))
    assert health == ["tapada"]

    pipeline.observe(_frame(3, _noisy_image(seed=99)))
    assert health == ["tapada", HEALTH_OK]


def test_each_pipeline_keeps_its_own_state():
    clips_a: list[OpenClip] = []
    clips_b: list[OpenClip] = []
    health_b: list[str] = []
    busy = ScriptedTracker({i: [_tracked(track_id=1, frame_index=i)] for i in range(10)})
    quiet = ScriptedTracker()
    pipeline_a = _pipeline(busy, clips=clips_a, camera_id="cam-a", min_frames=2, post_event_frames=1)
    pipeline_b = _pipeline(quiet, clips=clips_b, health=health_b, camera_id="cam-b", min_frames=2, post_event_frames=1)

    for index in range(4):
        frame = _frame(index)
        pipeline_a.observe(frame)
        pipeline_a.update(frame, [])
        pipeline_b.observe(frame)
        pipeline_b.update(frame, [])

    assert len(clips_a) == 1
    assert clips_b == []
    assert pipeline_b.recording_track_ids == set()
    assert health_b == []


def test_build_wires_the_components_from_the_camera_fps(monkeypatch):
    captured: dict = {}

    class FakeByteTrack(Tracker):
        def __init__(self, frame_rate: float, track_buffer_seconds: float) -> None:
            captured["frame_rate"] = frame_rate
            captured["track_buffer_seconds"] = track_buffer_seconds

        def update(self, detections, frame):
            return []

    monkeypatch.setattr(camera_pipeline, "ByteTrackTracker", FakeByteTrack)
    monkeypatch.setattr(camera_pipeline.settings, "frame_buffer_seconds", 4.0)
    monkeypatch.setattr(camera_pipeline.settings, "clip_post_event_seconds", 2.0)
    monkeypatch.setattr(camera_pipeline.settings, "track_buffer_seconds", 3.0)

    pipeline = CameraPipeline.build(
        "cam-build", fps=5.0, reconnect_delay_seconds=3.0, thresholds=None, on_clip=lambda c: None, on_health=lambda s: None
    )

    assert captured == {"frame_rate": 5.0, "track_buffer_seconds": 3.0}
    assert pipeline._post_event_frames == 10
    assert pipeline._buffer._buffer.maxlen == 20


def test_build_gives_the_trigger_engine_the_analysis_frequency(monkeypatch):
    class FakeByteTrack(Tracker):
        def __init__(self, frame_rate: float, track_buffer_seconds: float) -> None:
            pass

        def update(self, detections, frame):
            return []

    monkeypatch.setattr(camera_pipeline, "ByteTrackTracker", FakeByteTrack)

    pipeline = CameraPipeline.build(
        "cam-fps", fps=6.0, reconnect_delay_seconds=3.0, thresholds=None, on_clip=lambda c: None, on_health=lambda s: None
    )

    engine = pipeline._trigger_engine
    assert (engine.history_size, engine.stale_after_frames) == (30, 12)
    assert [rule.min_frames for rule in engine.rules if hasattr(rule, "min_frames")] == [3, 3]


def _compressed_frame(index: int) -> Frame:
    from monitoreo_urbano.vision_pipeline.frame_encoding import make_capture_hook

    frame = _frame(index)
    make_capture_hook(jpeg_quality=85)(frame)
    return frame


def test_clips_hold_compressed_frames_when_the_capture_compressed_them():
    from monitoreo_urbano.vision_pipeline.frame_encoding import EncodedFrame

    clips: list[OpenClip] = []
    tracker = ScriptedTracker({i: [_tracked(track_id=1, frame_index=i)] for i in range(10)})
    pipeline = _pipeline(tracker, clips=clips, min_frames=3, post_event_frames=3)

    for index in range(8):
        frame = _compressed_frame(index)
        pipeline.observe(frame)
        pipeline.update(frame, [])

    assert clips
    assert all(isinstance(f, EncodedFrame) for f in clips[0].frames)
    assert [f.frame_index for f in clips[0].frames] == [0, 1, 2, 3, 4]


def test_compressed_clips_keep_the_pre_event_context_without_repeating_the_trigger_frame():
    clips: list[OpenClip] = []
    tracker = ScriptedTracker({i: [_tracked(track_id=2, frame_index=i)] for i in range(12)})
    pipeline = _pipeline(tracker, clips=clips, min_frames=4, post_event_frames=2)

    for index in range(8):
        frame = _compressed_frame(index)
        pipeline.observe(frame)
        pipeline.update(frame, [])

    indices = [f.frame_index for f in clips[0].frames]
    assert indices == sorted(set(indices))
    assert indices[:4] == [0, 1, 2, 3]


def test_the_buffer_does_not_keep_the_raw_images_of_compressed_frames():
    pipeline = _pipeline()
    for index in range(3):
        frame = _compressed_frame(index)
        pipeline.observe(frame)

    assert all(not hasattr(stored, "image") for stored in pipeline._buffer.snapshot())


def _accident_burst_pipeline(monkeypatch, direct: bool):
    from monitoreo_urbano.common.config import settings

    monkeypatch.setattr(settings, "accident_direct_trigger", direct)
    monkeypatch.setattr(settings, "incident_min_seconds", {"fire": 0.8, "accident": 2 / 6})
    monkeypatch.setattr(settings, "incident_window_seconds", 1.5)
    monkeypatch.setattr(settings, "accident_review_enabled", False)
    monkeypatch.setattr(settings, "incident_strong_confidence", {})
    clips: list[OpenClip] = []
    pipeline = CameraPipeline(
        "cam-test",
        tracker=ScriptedTracker(),
        trigger_engine=EventTriggerEngine(fps=6.0, dedup_window_seconds=0.0),
        health_monitor=CameraHealthMonitor(
            fps=6.0, reconnect_delay_seconds=1.0, sustained_seconds=2.0, signal_loss_sustained_seconds=2.0
        ),
        buffer=FrameRingBuffer(max_seconds=10, fps=6.0),
        post_event_frames=2,
        on_clip=clips.append,
        on_health=[].append,
    )
    return pipeline, clips


def _run_accident_burst(pipeline: CameraPipeline, frames_with_detection: set[int], frame_count: int = 10) -> list[int]:
    from monitoreo_urbano.common.schemas import Detection

    triggered = []
    for index in range(frame_count):
        frame = _frame(index)
        pipeline.observe(frame)
        detections = (
            [Detection(class_name="accident", confidence=0.8, bbox=BoundingBox(x1=0, y1=0, x2=10, y2=10))]
            if index in frames_with_detection
            else []
        )
        pipeline.update(frame, detections)
        triggered.append(len(pipeline._open_clips))
    return triggered


def test_accident_burst_without_tracking_triggers_only_with_the_direct_path(monkeypatch):
    direct_pipeline, direct_clips = _accident_burst_pipeline(monkeypatch, direct=True)
    _run_accident_burst(direct_pipeline, {2, 4, 6})
    assert len(direct_clips) == 1
    assert direct_clips[0].trigger_reason == "accident"

    tracked_pipeline, tracked_clips = _accident_burst_pipeline(monkeypatch, direct=False)
    _run_accident_burst(tracked_pipeline, {2, 4, 6})
    assert tracked_clips == []


def test_fewer_accident_detections_than_the_engine_minimum_do_not_trigger_the_direct_path(monkeypatch):
    pipeline, clips = _accident_burst_pipeline(monkeypatch, direct=True)
    _run_accident_burst(pipeline, {3, 5})
    assert clips == []
