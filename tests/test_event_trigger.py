from __future__ import annotations

from datetime import datetime

from monitoreo_urbano.common.repository import set_runtime_override
from monitoreo_urbano.common.runtime_config import RuntimeThresholds
from monitoreo_urbano.common.schemas import INCIDENT_REASONS, BoundingBox, TrackedObject
from monitoreo_urbano.vision_pipeline.event_trigger import (
    EventTriggerEngine,
    MinDurationRule,
    PersonVehicleProximityRule,
    VehicleConvergenceRule,
)


def _tracked_object(
    track_id: int, frame_index: int, class_name: str = "car", center: tuple[float, float] = (5, 5)
) -> TrackedObject:
    cx, cy = center
    return TrackedObject(
        track_id=track_id,
        class_name=class_name,
        bbox=BoundingBox(x1=cx - 5, y1=cy - 5, x2=cx + 5, y2=cy + 5),
        confidence=0.9,
        frame_index=frame_index,
        timestamp=datetime.utcnow(),
    )


def test_does_not_trigger_before_min_frames():
    engine = EventTriggerEngine(rules=[MinDurationRule(min_frames=5)])
    for i in range(4):
        triggered = engine.update([_tracked_object(track_id=1, frame_index=i)])
    assert triggered == []


def test_triggers_once_min_frames_reached():
    engine = EventTriggerEngine(rules=[MinDurationRule(min_frames=3)])
    engine.update([_tracked_object(track_id=1, frame_index=0)])
    engine.update([_tracked_object(track_id=1, frame_index=1)])
    triggered = engine.update([_tracked_object(track_id=1, frame_index=2)])
    assert triggered == [1]


def test_does_not_trigger_twice_for_same_track():
    engine = EventTriggerEngine(rules=[MinDurationRule(min_frames=2)])
    engine.update([_tracked_object(track_id=1, frame_index=0)])
    first = engine.update([_tracked_object(track_id=1, frame_index=1)])
    second = engine.update([_tracked_object(track_id=1, frame_index=2)])
    assert first == [1]
    assert second == []


def test_tolerates_brief_occlusion_without_resetting_history():
    engine = EventTriggerEngine(rules=[MinDurationRule(min_frames=3)], stale_after_frames=5)
    engine.update([_tracked_object(track_id=1, frame_index=0)])
    engine.update([])
    engine.update([_tracked_object(track_id=1, frame_index=2)])
    triggered = engine.update([_tracked_object(track_id=1, frame_index=3)])
    assert triggered == [1]
    assert len(engine.history_for(1)) == 3


def _trigger_track(engine: EventTriggerEngine, track_id: int, class_name: str, center: tuple[float, float]) -> list[int]:
    engine.update([_tracked_object(track_id=track_id, frame_index=0, class_name=class_name, center=center)])
    return engine.update([_tracked_object(track_id=track_id, frame_index=1, class_name=class_name, center=center)])


def test_suppresses_new_track_id_in_same_spot_as_recent_trigger():
    engine = EventTriggerEngine(rules=[MinDurationRule(min_frames=2)], dedup_radius_px=50, dedup_window_seconds=60)
    first = _trigger_track(engine, track_id=1, class_name="person", center=(100, 100))
    second = _trigger_track(engine, track_id=2, class_name="person", center=(105, 98))
    assert first == [1]
    assert second == []


def test_does_not_suppress_different_location():
    engine = EventTriggerEngine(rules=[MinDurationRule(min_frames=2)], dedup_radius_px=50, dedup_window_seconds=60)
    first = _trigger_track(engine, track_id=1, class_name="person", center=(50, 50))
    second = _trigger_track(engine, track_id=2, class_name="person", center=(500, 400))
    assert first == [1]
    assert second == [2]


def test_does_not_suppress_different_class_in_same_spot():
    engine = EventTriggerEngine(rules=[MinDurationRule(min_frames=2)], dedup_radius_px=50, dedup_window_seconds=60)
    first = _trigger_track(engine, track_id=1, class_name="person", center=(100, 100))
    second = _trigger_track(engine, track_id=2, class_name="car", center=(100, 100))
    assert first == [1]
    assert second == [2]


def test_context_class_alone_does_not_trigger_by_default():
    engine = EventTriggerEngine()
    all_triggered: list[int] = []
    for i in range(20):
        all_triggered += engine.update([_tracked_object(track_id=1, frame_index=i, class_name="person", center=(50, 50))])
    assert all_triggered == []


def test_incident_class_triggers_alone_by_default():
    engine = EventTriggerEngine()
    all_triggered: list[int] = []
    for i in range(9):
        all_triggered += engine.update([_tracked_object(track_id=1, frame_index=i, class_name="fire", center=(50, 50))])
    assert all_triggered == [1]


def test_person_near_vehicle_triggers_but_not_alone():
    engine = EventTriggerEngine(rules=[PersonVehicleProximityRule(radius_px=60, min_frames=3)])
    all_triggered: list[int] = []
    for i in range(3):
        all_triggered += engine.update(
            [
                _tracked_object(track_id=1, frame_index=i, class_name="person", center=(100, 100)),
                _tracked_object(track_id=2, frame_index=i, class_name="car", center=(120, 100)),
            ]
        )
    assert all_triggered == [1]
    assert engine.reason_for(1) == "persona_junto_a_vehiculo"


def test_vehicle_convergence_triggers_both_vehicles():
    engine = EventTriggerEngine(rules=[VehicleConvergenceRule(radius_px=60, min_frames=3)])
    all_triggered: list[int] = []
    for i in range(3):
        all_triggered += engine.update(
            [
                _tracked_object(track_id=1, frame_index=i, class_name="car", center=(100, 100)),
                _tracked_object(track_id=2, frame_index=i, class_name="motorcycle", center=(130, 100)),
            ]
        )
    assert set(all_triggered) == {1, 2}
    assert engine.reason_for(1) == "vehiculos_muy_cerca"


def test_uses_live_dedup_thresholds_when_provided(db_conn):
    thresholds = RuntimeThresholds()
    set_runtime_override(db_conn, "event_dedup_radius_px", 5000.0)
    set_runtime_override(db_conn, "event_dedup_window_seconds", 60.0)
    thresholds.refresh(db_conn)

    engine = EventTriggerEngine(rules=[MinDurationRule(min_frames=2)], thresholds=thresholds)
    first = _trigger_track(engine, track_id=1, class_name="person", center=(0, 0))
    second = _trigger_track(engine, track_id=2, class_name="person", center=(2000, 2000))
    assert first == [1]
    assert second == []


def test_incident_is_suppressed_by_class_and_time_regardless_of_position():
    engine = EventTriggerEngine(rules=[MinDurationRule(min_frames=2)], dedup_radius_px=50, dedup_window_seconds=60)
    engine.update([_tracked_object(track_id=1, frame_index=0, class_name="accident", center=(10, 10))])
    assert engine.update([_tracked_object(track_id=1, frame_index=1, class_name="accident", center=(10, 10))]) == [1]

    engine.update([_tracked_object(track_id=2, frame_index=2, class_name="accident", center=(510, 10))])
    assert engine.update([_tracked_object(track_id=2, frame_index=3, class_name="accident", center=(510, 10))]) == []


def test_incident_triggers_again_after_the_dedup_window():
    engine = EventTriggerEngine(rules=[MinDurationRule(min_frames=2)], dedup_window_seconds=0)
    engine.update([_tracked_object(track_id=1, frame_index=0, class_name="fire")])
    assert engine.update([_tracked_object(track_id=1, frame_index=1, class_name="fire")]) == [1]
    engine.update([_tracked_object(track_id=2, frame_index=2, class_name="fire")])
    assert engine.update([_tracked_object(track_id=2, frame_index=3, class_name="fire")]) == [2]


def test_smoke_does_not_trigger_events_or_alerts():
    assert "smoke" not in INCIDENT_REASONS
    engine = EventTriggerEngine()
    for i in range(20):
        assert engine.update([_tracked_object(track_id=1, frame_index=i, class_name="smoke")]) == []


def _scene_with_incident_detections(engine: EventTriggerEngine, clock: list[float], seconds: list[float], monkeypatch) -> list[float]:
    from monitoreo_urbano.vision_pipeline import event_trigger

    class _Clock:
        @staticmethod
        def monotonic() -> float:
            return clock[0]

    monkeypatch.setattr(event_trigger, "time", _Clock)
    fired = []
    for index, t in enumerate(seconds):
        clock[0] = t
        if engine.update([_tracked_object(track_id=1_000 + index, frame_index=index, class_name="accident")]):
            fired.append(t)
    return fired


def test_a_scene_that_keeps_showing_the_incident_fires_again_after_the_dedup_window_by_default(monkeypatch):
    from monitoreo_urbano.common.config import settings

    monkeypatch.setattr(settings, "incident_dedup_mode", "global")
    engine = EventTriggerEngine(rules=[MinDurationRule(min_frames=1)], dedup_window_seconds=30)
    fired = _scene_with_incident_detections(engine, [0.0], [float(t) for t in range(0, 100, 5)], monkeypatch)
    assert len(fired) >= 3


def test_dedup_extension_keeps_one_event_while_the_scene_still_shows_the_incident(monkeypatch):
    from monitoreo_urbano.common.config import settings

    monkeypatch.setattr(settings, "incident_dedup_mode", "renovar")
    engine = EventTriggerEngine(rules=[MinDurationRule(min_frames=1)], dedup_window_seconds=30)
    fired = _scene_with_incident_detections(engine, [0.0], [float(t) for t in range(0, 100, 5)], monkeypatch)
    assert fired == [0.0]


def test_dedup_extension_lets_a_new_incident_fire_after_the_scene_goes_quiet(monkeypatch):
    from monitoreo_urbano.common.config import settings

    monkeypatch.setattr(settings, "incident_dedup_mode", "renovar")
    engine = EventTriggerEngine(rules=[MinDurationRule(min_frames=1)], dedup_window_seconds=30)
    seconds = [0.0, 5.0, 10.0, 100.0, 105.0]
    fired = _scene_with_incident_detections(engine, [0.0], seconds, monkeypatch)
    assert fired == [0.0, 100.0]


def test_location_dedup_lets_an_incident_elsewhere_fire_while_the_first_scene_is_still_visible(monkeypatch):
    from monitoreo_urbano.common.config import settings
    from monitoreo_urbano.vision_pipeline import event_trigger

    monkeypatch.setattr(settings, "incident_dedup_mode", "ubicacion")
    clock = [0.0]

    class _Clock:
        @staticmethod
        def monotonic() -> float:
            return clock[0]

    monkeypatch.setattr(event_trigger, "time", _Clock)
    engine = EventTriggerEngine(rules=[MinDurationRule(min_frames=1)], dedup_window_seconds=30)
    fired = []
    for index, (t, center) in enumerate([(0.0, (50, 50)), (5.0, (52, 50)), (10.0, (51, 49)), (12.0, (900, 600)), (14.0, (902, 601))]):
        clock[0] = t
        if engine.update([_tracked_object(track_id=5_000 + index, frame_index=index, class_name="accident", center=center)]):
            fired.append((t, center))
    assert fired == [(0.0, (50, 50)), (12.0, (900, 600))]
