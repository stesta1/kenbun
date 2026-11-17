from __future__ import annotations

from datetime import datetime

import pytest

from monitoreo_urbano.common.config import settings
from monitoreo_urbano.common.schemas import BoundingBox, TrackedObject
from monitoreo_urbano.vision_pipeline.event_trigger import DirectIncidentRule, EventTriggerEngine


@pytest.fixture(autouse=True)
def _persistence_only(monkeypatch):
    monkeypatch.setattr(settings, "accident_review_enabled", False)
    monkeypatch.setattr(settings, "incident_strong_confidence", {})
    monkeypatch.setattr(settings, "incident_alert_confidence", {})


def _obj(track_id: int, frame_index: int, class_name: str = "accident", center=(300.0, 200.0), size: float = 60.0) -> TrackedObject:
    cx, cy = center
    return TrackedObject(
        track_id=track_id,
        class_name=class_name,
        bbox=BoundingBox(x1=cx - size / 2, y1=cy - size / 2, x2=cx + size / 2, y2=cy + size / 2),
        confidence=0.9,
        frame_index=frame_index,
        timestamp=datetime.utcnow(),
    )


def _engine(fps: float = 6.0) -> EventTriggerEngine:
    return EventTriggerEngine(fps=fps, dedup_window_seconds=0)


def test_a_burst_split_in_several_track_ids_triggers_when_it_adds_up():
    engine = _engine(6)
    triggered = []
    for index in range(4):
        triggered += engine.update([_obj(track_id=100 + index, frame_index=index)])
    assert triggered, "una ráfaga de 0,67 s dividida en identificadores distintos debía disparar"


def test_the_same_burst_does_not_trigger_when_it_is_too_short():
    engine = _engine(6)
    triggered = []
    for index in range(3):
        triggered += engine.update([_obj(track_id=100 + index, frame_index=index)])
    assert triggered == []


def test_detections_far_from_each_other_do_not_add_up():
    engine = _engine(6)
    triggered = []
    for index in range(6):
        triggered += engine.update([_obj(track_id=index, frame_index=index, center=(100.0 + 900.0 * (index % 2), 200.0))])
    assert triggered == []


def test_detections_older_than_the_window_expire():
    engine = _engine(6)
    triggered = []
    for index in (0, 1, 2, 40, 41, 42):
        triggered += engine.update([_obj(track_id=index, frame_index=index)])
    assert triggered == []


def test_other_classes_use_the_per_track_count():
    rule = DirectIncidentRule(window_frames={"accident": 12}, min_frames_by_class={"accident": 4})
    engine = EventTriggerEngine(rules=[rule], dedup_window_seconds=0)
    triggered = []
    for index in range(4):
        triggered += engine.update([_obj(track_id=7, frame_index=index, class_name="fire")])
    assert triggered == []


@pytest.mark.parametrize("fps,needed", [(30, 20), (15, 10), (6, 4)])
def test_a_single_track_needs_the_same_time_at_any_frequency(fps, needed):
    engine = _engine(fps)
    for index in range(needed - 1):
        assert engine.update([_obj(track_id=1, frame_index=index)]) == []
    assert engine.update([_obj(track_id=1, frame_index=needed - 1)]) == [1]


def test_default_rules_carry_the_window_per_class():
    rule = EventTriggerEngine(fps=6).rules[0]

    assert set(rule.window_frames) == {"fire", "accident"}
    assert rule.window_frames["accident"] >= 12
