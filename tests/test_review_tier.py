
from __future__ import annotations

from datetime import datetime

import pytest

from monitoreo_urbano.common.config import settings
from monitoreo_urbano.common.schemas import ALERTABLE_REASONS, CATEGORY_LABELS, SELECTABLE_CATEGORIES, BoundingBox, TrackedObject
from monitoreo_urbano.vision_pipeline import event_trigger
from monitoreo_urbano.vision_pipeline.event_trigger import REVIEW_REASON, EventTriggerEngine


def _det(index: int, confidence: float, center=(300.0, 200.0), size: float = 60.0) -> TrackedObject:
    cx, cy = center
    return TrackedObject(
        track_id=1_000_000 + index, class_name="accident",
        bbox=BoundingBox(x1=cx - size / 2, y1=cy - size / 2, x2=cx + size / 2, y2=cy + size / 2),
        confidence=confidence, frame_index=index, timestamp=datetime.utcnow(),
    )


@pytest.fixture
def clock(monkeypatch):
    now = [0.0]

    class _Clock:
        @staticmethod
        def monotonic() -> float:
            return now[0]

    monkeypatch.setattr(event_trigger, "time", _Clock)
    monkeypatch.setattr(settings, "incident_dedup_mode", "renovar")
    return now


def _run(detections: list[tuple[float, float, tuple[float, float]]], clock, tail_seconds: float = 3.0) -> list[tuple[float, str]]:
    engine = EventTriggerEngine(fps=6.0, dedup_window_seconds=90.0)
    by_tick = {round(second * 6): (confidence, center) for second, confidence, center in detections}
    fired = []
    for tick in range(round((max(s for s, _, _ in detections) + tail_seconds) * 6) + 1):
        clock[0] = tick / 6
        objects = [_det(tick, *by_tick[tick][:1], by_tick[tick][1])] if tick in by_tick else []
        for track_id in engine.update(objects):
            fired.append((round(tick / 6, 2), engine.reason_for(track_id)))
    return fired


def test_the_review_reason_is_not_alertable_but_it_is_selectable_and_labelled():
    assert REVIEW_REASON not in ALERTABLE_REASONS
    assert REVIEW_REASON in SELECTABLE_CATEGORIES
    assert REVIEW_REASON in CATEGORY_LABELS


def test_a_weak_short_burst_goes_to_review_and_not_to_alert(clock):
    fired = _run([(0.0, 0.3, (300, 200)), (0.33, 0.3, (302, 200))], clock, tail_seconds=4.0)
    assert [reason for _, reason in fired] == [REVIEW_REASON]


def test_a_single_weak_detection_registers_nothing(clock):
    assert _run([(0.0, 0.3, (300, 200))], clock, tail_seconds=4.0) == []


def test_detections_below_the_review_confidence_register_nothing(clock):
    assert _run([(0.0, 0.2, (300, 200)), (0.17, 0.2, (300, 200)), (0.33, 0.2, (300, 200))], clock, tail_seconds=4.0) == []


def test_a_single_very_sure_detection_alerts_on_its_own(clock):
    fired = _run([(0.0, 0.8, (300, 200))], clock)
    assert [reason for _, reason in fired] == ["accident"]


def test_sustained_detections_above_the_alert_confidence_alert_without_a_duplicate_review_event(clock):
    fired = _run([(i / 6, 0.6, (300, 200)) for i in range(6)], clock)
    assert [reason for _, reason in fired] == ["accident"]


def test_the_review_event_waits_for_the_hold_before_opening(clock):
    fired = _run([(0.0, 0.3, (300, 200)), (0.17, 0.3, (302, 200))], clock, tail_seconds=4.0)
    assert len(fired) == 1 and fired[0][1] == REVIEW_REASON
    assert fired[0][0] >= settings.incident_review_hold_seconds


def test_a_review_event_does_not_mask_a_later_alert(clock):
    fired = _run([(0.0, 0.3, (300, 200)), (0.33, 0.3, (300, 200)), (5.0, 0.85, (310, 200))], clock)
    assert [reason for _, reason in fired] == [REVIEW_REASON, "accident"]


def test_once_the_incident_alerted_weak_detections_of_the_same_scene_do_not_open_review_events(clock):
    fired = _run([(0.0, 0.85, (300, 200)), (2.0, 0.3, (302, 200)), (2.17, 0.3, (301, 201)), (2.33, 0.3, (300, 200))], clock, tail_seconds=4.0)
    assert [reason for _, reason in fired] == ["accident"]


def test_review_can_be_turned_off(clock, monkeypatch):
    monkeypatch.setattr(settings, "accident_review_enabled", False)
    assert _run([(0.0, 0.3, (300, 200)), (0.33, 0.3, (302, 200))], clock, tail_seconds=4.0) == []
