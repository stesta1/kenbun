from __future__ import annotations

from datetime import datetime

import pytest

from monitoreo_urbano.common.config import settings
from monitoreo_urbano.common.schemas import BoundingBox, TrackedObject
from monitoreo_urbano.vision_pipeline.event_trigger import (
    DirectIncidentRule,
    EventTriggerEngine,
    PersonAltercationRule,
    default_rules,
    frames_for_duration,
)


def _tracked(track_id: int, frame_index: int, class_name: str = "fire") -> TrackedObject:
    return TrackedObject(
        track_id=track_id,
        class_name=class_name,
        bbox=BoundingBox(x1=0, y1=0, x2=10, y2=10),
        confidence=0.9,
        frame_index=frame_index,
        timestamp=datetime.utcnow(),
    )


def _min_frames(rules) -> list[int]:
    return [rule.min_frames for rule in rules if hasattr(rule, "min_frames")]


@pytest.mark.parametrize(
    "seconds, fps, minimum, expected",
    [(0.27, 30, 1, 8), (5.0, 30, 1, 150), (2.0, 30, 1, 60), (5.0, 6, 1, 30), (0.27, 6, 1, 2), (0.27, 6, 3, 3)],
)
def test_frames_for_duration(seconds, fps, minimum, expected):
    assert frames_for_duration(seconds, fps, minimum=minimum) == expected


def test_default_rules_without_fps_keep_the_historical_eight_frames():
    assert _min_frames(default_rules()) == [8, 8]


def test_default_rules_at_30_fps_reproduce_the_historical_value():
    assert _min_frames(default_rules(fps=30)) == [8, 8]


def test_default_rules_never_ask_for_fewer_than_the_minimum_samples():
    assert _min_frames(default_rules(fps=6)) == [settings.trigger_min_samples] * 2
    assert _min_frames(default_rules(fps=1)) == [settings.trigger_min_samples] * 2


def test_default_rules_keep_both_rule_types():
    rules = default_rules(fps=10)
    assert isinstance(rules[0], DirectIncidentRule)
    assert any(isinstance(rule, PersonAltercationRule) for rule in rules)


def test_engine_without_fps_keeps_the_historical_windows():
    engine = EventTriggerEngine()
    assert (engine.history_size, engine.stale_after_frames) == (150, 60)


def test_engine_at_30_fps_reproduces_the_historical_windows():
    engine = EventTriggerEngine(fps=30)
    assert (engine.history_size, engine.stale_after_frames) == (150, 60)


def test_engine_windows_follow_the_analysis_frequency():
    engine = EventTriggerEngine(fps=6)
    assert (engine.history_size, engine.stale_after_frames) == (30, 12)


def test_explicit_windows_win_over_the_seconds_derived_ones():
    engine = EventTriggerEngine(fps=6, history_size=7, stale_after_frames=4)
    assert (engine.history_size, engine.stale_after_frames) == (7, 4)


def test_incident_needs_the_same_persistence_in_time_at_any_frequency():
    for fps, needed in ((30, 24), (6, 5)):
        engine = EventTriggerEngine(fps=fps)
        triggered = []
        for index in range(needed):
            triggered = engine.update([_tracked(track_id=1, frame_index=index)])
        assert triggered == [1], f"a {fps} fps debía disparar con {needed} observaciones"

        engine = EventTriggerEngine(fps=fps)
        for index in range(needed - 1):
            assert engine.update([_tracked(track_id=1, frame_index=index)]) == []


def test_a_track_missing_for_the_stale_time_is_forgotten_at_any_frequency():
    for fps, absent_frames in ((30, 60), (6, 12)):
        engine = EventTriggerEngine(fps=fps)
        engine.update([_tracked(track_id=1, frame_index=0, class_name="person")])
        for _ in range(absent_frames - 1):
            engine.update([])
        assert 1 in engine._history
        engine.update([])
        assert 1 not in engine._history
