from __future__ import annotations

import math
import random
from datetime import datetime

import pytest

from monitoreo_urbano.common.schemas import BoundingBox, TrackedObject
from monitoreo_urbano.vision_pipeline.event_trigger import (
    EventTriggerEngine,
    PersonAltercationRule,
    default_rules,
)

HEIGHT = 100.0


def _person(track_id: int, frame_index: int, center: tuple[float, float], height: float = HEIGHT) -> TrackedObject:
    cx, cy = center
    return TrackedObject(
        track_id=track_id,
        class_name="person",
        bbox=BoundingBox(x1=cx - height / 4, y1=cy - height / 2, x2=cx + height / 4, y2=cy + height / 2),
        confidence=0.9,
        frame_index=frame_index,
        timestamp=datetime.utcnow(),
    )


def _engine(fps: float) -> EventTriggerEngine:
    return EventTriggerEngine(rules=[PersonAltercationRule(fps=fps)], fps=fps)


def _run(engine: EventTriggerEngine, seconds: float, fps: float, positions) -> list[int]:
    triggered: list[int] = []
    for index in range(int(seconds * fps)):
        t = index / fps
        a, b = positions(t)
        triggered += engine.update([_person(1, index, a), _person(2, index, b)])
    return triggered


def _still(jitter: float = 0.0, seed: int = 0):
    rng = random.Random(seed)

    def positions(t: float):
        return (
            (200 + rng.uniform(-jitter, jitter), 300 + rng.uniform(-jitter, jitter)),
            (260 + rng.uniform(-jitter, jitter), 300 + rng.uniform(-jitter, jitter)),
        )

    return positions


def _scuffle(speed_heights: float = 3.0, period: float = 0.5):
    amplitude = speed_heights * HEIGHT * period / 4

    def positions(t: float):
        sway = amplitude * math.sin(2 * math.pi * t / period)
        return (200 + sway, 300), (260 - sway, 300)

    return positions


@pytest.mark.parametrize("fps", [30, 15, 6])
def test_two_people_standing_still_never_trigger(fps):
    assert _run(_engine(fps), 20, fps, _still()) == []


@pytest.mark.parametrize("fps", [30, 15, 6])
def test_box_jitter_on_still_people_does_not_trigger(fps):
    assert _run(_engine(fps), 20, fps, _still(jitter=3.0)) == []


@pytest.mark.parametrize("fps", [30, 15, 6])
def test_a_sustained_scuffle_triggers(fps):
    triggered = _run(_engine(fps), 10, fps, _scuffle())
    assert triggered, f"a {fps} fps un forcejeo de 10 s debía disparar"


def test_a_scuffle_needs_to_last_several_seconds():
    assert _run(_engine(15), 2.5, 15, _scuffle()) == []


def test_it_triggers_with_the_same_persistence_in_seconds_at_any_frequency():
    first = {}
    for fps in (30, 6):
        engine = _engine(fps)
        for index in range(int(12 * fps)):
            a, b = _scuffle()(index / fps)
            if engine.update([_person(1, index, a), _person(2, index, b)]):
                first[fps] = index / fps
                break
    assert set(first) == {30, 6}
    assert abs(first[30] - first[6]) <= 1.0


def test_two_people_walking_straight_side_by_side_do_not_trigger():
    def positions(t: float):
        return (100 + 80 * t, 300), (160 + 80 * t, 300)

    assert _run(_engine(15), 12, 15, positions) == []


def test_two_people_running_straight_together_do_not_trigger():
    def positions(t: float):
        return (100 + 250 * t, 300), (160 + 250 * t, 300)

    assert _run(_engine(15), 6, 15, positions) == []


def test_one_agitated_person_next_to_a_still_one_does_not_trigger():
    def positions(t: float):
        sway = 120 * math.sin(2 * math.pi * t / 0.5)
        return (200 + sway, 300), (260, 300)

    assert _run(_engine(15), 12, 15, positions) == []


def test_two_agitated_people_far_apart_do_not_trigger():
    def positions(t: float):
        sway = 120 * math.sin(2 * math.pi * t / 0.5)
        return (200 + sway, 300), (900 - sway, 300)

    assert _run(_engine(15), 12, 15, positions) == []


def test_distance_is_measured_in_person_heights_not_pixels():
    def small(t: float):
        sway = 0.5 * 3.0 * HEIGHT * 0.5 / 4 * math.sin(2 * math.pi * t / 0.5)
        return (100 + sway, 150), (130 - sway, 150)

    engine = _engine(15)
    triggered = []
    for index in range(10 * 15):
        a, b = small(index / 15)
        triggered += engine.update([_person(1, index, a, HEIGHT / 2), _person(2, index, b, HEIGHT / 2)])
    assert triggered


def test_it_only_considers_people():
    engine = _engine(15)
    for index in range(150):
        a, b = _scuffle()(index / 15)
        cars = [
            TrackedObject(
                track_id=track_id, class_name="car", bbox=BoundingBox(x1=x - 25, y1=275, x2=x + 25, y2=325),
                confidence=0.9, frame_index=index, timestamp=datetime.utcnow(),
            )
            for track_id, x in ((1, a[0]), (2, b[0]))
        ]
        assert engine.update(cars) == []


def test_the_reason_reported_is_posible_altercado():
    engine = _engine(15)
    for index in range(150):
        a, b = _scuffle()(index / 15)
        if engine.update([_person(1, index, a), _person(2, index, b)]):
            assert engine.reason_for(1) == "posible_altercado"
            return
    pytest.fail("no disparó")


def test_default_rules_build_the_rule_with_the_engine_frequency():
    rules = default_rules(6.0)

    rule = next(r for r in rules if isinstance(r, PersonAltercationRule))
    assert rule.fps == 6.0
