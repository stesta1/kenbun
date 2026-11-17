from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np

from monitoreo_urbano.vlm_describer.describer import EVENT_FRAME_OFFSETS, MAX_FRAMES, select_event_frames
from monitoreo_urbano.vlm_describer.worker import trigger_timing


def _frames(count: int) -> list[np.ndarray]:
    return [np.full((2, 2, 3), i % 255, dtype=np.uint8) for i in range(count)]


def test_short_clips_are_returned_whole():
    frames = _frames(len(EVENT_FRAME_OFFSETS))

    assert select_event_frames(frames, 5.0, 10.0) is frames


def test_without_timing_the_whole_clip_goes_to_uniform_sampling():
    frames = _frames(120)

    assert select_event_frames(frames, None, 60.0) is frames
    assert select_event_frames(frames, 5.0, None) is frames
    assert select_event_frames(frames, 5.0, 0.0) is frames


def test_a_trigger_outside_the_clip_falls_back_to_the_whole_clip():
    frames = _frames(120)

    assert select_event_frames(frames, -1.0, 60.0) is frames
    assert select_event_frames(frames, 90.0, 60.0) is frames


def test_frames_are_taken_around_the_trigger_in_chronological_order():
    frames = _frames(121)
    trigger = 30.0

    selected = select_event_frames(frames, trigger, 60.0)

    indices = [int(f[0, 0, 0]) for f in selected]
    assert indices == sorted(indices)
    assert len(selected) == len(EVENT_FRAME_OFFSETS) <= MAX_FRAMES
    assert indices[0] == round((trigger - 3.0) * 2) and indices[-1] == round((trigger + 10.0) * 2)
    assert round(trigger * 2) in indices


def test_offsets_that_leave_the_clip_are_clamped_and_deduplicated():
    frames = _frames(40)

    selected = select_event_frames(frames, 19.5, 20.0)

    indices = [int(f[0, 0, 0]) for f in selected]
    assert indices == sorted(set(indices))
    assert max(indices) == 39


def test_trigger_timing_uses_the_event_marks():
    start = datetime(2026, 10, 19, 12, 0, 0)
    event = SimpleNamespace(start_ts=start, end_ts=start + timedelta(seconds=60), created_at=start + timedelta(seconds=30))

    assert trigger_timing(event) == (30.0, 60.0)


def test_trigger_timing_accepts_aware_and_naive_marks():
    start = datetime(2026, 10, 19, 12, 0, 0)
    event = SimpleNamespace(
        start_ts=start,
        end_ts=(start + timedelta(seconds=60)).replace(tzinfo=timezone.utc),
        created_at=(start + timedelta(seconds=12)).replace(tzinfo=timezone.utc),
    )

    assert trigger_timing(event) == (12.0, 60.0)


def test_trigger_timing_without_marks_is_unknown():
    assert trigger_timing(SimpleNamespace()) == (None, None)
