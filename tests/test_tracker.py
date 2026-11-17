from __future__ import annotations

import pytest

from monitoreo_urbano.vision_pipeline.tracker import ByteTrackTracker

pytest.importorskip("boxmot")


@pytest.mark.parametrize("fps", [30.0, 15.0, 10.0, 6.0, 5.0])
def test_lost_track_memory_lasts_the_configured_seconds_at_any_frequency(fps):
    tracker = ByteTrackTracker(frame_rate=fps, track_buffer_seconds=3.0)

    frames_remembered = tracker._tracker.max_time_lost

    assert frames_remembered == int(3.0 * fps)


def test_memory_in_seconds_is_the_same_at_30_fps_as_before_the_change():
    tracker = ByteTrackTracker(frame_rate=30.0, track_buffer_seconds=3.0)

    assert tracker._tracker.max_time_lost == 90
