from __future__ import annotations

from monitoreo_urbano.common.schemas import EventStatus, ReviewStatus
from tests.conftest import make_event


def test_event_defaults_to_pending_and_unreviewed():
    event = make_event()
    assert event.status == EventStatus.PENDING_DESCRIPTION
    assert event.review_status == ReviewStatus.PENDING
    assert event.description_nl is None


def test_event_accepts_overrides():
    event = make_event(camera_id="cam-02", object_classes=["person", "bicycle"])
    assert event.camera_id == "cam-02"
    assert event.object_classes == ["person", "bicycle"]
