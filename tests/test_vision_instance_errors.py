from __future__ import annotations

from monitoreo_urbano.vision_pipeline import instance as instance_module
from monitoreo_urbano.vision_pipeline.camera_pipeline import OpenClip

from .test_vision_instance import (  # noqa: F401  (los fixtures se reutilizan)
    FakeCameraStream,
    FakeStore,
    _frame,
    _wait_for,
    make_instance,
    short_clips,
)


def _clip(reason: str | None = "accident") -> OpenClip:
    return OpenClip(
        track_id=3,
        object_classes={"car"},
        trigger_reason=reason,
        frames=[_frame(0), _frame(1)],
        frames_remaining_after_trigger=0,
        confidences=[0.6, 0.8],
    )


def _camera_of(instance, camera_id: str = "cam-a"):
    return instance._cameras[camera_id]


class FailingStore(FakeStore):

    def __init__(self, *failing: str) -> None:
        super().__init__()
        self.failing = set(failing)
        self.calls: dict[str, int] = {}

    def _maybe_fail(self, name: str) -> None:
        self.calls[name] = self.calls.get(name, 0) + 1
        if name in self.failing:
            raise RuntimeError(f"{name} no disponible")

    def set_health(self, camera_id, status):
        self._maybe_fail("set_health")
        super().set_health(camera_id, status)

    def open_event(self, event):
        self._maybe_fail("open_event")
        super().open_event(event)

    def absorb_review_events(self, camera_id, alert_event_id, within_seconds):
        self._maybe_fail("absorb_review_events")
        return super().absorb_review_events(camera_id, alert_event_id, within_seconds)

    def refresh_thresholds(self, thresholds):
        self._maybe_fail("refresh_thresholds")
        super().refresh_thresholds(thresholds)

    def pending_description_count(self):
        self._maybe_fail("pending_description_count")
        return 7


def test_a_failure_recording_health_is_logged_and_does_not_propagate(make_instance):
    store = FailingStore("set_health")
    instance, _ = make_instance(["cam-a"], {"cam-a": FakeCameraStream("cam-a")}, store=store)

    instance._report_health("cam-a", "ok")

    assert store.calls["set_health"] == 1


def test_an_alert_absorbs_the_review_signals_that_came_before_it(make_instance):
    instance, store = make_instance(["cam-a"], {"cam-a": FakeCameraStream("cam-a")})
    clip = _clip("accident")

    instance._open_event(_camera_of(instance), clip)

    assert [event.event_id for event in store.opened] == [clip.event_id]
    assert store.absorbed and store.absorbed[0][:2] == ("cam-a", clip.event_id)


def test_other_incidents_do_not_absorb_anything(make_instance):
    instance, store = make_instance(["cam-a"], {"cam-a": FakeCameraStream("cam-a")})

    instance._open_event(_camera_of(instance), _clip("fire"))

    assert len(store.opened) == 1
    assert store.absorbed == []


def test_if_the_event_cannot_be_registered_at_trigger_time_nothing_is_absorbed(make_instance):
    store = FailingStore("open_event")
    instance, _ = make_instance(["cam-a"], {"cam-a": FakeCameraStream("cam-a")}, store=store)

    instance._open_event(_camera_of(instance), _clip("accident"))

    assert store.opened == []
    assert "absorb_review_events" not in store.calls


def test_a_failure_absorbing_review_signals_leaves_the_alert_registered(make_instance):
    store = FailingStore("absorb_review_events")
    instance, _ = make_instance(["cam-a"], {"cam-a": FakeCameraStream("cam-a")}, store=store)

    instance._open_event(_camera_of(instance), _clip("accident"))

    assert len(store.opened) == 1


def test_a_clip_that_closes_without_a_registered_event_is_inserted_as_a_new_one(make_instance):
    instance, store = make_instance(["cam-a"], {"cam-a": FakeCameraStream("cam-a")})
    clip = _clip("fire")

    instance._finalize_clip(_camera_of(instance), clip)

    assert [event.event_id for event in store.events] == [clip.event_id]
    assert store.events[0].confidence == 0.7
    assert store.events[0].clip_path == f"cam-a/{clip.event_id}.mp4"


def test_frames_arriving_before_a_camera_is_ready_are_ignored(make_instance):
    stream = FakeCameraStream("cam-a", ready=False)
    instance, store = make_instance(["cam-a"], {"cam-a": stream})
    camera = _camera_of(instance)
    assert camera.pipeline is None

    instance._bootstrap(camera)

    assert camera.pipeline is None
    assert store.health == []


def test_the_control_loop_survives_failures_of_the_database(make_instance, monkeypatch):
    monkeypatch.setattr(instance_module, "_THRESHOLDS_REFRESH_SECONDS", 0.0)
    monkeypatch.setattr(instance_module, "_QUEUE_DEPTH_REFRESH_SECONDS", 0.0)
    monkeypatch.setattr(instance_module.settings, "preview_enabled", True)
    monkeypatch.setattr(instance_module.settings, "stats_log_interval_seconds", 0.0)
    store = FailingStore()
    instance, _ = make_instance(["cam-a"], {"cam-a": FakeCameraStream("cam-a")}, store=store)

    instance.start()
    store.failing.update({"refresh_thresholds", "pending_description_count"})

    assert _wait_for(lambda: store.calls.get("refresh_thresholds", 0) >= 3 and store.calls.get("pending_description_count", 0) >= 2)
    assert instance.wait(0) is False


def test_the_queue_depth_shown_in_the_preview_comes_from_the_database(make_instance, monkeypatch):
    monkeypatch.setattr(instance_module, "_QUEUE_DEPTH_REFRESH_SECONDS", 0.0)
    monkeypatch.setattr(instance_module.settings, "preview_enabled", True)
    store = FailingStore()
    instance, _ = make_instance(["cam-a"], {"cam-a": FakeCameraStream("cam-a")}, store=store)

    instance.start()

    assert _wait_for(lambda: instance._queue_depth == 7)


def test_with_the_preview_enabled_each_camera_publishes_its_annotated_frame(make_instance, monkeypatch):
    published = []
    monkeypatch.setattr(instance_module.settings, "preview_enabled", True)
    monkeypatch.setattr(instance_module.settings, "preview_interval_seconds", 0.0)
    monkeypatch.setattr(instance_module, "publish_preview", lambda camera_id, image: published.append((camera_id, image.shape)))
    streams = {"cam-a": FakeCameraStream("cam-a", fps=30.0)}
    instance, _ = make_instance(["cam-a"], streams)
    instance.start()

    for index in range(10):
        streams["cam-a"].push(_frame(index))

    assert _wait_for(lambda: len(published) >= 1)
    assert published[0][0] == "cam-a"
    assert published[0][1] == (8, 8, 3)
