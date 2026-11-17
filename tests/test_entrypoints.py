from __future__ import annotations

import signal
import sys

import pytest

from monitoreo_urbano.common.config import settings
from monitoreo_urbano.common.schemas import CameraInfo
from monitoreo_urbano.event_store import main as store_main
from monitoreo_urbano.query_service import main as query_main
from monitoreo_urbano.vision_pipeline import main as vision_main
from monitoreo_urbano.vlm_describer import main as vlm_main


class Closable:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_the_indexer_closes_the_store_when_the_loop_ends(monkeypatch):
    store = Closable()
    monkeypatch.setattr(store_main, "EventStore", lambda: store)
    monkeypatch.setattr(store_main, "run_indexer_loop", lambda received: None)

    store_main.main()

    assert store.closed is True


def test_the_indexer_closes_the_store_even_if_the_loop_fails(monkeypatch):
    store = Closable()
    monkeypatch.setattr(store_main, "EventStore", lambda: store)

    def broken(received):
        raise RuntimeError("falló el indexado")

    monkeypatch.setattr(store_main, "run_indexer_loop", broken)

    with pytest.raises(RuntimeError):
        store_main.main()

    assert store.closed is True


def test_the_query_service_starts_the_api_on_the_configured_address(monkeypatch):
    calls = []
    monkeypatch.setattr(query_main.uvicorn, "run", lambda app, **kwargs: calls.append((app, kwargs)))

    query_main.main()

    assert calls == [
        (
            "monitoreo_urbano.query_service.api:app",
            {"host": settings.api_host, "port": settings.api_port, "reload": False},
        )
    ]


@pytest.fixture
def vlm_parts(monkeypatch):
    conn = Closable()
    seen = {}
    monkeypatch.setattr(vlm_main, "get_connection", lambda: conn)
    monkeypatch.setattr(vlm_main, "init_db", lambda received: seen.setdefault("init", received))
    monkeypatch.setattr(vlm_main, "build_describer", lambda: "descriptor")
    monkeypatch.setattr(vlm_main, "run_worker_loop", lambda received, describer: seen.update(loop=(received, describer)))
    monkeypatch.setattr(settings, "vlm_device", "cuda:0")
    seen["conn"] = conn
    return seen


def test_the_vlm_worker_runs_the_loop_and_closes_the_connection(monkeypatch, vlm_parts):
    monkeypatch.setattr(sys, "argv", ["main"])

    vlm_main.main()

    assert vlm_parts["loop"] == (vlm_parts["conn"], "descriptor")
    assert vlm_parts["conn"].closed is True
    assert settings.vlm_device == "cuda:0"


def test_the_vlm_worker_takes_its_gpu_from_the_command_line(monkeypatch, vlm_parts):
    monkeypatch.setattr(sys, "argv", ["main", "--device", "cuda:1"])

    vlm_main.main()

    assert settings.vlm_device == "cuda:1"


class FakeInstance:
    def __init__(self, cameras, **kwargs) -> None:
        self.cameras = cameras
        self.kwargs = kwargs
        self.events: list[str] = []

    def start(self) -> None:
        self.events.append("start")

    def wait(self, timeout) -> bool:
        self.events.append("wait")
        return True

    def stop(self) -> None:
        self.events.append("stop")

    def request_stop(self) -> None:
        self.events.append("request_stop")


@pytest.fixture
def vision_parts(monkeypatch):
    parts = {"handlers": {}}

    class FakeStore:
        def refresh_thresholds(self, thresholds) -> None:
            parts["refreshed"] = True

    def fake_instance(cameras, **kwargs):
        parts["instance"] = FakeInstance(cameras, **kwargs)
        return parts["instance"]

    monkeypatch.setattr(
        vision_main,
        "get_camera",
        lambda camera_id: CameraInfo(camera_id=camera_id, label=camera_id, source=f"rtsp://x/{camera_id}", comuna="1"),
    )
    monkeypatch.setattr(vision_main, "PostgresInstanceStore", FakeStore)
    monkeypatch.setattr(vision_main, "YOLODetector", lambda thresholds, device: ("detector", device))
    monkeypatch.setattr(vision_main, "VisionInstance", fake_instance)
    monkeypatch.setattr(vision_main.signal, "signal", lambda number, handler: parts["handlers"].update({number: handler}))
    monkeypatch.setattr(settings, "preview_enabled", False)
    return parts


def test_run_builds_the_instance_with_its_cameras_and_stops_it_at_the_end(vision_parts):
    vision_main.run(["cam-a", "cam-b"], analysis_fps=6, preview=True, device="cuda:1", name="vision-1")

    instance = vision_parts["instance"]
    assert [camera.camera_id for camera in instance.cameras] == ["cam-a", "cam-b"]
    assert instance.kwargs["analysis_fps"] == 6
    assert instance.kwargs["name"] == "vision-1"
    assert instance.kwargs["detector"] == ("detector", "cuda:1")
    assert vision_parts["refreshed"] is True
    assert settings.preview_enabled is True
    assert instance.events == ["start", "wait", "stop"]


def test_run_stops_the_instance_cleanly_when_a_stop_signal_arrives(vision_parts):
    vision_main.run(["cam-a"])

    handler = vision_parts["handlers"][signal.SIGINT]
    handler(signal.SIGINT, None)

    assert "request_stop" in vision_parts["instance"].events
    assert signal.SIGTERM in vision_parts["handlers"]
