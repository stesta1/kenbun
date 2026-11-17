from __future__ import annotations

import os
import time
from datetime import datetime

import cv2
import numpy as np
import pytest

from monitoreo_urbano.common.config import settings
from monitoreo_urbano.common.schemas import BoundingBox, TrackedObject
from monitoreo_urbano.vision_pipeline import visualizer


@pytest.fixture
def preview_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "live_preview_dir", tmp_path / "live")
    return tmp_path / "live"


def _object(track_id: int = 1, x1: float = 10, y1: float = 20, x2: float = 60, y2: float = 80) -> TrackedObject:
    return TrackedObject(
        track_id=track_id,
        class_name="car",
        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
        confidence=0.87,
        frame_index=1,
        timestamp=datetime.utcnow(),
    )


def _image(value: int = 0) -> np.ndarray:
    return np.full((120, 160, 3), value, dtype=np.uint8)


def test_overlay_does_not_modify_the_original_image():
    original = _image()
    snapshot = original.copy()

    drawn = visualizer.draw_overlay(original, [_object()], set(), fps_estimate=12.0, queue_depth=0)

    assert np.array_equal(original, snapshot)
    assert not np.array_equal(drawn, snapshot)


def test_overlay_marks_objects_that_are_being_recorded():
    quiet = visualizer.draw_overlay(_image(), [_object()], set(), 10.0, 0)
    recording = visualizer.draw_overlay(_image(), [_object()], {1}, 10.0, 0)

    assert not np.array_equal(quiet, recording)
    assert tuple(recording[24, 24]) == (0, 0, 255)


def test_each_track_keeps_its_own_color():
    assert visualizer._color_for_track(0) == visualizer._color_for_track(len(visualizer._PALETTE))
    assert visualizer._color_for_track(0) != visualizer._color_for_track(1)


def test_preview_is_written_atomically_and_can_be_loaded_back(preview_dir):
    visualizer.publish_preview("cam-01", _image(120))

    assert (preview_dir / "cam-01.jpg").exists()
    assert not (preview_dir / "cam-01.tmp.jpg").exists()
    loaded = visualizer.load_preview("cam-01")
    assert loaded is not None and loaded.shape == (120, 160, 3)


def test_preview_retries_when_the_file_is_locked_by_a_reader(preview_dir, monkeypatch):
    real_replace = os.replace
    attempts = {"n": 0}

    def flaky(src, dst):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise PermissionError("archivo en uso")
        real_replace(src, dst)

    monkeypatch.setattr(visualizer.os, "replace", flaky)

    visualizer.publish_preview("cam-01", _image())

    assert attempts["n"] == 3
    assert (preview_dir / "cam-01.jpg").exists()


def test_preview_failures_never_interrupt_the_analysis(preview_dir, monkeypatch):
    def always_locked(src, dst):
        raise PermissionError("archivo en uso")

    monkeypatch.setattr(visualizer.os, "replace", always_locked)
    monkeypatch.setattr(visualizer.time, "sleep", lambda seconds: None)

    visualizer.publish_preview("cam-01", _image())


def test_a_frame_that_cannot_be_compressed_is_skipped(preview_dir, monkeypatch):
    monkeypatch.setattr(visualizer.cv2, "imencode", lambda *args, **kwargs: (False, None))

    visualizer.publish_preview("cam-01", _image())

    assert not (preview_dir / "cam-01.jpg").exists()


def test_load_preview_returns_none_when_there_is_no_file(preview_dir):
    assert visualizer.load_preview("cam-99") is None


def test_load_preview_ignores_stale_images(preview_dir):
    visualizer.publish_preview("cam-01", _image())
    old = time.time() - 60
    os.utime(preview_dir / "cam-01.jpg", (old, old))

    assert visualizer.load_preview("cam-01") is None


def test_load_preview_ignores_empty_files(preview_dir):
    preview_dir.mkdir(parents=True)
    (preview_dir / "cam-01.jpg").write_bytes(b"")

    assert visualizer.load_preview("cam-01") is None


def test_load_preview_gives_up_after_repeated_read_errors(preview_dir, monkeypatch):
    visualizer.publish_preview("cam-01", _image())

    def unreadable(*args, **kwargs):
        raise PermissionError("archivo en uso")

    monkeypatch.setattr(visualizer.np, "fromfile", unreadable)
    monkeypatch.setattr(visualizer.time, "sleep", lambda seconds: None)

    assert visualizer.load_preview("cam-01") is None


def test_load_preview_recovers_from_a_transient_read_error(preview_dir, monkeypatch):
    visualizer.publish_preview("cam-01", _image())
    real_fromfile = np.fromfile
    attempts = {"n": 0}

    def flaky(*args, **kwargs):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise OSError("lectura interrumpida")
        return real_fromfile(*args, **kwargs)

    monkeypatch.setattr(visualizer.np, "fromfile", flaky)
    monkeypatch.setattr(visualizer.time, "sleep", lambda seconds: None)

    assert visualizer.load_preview("cam-01") is not None


def test_wrap_text_breaks_lines_that_exceed_the_width():
    lines = visualizer.wrap_text("uno dos tres cuatro cinco seis siete ocho", max_width_px=120)

    assert len(lines) > 1
    assert " ".join(lines) == "uno dos tres cuatro cinco seis siete ocho"


def test_wrap_text_of_blank_text_is_one_empty_line():
    assert visualizer.wrap_text("   ", max_width_px=100) == [""]


def test_text_panel_has_the_requested_size_with_and_without_title():
    plain = visualizer.render_text_panel(["Primera línea", "Segunda\ncon salto"], width=300, height=200)
    titled = visualizer.render_text_panel(["Primera línea"], width=300, height=200, title="Eventos")

    assert plain.shape == titled.shape == (200, 300, 3)
    assert not np.array_equal(plain, titled)


def test_text_panel_keeps_only_the_last_lines_that_fit():
    entries = [f"entrada {i}" for i in range(100)]

    panel = visualizer.render_text_panel(entries, width=300, height=100)

    assert panel.shape == (100, 300, 3)


def test_tile_of_nothing_is_a_blank_canvas():
    grid = visualizer.tile_images([], tile_width=40, tile_height=30)

    assert grid.shape == (30, 40, 3)


def test_tile_places_cameras_in_a_grid_and_marks_the_ones_without_signal():
    grid = visualizer.tile_images(
        [("cam-1", _image(200)), ("cam-2", None), ("cam-3", _image(50))], tile_width=80, tile_height=60
    )

    assert grid.shape == (120, 160, 3)
    assert grid[100, 100].tolist() == [15, 15, 15]
