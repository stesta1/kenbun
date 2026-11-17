from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pytest

from monitoreo_urbano.vision_pipeline import clip_writer
from monitoreo_urbano.vision_pipeline.camera_health import CameraHealthMonitor
from monitoreo_urbano.vision_pipeline.camera_source import Frame
from monitoreo_urbano.vision_pipeline.frame_encoding import (
    EncodedFrame,
    compute_health_signals,
    decode_image,
    encode_jpeg,
    make_capture_hook,
    stored_form,
)


def _noisy(seed: int = 0, height: int = 72, width: int = 128) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 255, size=(height, width, 3), dtype=np.uint8)


def _smooth(height: int = 72, width: int = 128) -> np.ndarray:
    gradient = np.tile(np.linspace(0, 255, width, dtype=np.uint8), (height, 1))
    return np.stack([gradient] * 3, axis=-1)


def _frame(image: np.ndarray, index: int = 0) -> Frame:
    return Frame(image=image, timestamp=datetime.utcnow(), frame_index=index)


def test_jpeg_round_trip_keeps_the_shape_and_is_much_smaller_than_raw():
    image = _smooth(720, 1280)

    jpeg = encode_jpeg(image, quality=85)
    decoded = decode_image(EncodedFrame(jpeg=jpeg, timestamp=datetime.utcnow(), frame_index=0))

    assert decoded.shape == image.shape
    assert len(jpeg) < image.nbytes / 10


def test_decoding_a_plain_frame_returns_its_image():
    image = _noisy()
    assert decode_image(_frame(image)) is image


def test_decoding_a_corrupt_jpeg_fails_clearly():
    with pytest.raises(ValueError):
        decode_image(EncodedFrame(jpeg=b"no es un jpeg", timestamp=datetime.utcnow(), frame_index=3))


def test_stored_form_is_the_frame_itself_when_nothing_was_compressed():
    frame = _frame(_noisy())
    assert stored_form(frame) is frame


def test_stored_form_drops_the_raw_image_when_the_frame_comes_compressed():
    frame = _frame(_noisy(), index=7)
    frame.jpeg = encode_jpeg(frame.image, 85)

    stored = stored_form(frame)

    assert isinstance(stored, EncodedFrame)
    assert (stored.frame_index, stored.timestamp) == (7, frame.timestamp)
    assert not hasattr(stored, "image")


def test_capture_hook_fills_the_jpeg_and_the_health_signals():
    frame = _frame(_noisy())

    make_capture_hook(jpeg_quality=85)(frame)

    assert frame.jpeg is not None
    assert frame.health is not None and len(frame.health) == 2


def test_capture_hook_does_not_compress_when_the_quality_is_zero():
    frame = _frame(_noisy())

    make_capture_hook(jpeg_quality=0)(frame)

    assert frame.jpeg is None
    assert frame.health is not None


@pytest.mark.parametrize("image", [np.zeros((72, 128, 3), dtype=np.uint8), _noisy(1), _noisy(2), _smooth()])
def test_precomputed_health_signals_give_the_same_decision_as_computing_them_in_place(image):
    def decisions(precompute: bool) -> list[str | None]:
        monitor = CameraHealthMonitor(fps=1.0, reconnect_delay_seconds=1.0, sustained_seconds=3.0)
        out = []
        for index in range(5):
            frame = _frame(image, index)
            if precompute:
                frame.health = compute_health_signals(image)
            out.append(monitor.check(frame))
        return out

    assert decisions(precompute=True) == decisions(precompute=False)


def test_health_signals_tell_a_black_frame_from_a_busy_one():
    black_std, _ = compute_health_signals(np.zeros((72, 128, 3), dtype=np.uint8))
    busy_std, busy_sharpness = compute_health_signals(_noisy())

    assert black_std < 1.0
    assert busy_std > 20.0 and busy_sharpness > 100.0


def test_a_clip_can_be_written_from_compressed_frames():
    frames = []
    for index in range(6):
        image = _noisy(index, 72, 128)
        frames.append(
            EncodedFrame(jpeg=encode_jpeg(image, 90), timestamp=datetime.utcnow(), frame_index=index)
        )

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "clip.mp4"
        clip_writer._write_with_opencv(frames, 5.0, out)
        capture = cv2.VideoCapture(str(out))
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        capture.release()

    assert count == 6
    assert width == 128
