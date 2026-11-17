from __future__ import annotations

import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pytest

from monitoreo_urbano.common.config import settings
from monitoreo_urbano.vision_pipeline import clip_writer
from monitoreo_urbano.vision_pipeline.camera_source import Frame
from monitoreo_urbano.vlm_describer import worker


class FakeMinio:

    def __init__(self, root: Path) -> None:
        self.root = root
        self.uploads: list[tuple[str, str]] = []

    def fput_object(self, bucket: str, key: str, path: str) -> None:
        target = self.root / key
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        self.uploads.append((bucket, key))

    def fget_object(self, bucket: str, key: str, path: str) -> None:
        shutil.copyfile(self.root / key, path)


def _frames(count: int = 6) -> list[Frame]:
    rng = np.random.default_rng(0)
    return [
        Frame(image=rng.integers(0, 255, size=(48, 64, 3), dtype=np.uint8), timestamp=datetime.utcnow(), frame_index=i)
        for i in range(count)
    ]


def _count_frames(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total


@pytest.fixture
def storage(monkeypatch, tmp_path) -> FakeMinio:
    fake = FakeMinio(tmp_path / "bucket")
    monkeypatch.setattr(clip_writer, "get_clips_client", lambda: fake)
    monkeypatch.setattr(worker, "get_clips_client", lambda: fake)
    return fake


def test_a_clip_without_frames_is_rejected():
    with pytest.raises(ValueError, match="sin frames"):
        clip_writer.write_clip("cam-01", "ev-1", [], fps=10)


def test_write_clip_stores_a_readable_video_under_camera_and_event(monkeypatch, storage):
    monkeypatch.setattr(settings, "clip_use_nvenc", False)

    key = clip_writer.write_clip("cam-01", "ev-1", _frames(6), fps=10)

    assert key == "cam-01/ev-1.mp4"
    assert storage.uploads == [(settings.s3_bucket_clips, "cam-01/ev-1.mp4")]
    assert _count_frames(storage.root / key) == 6


def test_write_clip_uses_nvenc_only_when_enabled_and_available(monkeypatch, storage):
    used = []
    monkeypatch.setattr(clip_writer, "_has_ffmpeg_nvenc", lambda: True)
    monkeypatch.setattr(clip_writer, "_write_with_ffmpeg_nvenc", lambda frames, fps, out: used.append("nvenc") or out.write_bytes(b"x"))
    monkeypatch.setattr(settings, "clip_use_nvenc", True)

    clip_writer.write_clip("cam-01", "ev-2", _frames(2), fps=10)

    assert used == ["nvenc"]


@pytest.fixture(autouse=True)
def reset_nvenc_cache(monkeypatch):
    monkeypatch.setattr(clip_writer, "_ffmpeg_nvenc_checked", None)


def test_nvenc_is_unavailable_without_ffmpeg(monkeypatch):
    monkeypatch.setattr(clip_writer.shutil, "which", lambda name: None)

    assert clip_writer._has_ffmpeg_nvenc() is False


def test_nvenc_is_detected_from_the_encoder_list(monkeypatch):
    monkeypatch.setattr(clip_writer.shutil, "which", lambda name: "ffmpeg")
    calls = []

    def fake_run(*args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, stdout=" V..... h264_nvenc NVIDIA NVENC H.264", stderr="")

    monkeypatch.setattr(clip_writer.subprocess, "run", fake_run)

    assert clip_writer._has_ffmpeg_nvenc() is True
    assert clip_writer._has_ffmpeg_nvenc() is True
    assert len(calls) == 1


def test_nvenc_is_unavailable_if_the_encoder_is_not_listed(monkeypatch):
    monkeypatch.setattr(clip_writer.shutil, "which", lambda name: "ffmpeg")
    monkeypatch.setattr(
        clip_writer.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout=" V..... libx264", stderr="")
    )

    assert clip_writer._has_ffmpeg_nvenc() is False


def test_nvenc_is_unavailable_if_ffmpeg_cannot_be_queried(monkeypatch):
    monkeypatch.setattr(clip_writer.shutil, "which", lambda name: "ffmpeg")

    def broken(*args, **kwargs):
        raise OSError("sin permisos")

    monkeypatch.setattr(clip_writer.subprocess, "run", broken)

    assert clip_writer._has_ffmpeg_nvenc() is False


class FakeStdin:
    def __init__(self) -> None:
        self.written = 0
        self.closed = False

    def write(self, data: bytes) -> None:
        self.written += len(data)

    def close(self) -> None:
        self.closed = True


class FakeProcess:
    def __init__(self, return_code: int) -> None:
        self.stdin = FakeStdin()
        self.stderr = type("Err", (), {"read": staticmethod(lambda: b"error de codificacion")})()
        self._return_code = return_code

    def wait(self) -> int:
        return self._return_code


def test_ffmpeg_receives_every_frame_as_raw_video(monkeypatch, tmp_path):
    process = FakeProcess(0)
    commands = []
    monkeypatch.setattr(clip_writer.subprocess, "Popen", lambda cmd, **kwargs: commands.append(cmd) or process)
    frames = _frames(3)

    clip_writer._write_with_ffmpeg_nvenc(frames, 10, tmp_path / "out.mp4")

    assert process.stdin.written == 3 * 48 * 64 * 3
    assert process.stdin.closed is True
    assert "h264_nvenc" in commands[0] and "64x48" in commands[0]


def test_a_failed_ffmpeg_run_falls_back_to_opencv(monkeypatch, tmp_path):
    monkeypatch.setattr(clip_writer.subprocess, "Popen", lambda cmd, **kwargs: FakeProcess(1))
    out = tmp_path / "out.mp4"

    clip_writer._write_with_ffmpeg_nvenc(_frames(4), 10, out)

    assert _count_frames(out) == 4


def test_non_contiguous_images_are_copied_before_being_sent_to_ffmpeg():
    image = np.zeros((4, 6, 3), dtype=np.uint8)[:, ::2]

    assert not image.flags["C_CONTIGUOUS"]
    assert clip_writer._ensure_contiguous(image).flags["C_CONTIGUOUS"]


def test_the_worker_reads_back_every_frame_of_a_stored_clip(monkeypatch, storage):
    monkeypatch.setattr(settings, "clip_use_nvenc", False)
    key = clip_writer.write_clip("cam-01", "ev-3", _frames(5), fps=10)

    frames = worker._read_clip_frames(key)

    assert len(frames) == 5
    assert frames[0].shape == (48, 64, 3)
