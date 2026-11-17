from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

from monitoreo_urbano.common.config import settings
from monitoreo_urbano.common.logging import get_logger
from monitoreo_urbano.common.object_storage import get_clips_client
from monitoreo_urbano.vision_pipeline.camera_source import Frame
from monitoreo_urbano.vision_pipeline.frame_encoding import EncodedFrame, decode_image

logger = get_logger(__name__)

_ffmpeg_nvenc_checked: bool | None = None


def _has_ffmpeg_nvenc() -> bool:
    global _ffmpeg_nvenc_checked
    if _ffmpeg_nvenc_checked is not None:
        return _ffmpeg_nvenc_checked

    if shutil.which("ffmpeg") is None:
        _ffmpeg_nvenc_checked = False
        return False

    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=10
        )
        _ffmpeg_nvenc_checked = "h264_nvenc" in result.stdout
    except Exception:
        _ffmpeg_nvenc_checked = False

    return _ffmpeg_nvenc_checked


def write_clip(camera_id: str, event_id: str, frames: list[Frame | EncodedFrame], fps: float) -> str:
    if not frames:
        raise ValueError("No se puede escribir un clip sin frames")

    object_key = f"{camera_id}/{event_id}.mp4"

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / f"{event_id}.mp4"

        if settings.clip_use_nvenc and _has_ffmpeg_nvenc():
            _write_with_ffmpeg_nvenc(frames, fps, tmp_path)
        else:
            _write_with_opencv(frames, fps, tmp_path)

        client = get_clips_client()
        client.fput_object(settings.s3_bucket_clips, object_key, str(tmp_path))

    return object_key


def _write_with_opencv(frames: list[Frame | EncodedFrame], fps: float, out_path: Path) -> None:
    height, width = decode_image(frames[0]).shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    try:
        for frame in frames:
            writer.write(decode_image(frame))
    finally:
        writer.release()


def _write_with_ffmpeg_nvenc(frames: list[Frame | EncodedFrame], fps: float, out_path: Path) -> None:
    height, width = decode_image(frames[0]).shape[:2]
    cmd = [
        "ffmpeg",
        "-y",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "-",
        "-c:v", "h264_nvenc",
        "-preset", "p4",
        "-pix_fmt", "yuv420p",
        str(out_path),
    ]
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    assert process.stdin is not None
    try:
        for frame in frames:
            process.stdin.write(_ensure_contiguous(decode_image(frame)).tobytes())
    finally:
        process.stdin.close()
        stderr = process.stderr.read() if process.stderr else b""
        return_code = process.wait()
        if return_code != 0:
            logger.warning("ffmpeg NVENC falló (code=%s), reintentando con OpenCV: %s", return_code, stderr[-500:])
            _write_with_opencv(frames, fps, out_path)


def _ensure_contiguous(image: np.ndarray) -> np.ndarray:
    return image if image.flags["C_CONTIGUOUS"] else np.ascontiguousarray(image)
