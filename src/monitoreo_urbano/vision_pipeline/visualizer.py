from __future__ import annotations

import os
import time
from pathlib import Path

import cv2
import numpy as np

from monitoreo_urbano.common.config import settings
from monitoreo_urbano.common.logging import get_logger
from monitoreo_urbano.common.schemas import TrackedObject

logger = get_logger(__name__)

_PALETTE = [
    (66, 135, 245),
    (66, 245, 135),
    (245, 135, 66),
    (245, 66, 200),
    (66, 220, 245),
    (200, 245, 66),
    (135, 66, 245),
]


def _color_for_track(track_id: int) -> tuple[int, int, int]:
    return _PALETTE[track_id % len(_PALETTE)]


def draw_overlay(
    image: np.ndarray,
    tracked_objects: list[TrackedObject],
    recording_track_ids: set[int],
    fps_estimate: float,
    queue_depth: int,
) -> np.ndarray:
    frame = image.copy()

    for obj in tracked_objects:
        color = _color_for_track(obj.track_id)
        x1, y1, x2, y2 = (int(obj.bbox.x1), int(obj.bbox.y1), int(obj.bbox.x2), int(obj.bbox.y2))
        is_recording = obj.track_id in recording_track_ids
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3 if is_recording else 2)

        label = f"{obj.class_name} #{obj.track_id} {obj.confidence:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), color, -1)
        cv2.putText(frame, label, (x1 + 3, max(12, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    if recording_track_ids:
        cv2.circle(frame, (24, 24), 8, (0, 0, 255), -1)
        cv2.putText(
            frame, f"REC ({len(recording_track_ids)})", (40, 32),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA,
        )

    status = f"{fps_estimate:.1f} fps | en cola para describir: {queue_depth}"
    cv2.putText(frame, status, (10, frame.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    return frame


def publish_preview(camera_id: str, frame: np.ndarray) -> None:
    try:
        settings.live_preview_dir.mkdir(parents=True, exist_ok=True)
        target = settings.live_preview_dir / f"{camera_id}.jpg"
        tmp_path = target.with_suffix(".tmp.jpg")
        ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            return
        tmp_path.write_bytes(buffer.tobytes())

        for attempt in range(5):
            try:
                os.replace(tmp_path, target)
                return
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.01)
    except Exception:
        logger.debug("No se pudo publicar el preview de %s", camera_id, exc_info=True)


def wrap_text(text: str, max_width_px: int, font_scale: float = 0.5, thickness: int = 1) -> list[str]:
    words = text.split()
    if not words:
        return [""]

    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        (w, _), _ = cv2.getTextSize(candidate, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        if w <= max_width_px:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def render_text_panel(
    entries: list[str],
    width: int = 900,
    height: int = 700,
    title: str = "",
    font_scale: float = 0.5,
) -> np.ndarray:
    canvas = np.full((height, width, 3), (30, 30, 30), dtype=np.uint8)
    line_height = 20
    margin = 12

    if title:
        cv2.rectangle(canvas, (0, 0), (width, 34), (50, 50, 50), -1)
        cv2.putText(canvas, title, (margin, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        top = 34 + margin
    else:
        top = margin

    max_lines = max(1, (height - top - margin) // line_height)

    wrapped: list[str] = []
    for entry in entries:
        for sub_line in entry.split("\n"):
            wrapped.extend(wrap_text(sub_line, width - 2 * margin, font_scale))
        wrapped.append("")

    visible = wrapped[-max_lines:]
    for i, line in enumerate(visible):
        y = top + (i + 1) * line_height
        cv2.putText(canvas, line, (margin, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (220, 220, 220), 1, cv2.LINE_AA)

    return canvas


def tile_images(images: list[tuple[str, np.ndarray | None]], tile_width: int = 480, tile_height: int = 360) -> np.ndarray:
    if not images:
        return np.full((tile_height, tile_width, 3), (20, 20, 20), dtype=np.uint8)

    cols = max(1, int(np.ceil(np.sqrt(len(images)))))
    rows = int(np.ceil(len(images) / cols))
    grid = np.full((rows * tile_height, cols * tile_width, 3), (15, 15, 15), dtype=np.uint8)

    for idx, (label, image) in enumerate(images):
        row, col = divmod(idx, cols)
        if image is None:
            tile = np.full((tile_height, tile_width, 3), (40, 40, 40), dtype=np.uint8)
            text = "SIN SENAL"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.putText(
                tile, text, ((tile_width - tw) // 2, (tile_height + th) // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2, cv2.LINE_AA,
            )
        else:
            tile = cv2.resize(image, (tile_width, tile_height))

        cv2.rectangle(tile, (0, 0), (tile_width - 1, 28), (0, 0, 0), -1)
        cv2.putText(tile, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.rectangle(tile, (0, 0), (tile_width - 1, tile_height - 1), (70, 70, 70), 1)

        y0, x0 = row * tile_height, col * tile_width
        grid[y0 : y0 + tile_height, x0 : x0 + tile_width] = tile

    return grid


def load_preview(camera_id: str, stale_after_seconds: float = 5.0) -> np.ndarray | None:
    path: Path = settings.live_preview_dir / f"{camera_id}.jpg"
    if not path.exists():
        return None
    if (time.time() - path.stat().st_mtime) > stale_after_seconds:
        return None

    data = None
    for attempt in range(3):
        try:
            data = np.fromfile(str(path), dtype=np.uint8)
            break
        except (PermissionError, OSError):
            if attempt == 2:
                return None
            time.sleep(0.01)

    if data is None or data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)
