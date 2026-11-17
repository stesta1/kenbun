from __future__ import annotations

from datetime import datetime

import numpy as np

from monitoreo_urbano.common.repository import set_runtime_override
from monitoreo_urbano.common.runtime_config import RuntimeThresholds
from monitoreo_urbano.vision_pipeline.camera_health import CameraHealthMonitor
from monitoreo_urbano.vision_pipeline.camera_source import Frame


def _frame(image: np.ndarray, index: int = 0) -> Frame:
    return Frame(image=image, timestamp=datetime.utcnow(), frame_index=index)


def _black_image(size: int = 20) -> np.ndarray:
    return np.zeros((size, size, 3), dtype=np.uint8)


def _noisy_image(seed: int, size: int = 20) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(size, size, 3), dtype=np.uint8)


def _smooth_image(size: int = 40) -> np.ndarray:
    gradient = np.tile(np.linspace(0, 255, size, dtype=np.uint8), (size, 1))
    return np.stack([gradient] * 3, axis=-1)


def _monitor(**overrides) -> CameraHealthMonitor:
    defaults = dict(fps=1.0, reconnect_delay_seconds=1.0, sustained_seconds=3.0, signal_loss_sustained_seconds=3.0)
    defaults.update(overrides)
    return CameraHealthMonitor(**defaults)


def test_does_not_trigger_on_normal_varying_frames():
    monitor = _monitor()
    reasons = [monitor.check(_frame(_noisy_image(seed=i), index=i)) for i in range(10)]
    assert all(r is None for r in reasons)


def test_does_not_repeat_blank_notification_while_still_sustained():
    monitor = _monitor()
    for i in range(3):
        monitor.check(_frame(_black_image(), index=i))
    again = monitor.check(_frame(_black_image(), index=3))
    assert again is None


def test_triggers_blank_after_sustained_black_frames():
    monitor = _monitor()
    reasons = [monitor.check(_frame(_black_image(), index=i)) for i in range(3)]
    assert reasons == [None, None, "tapada"]


def test_recovers_and_can_retrigger_after_normal_frame():
    monitor = _monitor()
    for i in range(3):
        monitor.check(_frame(_black_image(), index=i))

    monitor.check(_frame(_noisy_image(seed=99), index=3))

    reasons = [monitor.check(_frame(_black_image(), index=i)) for i in range(4, 7)]
    assert reasons == [None, None, "tapada"]


def test_signal_loss_triggers_only_after_sustained_retries():
    monitor = _monitor()
    reasons = [monitor.on_signal_loss() for _ in range(3)]
    assert reasons == [None, None, "corte_de_señal"]


def test_uses_live_thresholds_when_provided(db_conn):
    thresholds = RuntimeThresholds()
    set_runtime_override(db_conn, "camera_health_sustained_seconds", 1.0)
    thresholds.refresh(db_conn)

    monitor = CameraHealthMonitor(fps=1.0, reconnect_delay_seconds=1.0, thresholds=thresholds)
    reasons = [monitor.check(_frame(_black_image(), index=i)) for i in range(2)]
    assert reasons == ["tapada", None]


def test_real_frame_after_signal_loss_resets_the_streak():
    monitor = _monitor()
    monitor.on_signal_loss()
    monitor.on_signal_loss()
    monitor.check(_frame(_noisy_image(seed=1), index=0))

    reasons = [monitor.on_signal_loss() for _ in range(2)]
    assert reasons == [None, None]


def test_reports_ok_when_normal_frames_return_after_blank():
    monitor = _monitor()
    for i in range(3):
        monitor.check(_frame(_black_image(), index=i))
    assert monitor.check(_frame(_noisy_image(seed=5), index=3)) == "ok"


def test_reports_ok_only_once():
    monitor = _monitor()
    for i in range(3):
        monitor.check(_frame(_black_image(), index=i))
    monitor.check(_frame(_noisy_image(seed=5), index=3))
    assert monitor.check(_frame(_noisy_image(seed=6), index=4)) is None


def test_reports_ok_when_the_signal_returns_after_a_cut():
    monitor = _monitor()
    for _ in range(3):
        monitor.on_signal_loss()
    assert monitor.check(_frame(_noisy_image(seed=7), index=0)) == "ok"


def test_stays_in_signal_loss_while_returning_frames_are_still_black():
    monitor = _monitor()
    for _ in range(3):
        monitor.on_signal_loss()
    assert monitor.check(_frame(_black_image(), index=0)) is None


def test_never_reports_ok_without_a_previous_problem():
    monitor = _monitor()
    reasons = [monitor.check(_frame(_noisy_image(seed=i), index=i)) for i in range(5)]
    assert reasons == [None] * 5


def test_a_smooth_blurred_image_counts_as_covered_even_if_not_black():
    monitor = _monitor()
    reasons = [monitor.check(_frame(_smooth_image(), index=i)) for i in range(3)]
    assert reasons == [None, None, "tapada"]


def test_a_sharp_image_with_detail_is_not_covered():
    monitor = _monitor()
    reasons = [monitor.check(_frame(_noisy_image(seed=1, size=40), index=i)) for i in range(5)]
    assert reasons == [None] * 5


def test_covered_sharpness_threshold_can_be_relaxed():
    monitor = _monitor(covered_sharpness_threshold=0.0)
    reasons = [monitor.check(_frame(_smooth_image(), index=i)) for i in range(5)]
    assert reasons == [None] * 5
