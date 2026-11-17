from __future__ import annotations

from monitoreo_urbano.common.config import settings
from monitoreo_urbano.common.repository import (
    delete_runtime_override,
    get_runtime_overrides,
    set_runtime_override,
)
from monitoreo_urbano.common.runtime_config import RuntimeThresholds


def test_get_runtime_overrides_empty_by_default(db_conn):
    assert get_runtime_overrides(db_conn) == {}


def test_set_and_get_runtime_override(db_conn):
    set_runtime_override(db_conn, "yolo_confidence_threshold", 0.5)
    assert get_runtime_overrides(db_conn) == {"yolo_confidence_threshold": "0.5"}


def test_set_runtime_override_upserts(db_conn):
    set_runtime_override(db_conn, "yolo_confidence_threshold", 0.5)
    set_runtime_override(db_conn, "yolo_confidence_threshold", 0.8)
    assert get_runtime_overrides(db_conn) == {"yolo_confidence_threshold": "0.8"}


def test_delete_runtime_override(db_conn):
    set_runtime_override(db_conn, "yolo_confidence_threshold", 0.5)
    delete_runtime_override(db_conn, "yolo_confidence_threshold")
    assert get_runtime_overrides(db_conn) == {}


def test_thresholds_use_settings_default_before_refresh():
    thresholds = RuntimeThresholds()
    assert thresholds.yolo_confidence_threshold == settings.yolo_confidence_threshold
    assert thresholds.event_dedup_radius_px == settings.event_dedup_radius_px


def test_thresholds_refresh_picks_up_override(db_conn):
    thresholds = RuntimeThresholds()
    set_runtime_override(db_conn, "event_dedup_radius_px", 250.0)

    thresholds.refresh(db_conn)

    assert thresholds.event_dedup_radius_px == 250.0
    assert thresholds.yolo_confidence_threshold == settings.yolo_confidence_threshold


def test_thresholds_refresh_reverts_after_override_removed(db_conn):
    thresholds = RuntimeThresholds()
    set_runtime_override(db_conn, "event_dedup_radius_px", 250.0)
    thresholds.refresh(db_conn)
    assert thresholds.event_dedup_radius_px == 250.0

    delete_runtime_override(db_conn, "event_dedup_radius_px")
    thresholds.refresh(db_conn)

    assert thresholds.event_dedup_radius_px == settings.event_dedup_radius_px


def test_thresholds_ignores_unknown_keys_in_table(db_conn):
    set_runtime_override(db_conn, "algo_que_no_es_un_umbral", 999.0)
    thresholds = RuntimeThresholds()
    thresholds.refresh(db_conn)
    assert thresholds.yolo_confidence_threshold == settings.yolo_confidence_threshold
