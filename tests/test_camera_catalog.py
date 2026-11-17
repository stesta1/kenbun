from __future__ import annotations

import pytest

from monitoreo_urbano.common.cameras import get_camera, load_camera_catalog

CATALOG = """
cameras:
  - camera_id: cam-a
    label: "Av. Uno"
    source: "rtsp://localhost:8554/cam-a"
    comuna: "1"
  - camera_id: cam-b
    label: "Av. Dos"
    source: "rtsp://localhost:8554/cam-b"
    comuna: "2"
"""


def test_load_catalog_indexes_cameras_by_id(tmp_path):
    path = tmp_path / "cameras.yaml"
    path.write_text(CATALOG, encoding="utf-8")

    catalog = load_camera_catalog(path)

    assert set(catalog) == {"cam-a", "cam-b"}
    assert catalog["cam-a"].label == "Av. Uno"


def test_load_catalog_without_cameras_key_is_empty(tmp_path):
    path = tmp_path / "cameras.yaml"
    path.write_text("", encoding="utf-8")

    assert load_camera_catalog(path) == {}


def test_load_catalog_missing_file_explains_how_to_fix(tmp_path):
    with pytest.raises(FileNotFoundError, match="cameras.example.yaml"):
        load_camera_catalog(tmp_path / "no_existe.yaml")


def test_get_camera_returns_the_requested_one(tmp_path):
    path = tmp_path / "cameras.yaml"
    path.write_text(CATALOG, encoding="utf-8")

    assert get_camera("cam-b", path).label == "Av. Dos"


def test_get_camera_unknown_id_raises_key_error(tmp_path):
    path = tmp_path / "cameras.yaml"
    path.write_text(CATALOG, encoding="utf-8")

    with pytest.raises(KeyError, match="cam-x"):
        get_camera("cam-x", path)
