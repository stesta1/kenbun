from __future__ import annotations

import textwrap

import pytest

from monitoreo_urbano.common.instances import DEFAULT_INSTANCE_NAME, get_instance, load_instances

CATALOG = ["cam-a", "cam-b", "cam-c"]


def _write(tmp_path, text: str):
    path = tmp_path / "instances.yaml"
    path.write_text(textwrap.dedent(text), encoding="utf-8")
    return path


def test_without_a_file_there_is_one_instance_with_every_camera(tmp_path):
    instances = load_instances(tmp_path / "no_existe.yaml", CATALOG)

    assert [(i.name, i.cameras) for i in instances] == [(DEFAULT_INSTANCE_NAME, CATALOG)]


def test_without_a_file_and_without_cameras_it_fails_clearly(tmp_path):
    with pytest.raises(ValueError):
        load_instances(tmp_path / "no_existe.yaml", [])


def test_reads_instances_with_their_device_and_frequency(tmp_path):
    path = _write(
        tmp_path,
        """
        instances:
          - name: vision-1
            device: "cuda:0"
            analysis_fps: 6
            cameras: [cam-a, cam-b]
          - name: vision-2
            device: "cuda:1"
            cameras: [cam-c]
        """,
    )

    first, second = load_instances(path, CATALOG)

    assert (first.name, first.device, first.analysis_fps, first.cameras) == ("vision-1", "cuda:0", 6, ["cam-a", "cam-b"])
    assert (second.device, second.analysis_fps) == ("cuda:1", None)


def test_a_star_assigns_every_camera_of_the_catalog(tmp_path):
    path = _write(tmp_path, 'instances:\n  - name: unica\n    cameras: "*"\n')

    assert load_instances(path, CATALOG)[0].cameras == CATALOG


def test_cameras_can_be_written_as_a_comma_separated_string(tmp_path):
    path = _write(tmp_path, 'instances:\n  - name: v\n    cameras: "cam-a, cam-b"\n')

    assert load_instances(path, CATALOG)[0].cameras == ["cam-a", "cam-b"]


def test_a_camera_cannot_belong_to_two_instances(tmp_path):
    path = _write(
        tmp_path,
        """
        instances:
          - {name: uno, cameras: [cam-a, cam-b]}
          - {name: dos, cameras: [cam-b]}
        """,
    )

    with pytest.raises(ValueError, match="cam-b"):
        load_instances(path, CATALOG)


def test_an_unknown_camera_is_rejected(tmp_path):
    path = _write(tmp_path, "instances:\n  - {name: uno, cameras: [cam-zzz]}\n")

    with pytest.raises(ValueError, match="cam-zzz"):
        load_instances(path, CATALOG)


def test_repeated_names_are_rejected(tmp_path):
    path = _write(
        tmp_path,
        """
        instances:
          - {name: uno, cameras: [cam-a]}
          - {name: uno, cameras: [cam-b]}
        """,
    )

    with pytest.raises(ValueError, match="uno"):
        load_instances(path, CATALOG)


def test_an_instance_without_cameras_or_an_empty_file_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        load_instances(_write(tmp_path, "instances:\n  - {name: vacia, cameras: []}\n"), CATALOG)
    with pytest.raises(ValueError):
        load_instances(_write(tmp_path, "instances: []\n"), CATALOG)


def test_get_instance_by_name_lists_the_available_ones_when_missing(tmp_path):
    path = _write(tmp_path, "instances:\n  - {name: uno, cameras: [cam-a]}\n")

    assert get_instance("uno", path, CATALOG).cameras == ["cam-a"]
    with pytest.raises(KeyError, match="uno"):
        get_instance("otra", path, CATALOG)
