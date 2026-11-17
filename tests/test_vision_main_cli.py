from __future__ import annotations

import sys

import pytest

from monitoreo_urbano.common.instances import InstanceConfig
from monitoreo_urbano.vision_pipeline import main as vision_main


@pytest.fixture
def captured(monkeypatch):
    calls: list[dict] = []
    monkeypatch.setattr(vision_main, "run", lambda camera_ids, **kwargs: calls.append({"cameras": camera_ids, **kwargs}))
    monkeypatch.setattr(
        vision_main,
        "get_instance",
        lambda name: InstanceConfig(name=name, cameras=["cam-a", "cam-b"], device="cuda:1", analysis_fps=6),
    )
    return calls


def _main(monkeypatch, *args: str) -> None:
    monkeypatch.setattr(sys, "argv", ["main", *args])
    vision_main.main()


def test_an_instance_brings_its_cameras_device_and_frequency(monkeypatch, captured):
    _main(monkeypatch, "--instance", "vision-2")

    assert captured == [
        {"cameras": ["cam-a", "cam-b"], "analysis_fps": 6, "preview": None, "device": "cuda:1", "name": "vision-2"}
    ]


def test_the_command_line_frequency_overrides_the_instances(monkeypatch, captured):
    _main(monkeypatch, "--instance", "vision-2", "--analysis-fps", "10")

    assert captured[0]["analysis_fps"] == 10


def test_cameras_can_still_be_given_one_by_one_or_comma_separated(monkeypatch, captured):
    _main(monkeypatch, "--camera-id", "cam-x,cam-y", "--camera-id", "cam-z")

    assert captured == [
        {"cameras": ["cam-x", "cam-y", "cam-z"], "analysis_fps": None, "preview": None, "device": None, "name": "adhoc"}
    ]


@pytest.mark.parametrize("args", [[], ["--instance", "v", "--camera-id", "cam-a"]])
def test_exactly_one_of_instance_or_cameras_is_required(monkeypatch, captured, args):
    with pytest.raises(SystemExit):
        _main(monkeypatch, *args)
    assert captured == []
